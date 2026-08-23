"""End-to-end tests for the execution path.

These are the tests that would catch a regression nobody else would: the layers are
individually correct and wired together wrongly. Most of them assert on what did
NOT happen -- no message sent, no second order, no contact in the holdout.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import httpx
import pytest

from recoup.domain import ActionKind, Arm, AtRiskEvent, Channel, Customer, RiskKind
from recoup.engine import RecoveryEngine
from recoup.execution import ExecutionStatus, Executor, RecordingNotifier, WorkQueue
from recoup.ledger import Ledger, verify_chain
from recoup.policy.gates import IST, CustomerState, EventState, PolicyConfig
from recoup.razorpay.client import RazorpayClient

MIDDAY = datetime(2026, 9, 1, 11, 0, tzinfo=IST)
NIGHT = datetime(2026, 9, 1, 2, 30, tzinfo=IST)

ORDER = {"id": "order_test123", "entity": "order", "amount": 99900, "status": "created"}
LINK = {"id": "plink_test123", "short_url": "https://rzp.io/i/abc123"}


class Stub:
    """Answers by endpoint rather than by sequence, so tests do not depend on the
    exact number of calls the engine happens to make."""

    def __init__(self, **overrides: Any) -> None:
        self.overrides = overrides
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        for key, value in self.overrides.items():
            if key in path:
                if isinstance(value, Exception):
                    raise value
                return value
        if "orders" in path and request.method == "GET":
            return _json(200, {"entity": "collection", "count": 0, "items": []})
        if "orders" in path:
            return _json(200, ORDER)
        if "payment_links" in path:
            return _json(200, LINK)
        return _json(200, {})

    def posts_to(self, fragment: str) -> list[httpx.Request]:
        return [r for r in self.requests if fragment in r.url.path and r.method == "POST"]


def _json(status: int, body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status, content=json.dumps(body), headers={"content-type": "application/json"}
    )


def _engine(
    stub: Stub | None = None, *, holdout_rate: float = 0.0, config: PolicyConfig | None = None
) -> tuple[RecoveryEngine, Stub, RecordingNotifier, Ledger]:
    stub = stub or Stub()
    client = RazorpayClient(
        "rzp_test_key", "secret", transport=httpx.MockTransport(stub.handler), max_attempts=2
    )
    notifier = RecordingNotifier()
    ledger = Ledger()
    engine = RecoveryEngine(
        Executor(client, notifier), ledger, holdout_rate=holdout_rate, config=config
    )
    return engine, stub, notifier, ledger


def _event(
    reason: str = "insufficient_funds", amount: int = 99900, attempt: int = 1
) -> AtRiskEvent:
    return AtRiskEvent(
        event_id="evt_0001",
        customer_id="cust_0001",
        kind=RiskKind.FAILED_PAYMENT,
        amount_paise=amount,
        occurred_at=MIDDAY,
        error_reason=reason,
        method="card",
        attempt_number=attempt,
    )


def _customer(**kw: Any) -> Customer:
    base: dict[str, Any] = {
        "customer_id": "cust_0001",
        "segment": "loyal",
        "has_consent": True,
        "on_dnd_registry": False,
        "preferred_channel": Channel.WHATSAPP,
        "language": "en",
    }
    return Customer(**{**base, **kw})


def _states(**kw: Any) -> tuple[CustomerState, EventState]:
    cs = CustomerState(
        customer_id="cust_0001",
        has_consent=kw.get("has_consent", True),
        on_dnd_registry=kw.get("on_dnd_registry", False),
        last_contact_at=kw.get("last_contact_at"),
        contacts_in_window=kw.get("contacts_in_window", ()),
    )
    es = EventState(
        event_id="evt_0001",
        attempts_so_far=kw.get("attempts_so_far", 0),
        dispute_open=kw.get("dispute_open", False),
        promise_to_pay_until=kw.get("promise_to_pay_until"),
    )
    return cs, es


def _run(engine: RecoveryEngine, event: AtRiskEvent, now: datetime = MIDDAY, **kw: Any):
    cs, es = _states(**kw)
    return engine.handle(event, _customer(), cs, es, now)


# --- the happy path ---------------------------------------------------------


def test_insufficient_funds_schedules_a_silent_retry() -> None:
    """FUNDS means the money will exist later. Retry, do not message."""
    engine, _stub, notifier, _ = _engine()
    handled = _run(engine, _event("insufficient_funds"))

    assert handled.intent.action is ActionKind.RETRY_SCHEDULED
    assert handled.result.status is ExecutionStatus.DONE
    assert handled.result.artifacts["order_id"] == "order_test123"
    assert notifier.sent == [], "a silent retry must not message anybody"


def test_expired_card_asks_for_a_different_instrument() -> None:
    """Retrying is futile, so the only thing that recovers this is a new method."""
    engine, stub, notifier, _ = _engine()
    handled = _run(engine, _event("card_expired"))

    assert handled.intent.action is ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH
    assert handled.result.status is ExecutionStatus.DONE
    assert len(notifier.sent) == 1
    assert "different method" in notifier.sent[0]["body"]
    assert notifier.sent[0]["link"] == "https://rzp.io/i/abc123"
    assert stub.posts_to("orders") == [], "must not create an order it cannot charge"


def test_our_own_bug_never_reaches_the_customer() -> None:
    """invalid_order_id is our fault. Dunning someone for it is indefensible."""
    engine, stub, notifier, _ = _engine()
    handled = _run(engine, _event("invalid_order_id"))

    assert handled.executed is ActionKind.ROUTE_TO_OPS
    assert handled.result.status is ExecutionStatus.QUEUED
    assert notifier.sent == []
    assert stub.posts_to("payment_links") == []


def test_unmapped_reason_escalates_rather_than_guessing() -> None:
    engine, _, notifier, _ = _engine()
    handled = _run(engine, _event("some_reason_from_2027"))

    assert handled.intent.diagnosis is None
    assert handled.executed is ActionKind.ROUTE_TO_OPS
    assert notifier.sent == []


# --- gates actually bite ----------------------------------------------------


def test_quiet_hours_blocks_the_message_but_not_a_retry() -> None:
    engine, _, notifier, _ = _engine()
    blocked = _run(engine, _event("card_expired"), now=NIGHT)
    assert blocked.executed is ActionKind.NO_ACTION
    assert notifier.sent == []

    engine2, _, _notifier2, _ = _engine()
    allowed = _run(engine2, _event("insufficient_funds"), now=NIGHT)
    assert allowed.executed is ActionKind.RETRY_SCHEDULED
    assert allowed.result.status is ExecutionStatus.DONE


def test_no_consent_stops_contact_before_any_api_call() -> None:
    """A blocked message must not even raise a payment link -- that is a wasted
    write against a customer we are not allowed to talk to."""
    engine, stub, notifier, _ = _engine()
    handled = _run(engine, _event("card_expired"), has_consent=False)

    assert handled.executed is ActionKind.NO_ACTION
    assert notifier.sent == []
    assert stub.posts_to("payment_links") == []


def test_promise_to_pay_stops_everything() -> None:
    engine, stub, _notifier, _ = _engine()
    handled = _run(
        engine, _event("insufficient_funds"), promise_to_pay_until=MIDDAY + timedelta(days=3)
    )
    assert handled.executed is ActionKind.NO_ACTION
    assert stub.posts_to("orders") == []


def test_high_value_is_parked_for_a_human_not_dropped() -> None:
    engine, _stub, notifier, _ = _engine()
    handled = _run(engine, _event("card_expired", amount=50_000_00))

    assert handled.executed is ActionKind.QUEUED_FOR_APPROVAL
    assert handled.result.status is ExecutionStatus.QUEUED
    assert engine.executor.approval_queue.total_paise == 50_000_00
    assert notifier.sent == []


# --- holdout integrity ------------------------------------------------------


def test_holdout_never_acts_even_when_all_gates_allow() -> None:
    """Contaminating the control arm invalidates every incrementality number."""
    engine, stub, notifier, _ledger = _engine(holdout_rate=1.0)
    handled = _run(engine, _event("card_expired"))

    assert handled.intent.arm is Arm.HOLDOUT
    assert handled.executed is ActionKind.NO_ACTION
    assert notifier.sent == []
    assert stub.posts_to("payment_links") == []
    assert handled.verdict.allowed, "gates allowed it; the holdout is what stopped it"


def test_holdout_still_records_what_it_would_have_done() -> None:
    """A hole in the data is not a counterfactual."""
    engine, _, _, _ = _engine(holdout_rate=1.0)
    handled = _run(engine, _event("card_expired"))
    assert handled.record.intended_action == "nudge_with_instrument_switch"
    assert handled.record.executed_action == "no_action"
    assert handled.record.arm == "holdout"


# --- uncertain outcomes -----------------------------------------------------


def test_timeout_that_actually_created_the_order_is_reconciled_not_repeated() -> None:
    """The double-charge scenario. Must resolve by looking, not by retrying."""
    stub = Stub()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        stub.requests.append(request)
        if request.method == "POST" and "orders" in request.url.path:
            calls["n"] += 1
            raise httpx.ReadTimeout("lost the response")
        if request.method == "GET" and "orders" in request.url.path:
            return _json(
                200,
                {
                    "entity": "collection",
                    "count": 1,
                    "items": [{**ORDER, "receipt": "rcp-order-evt_0001-1"}],
                },
            )
        return _json(200, {})

    client = RazorpayClient(
        "rzp_test_key", "secret", transport=httpx.MockTransport(handler), max_attempts=2
    )
    engine = RecoveryEngine(Executor(client, RecordingNotifier()), Ledger(), holdout_rate=0.0)
    handled = _run(engine, _event("insufficient_funds"))

    assert handled.result.status is ExecutionStatus.RECONCILED
    assert "not repeated" in handled.result.detail
    assert calls["n"] == 1, "must not retry a POST whose outcome is unknown"


def test_timeout_that_created_nothing_is_reported_as_failed() -> None:
    """Only after confirming nothing exists is a later retry safe."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            raise httpx.ReadTimeout("gone")
        return _json(200, {"entity": "collection", "count": 0, "items": []})

    client = RazorpayClient(
        "rzp_test_key", "secret", transport=httpx.MockTransport(handler), max_attempts=2
    )
    engine = RecoveryEngine(Executor(client, RecordingNotifier()), Ledger(), holdout_rate=0.0)
    handled = _run(engine, _event("insufficient_funds"))

    assert handled.result.status is ExecutionStatus.FAILED
    assert "nothing was created" in handled.result.detail


def test_uncertain_payment_link_sends_no_message() -> None:
    """A message telling someone to pay, without a working link, is worse than
    silence."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "payment_links" in request.url.path:
            raise httpx.ReadTimeout("unknown")
        return _json(200, {})

    client = RazorpayClient(
        "rzp_test_key", "secret", transport=httpx.MockTransport(handler), max_attempts=2
    )
    notifier = RecordingNotifier()
    engine = RecoveryEngine(Executor(client, notifier), Ledger(), holdout_rate=0.0)
    handled = _run(engine, _event("card_expired"))

    assert handled.result.status is ExecutionStatus.UNRESOLVED
    assert handled.result.needs_human
    assert notifier.sent == [], "never send a message we could not attach a link to"


# --- the ledger sees everything ---------------------------------------------


def test_every_path_writes_a_record_including_the_boring_ones() -> None:
    """'Why did nobody chase this invoice' is the question an audit trail must
    answer, and it is the one a log of actions-only goes silent on."""
    engine, _, _, ledger = _engine()
    for reason, now in [
        ("insufficient_funds", MIDDAY),
        ("card_expired", NIGHT),
        ("invalid_order_id", MIDDAY),
        ("gateway_technical_error", MIDDAY),
    ]:
        _run(engine, _event(reason), now=now)
    assert len(ledger) == 4
    ledger.verify()


def test_ledger_record_carries_the_execution_artifacts() -> None:
    engine, _, _, _ = _engine()
    handled = _run(engine, _event("card_expired"))
    meta = handled.record.metadata
    assert meta["execution_status"] == "done"
    assert meta["payment_link_id"] == "plink_test123"
    assert meta["short_url"] == "https://rzp.io/i/abc123"
    assert meta["message_id"] == "msg_000000"


def test_chain_survives_a_full_batch() -> None:
    engine, _, _, ledger = _engine()
    for i in range(25):
        reason = ["insufficient_funds", "card_expired", "invalid_order_id"][i % 3]
        _run(engine, _event(reason))
    verify_chain(list(ledger))
    assert len(ledger) == 25


def test_settlement_appends_rather_than_editing() -> None:
    """An outcome arriving late is a fact added to history, never a correction
    applied to it -- editing would break the chain."""
    engine, _, _, ledger = _engine()
    handled = _run(engine, _event("card_expired"))
    before = handled.record.record_hash

    settled = engine.settle("evt_0001", recovered=True, now=MIDDAY + timedelta(days=1))
    assert settled is not None
    assert settled.recovered is True
    assert settled.metadata["settlement_for"] == before
    assert len(ledger) == 2
    ledger.verify()
    assert next(iter(ledger)).record_hash == before, "the original record is untouched"


def test_settling_an_unknown_event_is_a_noop() -> None:
    engine, _, _, ledger = _engine()
    assert engine.settle("evt_nope", recovered=True, now=MIDDAY) is None
    assert len(ledger) == 0


# --- queues -----------------------------------------------------------------


def test_ops_and_approval_are_separate_queues() -> None:
    """Different people, different authority."""
    ops, approvals = WorkQueue("ops"), WorkQueue("approval")
    stub = Stub()
    client = RazorpayClient(
        "rzp_test_key", "secret", transport=httpx.MockTransport(stub.handler), max_attempts=2
    )
    engine = RecoveryEngine(
        Executor(client, RecordingNotifier(), ops_queue=ops, approval_queue=approvals),
        Ledger(),
        holdout_rate=0.0,
    )
    _run(engine, _event("invalid_order_id"))
    _run(engine, _event("card_expired", amount=50_000_00))

    assert len(ops) == 1
    assert len(approvals) == 1
    assert ops.items[0]["event_id"] == "evt_0001"


def test_empty_queues_are_not_silently_replaced() -> None:
    """WorkQueue defines __len__, so an empty one is falsy. Same trap that hit
    IdempotencyStore; `or` here would discard the caller's queue."""
    ops = WorkQueue("ops")
    stub = Stub()
    client = RazorpayClient(
        "rzp_test_key", "secret", transport=httpx.MockTransport(stub.handler), max_attempts=2
    )
    executor = Executor(client, RecordingNotifier(), ops_queue=ops)
    assert executor.ops_queue is ops


# --- the full trace ---------------------------------------------------------


def test_explain_renders_the_whole_journey() -> None:
    engine, _, _, _ = _engine()
    text = _run(engine, _event("card_expired")).explain()
    for fragment in ("card_expired", "INSTRUMENT_INVALID", "nudge_with_instrument_switch", "allow"):
        assert fragment in text


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("insufficient_funds", ActionKind.RETRY_SCHEDULED),
        ("gateway_technical_error", ActionKind.RETRY_NOW),
        ("card_expired", ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH),
        ("invalid_vpa", ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH),
        ("invalid_order_id", ActionKind.ROUTE_TO_OPS),
        ("payment_method_not_enabled", ActionKind.ROUTE_TO_OPS),
    ],
)
def test_root_cause_drives_the_action(reason: str, expected: ActionKind) -> None:
    engine, _, _, _ = _engine()
    assert _run(engine, _event(reason)).intent.action is expected
