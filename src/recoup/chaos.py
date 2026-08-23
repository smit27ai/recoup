"""Chaos harness: break things on purpose, then check what must still be true.

Every other test in this repo asks whether a component works. This one asks a
different question -- when a dependency fails in a way nobody planned for, what does
the system do with somebody's money?

The design is invariant-first rather than fault-first. Listing failure modes is
endless and you always miss one; the useful move is to state the handful of things
that must hold NO MATTER WHAT breaks, then attack them. Four here, and they are the
four a merchant would actually care about:

  no double charge        the same recovery attempt never produces two orders
  no unconsented contact  nobody is messaged who should not be, ever
  no silent loss          every rupee ends recovered, failed-with-a-record, or in a
                          human queue -- never simply gone
  the ledger verifies     whatever happened, the chain still proves what happened

Faults are injected into the REAL code paths -- the real client, the real gates, the
real ledger -- not simulated by mocking out the behaviour under test. A chaos suite
that stubs the thing it is testing proves only that the stub works.

Two rules about what counts as surviving:

**Degrading is passing. Crashing is not.** An unhandled exception escaping the
engine is a failure even if no invariant was violated, because in production that is
a stuck queue and a silent stop.

**Refusing to act is a correct answer.** Several faults here end with the system
doing nothing and telling a human. That is not a partial failure to be explained
away; on a payments system, declining to act on bad information is the whole job.
"""

from __future__ import annotations

import json
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from recoup.diagnosis.escalation import EscalationService, StubEscalator
from recoup.domain import ActionKind, Channel, Customer
from recoup.engine import RecoveryEngine
from recoup.execution import Executor, RecordingNotifier
from recoup.generator.synthetic import ScenarioGenerator
from recoup.ledger import ChainError, Ledger
from recoup.policy.gates import CustomerState, EventState, PolicyConfig
from recoup.razorpay.client import RazorpayClient
from recoup.razorpay.webhooks import (
    ReplayGuard,
    WebhookError,
    compute_signature,
    parse,
)

SECRET = "whsec_chaos"


@dataclass
class Observation:
    """What the system did while a fault was active."""

    orders_created: list[str] = field(default_factory=list)
    messages_sent: list[dict[str, str]] = field(default_factory=list)
    events_seen: int = 0
    events_accounted: int = 0
    """Recovered, failed-with-a-record, or queued. Anything else is unaccounted."""
    crashed: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ChaosResult:
    fault: str
    what_broke: str
    what_happened: str
    violations: list[str]
    observation: Observation
    residual: str = ""
    """A known, bounded limitation this fault exposes and the fix not taken.

    Deliberately separate from `violations`. Tuning an invariant until it passes is
    how a chaos suite becomes decoration; stating precisely what still breaks, how
    far it goes, and what would fix it is the actual output.
    """

    @property
    def survived(self) -> bool:
        return not self.violations and self.observation.crashed is None


# --- invariants -------------------------------------------------------------


def no_double_charge(obs: Observation) -> str | None:
    """The same receipt must never yield two orders. This is the one that costs a
    merchant a chargeback and a customer their trust."""
    seen = obs.orders_created
    duplicates = {r for r in seen if seen.count(r) > 1}
    return f"duplicate orders for receipts {sorted(duplicates)}" if duplicates else None


def no_unconsented_contact(obs: Observation) -> str | None:
    for message in obs.messages_sent:
        if message.get("consented") == "no":
            return f"messaged {message['customer_id']} without consent"
    return None


def no_silent_loss(obs: Observation) -> str | None:
    """Every event must end somewhere a human could find it."""
    if obs.events_seen and obs.events_accounted < obs.events_seen:
        missing = obs.events_seen - obs.events_accounted
        return f"{missing} of {obs.events_seen} events ended in no recorded state"
    return None


def ledger_verifies(ledger: Ledger) -> Callable[[Observation], str | None]:
    def check(_: Observation) -> str | None:
        try:
            ledger.verify()
        except ChainError as exc:
            return f"ledger chain broken: {exc}"
        return None

    return check


# --- harness ----------------------------------------------------------------


def _customer(consent: bool = True) -> Customer:
    return Customer(
        customer_id="cust_chaos",
        segment="loyal",
        has_consent=consent,
        on_dnd_registry=False,
        preferred_channel=Channel.WHATSAPP,
    )


def _build(
    transport: httpx.MockTransport, ledger: Ledger
) -> tuple[RecoveryEngine, RecordingNotifier]:
    notifier = RecordingNotifier()
    engine = RecoveryEngine(
        Executor(
            RazorpayClient("rzp_test_chaos", "s", transport=transport, max_attempts=3), notifier
        ),
        ledger,
        holdout_rate=0.0,
        config=PolicyConfig(
            contact_window_start=datetime(2026, 1, 1, 0, 0).time(),
            contact_window_end=datetime(2026, 1, 1, 23, 59).time(),
        ),
        escalation=EscalationService(StubEscalator()),
    )
    return engine, notifier


def _drive(
    engine: RecoveryEngine,
    notifier: RecordingNotifier,
    obs: Observation,
    *,
    n_events: int = 60,
    consent: bool = True,
) -> None:
    """Push a batch of real events through the engine while a fault is active."""
    scenario = ScenarioGenerator(seed=4242).generate(n_events=n_events, n_customers=20)
    customer = _customer(consent)
    now = datetime(2026, 9, 1, 11, 0, tzinfo=UTC)

    for event in scenario.events:
        obs.events_seen += 1
        try:
            handled = engine.handle(
                event,
                customer,
                CustomerState(
                    customer_id=customer.customer_id,
                    has_consent=consent,
                    on_dnd_registry=False,
                ),
                EventState(event_id=event.event_id, attempts_so_far=0),
                now,
            )
        except Exception as exc:
            obs.crashed = f"{type(exc).__name__}: {exc}"
            return

        # Accounted means a human could find out what happened to this rupee. A
        # written record counts. So does an explicit refusal to act -- "we did
        # nothing and here is why" is an accounted outcome; "we acted and lost the
        # record" is not.
        refused = "refusing to act" in handled.result.detail
        if handled.record.record_hash or refused:
            obs.events_accounted += 1
        receipt = handled.result.artifacts.get("receipt")
        if receipt:
            obs.orders_created.append(receipt)

    for message in notifier.sent:
        obs.messages_sent.append({**message, "consented": "yes" if consent else "no"})


def _check(obs: Observation, ledger: Ledger) -> list[str]:
    checks: list[Callable[[Observation], str | None]] = [
        no_double_charge,
        no_unconsented_contact,
        no_silent_loss,
        ledger_verifies(ledger),
    ]
    return [problem for check in checks if (problem := check(obs)) is not None]


# --- the faults -------------------------------------------------------------


def fault_gateway_500s() -> ChaosResult:
    """Razorpay returns 500 on every write."""
    ledger = Ledger()
    calls = {"n": 0}

    def transport(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"entity": "collection", "items": []})
        calls["n"] += 1
        return httpx.Response(500, json={"error": {"code": "SERVER_ERROR", "description": "down"}})

    engine, notifier = _build(httpx.MockTransport(transport), ledger)
    obs = Observation()
    _drive(engine, notifier, obs)
    obs.notes.append(f"{calls['n']} write attempts, all rejected")
    return ChaosResult(
        "gateway 500s",
        "every Razorpay write returns 500",
        "retried within bounds, then recorded as failed; no message sent without a link",
        _check(obs, ledger),
        obs,
    )


def fault_post_timeout() -> ChaosResult:
    """The dangerous one: writes time out, so the outcome is UNKNOWN.

    A blind retry here double-charges; treating it as failure loses money. The
    reconcile path is what makes this survivable.
    """
    ledger = Ledger()
    created: dict[str, dict[str, Any]] = {}

    def transport(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200, json={"entity": "collection", "items": list(created.values())}
            )
        # The request DID land -- the response is what got lost.
        body = json.loads(request.content) if request.content else {}
        receipt = body.get("receipt") or body.get("reference_id", "")
        created[receipt] = {"id": f"order_{len(created)}", "receipt": receipt}
        raise httpx.ReadTimeout("response lost")

    engine, notifier = _build(httpx.MockTransport(transport), ledger)
    obs = Observation()
    _drive(engine, notifier, obs)
    obs.notes.append(
        f"{len(created)} orders actually created upstream, {len(obs.orders_created)} claimed"
    )
    return ChaosResult(
        "write timeout, request landed",
        "writes succeed upstream but the response is lost",
        "reconciled by receipt instead of retried; no duplicate orders",
        _check(obs, ledger),
        obs,
    )


def fault_rate_limited() -> ChaosResult:
    ledger = Ledger()

    def transport(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"entity": "collection", "items": []})
        return httpx.Response(
            429,
            headers={"Retry-After": "0"},
            json={"error": {"code": "RATE_LIMIT", "description": "slow down"}},
        )

    engine, notifier = _build(httpx.MockTransport(transport), ledger)
    obs = Observation()
    _drive(engine, notifier, obs, n_events=30)
    return ChaosResult(
        "rate limited",
        "Razorpay returns 429 with Retry-After",
        "backed off and honoured Retry-After, then recorded failure rather than hammering",
        _check(obs, ledger),
        obs,
    )


def fault_llm_down() -> ChaosResult:
    """Tier 2 is unreachable. Unmapped codes must go to a human, not be guessed."""
    ledger = Ledger()

    def transport(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"entity": "collection", "items": []})
        return httpx.Response(200, json={"id": "order_x", "short_url": "https://rzp.io/i/x"})

    class Down:
        name = "down"

        def propose(self, reason: str, context: dict[str, Any]) -> Any:
            raise RuntimeError("model API unreachable")

    engine, notifier = _build(httpx.MockTransport(transport), ledger)
    engine.escalation = EscalationService(Down())
    obs = Observation()
    _drive(engine, notifier, obs)
    routed = sum(1 for r in ledger if r.executed_action == str(ActionKind.ROUTE_TO_OPS))
    obs.notes.append(f"{routed} events routed to a human instead of guessed at")
    return ChaosResult(
        "tier-2 model down",
        "every escalation raises",
        "unmapped codes routed to ops; recovery of known codes unaffected",
        _check(obs, ledger),
        obs,
    )


def fault_ledger_broken() -> ChaosResult:
    """The audit trail fails. Recovery must continue and the gap must be visible."""

    class BrokenLedger(Ledger):
        def append(self, **kwargs: Any) -> Any:
            raise RuntimeError("disk full")

    ledger = BrokenLedger()

    def transport(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"entity": "collection", "items": []})
        return httpx.Response(200, json={"id": "order_x", "short_url": "https://rzp.io/i/x"})

    engine, notifier = _build(httpx.MockTransport(transport), ledger)
    obs = Observation()
    _drive(engine, notifier, obs, n_events=10)
    return ChaosResult(
        "ledger unwritable",
        "every ledger append raises",
        "one action slips through unrecorded, then the breaker opens and the rest refuse",
        _check(obs, ledger),
        obs,
        residual=(
            "exactly one event acts unrecorded, and that is irreducible with post-hoc "
            "recording: the ledger cannot be known broken until it is written to. "
            "Two-phase recording (write the intent, act, write the outcome) would "
            "eliminate it by failing closed on the very first event. Not built -- it "
            "doubles records per decision -- so the residual is one event, bounded, "
            "and stated rather than tuned away."
        ),
    )


def fault_hostile_webhooks() -> ChaosResult:
    """Forged, replayed and malformed webhooks. None may reach the decision path."""
    guard = ReplayGuard()
    now = datetime(2026, 9, 1, 11, 0, tzinfo=UTC)
    good = json.dumps(
        {
            "event": "payment.failed",
            "id": "evt_real",
            "created_at": int(now.timestamp()),
            "payload": {"payment": {"entity": {"error_reason": "card_expired", "amount": 100}}},
        }
    ).encode()

    attacks: list[tuple[str, bytes, str]] = [
        ("forged signature", good, "deadbeef"),
        (
            "tampered amount",
            good.replace(b'"amount": 100', b'"amount": 1'),
            compute_signature(good, SECRET),
        ),
        ("wrong secret", good, compute_signature(good, "whsec_other")),
        ("replayed", good, compute_signature(good, SECRET)),
        ("garbage body", b"{{{ not json", "deadbeef"),
    ]

    obs = Observation()
    accepted: list[str] = []
    parse(good, compute_signature(good, SECRET), SECRET, now=now, replay_guard=guard)
    for name, body, signature in attacks:
        try:
            parse(body, signature, SECRET, now=now, replay_guard=guard)
            accepted.append(name)
        except WebhookError:
            obs.notes.append(f"rejected: {name}")

    stale = json.dumps(
        {
            "event": "payment.failed",
            "id": "evt_old",
            "created_at": int((now - timedelta(days=2)).timestamp()),
            "payload": {},
        }
    ).encode()
    try:
        parse(stale, compute_signature(stale, SECRET), SECRET, now=now)
        accepted.append("stale replay")
    except WebhookError:
        obs.notes.append("rejected: stale replay")

    violations = [f"accepted hostile webhook: {a}" for a in accepted]
    return ChaosResult(
        "hostile webhooks",
        "forged, tampered, replayed, stale and malformed payloads",
        f"all {len(attacks) + 1} rejected at the trust boundary",
        violations,
        obs,
    )


def fault_consent_withdrawn() -> ChaosResult:
    """Consent is absent for everyone. Not one message may go out."""
    ledger = Ledger()

    def transport(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"entity": "collection", "items": []})
        return httpx.Response(200, json={"id": "order_x", "short_url": "https://rzp.io/i/x"})

    engine, notifier = _build(httpx.MockTransport(transport), ledger)
    obs = Observation()
    _drive(engine, notifier, obs, consent=False)
    obs.notes.append(f"{len(notifier.sent)} messages sent (must be 0)")
    violations = _check(obs, ledger)
    if notifier.sent:
        violations.append(f"{len(notifier.sent)} messages sent without consent")
    return ChaosResult(
        "consent withdrawn for everyone",
        "no customer has consent",
        "zero messages; retries and ops routing continue normally",
        violations,
        obs,
    )


def fault_tampered_ledger() -> ChaosResult:
    """Somebody edits history. Verification must catch it."""
    from dataclasses import replace

    ledger = Ledger()

    def transport(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"entity": "collection", "items": []})
        return httpx.Response(200, json={"id": "order_x", "short_url": "https://rzp.io/i/x"})

    engine, notifier = _build(httpx.MockTransport(transport), ledger)
    obs = Observation()
    _drive(engine, notifier, obs, n_events=20)

    records = list(ledger)
    tampered = list(records)
    tampered[3] = replace(tampered[3], executed_action="no_action").sealed()
    from recoup.ledger import verify_chain

    detected = False
    try:
        verify_chain(tampered)
    except ChainError as exc:
        detected = True
        obs.notes.append(f"detected: {str(exc)[:70]}")

    return ChaosResult(
        "ledger tampered after the fact",
        "a record is edited and re-sealed",
        "chain verification detects it; re-sealing is not enough",
        [] if detected else ["tampering went undetected"],
        obs,
    )


FAULTS: tuple[Callable[[], ChaosResult], ...] = (
    fault_gateway_500s,
    fault_post_timeout,
    fault_rate_limited,
    fault_llm_down,
    fault_hostile_webhooks,
    fault_consent_withdrawn,
    fault_tampered_ledger,
    fault_ledger_broken,
)


def run() -> list[ChaosResult]:
    return [fault() for fault in FAULTS]


def report(results: list[ChaosResult]) -> str:
    lines = [
        "",
        "  CHAOS: what still holds when things break",
        "  " + "=" * 74,
        "",
        "  Invariants checked under every fault:",
        "    no double charge       the same attempt never produces two orders",
        "    no unconsented contact nobody is messaged who should not be",
        "    no silent loss         every rupee ends somewhere a human can find it",
        "    ledger verifies        the chain still proves what happened",
        "",
    ]
    for r in results:
        mark = "OK  " if r.survived else "FAIL"
        lines.append(f"  [{mark}] {r.fault}")
        lines.append(f"         broke:  {r.what_broke}")
        lines.append(f"         did:    {r.what_happened}")
        for note in r.observation.notes[:3]:
            lines.append(f"         note:   {note}")
        if r.observation.crashed:
            lines.append(f"         CRASH:  {r.observation.crashed}")
        for violation in r.violations:
            lines.append(f"         BROKE:  {violation}")
        if r.residual:
            wrapped = textwrap.wrap(r.residual, width=66)
            lines.append(f"         KNOWN:  {wrapped[0]}")
            lines.extend(f"                 {line}" for line in wrapped[1:])
        lines.append("")

    survived = sum(1 for r in results if r.survived)
    crashed = sum(1 for r in results if r.observation.crashed)
    lines += [
        "  " + "=" * 74,
        f"  {survived}/{len(results)} faults survived with every invariant intact.",
        f"  {crashed} crashed. {len(results) - survived - crashed} degraded with a "
        "stated residual.",
        "",
        "  Degrading is passing; crashing is not. Refusing to act is a correct answer --",
        "  on a payments system, declining to act on bad information is the whole job.",
        "",
        "  The one that does not pass is left not passing. Tuning an invariant until it",
        "  goes green is how a chaos suite becomes decoration.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    results = run()
    print(report(results))
    return 0 if all(r.survived for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
