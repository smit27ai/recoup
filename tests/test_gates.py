"""Gate tests.

The property that matters most here is not that a compliant action passes -- it is
that a non-compliant one is blocked for EVERY reason it violates, so the audit
ledger records the full picture rather than whichever gate happened to run first.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from recoup.policy.gates import (
    IST,
    CustomerState,
    Disposition,
    EventState,
    GateContext,
    GateID,
    PolicyConfig,
    ProposedAction,
    evaluate,
)

MIDDAY = datetime(2026, 9, 1, 11, 0, tzinfo=IST)
MIDNIGHT = datetime(2026, 9, 1, 2, 30, tzinfo=IST)


def _action(**kw: object) -> ProposedAction:
    base: dict[str, object] = {
        "event_id": "evt_1",
        "customer_id": "cust_1",
        "kind": "whatsapp_nudge",
        "is_contact": True,
        "amount_paise": 99_900,
        "idempotency_key": "evt_1:whatsapp_nudge:1",
    }
    base.update(kw)
    return ProposedAction(**base)  # type: ignore[arg-type]


def _customer(**kw: object) -> CustomerState:
    base: dict[str, object] = {
        "customer_id": "cust_1",
        "has_consent": True,
        "on_dnd_registry": False,
    }
    base.update(kw)
    return CustomerState(**base)  # type: ignore[arg-type]


def _event(**kw: object) -> EventState:
    base: dict[str, object] = {"event_id": "evt_1", "attempts_so_far": 0}
    base.update(kw)
    return EventState(**base)  # type: ignore[arg-type]


def _ctx(
    action: ProposedAction | None = None,
    customer: CustomerState | None = None,
    event: EventState | None = None,
    now: datetime = MIDDAY,
    config: PolicyConfig | None = None,
) -> GateContext:
    return GateContext(
        action=action or _action(),
        customer=customer or _customer(),
        event=event or _event(),
        now=now,
        config=config or PolicyConfig(),
    )


def test_clean_contact_is_allowed() -> None:
    verdict = evaluate(_ctx())
    assert verdict.allowed, verdict.explain()
    assert verdict.disposition is Disposition.ALLOW


def test_every_gate_runs_on_every_evaluation() -> None:
    verdict = evaluate(_ctx())
    assert {r.gate for r in verdict.results} == set(GateID)


def test_quiet_hours_blocks_night_contact() -> None:
    verdict = evaluate(_ctx(now=MIDNIGHT))
    assert not verdict.allowed
    assert GateID.QUIET_HOURS in {r.gate for r in verdict.denials}


def test_quiet_hours_does_not_block_a_silent_retry() -> None:
    """A card retry at 02:30 disturbs nobody. Only CONTACT is time-restricted."""
    verdict = evaluate(_ctx(action=_action(kind="retry_charge", is_contact=False), now=MIDNIGHT))
    assert verdict.allowed, verdict.explain()


def test_no_consent_blocks_contact() -> None:
    verdict = evaluate(_ctx(customer=_customer(has_consent=False)))
    assert GateID.CONSENT in {r.gate for r in verdict.denials}


def test_dnd_blocks_contact() -> None:
    verdict = evaluate(_ctx(customer=_customer(on_dnd_registry=True)))
    assert GateID.DND in {r.gate for r in verdict.denials}


def test_promise_to_pay_stops_recovery() -> None:
    verdict = evaluate(_ctx(event=_event(promise_to_pay_until=MIDDAY + timedelta(days=3))))
    assert GateID.STOPPING_RULE in {r.gate for r in verdict.denials}


def test_open_dispute_stops_recovery() -> None:
    verdict = evaluate(_ctx(event=_event(dispute_open=True)))
    assert GateID.STOPPING_RULE in {r.gate for r in verdict.denials}


def test_attempt_cap_stops_recovery() -> None:
    verdict = evaluate(_ctx(event=_event(attempts_so_far=4)))
    assert GateID.STOPPING_RULE in {r.gate for r in verdict.denials}


def test_contact_budget_exhausted() -> None:
    recent = tuple(MIDDAY - timedelta(days=d) for d in (1, 2, 3))
    verdict = evaluate(_ctx(customer=_customer(contacts_in_window=recent)))
    assert GateID.CONTACT_BUDGET in {r.gate for r in verdict.denials}


def test_contacts_outside_the_window_do_not_count() -> None:
    stale = tuple(MIDDAY - timedelta(days=d) for d in (10, 20, 30))
    verdict = evaluate(_ctx(customer=_customer(contacts_in_window=stale)))
    assert verdict.allowed, verdict.explain()


def test_fatigue_blocks_same_day_second_contact() -> None:
    verdict = evaluate(_ctx(customer=_customer(last_contact_at=MIDDAY - timedelta(hours=2))))
    assert GateID.FATIGUE in {r.gate for r in verdict.denials}


def test_replayed_idempotency_key_is_blocked() -> None:
    key = "evt_1:whatsapp_nudge:1"
    verdict = evaluate(_ctx(event=_event(executed_idempotency_keys=frozenset({key}))))
    assert GateID.IDEMPOTENCY in {r.gate for r in verdict.denials}


def test_missing_idempotency_key_is_blocked() -> None:
    verdict = evaluate(_ctx(action=_action(idempotency_key="")))
    assert GateID.IDEMPOTENCY in {r.gate for r in verdict.denials}


def test_large_discount_needs_approval_not_denial() -> None:
    verdict = evaluate(_ctx(action=_action(discount_bps=2500)))
    assert verdict.disposition is Disposition.NEEDS_APPROVAL
    assert not verdict.denials


def test_high_value_action_needs_approval() -> None:
    verdict = evaluate(_ctx(action=_action(amount_paise=50_000_00)))
    assert verdict.disposition is Disposition.NEEDS_APPROVAL


def test_denial_beats_approval() -> None:
    """A high-value action at 2am is denied, not queued for a human to rubber-stamp."""
    verdict = evaluate(_ctx(action=_action(amount_paise=50_000_00), now=MIDNIGHT))
    assert verdict.disposition is Disposition.DENY


def test_all_violations_are_recorded_not_just_the_first() -> None:
    """The whole point of not short-circuiting."""
    verdict = evaluate(
        _ctx(
            customer=_customer(
                has_consent=False,
                on_dnd_registry=True,
                last_contact_at=MIDDAY - timedelta(minutes=30),
            ),
            event=_event(attempts_so_far=9, dispute_open=True),
            now=MIDNIGHT,
        )
    )
    denied = {r.gate for r in verdict.denials}
    assert denied >= {
        GateID.CONSENT,
        GateID.DND,
        GateID.QUIET_HOURS,
        GateID.FATIGUE,
        GateID.STOPPING_RULE,
    }
    assert len(verdict.denials) >= 5


@pytest.mark.parametrize("hour", [0, 3, 7, 19, 21, 23])
def test_hours_outside_window_always_blocked(hour: int) -> None:
    now = datetime(2026, 9, 1, hour, 0, tzinfo=IST)
    verdict = evaluate(_ctx(now=now))
    assert GateID.QUIET_HOURS in {r.gate for r in verdict.denials}


@pytest.mark.parametrize("hour", [8, 12, 15, 18])
def test_hours_inside_window_allowed(hour: int) -> None:
    now = datetime(2026, 9, 1, hour, 0, tzinfo=IST)
    verdict = evaluate(_ctx(now=now))
    assert verdict.allowed, verdict.explain()
