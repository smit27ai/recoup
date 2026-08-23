"""Ledger tests.

The tamper tests are the ones that matter. An audit trail nobody has tried to break
is a log file with extra steps.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pytest

from recoup.diagnosis.taxonomy import diagnose
from recoup.domain import ActionKind, Arm, AtRiskEvent, RiskKind
from recoup.ledger import (
    GENESIS,
    ChainError,
    Ledger,
    diff_replay,
    load,
    replay,
    verify_chain,
)
from recoup.policy.gates import (
    IST,
    CustomerState,
    EventState,
    GateContext,
    ProposedAction,
    evaluate,
)

NOW = datetime(2026, 9, 1, 11, 0, tzinfo=IST)


def _event(i: int = 0, reason: str = "insufficient_funds", amount: int = 99_900) -> AtRiskEvent:
    return AtRiskEvent(
        event_id=f"evt_{i:04d}",
        customer_id=f"cust_{i % 3:04d}",
        kind=RiskKind.FAILED_PAYMENT,
        amount_paise=amount,
        occurred_at=NOW,
        error_reason=reason,
        method="card",
    )


def _verdict(event: AtRiskEvent, *, consent: bool = True, now: datetime = NOW):
    return evaluate(
        GateContext(
            action=ProposedAction(
                event_id=event.event_id,
                customer_id=event.customer_id,
                kind="nudge",
                is_contact=True,
                amount_paise=event.amount_paise,
                idempotency_key=f"{event.event_id}:nudge:1",
            ),
            customer=CustomerState(
                customer_id=event.customer_id, has_consent=consent, on_dnd_registry=False
            ),
            event=EventState(event_id=event.event_id, attempts_so_far=0),
            now=now,
        )
    )


def _fill(ledger: Ledger, n: int = 5) -> None:
    for i in range(n):
        ev = _event(i)
        v = _verdict(ev)
        ledger.append(
            event=ev,
            diagnosis=diagnose(ev.error_reason),
            intended=ActionKind.NUDGE,
            verdict=v,
            executed=ActionKind.NUDGE if v.allowed else ActionKind.NO_ACTION,
            arm=Arm.TREATMENT,
            decided_at=NOW + timedelta(minutes=i),
            recovered=i % 2 == 0,
            policy_version="v1",
            taxonomy_version="2026-08-23",
        )


def test_chain_verifies() -> None:
    led = Ledger()
    _fill(led, 10)
    led.verify()
    assert len(led) == 10


def test_first_record_anchors_to_genesis() -> None:
    led = Ledger()
    _fill(led, 1)
    assert next(iter(led)).prev_hash == GENESIS


def test_each_record_chains_to_its_predecessor() -> None:
    led = Ledger()
    _fill(led, 6)
    records = list(led)
    for prev, cur in pairwise(records):
        assert cur.prev_hash == prev.record_hash


def test_editing_a_record_breaks_the_chain() -> None:
    """The whole point. Change a decision after the fact and verification fails."""
    led = Ledger()
    _fill(led, 5)
    records = list(led)
    records[2] = replace(records[2], executed_action="no_action")
    with pytest.raises(ChainError, match="do not match its hash"):
        verify_chain(records)


def test_editing_a_gate_reason_breaks_the_chain() -> None:
    """Rewriting WHY something was blocked must be as detectable as rewriting what."""
    led = Ledger()
    _fill(led, 4)
    records = list(led)
    gates = list(records[1].gates)
    gates[0] = replace(gates[0], reason="looked fine to me")
    records[1] = replace(records[1], gates=tuple(gates))
    with pytest.raises(ChainError):
        verify_chain(records)


def test_deleting_a_record_breaks_the_chain() -> None:
    """Quietly dropping an embarrassing decision must not verify."""
    led = Ledger()
    _fill(led, 6)
    records = list(led)
    del records[3]
    with pytest.raises(ChainError):
        verify_chain(records)


def test_reordering_records_breaks_the_chain() -> None:
    led = Ledger()
    _fill(led, 6)
    records = list(led)
    records[2], records[4] = records[4], records[2]
    with pytest.raises(ChainError):
        verify_chain(records)


def test_resealing_an_edited_record_still_breaks_the_chain() -> None:
    """A tamperer who recomputes the edited record hash still fails, because every
    later record commits to the old one. Partial forgery is not enough."""
    led = Ledger()
    _fill(led, 6)
    records = list(led)
    records[2] = replace(records[2], amount_paise=1).sealed()
    with pytest.raises(ChainError, match="prev_hash does not match"):
        verify_chain(records)


def test_empty_ledger_verifies() -> None:
    verify_chain([])


def test_hash_is_stable_across_serialisation(tmp_path: Path) -> None:
    """A record must hash identically after a JSONL round trip, or durability
    silently invalidates the chain."""
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path=path)
    _fill(led, 5)
    reloaded = list(load(path))
    assert [r.record_hash for r in reloaded] == [r.record_hash for r in led]
    verify_chain(reloaded)


def test_reopening_a_ledger_continues_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    first = Ledger(path=path)
    _fill(first, 3)
    head = first.head

    second = Ledger(path=path)
    assert len(second) == 3
    assert second.head == head
    _fill(second, 2)
    second.verify()
    assert len(second) == 5


def test_persisted_lines_are_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path=path)
    _fill(led, 3)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        json.loads(line)


def test_every_gate_is_recorded_not_only_failures() -> None:
    """Passing gates are evidence too -- they prove the check actually ran."""
    led = Ledger()
    _fill(led, 1)
    rec = next(iter(led))
    assert len(rec.gates) == 9
    assert any(g.disposition == "allow" for g in rec.gates)


def test_denial_reasons_are_stored_verbatim() -> None:
    ev = _event(0)
    v = _verdict(ev, consent=False)
    led = Ledger()
    led.append(
        event=ev,
        diagnosis=diagnose(ev.error_reason),
        intended=ActionKind.NUDGE,
        verdict=v,
        executed=ActionKind.NO_ACTION,
        arm=Arm.TREATMENT,
        decided_at=NOW,
    )
    rec = next(iter(led))
    assert "consent" in rec.denied_by
    reason = next(g.reason for g in rec.gates if g.gate == "consent")
    assert "DPDP" in reason


def test_explain_covers_all_six_questions() -> None:
    led = Ledger()
    _fill(led, 1)
    text = next(iter(led)).explain()
    for expected in ("saw", "wanted", "did", "outcome", "insufficient_funds", "FUNDS"):
        assert expected in text


def test_customer_history_query() -> None:
    led = Ledger()
    _fill(led, 9)
    history = led.for_customer("cust_0000")
    assert len(history) == 3
    assert all(r.customer_id == "cust_0000" for r in history)


def test_replay_detects_what_a_policy_change_would_alter() -> None:
    """Answering 'what would the new rules have done last month' without touching
    a single real customer."""
    led = Ledger()
    _fill(led, 10)

    def stricter(rec) -> str:
        # A proposed rule: never nudge above Rs.500.
        return "no_action" if rec.amount_paise > 50_000 else rec.executed_action

    pairs = replay(led, stricter)
    summary = diff_replay(pairs)
    assert summary["nudge -> no_action"] == 10


def test_replay_with_identical_policy_changes_nothing() -> None:
    led = Ledger()
    _fill(led, 8)
    summary = diff_replay(replay(led, lambda rec: rec.executed_action))
    assert summary == {"unchanged": 8}


def test_replay_does_not_mutate_the_ledger() -> None:
    led = Ledger()
    _fill(led, 5)
    before = [r.record_hash for r in led]
    replay(led, lambda rec: "no_action")
    assert [r.record_hash for r in led] == before
    led.verify()
