"""Chaos results as a permanent gate.

A chaos run that only happens when someone remembers to run it is a demo. These
assertions turn each fault into a regression: if a future change makes the system
double-charge under a timeout, or message someone once the ledger breaks, the build
fails rather than a script nobody ran.
"""

from __future__ import annotations

import pytest

from recoup.chaos import (
    FAULTS,
    fault_consent_withdrawn,
    fault_gateway_500s,
    fault_hostile_webhooks,
    fault_ledger_broken,
    fault_llm_down,
    fault_post_timeout,
    fault_rate_limited,
    fault_tampered_ledger,
    report,
    run,
)


def test_nothing_crashes_under_any_fault() -> None:
    """An unhandled exception escaping the engine is a stuck queue in production,
    even when no invariant was technically violated."""
    for result in run():
        assert result.observation.crashed is None, (
            f"{result.fault} crashed: {result.observation.crashed}"
        )


def test_a_lost_response_never_double_charges() -> None:
    """The expensive one. The write landed, the response did not -- a blind retry
    here charges a customer twice."""
    result = fault_post_timeout()
    assert result.survived, result.violations
    seen = result.observation.orders_created
    assert len(seen) == len(set(seen)), "the same receipt produced two orders"


def test_a_dead_gateway_loses_no_money_silently() -> None:
    result = fault_gateway_500s()
    assert result.survived, result.violations
    assert result.observation.events_accounted == result.observation.events_seen


def test_rate_limiting_is_absorbed_not_hammered() -> None:
    result = fault_rate_limited()
    assert result.survived, result.violations


def test_a_dead_model_never_guesses() -> None:
    """Tier 2 unreachable must route to a human, not fall back to a guess."""
    result = fault_llm_down()
    assert result.survived, result.violations
    assert result.observation.messages_sent == [] or all(
        m.get("consented") == "yes" for m in result.observation.messages_sent
    )


def test_hostile_webhooks_never_reach_the_decision_path() -> None:
    result = fault_hostile_webhooks()
    assert result.survived, result.violations
    assert len(result.observation.notes) >= 6, "every attack must be individually rejected"


def test_no_consent_means_no_messages_at_all() -> None:
    result = fault_consent_withdrawn()
    assert result.survived, result.violations
    assert result.observation.messages_sent == []


def test_tampering_is_always_detected() -> None:
    result = fault_tampered_ledger()
    assert result.survived, result.violations


def test_a_broken_ledger_is_bounded_at_one_unrecorded_action() -> None:
    """Deliberately asserts the LIMITATION, not the absence of one.

    This fault does not pass, and it is left not passing. What must hold is that the
    damage is bounded: the circuit breaker opens after the first failure and every
    later event refuses to act rather than acting unrecorded. If that bound ever
    slips, this fails.
    """
    result = fault_ledger_broken()
    obs = result.observation
    unaccounted = obs.events_seen - obs.events_accounted
    assert obs.crashed is None, "a broken audit trail must not crash the engine"
    assert unaccounted <= 1, f"{unaccounted} events acted unrecorded; the bound is 1"
    assert result.residual, "a known limitation must be stated, not hidden"


def test_the_report_states_what_did_not_pass() -> None:
    """A summary that hides its failures is worse than no summary."""
    text = report(run())
    assert "KNOWN:" in text
    assert "does not pass is left not passing" in text


@pytest.mark.parametrize("fault", FAULTS, ids=lambda f: f.__name__)
def test_every_fault_reports_what_broke_and_what_happened(fault) -> None:
    result = fault()
    assert result.what_broke and result.what_happened
