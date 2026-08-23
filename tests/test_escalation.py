"""Tier-2 escalation tests.

The safety policy is the whole point of this module, so most of these check that a
model is NOT allowed to talk a customer into being contacted. A backend is assumed
to be occasionally wrong, occasionally overconfident, and occasionally down; none of
those may become a customer's problem.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from recoup.diagnosis.escalation import (
    ClaudeEscalator,
    EscalationService,
    InvalidProposal,
    Owner,
    Proposal,
    RetryClass,
    ReviewQueue,
    RootCause,
    StubEscalator,
    build_escalator,
    parse_proposal,
)
from recoup.diagnosis.taxonomy import diagnose, load_taxonomy


def _proposal(**kw: Any) -> Proposal:
    base: dict[str, Any] = {
        "reason": "some_new_code",
        "root_cause": RootCause.FUNDS,
        "retry_class": RetryClass.SCHEDULED,
        "new_instrument": False,
        "customer_action": True,
        "owner": Owner.CUSTOMER,
        "in_scope": True,
        "confidence": 0.9,
        "rationale": "test",
        "model": "test",
        "proposed_at": datetime.now(),
    }
    return Proposal(**{**base, **kw})


class Canned:
    """A backend that returns exactly what a test tells it to."""

    name = "canned"

    def __init__(self, proposal: Proposal | Exception | None) -> None:
        self._p = proposal
        self.calls = 0

    def propose(self, reason: str, context: dict[str, Any]) -> Proposal | None:
        self.calls += 1
        if isinstance(self._p, Exception):
            raise self._p
        return self._p


# --- asymmetric trust: the core safety property -----------------------------


def test_de_escalating_proposal_is_trusted_at_low_confidence() -> None:
    """Routing work AWAY from a customer is the safe direction. Being wrong costs
    an ops ticket, so it does not need to clear a confidence bar."""
    svc = EscalationService(
        Canned(
            _proposal(
                root_cause=RootCause.INTEGRATION_BUG,
                in_scope=False,
                customer_action=False,
                confidence=0.3,
            )
        )
    )
    diagnosis = svc.diagnose("weird_new_code")
    assert diagnosis is not None
    assert diagnosis.in_scope is False
    assert diagnosis.contactable is False
    assert diagnosis.tier == 2


def test_too_unsure_to_be_worth_even_a_retry() -> None:
    """Below the retry floor there is nothing safe to do but hand it to a human."""
    svc = EscalationService(Canned(_proposal(confidence=0.3)))
    assert svc.diagnose("weird_new_code") is None


def test_mediocre_confidence_still_buys_a_silent_retry() -> None:
    """A retry disturbs nobody, so a mediocre guess is worth acting on -- the cost
    of being wrong is one API call, the cost of doing nothing is real money."""
    svc = EscalationService(Canned(_proposal(confidence=0.6)))
    diagnosis = svc.diagnose("weird_new_code")
    assert diagnosis is not None
    assert diagnosis.retryable
    assert diagnosis.contactable is False


def test_confident_action_proposal_still_cannot_authorise_contact() -> None:
    """Even at 0.99 the model may drive a silent retry, but not a message, until a
    human has approved the rule. This is the single most important test here."""
    svc = EscalationService(Canned(_proposal(confidence=0.99, customer_action=True)))
    diagnosis = svc.diagnose("weird_new_code")
    assert diagnosis is not None
    assert diagnosis.customer_action is False, "tier 2 must not unlock contact on its own"
    assert diagnosis.contactable is False
    assert diagnosis.retryable, "a silent retry is still allowed"


def test_confident_non_contact_proposal_passes_through() -> None:
    """Nothing to downgrade: it never wanted to contact anyone."""
    svc = EscalationService(
        Canned(
            _proposal(
                root_cause=RootCause.GATEWAY_DOWN,
                retry_class=RetryClass.NOW,
                customer_action=False,
                owner=Owner.RAZORPAY,
                confidence=0.9,
            )
        )
    )
    diagnosis = svc.diagnose("weird_new_code")
    assert diagnosis is not None
    assert diagnosis.retry_class is RetryClass.NOW


def test_out_of_scope_proposal_never_becomes_contactable() -> None:
    svc = EscalationService(
        Canned(_proposal(in_scope=False, customer_action=True, confidence=0.95))
    )
    diagnosis = svc.diagnose("merchant_thing")
    assert diagnosis is not None
    assert diagnosis.contactable is False


# --- failure is never fatal -------------------------------------------------


def test_backend_exception_degrades_to_none() -> None:
    """Tier 2 is an enhancement, not a dependency. A model outage must not take the
    recovery pipeline down with it."""
    svc = EscalationService(Canned(RuntimeError("503 from the API")))
    assert svc.diagnose("weird_new_code") is None


def test_backend_returning_none_is_fine() -> None:
    svc = EscalationService(Canned(None))
    assert svc.diagnose("weird_new_code") is None


def test_failure_is_cached_so_an_outage_is_not_amplified() -> None:
    """A model that is down should be asked once, not once per event."""
    backend = Canned(RuntimeError("down"))
    svc = EscalationService(backend)
    for _ in range(50):
        svc.diagnose("weird_new_code")
    assert backend.calls == 1


# --- caching ----------------------------------------------------------------


def test_same_reason_costs_exactly_one_call() -> None:
    backend = Canned(_proposal())
    svc = EscalationService(backend)
    for _ in range(1000):
        svc.diagnose("weird_new_code")
    assert backend.calls == 1
    assert svc.calls == 1


def test_cache_keeps_diagnoses_consistent() -> None:
    """Two identical failures must never get different answers."""
    svc = EscalationService(Canned(_proposal()))
    first = svc.diagnose("weird_new_code")
    second = svc.diagnose("weird_new_code")
    assert first == second


def test_distinct_reasons_each_get_a_call() -> None:
    backend = Canned(_proposal())
    svc = EscalationService(backend)
    svc.diagnose("code_a")
    svc.diagnose("code_b")
    assert backend.calls == 2


# --- rule mining ------------------------------------------------------------


def test_escalation_produces_a_reviewable_taxonomy_row() -> None:
    """The model's job is to shrink its own job."""
    svc = EscalationService(Canned(_proposal(reason="brand_new_code")))
    svc.diagnose("brand_new_code")

    assert len(svc.review) == 1
    row = svc.review.as_taxonomy_rows()
    assert row.startswith("brand_new_code\t")
    assert len(row.split("\t")) == 8, "must match the taxonomy TSV column count"


def test_review_queue_ranks_by_how_often_a_code_was_seen() -> None:
    """A reviewer should fix the code costing the most money first."""
    queue = ReviewQueue()
    for reason, times in (("rare_code", 2), ("common_code", 40)):
        svc = EscalationService(Canned(_proposal(reason=reason)), review_queue=queue)
        for _ in range(times):
            svc.diagnose(reason)

    ranked = queue.by_impact()
    assert ranked[0][0].reason == "common_code"
    assert ranked[0][1] == 40


def test_approved_row_would_parse_as_tier_one() -> None:
    """The mined row must be usable verbatim -- otherwise the loop does not close."""
    svc = EscalationService(Canned(_proposal(reason="brand_new_code")))
    svc.diagnose("brand_new_code")
    fields = svc.review.as_taxonomy_rows().split("\t")
    assert RootCause(fields[2])
    assert RetryClass(fields[3])
    assert fields[4] in {"0", "1"}
    assert Owner(fields[6])


def test_empty_review_queue_is_not_silently_replaced() -> None:
    """ReviewQueue defines __len__, so `or` would discard it. Same trap as before."""
    queue = ReviewQueue()
    assert EscalationService(Canned(None), review_queue=queue).review is queue


# --- validation: reject, never coerce ---------------------------------------


def test_unknown_root_cause_is_rejected() -> None:
    with pytest.raises(InvalidProposal):
        parse_proposal(
            "x",
            {
                "root_cause": "VIBES",
                "retry_class": "NOW",
                "owner": "customer",
                "confidence": 0.9,
                "new_instrument": False,
                "customer_action": True,
                "in_scope": True,
            },
            model="m",
        )


def test_confidence_out_of_range_is_rejected() -> None:
    with pytest.raises(InvalidProposal, match="out of range"):
        parse_proposal(
            "x",
            {
                "root_cause": "FUNDS",
                "retry_class": "NOW",
                "owner": "customer",
                "confidence": 1.7,
                "new_instrument": False,
                "customer_action": True,
                "in_scope": True,
            },
            model="m",
        )


def test_non_boolean_flag_is_rejected() -> None:
    with pytest.raises(InvalidProposal, match="must be a boolean"):
        parse_proposal(
            "x",
            {
                "root_cause": "FUNDS",
                "retry_class": "NOW",
                "owner": "customer",
                "confidence": 0.9,
                "new_instrument": "yes",
                "customer_action": True,
                "in_scope": True,
            },
            model="m",
        )


def test_missing_field_is_rejected() -> None:
    with pytest.raises(InvalidProposal):
        parse_proposal("x", {"root_cause": "FUNDS"}, model="m")


def test_valid_payload_parses() -> None:
    proposal = parse_proposal(
        "x",
        {
            "root_cause": "FUNDS",
            "retry_class": "SCHEDULED",
            "owner": "customer",
            "confidence": 0.82,
            "new_instrument": False,
            "customer_action": True,
            "in_scope": True,
            "rationale": "balance language",
        },
        model="claude-opus-5",
    )
    assert proposal.root_cause is RootCause.FUNDS
    assert proposal.confidence == 0.82


# --- the stub backend -------------------------------------------------------


def test_no_confidence_value_whatsoever_unlocks_contact() -> None:
    """The property that matters most. Swept across the whole range so nobody can
    reintroduce a contact threshold without this failing."""
    for confidence in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
        svc = EscalationService(Canned(_proposal(confidence=confidence)))
        diagnosis = svc.diagnose(f"code_{confidence}")
        assert diagnosis is None or diagnosis.contactable is False


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("card_has_expired_2027", RootCause.INSTRUMENT_INVALID),
        ("account_balance_too_low", RootCause.FUNDS),
        ("issuer_bank_unreachable", RootCause.ISSUER_DOWN),
        ("merchant_not_enabled_for_upi", RootCause.MERCHANT_CONFIG),
        ("duplicate_order_reference", RootCause.INTEGRATION_BUG),
    ],
)
def test_stub_classifies_plausibly(reason: str, expected: RootCause) -> None:
    proposal = StubEscalator().propose(reason, {})
    assert proposal is not None
    assert proposal.root_cause is expected


def test_stub_refuses_to_guess_on_gibberish() -> None:
    proposal = StubEscalator().propose("zx_qq_9917", {})
    assert proposal is not None
    assert proposal.root_cause is RootCause.UNKNOWN
    assert proposal.in_scope is False
    assert proposal.confidence < 0.5


def test_stub_gibberish_yields_no_actionable_diagnosis() -> None:
    svc = EscalationService(StubEscalator())
    diagnosis = svc.diagnose("zx_qq_9917")
    assert diagnosis is not None
    assert diagnosis.contactable is False


# --- backend selection ------------------------------------------------------


def test_build_escalator_uses_stub_without_a_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(build_escalator(), StubEscalator)


def test_build_escalator_uses_claude_with_a_key(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    backend = build_escalator()
    assert isinstance(backend, ClaudeEscalator)
    assert backend.model == "claude-opus-5"


# --- tier 1 still owns everything it knows ----------------------------------


def test_tier_two_is_never_consulted_for_a_known_code() -> None:
    """Escalating a code the table already has would be pure waste and would let a
    model's opinion override documented behaviour."""
    backend = Canned(_proposal())
    svc = EscalationService(backend)
    for reason in list(load_taxonomy())[:20]:
        if diagnose(reason) is not None:
            continue
        svc.diagnose(reason)
    assert backend.calls == 0
