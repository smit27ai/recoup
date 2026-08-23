"""Action selection.

Every strategy has the same signature so the harness can score them head to head,
including the deliberately bad ones. The baselines are not strawmen -- BLIND_RETRY
and BLAST are, respectively, what a retry-schedule product and a dunning-email
product actually do, and both look excellent until you measure them against a
holdout.

`taxonomy_policy` is Recoup tier 1. It calls no model. It reads the diagnosis and
routes. The bandit (Day 6-7) will sit ON TOP of this to choose among the contact
variants it leaves open -- it does not replace it, because which actions are even
admissible is a question of fact, not of exploration.
"""

from __future__ import annotations

from collections.abc import Callable

from recoup.diagnosis.taxonomy import Diagnosis, RetryClass
from recoup.domain import ActionKind, AtRiskEvent, Customer, RiskKind

A = ActionKind

Strategy = Callable[[AtRiskEvent, Diagnosis | None, Customer], ActionKind]


def no_action(_e: AtRiskEvent, _d: Diagnosis | None, _c: Customer) -> ActionKind:
    """Floor. Whatever this recovers, the world recovers without us."""
    return A.NO_ACTION


def blind_retry(_e: AtRiskEvent, _d: Diagnosis | None, _c: Customer) -> ActionKind:
    """Retry everything immediately. Ignores whether the instrument is even valid."""
    return A.RETRY_NOW


def blast(_e: AtRiskEvent, _d: Diagnosis | None, c: Customer) -> ActionKind:
    """Message everyone, every time. The gates will block a lot of this."""
    return A.NUDGE_WITH_INCENTIVE if c.segment == "at_risk" else A.NUDGE


def taxonomy_policy(e: AtRiskEvent, d: Diagnosis | None, _c: Customer) -> ActionKind:
    """Recoup tier 1: route on root cause, deterministically."""
    if e.kind is RiskKind.CHECKOUT_ABANDONED:
        return A.NUDGE
    if e.kind is RiskKind.INVOICE_OVERDUE:
        return A.NUDGE

    if d is None:
        # Unrecognised failure. We do NOT guess and we do NOT dun. A human looks.
        return A.ROUTE_TO_OPS
    if not d.in_scope:
        # Our bug or an ops ticket. Contacting the customer would be indefensible.
        return A.ROUTE_TO_OPS

    if d.new_instrument:
        # Retrying is futile by construction. Only a different instrument works.
        return A.NUDGE_WITH_INSTRUMENT_SWITCH if d.customer_action else A.ROUTE_TO_OPS

    if d.retry_class is RetryClass.NOW:
        return A.RETRY_NOW
    if d.retry_class in {RetryClass.SOON, RetryClass.SCHEDULED}:
        # Silent retry first -- it costs no contact budget and annoys nobody.
        return A.RETRY_SCHEDULED
    return A.NUDGE if d.contactable else A.NO_ACTION


STRATEGIES: dict[str, Strategy] = {
    "no_action": no_action,
    "blind_retry": blind_retry,
    "blast": blast,
    "taxonomy_policy": taxonomy_policy,
}
