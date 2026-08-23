"""Core domain types shared across diagnosis, policy, execution and measurement."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class RiskKind(StrEnum):
    """The four ways revenue slips away, per the track brief."""

    FAILED_PAYMENT = "failed_payment"
    CHECKOUT_ABANDONED = "checkout_abandoned"
    SUBSCRIPTION_FAILED = "subscription_failed"
    INVOICE_OVERDUE = "invoice_overdue"


class Channel(StrEnum):
    NONE = "none"
    WHATSAPP = "whatsapp"
    SMS = "sms"
    EMAIL = "email"


class ActionKind(StrEnum):
    """The action space the bandit chooses from.

    NO_ACTION is a first-class arm, not an absence of one. A recovery system that
    cannot choose to leave someone alone is a spam system, and on a large fraction
    of events it is genuinely the highest-EV choice.
    """

    NO_ACTION = "no_action"
    RETRY_NOW = "retry_now"
    RETRY_SCHEDULED = "retry_scheduled"
    """Timed to payday / issuer cutoff rather than fired blindly."""
    NUDGE = "nudge"
    """Message with a fresh payment link, no incentive."""
    NUDGE_WITH_INSTRUMENT_SWITCH = "nudge_with_instrument_switch"
    """Message that asks for a different payment method."""
    NUDGE_WITH_INCENTIVE = "nudge_with_incentive"
    """Message carrying a discount. Costs margin, so must earn its place."""
    ROUTE_TO_OPS = "route_to_ops"
    """Not the customer fault. Goes to a human queue, never to the customer."""

    QUEUED_FOR_APPROVAL = "queued_for_approval"
    """Permissible, but over an authority limit, so it waits on a human.

    Distinct from NO_ACTION on purpose. Collapsing the two was a real bug: every
    high-value event -- exactly the ones worth most -- was silently dropped and
    counted as a decision not to act, so the money vanished with no queue entry and
    nothing in the metrics to show it had ever been considered. "We chose not to"
    and "a human has not looked yet" are different states and must never share a
    representation.
    """

    @property
    def is_contact(self) -> bool:
        return self in {
            ActionKind.NUDGE,
            ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH,
            ActionKind.NUDGE_WITH_INCENTIVE,
        }

    @property
    def is_retry(self) -> bool:
        return self in {ActionKind.RETRY_NOW, ActionKind.RETRY_SCHEDULED}


class Arm(StrEnum):
    """Assignment for incrementality measurement."""

    TREATMENT = "treatment"
    HOLDOUT = "holdout"
    """Deliberately left alone so we can measure what we actually caused."""


@dataclass(frozen=True, slots=True)
class Customer:
    customer_id: str
    segment: str
    has_consent: bool
    on_dnd_registry: bool
    preferred_channel: Channel
    language: str = "en"
    """en | hi | hinglish -- drives message generation, not policy."""


@dataclass(frozen=True, slots=True)
class AtRiskEvent:
    """One unit of revenue that did not arrive."""

    event_id: str
    customer_id: str
    kind: RiskKind
    amount_paise: int
    occurred_at: datetime
    error_reason: str | None
    """None for abandonment/overdue, which have no gateway error."""
    method: str
    attempt_number: int = 1
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """The simulator answer key. NEVER visible to the policy engine.

    Held out of every code path except scoring. Its presence is what lets us report
    honest numbers -- including an upper bound on how much money was recoverable at
    all, which is the number that turns "we recovered Rs.X" into a meaningful claim.
    """

    event_id: str
    self_heal_probability: float
    """P(recovers with NO intervention). The reason a holdout is mandatory: without
    it, every rupee in this bucket gets falsely claimed as recovered by us."""
    lift_by_action: dict[ActionKind, float]
    """Incremental P(recovery) each action adds on top of self-heal. May be zero --
    retrying an expired card is futile no matter how confident the agent sounds."""
    max_recoverable_paise: int

    def probability(self, action: ActionKind) -> float:
        return min(1.0, max(0.0, self.self_heal_probability + self.lift_by_action.get(action, 0.0)))
