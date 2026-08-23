"""Compliance gates: the hard veto layer.

Nothing in this module is probabilistic and nothing in it calls a model. Gates are
pure predicates over an explicit context, and every one of them can say NO to an
action that the propensity model and the bandit both wanted to take.

Two design rules that matter more than the individual gates:

1. ALL gates are evaluated, always -- we never short-circuit on the first denial.
   A decision that was blocked for three independent reasons must be auditable as
   having been blocked for three reasons, not one. Short-circuiting saves
   microseconds and destroys the audit trail.

2. Gates are evaluated immediately BEFORE execution, never at planning time. A
   workflow that slept four days may have been planned inside quiet hours and woken
   outside them, or the customer may have revoked consent while it slept. Deciding
   at plan time and executing later is the classic way compliant systems emit
   non-compliant messages.

The thresholds below are defaults, not law. Each carries the source it derives from
so a reviewer can check our reading. Treat them as operator-configurable policy and
have counsel confirm before any production use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


class GateID(StrEnum):
    CONSENT = "consent"
    DND = "dnd"
    QUIET_HOURS = "quiet_hours"
    CONTACT_BUDGET = "contact_budget"
    FATIGUE = "fatigue"
    STOPPING_RULE = "stopping_rule"
    DISCOUNT_AUTHORITY = "discount_authority"
    VALUE_APPROVAL = "value_approval"
    IDEMPOTENCY = "idempotency"


class Disposition(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    NEEDS_APPROVAL = "needs_approval"
    """Not a denial. The action is permissible but a human must authorise it."""


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Operator-configurable policy. Defaults cite the regime they derive from."""

    # RBI Fair Practices Code on recovery: borrowers are not to be contacted for
    # recovery outside 08:00-19:00 local time.
    # https://www.rbi.org.in/Scripts/BS_ViewMasCirculardetails.aspx  (Fair Practices Code)
    contact_window_start: time = time(8, 0)
    contact_window_end: time = time(19, 0)
    contact_timezone: ZoneInfo = IST

    # TRAI TCCCPA 2018 / DLT: commercial communication requires scrubbing against
    # preference registries. https://trai.gov.in/  -- we model this as a hard flag.
    honour_dnd: bool = True

    # DPDP Act 2023: processing personal data for contact requires a lawful basis.
    # No recorded consent means no contact, full stop.
    require_consent: bool = True

    # Contact budgeting. Not legally mandated -- this is the difference between a
    # recovery system and a harassment machine, and it is what stops seven
    # single-purpose agents all messaging the same person on the same morning.
    max_contacts_per_window: int = 3
    contact_window: timedelta = timedelta(days=7)
    min_gap_between_contacts: timedelta = timedelta(hours=24)
    max_attempts_per_event: int = 4

    # Authority limits. Above these, a human signs off.
    max_discount_bps: int = 1000
    """Basis points off the outstanding amount an agent may offer unattended (10%)."""
    approval_threshold_paise: int = 5_000_00
    """Any action touching more than this needs human approval (Rs.5,000)."""


@dataclass(frozen=True, slots=True)
class ProposedAction:
    """What the policy engine wants to do, before anyone has said it may."""

    event_id: str
    customer_id: str
    kind: str
    """e.g. retry_charge, send_payment_link, whatsapp_nudge, email_nudge."""
    is_contact: bool
    amount_paise: int
    discount_bps: int = 0
    idempotency_key: str = ""


@dataclass(frozen=True, slots=True)
class CustomerState:
    """Everything the gates need to know about the person on the other end."""

    customer_id: str
    has_consent: bool
    on_dnd_registry: bool
    contacts_in_window: Sequence[datetime] = field(default_factory=tuple)
    last_contact_at: datetime | None = None
    opted_out: bool = False
    hardship_flag: bool = False
    """Bereavement, declared financial hardship, or an active grievance. Stop."""


@dataclass(frozen=True, slots=True)
class EventState:
    """Everything the gates need to know about the money."""

    event_id: str
    attempts_so_far: int
    already_recovered: bool = False
    dispute_open: bool = False
    promise_to_pay_until: datetime | None = None
    """Customer said they would pay by this date. Contacting before it is bad faith."""
    executed_idempotency_keys: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class GateContext:
    action: ProposedAction
    customer: CustomerState
    event: EventState
    now: datetime
    config: PolicyConfig = PolicyConfig()


@dataclass(frozen=True, slots=True)
class GateResult:
    gate: GateID
    disposition: Disposition
    reason: str
    """Human-readable, written to the audit ledger verbatim. Say WHY, with numbers."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """The full record of every gate that ran, not just the one that failed."""

    results: tuple[GateResult, ...]
    now: datetime

    @property
    def denials(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if r.disposition is Disposition.DENY)

    @property
    def approvals_needed(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if r.disposition is Disposition.NEEDS_APPROVAL)

    @property
    def disposition(self) -> Disposition:
        if self.denials:
            return Disposition.DENY
        if self.approvals_needed:
            return Disposition.NEEDS_APPROVAL
        return Disposition.ALLOW

    @property
    def allowed(self) -> bool:
        return self.disposition is Disposition.ALLOW

    def explain(self) -> str:
        blocking = self.denials + self.approvals_needed
        if not blocking:
            return f"allowed by all {len(self.results)} gates"
        return "; ".join(f"{r.gate}: {r.reason}" for r in blocking)


# --- individual gates -------------------------------------------------------
# Each returns a GateResult. They never raise and never mutate.


def _gate_consent(ctx: GateContext) -> GateResult:
    if not ctx.action.is_contact:
        return GateResult(GateID.CONSENT, Disposition.ALLOW, "not a contact action")
    if ctx.config.require_consent and not ctx.customer.has_consent:
        return GateResult(
            GateID.CONSENT, Disposition.DENY, "no recorded consent to contact (DPDP 2023)"
        )
    if ctx.customer.opted_out:
        return GateResult(GateID.CONSENT, Disposition.DENY, "customer has opted out")
    return GateResult(GateID.CONSENT, Disposition.ALLOW, "consent on record")


def _gate_dnd(ctx: GateContext) -> GateResult:
    if not ctx.action.is_contact:
        return GateResult(GateID.DND, Disposition.ALLOW, "not a contact action")
    if ctx.config.honour_dnd and ctx.customer.on_dnd_registry:
        return GateResult(GateID.DND, Disposition.DENY, "number on preference registry (TRAI)")
    return GateResult(GateID.DND, Disposition.ALLOW, "not on preference registry")


def _gate_quiet_hours(ctx: GateContext) -> GateResult:
    if not ctx.action.is_contact:
        return GateResult(GateID.QUIET_HOURS, Disposition.ALLOW, "not a contact action")
    local = ctx.now.astimezone(ctx.config.contact_timezone).time()
    start, end = ctx.config.contact_window_start, ctx.config.contact_window_end
    if start <= local < end:
        return GateResult(
            GateID.QUIET_HOURS, Disposition.ALLOW, f"{local:%H:%M} inside {start:%H:%M}-{end:%H:%M}"
        )
    return GateResult(
        GateID.QUIET_HOURS,
        Disposition.DENY,
        f"{local:%H:%M} IST outside permitted {start:%H:%M}-{end:%H:%M} (RBI FPC)",
    )


def _gate_contact_budget(ctx: GateContext) -> GateResult:
    if not ctx.action.is_contact:
        return GateResult(GateID.CONTACT_BUDGET, Disposition.ALLOW, "not a contact action")
    cutoff = ctx.now - ctx.config.contact_window
    recent = [t for t in ctx.customer.contacts_in_window if t >= cutoff]
    cap = ctx.config.max_contacts_per_window
    if len(recent) >= cap:
        return GateResult(
            GateID.CONTACT_BUDGET,
            Disposition.DENY,
            f"{len(recent)}/{cap} contacts already used in last {ctx.config.contact_window.days}d",
        )
    return GateResult(
        GateID.CONTACT_BUDGET, Disposition.ALLOW, f"{len(recent)}/{cap} contacts used"
    )


def _gate_fatigue(ctx: GateContext) -> GateResult:
    if not ctx.action.is_contact or ctx.customer.last_contact_at is None:
        return GateResult(GateID.FATIGUE, Disposition.ALLOW, "no prior contact")
    elapsed = ctx.now - ctx.customer.last_contact_at
    required = ctx.config.min_gap_between_contacts
    if elapsed < required:
        hours = elapsed.total_seconds() / 3600
        return GateResult(
            GateID.FATIGUE,
            Disposition.DENY,
            f"last contact {hours:.1f}h ago, minimum gap is {required.total_seconds() / 3600:.0f}h",
        )
    return GateResult(GateID.FATIGUE, Disposition.ALLOW, "outside fatigue window")


def _gate_stopping_rule(ctx: GateContext) -> GateResult:
    ev, cust = ctx.event, ctx.customer
    if ev.already_recovered:
        return GateResult(GateID.STOPPING_RULE, Disposition.DENY, "money already recovered")
    if ev.dispute_open:
        return GateResult(
            GateID.STOPPING_RULE, Disposition.DENY, "dispute open, recovery must not run"
        )
    if cust.hardship_flag:
        return GateResult(GateID.STOPPING_RULE, Disposition.DENY, "hardship/bereavement flag set")
    if ev.promise_to_pay_until is not None and ctx.now < ev.promise_to_pay_until:
        return GateResult(
            GateID.STOPPING_RULE,
            Disposition.DENY,
            f"promise-to-pay honoured until {ev.promise_to_pay_until:%Y-%m-%d}",
        )
    cap = ctx.config.max_attempts_per_event
    if ev.attempts_so_far >= cap:
        return GateResult(
            GateID.STOPPING_RULE,
            Disposition.DENY,
            f"attempt cap reached ({ev.attempts_so_far}/{cap})",
        )
    return GateResult(
        GateID.STOPPING_RULE, Disposition.ALLOW, f"attempt {ev.attempts_so_far + 1}/{cap}"
    )


def _gate_discount_authority(ctx: GateContext) -> GateResult:
    offered, cap = ctx.action.discount_bps, ctx.config.max_discount_bps
    if offered <= cap:
        return GateResult(
            GateID.DISCOUNT_AUTHORITY, Disposition.ALLOW, f"{offered}bps within {cap}bps authority"
        )
    return GateResult(
        GateID.DISCOUNT_AUTHORITY,
        Disposition.NEEDS_APPROVAL,
        f"{offered}bps exceeds {cap}bps unattended authority",
    )


def _gate_value_approval(ctx: GateContext) -> GateResult:
    amount, threshold = ctx.action.amount_paise, ctx.config.approval_threshold_paise
    if amount <= threshold:
        return GateResult(
            GateID.VALUE_APPROVAL, Disposition.ALLOW, f"Rs.{amount / 100:,.0f} under threshold"
        )
    return GateResult(
        GateID.VALUE_APPROVAL,
        Disposition.NEEDS_APPROVAL,
        f"Rs.{amount / 100:,.0f} exceeds unattended threshold Rs.{threshold / 100:,.0f}",
    )


def _gate_idempotency(ctx: GateContext) -> GateResult:
    key = ctx.action.idempotency_key
    if not key:
        return GateResult(GateID.IDEMPOTENCY, Disposition.DENY, "action carries no idempotency key")
    if key in ctx.event.executed_idempotency_keys:
        return GateResult(GateID.IDEMPOTENCY, Disposition.DENY, f"key {key} already executed")
    return GateResult(GateID.IDEMPOTENCY, Disposition.ALLOW, "key unused")


ALL_GATES = (
    _gate_consent,
    _gate_dnd,
    _gate_quiet_hours,
    _gate_contact_budget,
    _gate_fatigue,
    _gate_stopping_rule,
    _gate_discount_authority,
    _gate_value_approval,
    _gate_idempotency,
)


def evaluate(ctx: GateContext) -> Verdict:
    """Run every gate and return the complete record.

    Deliberately not short-circuited: see the module docstring.
    """
    return Verdict(results=tuple(gate(ctx) for gate in ALL_GATES), now=ctx.now)
