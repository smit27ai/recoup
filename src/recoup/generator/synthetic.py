"""Seeded synthetic population of at-risk revenue, with a hidden answer key.

Why a simulator at all, when Razorpay test-mode APIs exist? Because test mode can
produce a failure but it cannot tell you what WOULD have happened under a different
intervention. Measuring recovery honestly needs the counterfactual, and the only
place a counterfactual is observable is a world you built. Test-mode APIs are used
for the execution path (real orders, real payment links, real webhooks); the
simulator is used for the measurement path. Both are real, they answer different
questions.

The answer key lives in GroundTruth and is fenced off from every module except
scoring. Two numbers per event matter:

  self_heal_probability  -- P(recovers if we do absolutely nothing)
  lift_by_action         -- what each action ADDS on top of that

Almost every dunning product on the market reports gross recovery, which silently
counts the self-heal bucket as its own work. The lift matrix below is built so that
naive strategies look good on gross numbers and bad on incremental ones -- most
sharply for GATEWAY_DOWN, where the customer very often just retries by themselves
and any system that fires a retry gets to claim credit for it.

Second thing the matrix encodes: futility. Retrying an expired card has a lift of
exactly 0.0. No amount of retry budget recovers that rupee; only asking for a
different instrument does. A policy that cannot tell those apart burns its contact
budget and its issuer reputation for nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from recoup.diagnosis.taxonomy import RootCause, load_taxonomy
from recoup.domain import (
    ActionKind,
    AtRiskEvent,
    Channel,
    Customer,
    GroundTruth,
    RiskKind,
)
from recoup.policy.gates import IST

A = ActionKind
RC = RootCause

# P(recovers with no intervention at all), by root cause.
SELF_HEAL: dict[RootCause, float] = {
    RC.GATEWAY_DOWN: 0.52,
    RC.ISSUER_DOWN: 0.44,
    RC.PSP_DOWN: 0.40,
    RC.AUTH_FAILED: 0.31,
    RC.AUTH_ABANDONED: 0.26,
    RC.LIMIT_EXCEEDED: 0.18,
    RC.FUNDS: 0.15,
    RC.MANDATE_PROBLEM: 0.12,
    RC.UNKNOWN: 0.20,
    RC.INSTRUMENT_NOT_ENROLLED: 0.09,
    RC.INSTRUMENT_INVALID: 0.07,
    RC.INSTRUMENT_BLOCKED: 0.06,
    RC.CREDIT_INELIGIBLE: 0.05,
    RC.RISK_DECLINE: 0.03,
    RC.MERCHANT_CONFIG: 0.02,
    RC.INTEGRATION_BUG: 0.02,
    RC.COMPLIANCE: 0.01,
    RC.OPS: 0.01,
}

# Incremental P(recovery) each action adds. Absent entries are 0.0.
LIFT: dict[RootCause, dict[ActionKind, float]] = {
    # Money will exist later. Time the retry; do not hammer it now.
    RC.FUNDS: {
        A.RETRY_NOW: 0.02,
        A.RETRY_SCHEDULED: 0.34,
        A.NUDGE: 0.12,
        A.NUDGE_WITH_INSTRUMENT_SWITCH: 0.16,
        A.NUDGE_WITH_INCENTIVE: 0.21,
    },
    # Retrying is FUTILE. Only a different instrument moves this money.
    RC.INSTRUMENT_INVALID: {
        A.RETRY_NOW: 0.0,
        A.RETRY_SCHEDULED: 0.0,
        A.NUDGE: 0.14,
        A.NUDGE_WITH_INSTRUMENT_SWITCH: 0.41,
        A.NUDGE_WITH_INCENTIVE: 0.19,
    },
    RC.INSTRUMENT_BLOCKED: {
        A.RETRY_NOW: 0.0,
        A.RETRY_SCHEDULED: 0.0,
        A.NUDGE: 0.11,
        A.NUDGE_WITH_INSTRUMENT_SWITCH: 0.38,
        A.NUDGE_WITH_INCENTIVE: 0.15,
    },
    RC.INSTRUMENT_NOT_ENROLLED: {
        A.RETRY_NOW: 0.0,
        A.RETRY_SCHEDULED: 0.0,
        A.NUDGE: 0.13,
        A.NUDGE_WITH_INSTRUMENT_SWITCH: 0.31,
        A.NUDGE_WITH_INCENTIVE: 0.16,
    },
    # They were one tap away. A reminder is worth a lot; a retry is worth nothing.
    RC.AUTH_ABANDONED: {
        A.RETRY_NOW: 0.04,
        A.RETRY_SCHEDULED: 0.05,
        A.NUDGE: 0.30,
        A.NUDGE_WITH_INSTRUMENT_SWITCH: 0.22,
        A.NUDGE_WITH_INCENTIVE: 0.36,
    },
    RC.AUTH_FAILED: {
        A.RETRY_NOW: 0.09,
        A.RETRY_SCHEDULED: 0.11,
        A.NUDGE: 0.27,
        A.NUDGE_WITH_INSTRUMENT_SWITCH: 0.24,
        A.NUDGE_WITH_INCENTIVE: 0.29,
    },
    RC.LIMIT_EXCEEDED: {
        A.RETRY_NOW: 0.01,
        A.RETRY_SCHEDULED: 0.27,
        A.NUDGE: 0.10,
        A.NUDGE_WITH_INSTRUMENT_SWITCH: 0.25,
        A.NUDGE_WITH_INCENTIVE: 0.13,
    },
    # Not the customer fault. Messaging them is noise; waiting out the outage works.
    RC.ISSUER_DOWN: {
        A.RETRY_NOW: 0.19,
        A.RETRY_SCHEDULED: 0.44,
        A.NUDGE: 0.03,
        A.NUDGE_WITH_INSTRUMENT_SWITCH: 0.17,
        A.NUDGE_WITH_INCENTIVE: 0.04,
    },
    RC.PSP_DOWN: {
        A.RETRY_NOW: 0.16,
        A.RETRY_SCHEDULED: 0.39,
        A.NUDGE: 0.05,
        A.NUDGE_WITH_INSTRUMENT_SWITCH: 0.26,
        A.NUDGE_WITH_INCENTIVE: 0.06,
    },
    # High self-heal AND high retry lift: the trap bucket for gross-number reporting.
    RC.GATEWAY_DOWN: {
        A.RETRY_NOW: 0.33,
        A.RETRY_SCHEDULED: 0.28,
        A.NUDGE: 0.02,
        A.NUDGE_WITH_INSTRUMENT_SWITCH: 0.08,
        A.NUDGE_WITH_INCENTIVE: 0.03,
    },
    RC.MANDATE_PROBLEM: {
        A.RETRY_NOW: 0.05,
        A.RETRY_SCHEDULED: 0.08,
        A.NUDGE: 0.26,
        A.NUDGE_WITH_INSTRUMENT_SWITCH: 0.21,
        A.NUDGE_WITH_INCENTIVE: 0.24,
    },
    RC.CREDIT_INELIGIBLE: {
        A.NUDGE: 0.09,
        A.NUDGE_WITH_INSTRUMENT_SWITCH: 0.29,
        A.NUDGE_WITH_INCENTIVE: 0.12,
    },
    # Declined for risk. Chasing it is how you get fined, not paid.
    RC.RISK_DECLINE: {A.NUDGE_WITH_INSTRUMENT_SWITCH: 0.08, A.ROUTE_TO_OPS: 0.05},
    RC.UNKNOWN: {
        A.RETRY_NOW: 0.08,
        A.RETRY_SCHEDULED: 0.15,
        A.NUDGE: 0.12,
        A.NUDGE_WITH_INSTRUMENT_SWITCH: 0.16,
        A.NUDGE_WITH_INCENTIVE: 0.14,
    },
    # Our problem, not theirs. Only a human fixing it recovers this money.
    RC.MERCHANT_CONFIG: {A.ROUTE_TO_OPS: 0.62},
    RC.INTEGRATION_BUG: {A.ROUTE_TO_OPS: 0.71},
    RC.COMPLIANCE: {A.ROUTE_TO_OPS: 0.30},
    RC.OPS: {A.ROUTE_TO_OPS: 0.55},
}

SEGMENTS = ("new", "casual", "loyal", "at_risk", "business")
# Loyal customers respond better to everything; at-risk ones respond worse.
SEGMENT_MULTIPLIER = {
    "new": 0.85,
    "casual": 1.0,
    "loyal": 1.25,
    "at_risk": 0.70,
    "business": 1.10,
}
LANGUAGES = ("en", "hi", "hinglish")


@dataclass(frozen=True, slots=True)
class Scenario:
    """One generated world: the events, the people, and the answer key."""

    customers: dict[str, Customer]
    events: tuple[AtRiskEvent, ...]
    truth: dict[str, GroundTruth]
    seed: int

    @property
    def total_at_risk_paise(self) -> int:
        return sum(e.amount_paise for e in self.events)

    def ceiling_paise(self) -> int:
        """Upper bound on recoverable money: best action per event, in expectation.

        No policy can beat this, and quoting a recovery figure without it is how
        dunning vendors make 3% sound like a triumph.
        """
        total = 0.0
        for ev in self.events:
            gt = self.truth[ev.event_id]
            best = max((gt.probability(a) for a in ActionKind), default=0.0)
            total += best * ev.amount_paise
        return int(total)

    def self_heal_paise(self) -> int:
        """Money that arrives on its own. Anyone claiming this as recovery is lying."""
        return int(
            sum(self.truth[e.event_id].self_heal_probability * e.amount_paise for e in self.events)
        )


class ScenarioGenerator:
    """Deterministic given a seed. Same seed, same world, byte for byte."""

    def __init__(self, seed: int = 20260905) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def _amount(self, kind: RiskKind) -> int:
        r = self._rng
        match kind:
            case RiskKind.SUBSCRIPTION_FAILED:
                return int(r.lognormvariate(6.4, 0.6)) * 100
            case RiskKind.INVOICE_OVERDUE:
                return int(r.lognormvariate(10.2, 1.0)) * 100
            case _:
                return int(r.lognormvariate(7.1, 0.9)) * 100

    def _root_cause_for(self, kind: RiskKind, reason: str | None) -> RootCause:
        if kind is RiskKind.CHECKOUT_ABANDONED:
            return RC.AUTH_ABANDONED
        if kind is RiskKind.INVOICE_OVERDUE:
            return RC.FUNDS
        table = load_taxonomy()
        entry = table.get(reason or "")
        return entry.root_cause if entry else RC.UNKNOWN

    def _make_truth(self, event: AtRiskEvent, root_cause: RootCause, segment: str) -> GroundTruth:
        r = self._rng
        mult = SEGMENT_MULTIPLIER[segment]
        # Per-event jitter so the world is not a lookup table the policy can memorise.
        heal = SELF_HEAL.get(root_cause, 0.10) * r.uniform(0.75, 1.25)
        heal = min(0.95, max(0.0, heal))

        lifts: dict[ActionKind, float] = {}
        for action, base in LIFT.get(root_cause, {}).items():
            if base == 0.0:
                lifts[action] = 0.0  # futility is exact, never jittered
                continue
            lifts[action] = max(0.0, base * mult * r.uniform(0.8, 1.2))
        lifts[A.NO_ACTION] = 0.0

        # Repeat attempts decay hard. The fourth message is not as good as the first.
        decay = 0.72 ** (event.attempt_number - 1)
        lifts = {k: v * decay for k, v in lifts.items()}

        best = max((min(1.0, heal + v) for v in lifts.values()), default=heal)
        return GroundTruth(
            event_id=event.event_id,
            self_heal_probability=heal,
            lift_by_action=lifts,
            max_recoverable_paise=int(best * event.amount_paise),
        )

    def generate(self, n_events: int = 5000, n_customers: int = 1800) -> Scenario:
        r = self._rng
        table = load_taxonomy()
        # Weight reasons by rough real-world frequency rather than uniformly: a
        # uniform draw would make integration bugs as common as insufficient funds
        # and quietly make the whole evaluation meaningless.
        weighted_reasons: list[tuple[str, float]] = []
        for reason_code, entry in table.items():
            w = {
                RC.FUNDS: 14.0,
                RC.AUTH_ABANDONED: 10.0,
                RC.AUTH_FAILED: 8.0,
                RC.GATEWAY_DOWN: 7.0,
                RC.ISSUER_DOWN: 6.0,
                RC.PSP_DOWN: 5.0,
                RC.INSTRUMENT_INVALID: 5.0,
                RC.LIMIT_EXCEEDED: 4.0,
                RC.INSTRUMENT_BLOCKED: 3.0,
                RC.MANDATE_PROBLEM: 3.0,
                RC.UNKNOWN: 3.0,
                RC.RISK_DECLINE: 2.0,
                RC.CREDIT_INELIGIBLE: 1.5,
                RC.INSTRUMENT_NOT_ENROLLED: 1.5,
            }.get(entry.root_cause, 0.6)
            weighted_reasons.append((reason_code, w))
        reasons = [x[0] for x in weighted_reasons]
        weights = [x[1] for x in weighted_reasons]

        customers = {}
        for i in range(n_customers):
            cid = f"cust_{i:05d}"
            customers[cid] = Customer(
                customer_id=cid,
                segment=r.choices(SEGMENTS, weights=[20, 35, 22, 15, 8])[0],
                has_consent=r.random() < 0.86,
                on_dnd_registry=r.random() < 0.12,
                preferred_channel=r.choices(
                    [Channel.WHATSAPP, Channel.SMS, Channel.EMAIL], weights=[55, 20, 25]
                )[0],
                language=r.choices(LANGUAGES, weights=[45, 25, 30])[0],
            )
        cust_ids = list(customers)

        base = datetime(2026, 8, 1, tzinfo=IST)
        events: list[AtRiskEvent] = []
        truth: dict[str, GroundTruth] = {}
        for i in range(n_events):
            kind = r.choices(list(RiskKind), weights=[40, 30, 20, 10])[0]
            cid = r.choice(cust_ids)
            reason: str | None = (
                r.choices(reasons, weights=weights)[0]
                if kind in {RiskKind.FAILED_PAYMENT, RiskKind.SUBSCRIPTION_FAILED}
                else None
            )
            ev = AtRiskEvent(
                event_id=f"evt_{i:06d}",
                customer_id=cid,
                kind=kind,
                amount_paise=max(100, self._amount(kind)),
                occurred_at=base + timedelta(minutes=r.randrange(0, 30 * 24 * 60)),
                error_reason=reason,
                method=r.choices(
                    ["upi", "card", "netbanking", "wallet", "emandate"],
                    weights=[48, 30, 12, 5, 5],
                )[0],
                attempt_number=r.choices([1, 2, 3], weights=[75, 18, 7])[0],
            )
            rc = self._root_cause_for(kind, reason)
            events.append(ev)
            truth[ev.event_id] = self._make_truth(ev, rc, customers[cid].segment)

        return Scenario(customers=customers, events=tuple(events), truth=truth, seed=self.seed)
