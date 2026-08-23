"""The spine: webhook -> diagnose -> policy -> gates -> execute -> ledger.

One function, `handle`, carries a single at-risk rupee through every stage, and one
ledger record comes out the other end explaining the whole journey. Everything else
in Recoup exists to serve this path.

The ordering is not arbitrary. Two constraints fix it:

**Gates are evaluated immediately before execution, never at decision time.**
Diagnosis and policy can happen whenever -- they are pure functions of state. But a
workflow may have been planned days before it runs, and in between the customer can
revoke consent, open a dispute, promise to pay, or simply have the clock roll past
19:00. So `decide()` produces an *intent* and deliberately does not authorise
anything; `authorise()` runs the gates against the clock at the moment of action.
Splitting these is what makes the multi-day case correct rather than accidentally
correct for same-second execution.

**The ledger is written on every path, including the boring ones.**
A decision not to act is a decision, and it is the one a merchant is most likely to
ask about later ("why did nobody chase this invoice?"). Recording only the actions
produces an audit trail that answers the easy questions and goes silent on the hard
ones.

The holdout deserves a note. It is enforced here, at the top of the pipeline, and
not in policy -- a holdout event goes through diagnosis and policy exactly like any
other so the intent is recorded, then is forced to NO_ACTION before execution. That
way the ledger shows what we *would* have done, which is what makes the counterfactual
interpretable rather than a hole in the data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime

from recoup.diagnosis.taxonomy import Diagnosis, diagnose
from recoup.domain import ActionKind, Arm, AtRiskEvent, Customer
from recoup.execution import ExecutionResult, ExecutionStatus, Executor
from recoup.ledger import DecisionRecord, Ledger
from recoup.policy.gates import (
    CustomerState,
    Disposition,
    EventState,
    GateContext,
    PolicyConfig,
    ProposedAction,
    Verdict,
    evaluate,
)
from recoup.policy.strategies import Strategy, taxonomy_policy

POLICY_VERSION = "taxonomy-v1"
TAXONOMY_VERSION = "2026-08-23"
INCENTIVE_BPS = 1500


@dataclass(frozen=True, slots=True)
class Intent:
    """What we would like to do. Carries no authority whatsoever."""

    event: AtRiskEvent
    diagnosis: Diagnosis | None
    action: ActionKind
    arm: Arm

    @property
    def discount_bps(self) -> int:
        return INCENTIVE_BPS if self.action is ActionKind.NUDGE_WITH_INCENTIVE else 0


@dataclass(frozen=True, slots=True)
class Handled:
    """Everything that happened to one event, and the record proving it."""

    intent: Intent
    verdict: Verdict
    executed: ActionKind
    result: ExecutionResult
    record: DecisionRecord

    @property
    def acted(self) -> bool:
        return self.result.status is ExecutionStatus.DONE

    def explain(self) -> str:
        return self.record.explain()


class RecoveryEngine:
    """Binds the pure decision layers to the side-effecting one."""

    def __init__(
        self,
        executor: Executor,
        ledger: Ledger | None = None,
        *,
        strategy: Strategy = taxonomy_policy,
        config: PolicyConfig | None = None,
        holdout_rate: float = 0.20,
        seed: int = 20260905,
    ) -> None:
        self.executor = executor
        # Explicit None checks throughout: Ledger defines __len__, so `or` would
        # silently swap out a caller's empty ledger. See client.IdempotencyStore.
        self.ledger = ledger if ledger is not None else Ledger()
        self.strategy = strategy
        self.config = config if config is not None else PolicyConfig()
        self.holdout_rate = holdout_rate
        self._rng = random.Random(seed)

    # --- stage 1: decide (pure) --------------------------------------------

    def decide(self, event: AtRiskEvent, customer: Customer) -> Intent:
        """Diagnose and choose. Touches nothing and authorises nothing."""
        diagnosis = diagnose(event.error_reason)
        action = self.strategy(event, diagnosis, customer)
        arm = Arm.HOLDOUT if self._rng.random() < self.holdout_rate else Arm.TREATMENT
        return Intent(event=event, diagnosis=diagnosis, action=action, arm=arm)

    # --- stage 2: authorise (pure, but clock-sensitive) --------------------

    def authorise(
        self, intent: Intent, customer_state: CustomerState, event_state: EventState, now: datetime
    ) -> Verdict:
        """Run every gate against the clock as it is RIGHT NOW.

        Called immediately before execution, never cached from decision time.
        """
        return evaluate(
            GateContext(
                action=ProposedAction(
                    event_id=intent.event.event_id,
                    customer_id=intent.event.customer_id,
                    kind=str(intent.action),
                    is_contact=intent.action.is_contact,
                    amount_paise=intent.event.amount_paise,
                    discount_bps=intent.discount_bps,
                    idempotency_key=(
                        f"{intent.event.event_id}:{intent.action}:{intent.event.attempt_number}"
                    ),
                ),
                customer=customer_state,
                event=event_state,
                now=now,
                config=self.config,
            )
        )

    # --- stage 3: the whole path -------------------------------------------

    def handle(
        self,
        event: AtRiskEvent,
        customer: Customer,
        customer_state: CustomerState,
        event_state: EventState,
        now: datetime,
    ) -> Handled:
        intent = self.decide(event, customer)
        verdict = self.authorise(intent, customer_state, event_state, now)

        executed = self._resolve(intent, verdict)
        result = self.executor.execute(
            executed, event, customer, now=now, discount_bps=intent.discount_bps
        )

        record = self.ledger.append(
            event=event,
            diagnosis=intent.diagnosis,
            intended=intent.action,
            verdict=verdict,
            executed=executed,
            arm=intent.arm,
            decided_at=now,
            recovered=None,  # settled later, by a payment.captured webhook
            policy_version=POLICY_VERSION,
            taxonomy_version=TAXONOMY_VERSION,
            metadata={
                "execution_status": str(result.status),
                "execution_detail": result.detail,
                **result.artifacts,
            },
        )
        return Handled(
            intent=intent, verdict=verdict, executed=executed, result=result, record=record
        )

    @staticmethod
    def _resolve(intent: Intent, verdict: Verdict) -> ActionKind:
        """Reduce intent plus verdict to the one action that may actually run.

        Holdout first, and unconditionally: a holdout event must not act even when
        every gate would have allowed it, or the control arm is contaminated and the
        entire incrementality measurement becomes worthless.
        """
        if intent.arm is Arm.HOLDOUT:
            return ActionKind.NO_ACTION
        if verdict.allowed:
            return intent.action
        if verdict.disposition is Disposition.NEEDS_APPROVAL:
            return ActionKind.QUEUED_FOR_APPROVAL
        return ActionKind.NO_ACTION

    # --- settlement ---------------------------------------------------------

    def settle(self, event_id: str, recovered: bool, now: datetime) -> DecisionRecord | None:
        """Record the outcome once a payment.captured / invoice.paid webhook lands.

        Appends a NEW record rather than editing the old one. The ledger is
        append-only, so an outcome arriving later is a fact added to history, never
        a correction applied to it -- editing would break the chain, which is
        precisely the property we want.
        """
        prior = self.ledger.for_event(event_id)
        if not prior:
            return None
        last = prior[-1]
        return self.ledger.append(
            event=AtRiskEvent(
                event_id=event_id,
                customer_id=last.customer_id,
                kind=last.metadata.get("kind", "failed_payment"),  # type: ignore[arg-type]
                amount_paise=last.amount_paise,
                occurred_at=now,
                error_reason=last.error_reason,
                method=last.metadata.get("method", "unknown"),
            ),
            diagnosis=None,
            intended=ActionKind.NO_ACTION,
            verdict=Verdict(results=(), now=now),
            executed=ActionKind.NO_ACTION,
            arm=Arm(last.arm),
            decided_at=now,
            recovered=recovered,
            policy_version=POLICY_VERSION,
            taxonomy_version=TAXONOMY_VERSION,
            metadata={"settlement_for": last.record_hash, "execution_status": "settled"},
        )
