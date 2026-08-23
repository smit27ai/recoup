"""Durable recovery workflows.

A recovery sequence is not a request. It is a plan that unfolds over days: try the
card tonight, wait for payday, ask for a different instrument, wait, escalate, stop.
Between any two steps the process can be deployed over, the machine can die, and the
world can change -- the customer pays through another channel, opens a dispute,
promises to pay on the 15th, or revokes consent.

An in-process scheduler gets this wrong in a specific and expensive way: it holds the
plan in memory, so a restart either loses the sequence entirely (money quietly
abandoned) or replays it from the top (the customer gets the same message twice).
Temporal makes the sequence itself durable, so a workflow that went to sleep on
Tuesday wakes on Friday on a different machine with its history intact.

Three things this file is careful about.

**Gates are re-evaluated at every wake, never carried forward.** This is the claim
the rest of the codebase has been making since day one, and a multi-day workflow is
where it stops being theoretical. A step planned at 14:00 Tuesday that fires at 19:30
Friday must be checked against Friday at 19:30. `authorise_step` is an activity that
runs immediately before its action, every time.

**Determinism, which the gate design happened to get right for free.** Temporal
replays workflow code from history to rebuild state, so workflow code must be a pure
function of its inputs and history -- no clocks, no randomness, no I/O. Every gate in
this system already takes `now` as an explicit parameter rather than calling
`datetime.now()` internally, which makes them replay-safe by construction. Had they
read the clock themselves, every gate decision would have silently changed on replay
and the audit trail would have disagreed with itself.

**Signals stop the sequence, they do not just annotate it.** A `payment.captured`
webhook arriving on day two must halt the plan mid-flight -- the remaining scheduled
messages are for money that has already arrived, and sending them is the single most
visible way a recovery system embarrasses a merchant. Same for opt-out, disputes and
promises to pay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

with workflow.unsafe.imports_passed_through():
    from recoup.diagnosis.taxonomy import Diagnosis, RetryClass, diagnose
    from recoup.domain import ActionKind

TASK_QUEUE = "recoup-recovery"


def sandbox_runner() -> SandboxedWorkflowRunner:
    """Workflow sandbox with our own package passed through.

    Temporal sandboxes workflow code and blocks imports that could introduce
    non-determinism -- urllib among them. Importing this module drags in the parent
    `recoup.workflows` package, whose __init__ reaches the Razorpay client and
    therefore httpx and urllib, and validation fails before a single workflow runs.

    Passing `recoup` through is the right answer rather than a workaround, because
    the determinism guarantee it protects is one this codebase already keeps by
    construction: workflow code here calls no clock, no RNG and no I/O, and every
    gate takes `now` as an explicit parameter. What the sandbox would catch, the
    design already prevents. Side effects live in activities, which are not
    sandboxed at all.
    """
    return SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules("recoup")
    )


# How long to wait for a human before giving up on a parked action. Long enough to
# cover a weekend, short enough that money is not held hostage by an empty queue.
APPROVAL_TIMEOUT = timedelta(days=3)


@dataclass
class Step:
    """One scheduled attempt: wait this long, then try this."""

    delay: timedelta
    action: ActionKind
    note: str = ""


@dataclass
class RecoveryRequest:
    """Everything the workflow needs, captured at start.

    Deliberately a snapshot of IDs and amounts rather than live objects: workflow
    inputs are serialised into history and replayed, so anything that could change
    underneath must be re-fetched by an activity rather than trusted from here.
    """

    event_id: str
    customer_id: str
    amount_paise: int
    error_reason: str | None
    method: str = "card"
    attempt_number: int = 1


@dataclass
class GateOutcome:
    """One gate result, in the words it used at the time."""

    gate: str
    disposition: str
    reason: str


@dataclass
class RecordRequest:
    """Everything needed to write one ledger record for one step."""

    step: AuthoriseRequest
    error_reason: str | None
    gates: list[GateOutcome]
    executed_action: str
    detail: str
    arm: str


@dataclass
class StepOutcome:
    step: int
    action: str
    executed: bool
    detail: str
    at: str
    record_hash: str = ""
    """Ledger record this step produced. Empty only if recording itself failed."""


@dataclass
class RecoveryOutcome:
    event_id: str
    stopped_because: str
    steps: list[StepOutcome] = field(default_factory=list)
    recovered: bool = False

    @property
    def contacts_made(self) -> int:
        return sum(1 for s in self.steps if s.executed and "nudge" in s.action)


def plan_for(diagnosis: Diagnosis | None) -> list[Step]:
    """The escalation ladder for a root cause.

    Timing is part of the diagnosis, not a global constant. FUNDS waits for money to
    exist -- retrying an empty account four times in an hour is four failures and a
    worse issuer reputation, not four chances. A gateway outage is the opposite: it
    is likely over in minutes, so try again quickly and do not bother the customer at
    all, because it was never their problem.
    """
    if diagnosis is None or not diagnosis.in_scope:
        return [Step(timedelta(0), ActionKind.ROUTE_TO_OPS, "not customer-actionable")]

    if diagnosis.new_instrument:
        # Retrying is futile no matter how long we wait. Ask once, then again after a
        # few days, then stop -- a third ask is harassment, not recovery.
        return [
            Step(
                timedelta(hours=1),
                ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH,
                "ask for another method",
            ),
            Step(
                timedelta(days=3), ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH, "second and final ask"
            ),
        ]

    match diagnosis.retry_class:
        case RetryClass.NOW:
            # Transient. Retry quickly and silently; involve the customer only if the
            # quick retries fail, and only if they can actually do something.
            steps = [
                Step(timedelta(minutes=15), ActionKind.RETRY_NOW, "transient, retry soon"),
                Step(timedelta(hours=4), ActionKind.RETRY_NOW, "second quick retry"),
            ]
            if diagnosis.customer_action:
                steps.append(Step(timedelta(days=1), ActionKind.NUDGE, "retries exhausted"))
            return steps

        case RetryClass.SOON:
            return [
                Step(timedelta(hours=6), ActionKind.RETRY_SCHEDULED, "wait out the issue"),
                Step(timedelta(days=1), ActionKind.RETRY_SCHEDULED, "second attempt"),
                Step(timedelta(days=3), ActionKind.NUDGE, "still failing, tell the customer"),
            ]

        case RetryClass.SCHEDULED:
            # Money may not exist yet. Spread attempts across a pay cycle rather than
            # hammering an empty account.
            return [
                Step(timedelta(days=1), ActionKind.RETRY_SCHEDULED, "next day"),
                Step(timedelta(days=3), ActionKind.NUDGE, "let them choose the moment"),
                Step(timedelta(days=7), ActionKind.RETRY_SCHEDULED, "likely after payday"),
            ]

        case _:
            return (
                [Step(timedelta(hours=2), ActionKind.NUDGE, "only a human can fix this")]
                if diagnosis.customer_action
                else [
                    Step(timedelta(0), ActionKind.ROUTE_TO_OPS, "nothing to retry, nobody to tell")
                ]
            )


# --- activities: everything that touches the world --------------------------


@dataclass
class AuthoriseRequest:
    event_id: str
    customer_id: str
    amount_paise: int
    action: str
    attempt_number: int
    now_iso: str
    """Passed in from workflow.now(), so the gate decision is replay-stable."""


@dataclass
class AuthoriseResult:
    allowed: bool
    needs_approval: bool
    denials: list[str]
    explanation: str
    gates: list[GateOutcome] = field(default_factory=list)
    """Every gate that ran, passing ones included.

    Carried back through workflow history rather than cached in the activity, so the
    ledger record can be written by a DIFFERENT worker later without losing the
    reasons. A summary would make the audit trail say "blocked" without saying why,
    which is the difference between an audit trail and a log line.
    """


class RecoveryActivities:
    """Activity implementations, bound to a live engine.

    A class rather than free functions so the worker can be handed a configured
    engine -- with its Razorpay client, ledger and gates -- instead of reaching for
    module-level global state that tests cannot replace.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    @activity.defn(name="authorise_step")
    async def authorise_step(self, req: AuthoriseRequest) -> AuthoriseResult:
        """Run every gate against the clock AS IT IS NOW.

        The workflow passes its own notion of now, which on a replay is the original
        historical time -- so a decision made on Friday still reads as Friday when
        the history is replayed next year, and the audit trail does not drift.
        """
        from recoup.domain import ActionKind as AK
        from recoup.policy.gates import (
            Disposition,
            GateContext,
            ProposedAction,
            evaluate,
        )

        state = self.engine.load_state(req.customer_id, req.event_id)
        action = AK(req.action)
        verdict = evaluate(
            GateContext(
                action=ProposedAction(
                    event_id=req.event_id,
                    customer_id=req.customer_id,
                    kind=req.action,
                    is_contact=action.is_contact,
                    amount_paise=req.amount_paise,
                    idempotency_key=f"{req.event_id}:{req.action}:{req.attempt_number}",
                ),
                customer=state[0],
                event=state[1],
                now=datetime.fromisoformat(req.now_iso),
                config=self.engine.config,
            )
        )
        return AuthoriseResult(
            allowed=verdict.allowed,
            needs_approval=verdict.disposition is Disposition.NEEDS_APPROVAL,
            denials=[str(r.gate) for r in verdict.denials],
            explanation=verdict.explain(),
            gates=[
                GateOutcome(gate=str(r.gate), disposition=str(r.disposition), reason=r.reason)
                for r in verdict.results
            ],
        )

    @activity.defn(name="record_step")
    async def record_step(self, req: RecordRequest) -> str:
        """Write one ledger record for one step of the plan.

        Called on EVERY step, including the ones that did nothing. A workflow that
        recorded only its actions would answer "why did you message me twice" and go
        silent on "why did nobody chase this for a week" -- and the second question
        is the one a merchant actually asks.
        """
        return str(self.engine.record_step(req))

    @activity.defn(name="execute_step")
    async def execute_step(self, req: AuthoriseRequest) -> str:
        """Perform the authorised action. The only place a side effect happens.

        Idempotency is keyed on (event, action, attempt), so a Temporal activity
        retry after a lost response reuses the same key and cannot double-charge.
        """
        return str(self.engine.execute_for_workflow(req))


# --- the workflow -----------------------------------------------------------


async def _sleep_or_until(condition: Any, delay: timedelta) -> bool:
    """Sleep for `delay`, but wake early if `condition` becomes true.

    `workflow.wait_condition` RAISES on timeout rather than returning, which is the
    opposite of what a scheduled wait wants: here the timeout elapsing is the normal
    case and the condition firing is the exception. Returns True if the condition
    fired, False if the full delay elapsed.
    """
    if delay <= timedelta(0):
        return bool(condition())
    try:
        await workflow.wait_condition(condition, timeout=delay)
    except TimeoutError:
        return False
    return True


@workflow.defn
class RecoveryWorkflow:
    """One at-risk rupee, followed until it is recovered or the plan is exhausted."""

    def __init__(self) -> None:
        self._recovered = False
        self._stopped: str | None = None
        self._approved: bool | None = None
        self._promise_until: str | None = None

    # --- signals: the world changing under the plan ---

    @workflow.signal
    def payment_recovered(self) -> None:
        """A payment.captured webhook landed. Stop, now.

        The remaining scheduled steps are for money that has already arrived. Sending
        them is the most visible way a recovery system embarrasses a merchant.
        """
        self._recovered = True
        self._stopped = "recovered"

    @workflow.signal
    def customer_opted_out(self) -> None:
        self._stopped = "customer opted out"

    @workflow.signal
    def dispute_opened(self) -> None:
        self._stopped = "dispute opened"

    @workflow.signal
    def promised_to_pay(self, until_iso: str) -> None:
        """Honour a promise. Chasing someone before a date they gave you is bad faith
        and, under RBI conduct expectations, a complaint waiting to happen."""
        self._promise_until = until_iso

    @workflow.signal
    def approval_decided(self, approved: bool) -> None:
        self._approved = approved

    @workflow.query
    def status(self) -> dict[str, Any]:
        """Inspectable while running -- the ops console reads this."""
        return {
            "recovered": self._recovered,
            "stopped": self._stopped,
            "approved": self._approved,
            "promise_until": self._promise_until,
        }

    async def _record(
        self,
        req: AuthoriseRequest,
        request: RecoveryRequest,
        gates: list[GateOutcome],
        executed_action: str,
        detail: str,
    ) -> str:
        """Write the ledger record for one step, whatever happened.

        Recording is never allowed to fail the recovery: an audit trail that can halt
        a workflow is a liability, and a step that ran but went unrecorded is a
        smaller problem than a step that never ran because recording broke. The empty
        hash makes the omission visible rather than pretending it recorded.

        The bounded RetryPolicy is the load-bearing half of that, and its absence is
        a trap this workflow fell into: Temporal retries a failed activity FOREVER by
        default, so the try/except below never ran and a broken ledger wedged the
        whole recovery indefinitely -- the exact failure this method exists to
        prevent, arrived at by not configuring anything. A few retries cover a
        transient blip; past that the step is recorded as unrecorded and the plan
        moves on.
        """
        try:
            return str(
                await workflow.execute_activity(
                    "record_step",
                    RecordRequest(
                        step=req,
                        error_reason=request.error_reason,
                        gates=gates,
                        executed_action=executed_action,
                        detail=detail,
                        arm="treatment",
                    ),
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )
            )
        except Exception:
            return ""

    @workflow.run
    async def run(self, request: RecoveryRequest) -> RecoveryOutcome:
        diagnosis = diagnose(request.error_reason)
        plan = plan_for(diagnosis)
        outcome = RecoveryOutcome(event_id=request.event_id, stopped_because="plan exhausted")

        for index, step in enumerate(plan, start=1):
            # Sleep, but wake early if the world changes. A workflow that slept
            # through a payment.captured would send its next message into a void.
            await _sleep_or_until(lambda: self._stopped is not None, step.delay)
            if self._stopped is not None:
                outcome.stopped_because = self._stopped
                outcome.recovered = self._recovered
                return outcome

            # A promise to pay pushes the whole remaining plan out. Re-checked here
            # rather than at planning time, because it can arrive mid-sequence.
            if self._promise_until is not None:
                promised = _parse(self._promise_until)
                if workflow.now() < promised:
                    await _sleep_or_until(
                        lambda: self._stopped is not None, promised - workflow.now()
                    )
                    if self._stopped is not None:
                        outcome.stopped_because = self._stopped
                        outcome.recovered = self._recovered
                        return outcome

            req = AuthoriseRequest(
                event_id=request.event_id,
                customer_id=request.customer_id,
                amount_paise=request.amount_paise,
                action=str(step.action),
                attempt_number=index,
                # workflow.now() is replay-stable: on a replay it returns the
                # original historical time, not the wall clock of the replay.
                now_iso=workflow.now().isoformat(),
            )

            verdict = await workflow.execute_activity(
                "authorise_step",
                req,
                start_to_close_timeout=timedelta(seconds=30),
            )

            if verdict["needs_approval"]:
                self._approved = None
                await _sleep_or_until(
                    lambda: self._approved is not None or self._stopped is not None,
                    APPROVAL_TIMEOUT,
                )
                if self._stopped is not None:
                    outcome.stopped_because = self._stopped
                    outcome.recovered = self._recovered
                    return outcome
                if self._approved is not True:
                    # Rejected, or nobody looked in time. Either way the money stays
                    # unrecovered and that fact is recorded rather than retried around.
                    detail = "not approved within the review window"
                    outcome.steps.append(
                        StepOutcome(
                            step=index,
                            action=str(step.action),
                            executed=False,
                            detail=detail,
                            at=workflow.now().isoformat(),
                            record_hash=await self._record(
                                req,
                                request,
                                verdict["gates"],
                                str(ActionKind.QUEUED_FOR_APPROVAL),
                                detail,
                            ),
                        )
                    )
                    continue

            elif not verdict["allowed"]:
                detail = f"blocked: {verdict['explanation']}"
                outcome.steps.append(
                    StepOutcome(
                        step=index,
                        action=str(step.action),
                        executed=False,
                        detail=detail,
                        at=workflow.now().isoformat(),
                        record_hash=await self._record(
                            req, request, verdict["gates"], str(ActionKind.NO_ACTION), detail
                        ),
                    )
                )
                continue

            detail = await workflow.execute_activity(
                "execute_step",
                req,
                start_to_close_timeout=timedelta(minutes=2),
            )
            outcome.steps.append(
                StepOutcome(
                    step=index,
                    action=str(step.action),
                    executed=True,
                    detail=detail,
                    at=workflow.now().isoformat(),
                    record_hash=await self._record(
                        req, request, verdict["gates"], str(step.action), detail
                    ),
                )
            )

        outcome.recovered = self._recovered
        if self._stopped is not None:
            outcome.stopped_because = self._stopped
        return outcome


def _parse(iso: str) -> datetime:
    return datetime.fromisoformat(iso)
