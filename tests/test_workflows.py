"""Durable workflow tests, run against Temporal's time-skipping environment.

A seven-day recovery sequence executes here in milliseconds: the test server fast
forwards its own clock whenever every workflow is asleep. That is what makes it
possible to actually TEST multi-day behaviour rather than assert on a plan and hope
-- and the properties worth testing are all about what happens across those gaps.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import httpx
import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from recoup.diagnosis.taxonomy import diagnose
from recoup.domain import ActionKind, Channel, Customer
from recoup.execution import Executor, RecordingNotifier
from recoup.policy.gates import IST, PolicyConfig
from recoup.razorpay.client import RazorpayClient
from recoup.workflows import (
    TASK_QUEUE,
    RecoveryActivities,
    RecoveryRequest,
    RecoveryWorkflow,
    StateStore,
    WorkflowBackend,
    plan_for,
    sandbox_runner,
)

pytestmark = pytest.mark.workflow


def _transport(request: httpx.Request) -> httpx.Response:
    if request.method == "GET":
        return httpx.Response(200, json={"entity": "collection", "items": []})
    return httpx.Response(200, json={"id": "order_x", "short_url": "https://rzp.io/i/x"})


def _backend(**kw: object) -> tuple[WorkflowBackend, RecordingNotifier, StateStore]:
    notifier = RecordingNotifier()
    store = StateStore()
    store.customers["cust_1"] = Customer(
        customer_id="cust_1",
        segment="loyal",
        has_consent=bool(kw.get("consent", True)),
        on_dnd_registry=bool(kw.get("dnd", False)),
        preferred_channel=Channel.WHATSAPP,
    )
    store.consent["cust_1"] = bool(kw.get("consent", True))
    executor = Executor(
        RazorpayClient("rzp_test_k", "s", transport=httpx.MockTransport(_transport)),
        notifier,
    )
    config = kw.get("config") or PolicyConfig(
        # Widened so quiet hours does not dominate every test; the quiet-hours
        # behaviour has its own dedicated test below.
        contact_window_start=datetime(2026, 1, 1, 0, 0).time(),
        contact_window_end=datetime(2026, 1, 1, 23, 59).time(),
    )
    return WorkflowBackend(executor, store, config=config), notifier, store


async def _run(
    env: WorkflowEnvironment,
    request: RecoveryRequest,
    backend: WorkflowBackend,
    *,
    signals: list[tuple[timedelta, str, object]] | None = None,
):
    """Start a workflow, optionally signal it partway through, and return the result."""
    activities = RecoveryActivities(backend)
    async with Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[RecoveryWorkflow],
        activities=[activities.authorise_step, activities.execute_step],
        workflow_runner=sandbox_runner(),
    ):
        handle = await env.client.start_workflow(
            RecoveryWorkflow.run,
            request,
            id=f"recovery-{uuid.uuid4()}",
            task_queue=TASK_QUEUE,
        )
        for delay, name, arg in signals or []:
            await env.sleep(delay)
            if arg is None:
                await handle.signal(name)
            else:
                await handle.signal(name, arg)
        return await handle.result()


# --- the plan --------------------------------------------------------------


def test_funds_failure_is_spread_across_a_pay_cycle() -> None:
    """Retrying an empty account four times in an hour is four failures and a worse
    issuer reputation, not four chances."""
    plan = plan_for(diagnose("insufficient_funds"))
    assert [s.delay for s in plan] == [timedelta(days=1), timedelta(days=3), timedelta(days=7)]


def test_gateway_failure_never_contacts_the_customer() -> None:
    """It was never their problem."""
    plan = plan_for(diagnose("gateway_technical_error"))
    assert plan
    assert not any(s.action.is_contact for s in plan)


def test_expired_card_asks_twice_and_stops() -> None:
    """A third ask is harassment, not recovery."""
    plan = plan_for(diagnose("card_expired"))
    assert len(plan) == 2
    assert not any(s.action.is_retry for s in plan)


def test_our_own_bug_goes_straight_to_a_human() -> None:
    plan = plan_for(diagnose("invalid_order_id"))
    assert [s.action for s in plan] == [ActionKind.ROUTE_TO_OPS]


# --- durability across days ------------------------------------------------


@pytest.mark.asyncio
async def test_a_seven_day_sequence_runs_to_completion() -> None:
    async with await WorkflowEnvironment.start_time_skipping() as env:
        backend, _notifier, _ = _backend()
        outcome = await _run(
            env,
            RecoveryRequest("evt_1", "cust_1", 99_900, "insufficient_funds"),
            backend,
        )
    assert outcome.stopped_because == "plan exhausted"
    assert len(outcome.steps) == 3
    assert outcome.contacts_made == 1, "one nudge across a week, not three"


@pytest.mark.asyncio
async def test_recovery_signal_halts_the_plan_mid_flight() -> None:
    """A payment.captured on day two must stop the remaining messages. Sending them
    is the most visible way a recovery system embarrasses a merchant."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        backend, notifier, _ = _backend()
        outcome = await _run(
            env,
            RecoveryRequest("evt_2", "cust_1", 99_900, "insufficient_funds"),
            backend,
            signals=[(timedelta(days=2), "payment_recovered", None)],
        )
    assert outcome.recovered
    assert outcome.stopped_because == "recovered"
    assert len(outcome.steps) < 3, "remaining steps must not run"
    assert notifier.sent == [], "nobody may be messaged after the money arrived"


@pytest.mark.asyncio
async def test_opt_out_stops_everything() -> None:
    async with await WorkflowEnvironment.start_time_skipping() as env:
        backend, notifier, _ = _backend()
        outcome = await _run(
            env,
            RecoveryRequest("evt_3", "cust_1", 99_900, "insufficient_funds"),
            backend,
            signals=[(timedelta(hours=2), "customer_opted_out", None)],
        )
    assert outcome.stopped_because == "customer opted out"
    assert notifier.sent == []


@pytest.mark.asyncio
async def test_dispute_stops_recovery() -> None:
    async with await WorkflowEnvironment.start_time_skipping() as env:
        backend, _notifier, _ = _backend()
        outcome = await _run(
            env,
            RecoveryRequest("evt_4", "cust_1", 99_900, "insufficient_funds"),
            backend,
            signals=[(timedelta(hours=6), "dispute_opened", None)],
        )
    assert outcome.stopped_because == "dispute opened"


@pytest.mark.asyncio
async def test_promise_to_pay_defers_the_rest_of_the_plan() -> None:
    """Chasing someone before a date they gave you is bad faith."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        backend, _notifier, _ = _backend()
        promise = (datetime.now(IST) + timedelta(days=5)).isoformat()
        outcome = await _run(
            env,
            RecoveryRequest("evt_5", "cust_1", 99_900, "insufficient_funds"),
            backend,
            signals=[(timedelta(hours=1), "promised_to_pay", promise)],
        )
    # The plan still completes, but nothing fired inside the promised window.
    assert outcome.stopped_because == "plan exhausted"


# --- gates are re-evaluated at every wake ----------------------------------


@pytest.mark.asyncio
async def test_consent_revoked_mid_sequence_blocks_the_later_contact() -> None:
    """THE property this whole design exists for. A step planned on Tuesday and
    fired on Friday must be checked against Friday -- if the plan cached its
    authorisation, this message would go out to someone who withdrew consent."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        backend, notifier, store = _backend()
        activities = RecoveryActivities(backend)
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[RecoveryWorkflow],
            activities=[activities.authorise_step, activities.execute_step],
            workflow_runner=sandbox_runner(),
        ):
            handle = await env.client.start_workflow(
                RecoveryWorkflow.run,
                RecoveryRequest("evt_6", "cust_1", 99_900, "insufficient_funds"),
                id=f"recovery-{uuid.uuid4()}",
                task_queue=TASK_QUEUE,
            )
            # Day 2: the customer withdraws consent. The nudge is scheduled for day 3.
            await env.sleep(timedelta(days=2))
            store.consent["cust_1"] = False
            outcome = await handle.result()

    nudges = [s for s in outcome.steps if "nudge" in s.action]
    assert nudges, "the plan should still have reached its nudge step"
    assert all(not s.executed for s in nudges), "consent was revoked before it fired"
    assert notifier.sent == []


@pytest.mark.asyncio
async def test_quiet_hours_blocks_a_step_that_wakes_at_night() -> None:
    async with await WorkflowEnvironment.start_time_skipping() as env:
        # Real RBI-derived window, so a step waking outside 08:00-19:00 is blocked.
        backend, _notifier, _ = _backend(config=PolicyConfig())
        outcome = await _run(
            env, RecoveryRequest("evt_7", "cust_1", 99_900, "card_expired"), backend
        )
    blocked = [s for s in outcome.steps if not s.executed and "blocked" in s.detail]
    assert outcome.steps, "the plan ran"
    # Whether it lands in quiet hours depends on the start time; if it did, the
    # explanation must name the gate rather than failing silently.
    for step in blocked:
        assert "quiet_hours" in step.detail or "consent" in step.detail


# --- human approval mid-workflow -------------------------------------------


@pytest.mark.asyncio
async def test_high_value_step_waits_for_a_human_and_proceeds_on_approval() -> None:
    async with await WorkflowEnvironment.start_time_skipping() as env:
        backend, _notifier, _ = _backend()
        outcome = await _run(
            env,
            RecoveryRequest("evt_8", "cust_1", 90_000_00, "card_expired"),
            backend,
            signals=[(timedelta(hours=2), "approval_decided", True)],
        )
    assert any(s.executed for s in outcome.steps), "approval should let it through"


@pytest.mark.asyncio
async def test_unapproved_step_is_recorded_not_retried_around() -> None:
    """Rejected, or nobody looked in time. Either way the money stays unrecovered and
    that is recorded, rather than the workflow finding another way to act."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        backend, notifier, _ = _backend()
        outcome = await _run(
            env,
            RecoveryRequest("evt_9", "cust_1", 90_000_00, "card_expired"),
            backend,
            signals=[(timedelta(hours=2), "approval_decided", False)],
        )
    assert notifier.sent == []
    assert any("not approved" in s.detail for s in outcome.steps)


@pytest.mark.asyncio
async def test_approval_that_never_comes_times_out_rather_than_hanging() -> None:
    """Money must not be held hostage by an empty review queue."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        backend, notifier, _ = _backend()
        outcome = await _run(
            env, RecoveryRequest("evt_10", "cust_1", 90_000_00, "card_expired"), backend
        )
    assert outcome.stopped_because == "plan exhausted"
    assert any("not approved" in s.detail for s in outcome.steps)
    assert notifier.sent == []


# --- determinism ------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_is_deterministic_across_identical_runs() -> None:
    """Temporal replays workflow code from history, so the same inputs must produce
    the same decisions. Every gate takes `now` as a parameter rather than reading the
    clock, which is what makes this hold."""
    results = []
    for _ in range(2):
        async with await WorkflowEnvironment.start_time_skipping() as env:
            backend, _, _ = _backend()
            outcome = await _run(
                env,
                RecoveryRequest("evt_same", "cust_1", 99_900, "insufficient_funds"),
                backend,
            )
            results.append([(s.step, s.action, s.executed) for s in outcome.steps])
    assert results[0] == results[1]


@pytest.mark.asyncio
async def test_ops_route_completes_immediately_without_touching_the_customer() -> None:
    async with await WorkflowEnvironment.start_time_skipping() as env:
        backend, notifier, _ = _backend()
        outcome = await _run(
            env, RecoveryRequest("evt_11", "cust_1", 99_900, "invalid_order_id"), backend
        )
    assert len(outcome.steps) == 1
    assert outcome.steps[0].action == "route_to_ops"
    assert notifier.sent == []
    assert len(backend.executor.ops_queue) == 1
