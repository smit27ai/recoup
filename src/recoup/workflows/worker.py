"""`python -m recoup.workflows.worker` -- run a real Temporal worker.

Needs a Temporal server. The dev server is a single binary, no Docker:

    temporal server start-dev

The tests do not need any of this -- they use Temporal's time-skipping test
environment, which runs a seven-day recovery sequence in milliseconds. This module
exists for driving the real thing.
"""

from __future__ import annotations

import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from recoup.cli import build_client
from recoup.execution import Executor, RecordingNotifier
from recoup.workflows.backend import StateStore, WorkflowBackend
from recoup.workflows.recovery import (
    TASK_QUEUE,
    RecoveryActivities,
    RecoveryWorkflow,
    sandbox_runner,
)

TEMPORAL_ADDRESS = "localhost:7233"


async def run(address: str = TEMPORAL_ADDRESS) -> None:
    razorpay, mode = build_client()
    backend = WorkflowBackend(Executor(razorpay, RecordingNotifier()), StateStore())
    activities = RecoveryActivities(backend)

    client = await Client.connect(address)
    print(f"  worker on {TASK_QUEUE}  ({mode})")
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[RecoveryWorkflow],
        activities=[
            activities.authorise_step,
            activities.execute_step,
            activities.record_step,
        ],
        workflow_runner=sandbox_runner(),
    ):
        await asyncio.Event().wait()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n  stopped")
    except RuntimeError as exc:
        raise SystemExit(
            f"could not reach Temporal at {TEMPORAL_ADDRESS}: {exc}\n"
            "  start one with:  temporal server start-dev"
        ) from exc


if __name__ == "__main__":
    main()
