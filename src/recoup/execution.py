"""Executors: the only code in Recoup permitted to cause a side effect.

Everything upstream -- diagnosis, propensity, policy, gates -- is pure. It reads
state and returns decisions. This module is where a decision becomes an order, a
payment link, a message, or a queue entry, and it is deliberately the smallest
surface in the project.

The contract every executor obeys:

**Nothing here decides anything.** An executor receives an already-authorised action
and performs it. It never consults the taxonomy, never re-reads policy, and never
substitutes a different action because the requested one failed. If a payment link
cannot be raised, that is an execution failure to be recorded -- not licence to send
an SMS instead.

**Contact is impossible to reach except through the gate.** `Notifier.send` is the
only path to a customer, and the engine calls it only after a Verdict allowed it.
The Razorpay client separately disables the processor's own SMS/email, because that
would be a second, ungated path to the same person.

**Uncertain outcomes are surfaced, never swallowed.** When the Razorpay client
raises `UncertainOutcome`, the executor reconciles and reports what it found. It
does not retry, and it does not guess.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from recoup.domain import ActionKind, AtRiskEvent, Channel, Customer
from recoup.razorpay.client import (
    ApiError,
    RazorpayClient,
    UncertainOutcome,
    receipt_for,
)


class ExecutionStatus(StrEnum):
    DONE = "done"
    SKIPPED = "skipped"
    """Nothing to do -- NO_ACTION, or a holdout."""
    QUEUED = "queued"
    """Handed to a human queue: ops triage or approval."""
    FAILED = "failed"
    """The side effect definitively did not happen."""
    RECONCILED = "reconciled"
    """Outcome was unknown; we looked, and it HAD happened. Not retried."""
    UNRESOLVED = "unresolved"
    """Outcome unknown AND reconciliation failed. The one state a human must see."""


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    status: ExecutionStatus
    action: ActionKind
    detail: str
    """Written verbatim to the ledger. Say what happened, with identifiers."""
    artifacts: dict[str, str] = field(default_factory=dict)
    """order_id, payment_link_id, short_url -- whatever a human would need to look
    this up in the Razorpay dashboard."""

    @property
    def needs_human(self) -> bool:
        return self.status in {ExecutionStatus.UNRESOLVED, ExecutionStatus.QUEUED}


class Notifier(Protocol):
    """The only route to a customer.

    Kept as a Protocol so the real BSP integration can drop in without the engine
    changing, and so tests can assert on exactly what would have been sent.
    """

    def send(self, *, customer: Customer, channel: Channel, body: str, link: str | None) -> str: ...


@dataclass(slots=True)
class RecordingNotifier:
    """Records what would be sent instead of sending it.

    This is the default, and not only because there is no BSP wired up: a recovery
    system under development must not be one misconfiguration away from messaging
    real people. Swapping this out should be a deliberate act.
    """

    sent: list[dict[str, str]] = field(default_factory=list)

    def send(self, *, customer: Customer, channel: Channel, body: str, link: str | None) -> str:
        message_id = f"msg_{len(self.sent):06d}"
        self.sent.append(
            {
                "message_id": message_id,
                "customer_id": customer.customer_id,
                "channel": str(channel),
                "language": customer.language,
                "body": body,
                "link": link or "",
            }
        )
        return message_id

    def to(self, customer_id: str) -> list[dict[str, str]]:
        return [m for m in self.sent if m["customer_id"] == customer_id]


@dataclass(slots=True)
class WorkQueue:
    """Human queues. Ops triage and approvals are different queues on purpose --
    they are worked by different people with different authority."""

    name: str
    items: list[dict[str, str]] = field(default_factory=list)

    def push(self, event: AtRiskEvent, reason: str, extra: dict[str, str] | None = None) -> None:
        self.items.append(
            {
                "event_id": event.event_id,
                "customer_id": event.customer_id,
                "amount_paise": str(event.amount_paise),
                "reason": reason,
                **(extra or {}),
            }
        )

    def __len__(self) -> int:
        return len(self.items)

    @property
    def total_paise(self) -> int:
        return sum(int(i["amount_paise"]) for i in self.items)


# Copy is intentionally plain and free of urgency or threat. A recovery message
# that manufactures pressure is both worse-performing and, under RBI conduct rules,
# a liability. Tier 3 (an LLM) will localise these into Hindi/Hinglish; the English
# skeletons stay here so the non-LLM path is always functional.
TEMPLATES: dict[ActionKind, str] = {
    ActionKind.NUDGE: (
        "Your payment of {amount} did not go through. "
        "You can complete it here whenever convenient: {link}"
    ),
    ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH: (
        "Your payment of {amount} could not be completed with the saved payment method. "
        "You can pay using a different method here: {link}"
    ),
    ActionKind.NUDGE_WITH_INCENTIVE: (
        "Your payment of {amount} did not go through. "
        "Complete it here and we will apply a {discount} discount: {link}"
    ),
}


class Executor:
    """Turns an authorised action into a real side effect."""

    def __init__(
        self,
        client: RazorpayClient,
        notifier: Notifier | None = None,
        *,
        ops_queue: WorkQueue | None = None,
        approval_queue: WorkQueue | None = None,
    ) -> None:
        self.client = client
        self.notifier: Notifier = notifier if notifier is not None else RecordingNotifier()
        # Explicit `is None` rather than `or`: WorkQueue defines __len__, so an empty
        # queue is falsy and `or` would silently discard the caller's queue. Same
        # class of bug as the one that hit IdempotencyStore.
        self.ops_queue = ops_queue if ops_queue is not None else WorkQueue("ops")
        self.approval_queue = (
            approval_queue if approval_queue is not None else WorkQueue("approval")
        )

    def execute(
        self,
        action: ActionKind,
        event: AtRiskEvent,
        customer: Customer,
        *,
        now: datetime,
        discount_bps: int = 0,
    ) -> ExecutionResult:
        match action:
            case ActionKind.NO_ACTION:
                return ExecutionResult(ExecutionStatus.SKIPPED, action, "no action taken")
            case ActionKind.ROUTE_TO_OPS:
                self.ops_queue.push(
                    event, f"root cause not customer-actionable ({event.error_reason})"
                )
                return ExecutionResult(ExecutionStatus.QUEUED, action, "routed to ops triage queue")
            case ActionKind.QUEUED_FOR_APPROVAL:
                self.approval_queue.push(event, "exceeds unattended authority")
                return ExecutionResult(ExecutionStatus.QUEUED, action, "parked for human approval")
            case ActionKind.RETRY_NOW | ActionKind.RETRY_SCHEDULED:
                return self._retry(action, event)
            case _:
                return self._contact(action, event, customer, now=now, discount_bps=discount_bps)

    # --- side effects -------------------------------------------------------

    def _retry(self, action: ActionKind, event: AtRiskEvent) -> ExecutionResult:
        """A silent re-attempt. Creates a fresh order for the same amount.

        No message goes out, which is why the quiet-hours gate does not apply: this
        disturbs nobody.
        """
        receipt = receipt_for(event.event_id, event.attempt_number, "order")
        try:
            order = self.client.create_order(
                event.amount_paise,
                receipt=receipt,
                notes={"recoup_event": event.event_id, "recoup_action": str(action)},
            )
        except UncertainOutcome as exc:
            return self._reconcile(action, exc, receipt)
        except ApiError as exc:
            return ExecutionResult(
                ExecutionStatus.FAILED, action, f"order creation rejected: {exc}"
            )
        return ExecutionResult(
            ExecutionStatus.DONE,
            action,
            f"order {order.get('id')} created for retry",
            {"order_id": str(order.get("id", "")), "receipt": receipt},
        )

    def _contact(
        self,
        action: ActionKind,
        event: AtRiskEvent,
        customer: Customer,
        *,
        now: datetime,
        discount_bps: int,
    ) -> ExecutionResult:
        """Raise a payment link, then send exactly one message carrying it.

        Order matters: if the link cannot be raised there is nothing useful to send,
        and a message telling someone to pay without saying how is worse than
        silence. We never send first and hope.
        """
        reference = receipt_for(event.event_id, event.attempt_number, "link")
        try:
            link = self.client.create_payment_link(
                event.amount_paise,
                reference_id=reference,
                description=f"Payment for {event.kind}",
                customer_contact=None,
                notes={"recoup_event": event.event_id, "recoup_action": str(action)},
            )
        except UncertainOutcome as exc:
            # A link may exist, but we have nothing to put in a message and must not
            # send a broken one. Surface it; do not contact.
            return ExecutionResult(
                ExecutionStatus.UNRESOLVED,
                action,
                f"payment link outcome unknown, no message sent: {exc.cause!r}",
                {"reference_id": reference},
            )
        except ApiError as exc:
            return ExecutionResult(
                ExecutionStatus.FAILED, action, f"payment link rejected, no message sent: {exc}"
            )

        short_url = str(link.get("short_url", ""))
        template = TEMPLATES.get(action, TEMPLATES[ActionKind.NUDGE])
        body = template.format(
            amount=f"Rs.{event.amount_paise / 100:,.2f}",
            link=short_url,
            discount=f"{discount_bps / 100:.0f}%",
        )
        message_id = self.notifier.send(
            customer=customer,
            channel=customer.preferred_channel,
            body=body,
            link=short_url,
        )
        return ExecutionResult(
            ExecutionStatus.DONE,
            action,
            f"payment link {link.get('id')} raised and {customer.preferred_channel} sent",
            {
                "payment_link_id": str(link.get("id", "")),
                "short_url": short_url,
                "message_id": message_id,
                "reference_id": reference,
            },
        )

    def _reconcile(
        self, action: ActionKind, exc: UncertainOutcome, receipt: str
    ) -> ExecutionResult:
        """Resolve an unknown outcome by looking, not by guessing.

        Three ways out, and all three are recorded honestly:
          - it HAD happened      -> RECONCILED, and we do not do it again
          - it had not           -> FAILED, safe for the caller to try later
          - we could not tell    -> UNRESOLVED, a human has to look
        """
        try:
            existing = self.client.reconcile_order(receipt)
        except Exception as probe_error:
            return ExecutionResult(
                ExecutionStatus.UNRESOLVED,
                action,
                f"outcome unknown and reconciliation failed: {probe_error!r}",
                {"receipt": receipt},
            )

        if existing is not None:
            return ExecutionResult(
                ExecutionStatus.RECONCILED,
                action,
                f"outcome was unknown; order {existing.get('id')} already existed, not repeated",
                {"order_id": str(existing.get("id", "")), "receipt": receipt},
            )
        return ExecutionResult(
            ExecutionStatus.FAILED,
            action,
            f"outcome was unknown; confirmed nothing was created ({exc.cause!r})",
            {"receipt": receipt},
        )


def summarise(results: Sequence[ExecutionResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        counts[str(r.status)] = counts.get(str(r.status), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
