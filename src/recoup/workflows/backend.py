"""What the workflow activities talk to.

Kept separate from `RecoveryEngine` on purpose. The engine handles one event start to
finish in a single call; a workflow needs something different -- the ability to look
up current state at an arbitrary later moment, and to execute one step of a plan
without re-deciding the whole thing.

The state lookup is the important part. A workflow that slept four days must not act
on the state it was started with. Consent, contact history, disputes and promises all
change while it sleeps, and reading them fresh at each wake is the entire reason the
gates are re-evaluated rather than cached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from recoup.domain import ActionKind, AtRiskEvent, Channel, Customer, RiskKind
from recoup.execution import Executor
from recoup.policy.gates import CustomerState, EventState, PolicyConfig

if TYPE_CHECKING:
    from recoup.workflows.recovery import AuthoriseRequest


@dataclass
class StateStore:
    """Current truth about customers and events.

    In-memory here; a table in production. The contract is that it reflects the world
    NOW, not when the workflow started -- which is why activities call it on every
    wake instead of the workflow caching a snapshot.
    """

    customers: dict[str, Customer] = field(default_factory=dict)
    contacts: dict[str, list[datetime]] = field(default_factory=dict)
    opted_out: set[str] = field(default_factory=set)
    consent: dict[str, bool] = field(default_factory=dict)
    disputes: set[str] = field(default_factory=set)
    recovered: set[str] = field(default_factory=set)
    attempts: dict[str, int] = field(default_factory=dict)
    promises: dict[str, datetime] = field(default_factory=dict)
    events: dict[str, AtRiskEvent] = field(default_factory=dict)

    def customer_state(self, customer_id: str) -> CustomerState:
        customer = self.customers.get(customer_id)
        history = self.contacts.get(customer_id, [])
        return CustomerState(
            customer_id=customer_id,
            has_consent=self.consent.get(customer_id, customer.has_consent if customer else True),
            on_dnd_registry=customer.on_dnd_registry if customer else False,
            contacts_in_window=tuple(history),
            last_contact_at=history[-1] if history else None,
            opted_out=customer_id in self.opted_out,
        )

    def event_state(self, event_id: str) -> EventState:
        return EventState(
            event_id=event_id,
            attempts_so_far=self.attempts.get(event_id, 0),
            already_recovered=event_id in self.recovered,
            dispute_open=event_id in self.disputes,
            promise_to_pay_until=self.promises.get(event_id),
        )

    def record_contact(self, customer_id: str, at: datetime) -> None:
        self.contacts.setdefault(customer_id, []).append(at)

    def record_attempt(self, event_id: str) -> None:
        self.attempts[event_id] = self.attempts.get(event_id, 0) + 1


class WorkflowBackend:
    """Binds a workflow's activities to a real executor, ledger and state store."""

    def __init__(
        self,
        executor: Executor,
        store: StateStore | None = None,
        *,
        config: PolicyConfig | None = None,
    ) -> None:
        self.executor = executor
        # Explicit None checks: several of these define __len__ elsewhere in the
        # codebase and `or` has already caused one double-charge bug here.
        self.store = store if store is not None else StateStore()
        self.config = config if config is not None else PolicyConfig()

    def load_state(self, customer_id: str, event_id: str) -> tuple[CustomerState, EventState]:
        """Read the world as it is right now, not as it was when the plan was made."""
        return self.store.customer_state(customer_id), self.store.event_state(event_id)

    def execute_for_workflow(self, req: AuthoriseRequest) -> str:
        """Run one authorised step.

        Deliberately does NOT re-check the gates: authorisation already happened in
        its own activity moments ago, and re-deciding here would mean two different
        answers could exist for one step with no record of which one applied.
        """
        action = ActionKind(req.action)
        event = self.store.events.get(req.event_id) or AtRiskEvent(
            event_id=req.event_id,
            customer_id=req.customer_id,
            kind=RiskKind.FAILED_PAYMENT,
            amount_paise=req.amount_paise,
            occurred_at=datetime.fromisoformat(req.now_iso),
            error_reason=None,
            method="card",
            attempt_number=req.attempt_number,
        )
        customer = self.store.customers.get(req.customer_id) or Customer(
            customer_id=req.customer_id,
            segment="casual",
            has_consent=True,
            on_dnd_registry=False,
            preferred_channel=Channel.WHATSAPP,
        )

        now = datetime.fromisoformat(req.now_iso)
        result = self.executor.execute(action, event, customer, now=now)

        self.store.record_attempt(req.event_id)
        if action.is_contact and result.status.value == "done":
            self.store.record_contact(req.customer_id, now)
        return f"{result.status}: {result.detail}"
