"""`python -m recoup.console.server` -- run the console API with seeded data.

Seeds from the same synthetic scenario the CLI demo uses, so the console has real
decisions, a real approval queue and real mined rules to work with the moment it
starts. Without keys it drives the local double; with them, real test mode.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import uvicorn

from recoup.cli import build_client
from recoup.console.api import ConsoleState, create_app
from recoup.diagnosis.escalation import EscalationService, build_escalator
from recoup.engine import RecoveryEngine
from recoup.execution import Executor, RecordingNotifier
from recoup.generator.synthetic import ScenarioGenerator
from recoup.ledger import Ledger
from recoup.policy.gates import CustomerState, EventState

UNKNOWN_CODES = (
    "acct_balance_shortfall",
    "issuer_host_unreachable",
    "merchant_kyc_pending",
    "psp_handle_retired",
)


def seed(n_events: int = 400) -> ConsoleState:
    client, _ = build_client()
    escalation = EscalationService(build_escalator())
    engine = RecoveryEngine(
        Executor(client, RecordingNotifier()),
        Ledger(),
        holdout_rate=0.20,
        escalation=escalation,
    )

    scenario = ScenarioGenerator(seed=20260905).generate(n_events=n_events, n_customers=250)
    contacts: dict[str, list[datetime]] = {}
    for i, event in enumerate(scenario.events):
        # Salt a few events with codes the taxonomy does not know, so the rule
        # review queue has something real in it rather than being an empty tab.
        if i % 47 == 0 and event.error_reason:
            event = replace(event, error_reason=UNKNOWN_CODES[i % len(UNKNOWN_CODES)])
        customer = scenario.customers[event.customer_id]
        prior = contacts.setdefault(customer.customer_id, [])
        result = engine.handle(
            event,
            customer,
            CustomerState(
                customer_id=customer.customer_id,
                has_consent=customer.has_consent,
                on_dnd_registry=customer.on_dnd_registry,
                contacts_in_window=tuple(prior),
                last_contact_at=prior[-1] if prior else None,
            ),
            EventState(event_id=event.event_id, attempts_so_far=event.attempt_number - 1),
            event.occurred_at,
        )
        if result.executed.is_contact and result.acted:
            prior.append(event.occurred_at)

    return ConsoleState(engine=engine, escalation=escalation)


def main() -> None:
    state = seed()
    print(f"  seeded {len(state.engine.ledger)} decisions")
    print(f"  approval queue: {len(state.engine.executor.approval_queue)} items")
    print(f"  pending rules:  {len(state.escalation.review) if state.escalation else 0}")
    print("  API on http://127.0.0.1:8000  (console dev server: npm run dev in console/)")
    uvicorn.run(create_app(state), host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
