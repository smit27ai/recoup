"""`python -m recoup` -- run the whole path and show what happened.

Works with no configuration at all. When `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET`
are present it drives real test-mode Razorpay; without them it drives a local
in-process double that speaks the same HTTP shapes. Same engine, same gates, same
ledger either way -- only the transport differs, which is the point: a reviewer who
clones this repo can see it work in one command, and adding keys changes nothing
except where the orders end up.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from datetime import datetime
from typing import Any

import httpx

from recoup.domain import ActionKind, Arm
from recoup.engine import Handled, RecoveryEngine
from recoup.execution import ExecutionStatus, Executor, RecordingNotifier
from recoup.generator.synthetic import ScenarioGenerator
from recoup.ledger import Ledger
from recoup.policy.gates import CustomerState, EventState
from recoup.razorpay.client import RazorpayClient


class LocalRazorpay:
    """In-process stand-in for the Razorpay API.

    Not a mock in the testing sense -- it holds state, so a receipt that was already
    used comes back as the same order, which is what makes the reconciliation path
    demonstrable without an internet connection.
    """

    def __init__(self) -> None:
        self.orders: dict[str, dict[str, Any]] = {}
        self.links: dict[str, dict[str, Any]] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        body = json.loads(request.content) if request.content else {}

        if path.endswith("/orders") and method == "POST":
            receipt = body.get("receipt", "")
            if receipt in self.orders:
                return _ok(self.orders[receipt])
            order = {
                "id": f"order_{len(self.orders):08d}",
                "entity": "order",
                "amount": body.get("amount"),
                "currency": "INR",
                "receipt": receipt,
                "status": "created",
            }
            self.orders[receipt] = order
            return _ok(order)

        if path.endswith("/orders") and method == "GET":
            return _ok({"entity": "collection", "items": list(self.orders.values())})

        if path.endswith("/payment_links") and method == "POST":
            ref = body.get("reference_id", "")
            if ref in self.links:
                return _ok(self.links[ref])
            link = {
                "id": f"plink_{len(self.links):08d}",
                "reference_id": ref,
                "amount": body.get("amount"),
                "short_url": f"https://rzp.io/i/{len(self.links):06d}",
                "status": "created",
            }
            self.links[ref] = link
            return _ok(link)

        return _ok({})


def _ok(body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200, content=json.dumps(body), headers={"content-type": "application/json"}
    )


def build_client() -> tuple[RazorpayClient, str]:
    key_id = os.environ.get("RAZORPAY_KEY_ID", "").strip()
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET", "").strip()
    if key_id and key_secret:
        # allow_live stays False: if someone puts a live key here, refuse loudly.
        return RazorpayClient(key_id, key_secret), f"Razorpay test mode ({key_id[:12]}...)"
    fake = LocalRazorpay()
    return (
        RazorpayClient("rzp_test_local", "local", transport=httpx.MockTransport(fake.handler)),
        "local in-process double (set RAZORPAY_KEY_ID/SECRET for real test mode)",
    )


def demo(n_events: int = 300, holdout_rate: float = 0.20) -> int:
    client, mode = build_client()
    notifier = RecordingNotifier()
    ledger = Ledger()
    engine = RecoveryEngine(Executor(client, notifier), ledger, holdout_rate=holdout_rate)

    scenario = ScenarioGenerator(seed=20260905).generate(n_events=n_events, n_customers=200)
    print(f"\n  transport   {mode}")
    print(f"  events      {n_events:,}   holdout {holdout_rate:.0%}\n")

    contacts: dict[str, list[datetime]] = {}
    handled: list[Handled] = []
    for event in scenario.events:
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
        handled.append(result)

    _funnel(handled)
    _gates(handled)
    _traces(handled)
    _integrity(ledger, engine, handled)
    return 0


def _funnel(handled: list[Handled]) -> None:
    total = len(handled)
    at_risk = sum(h.intent.event.amount_paise for h in handled)
    holdout = sum(1 for h in handled if h.intent.arm is Arm.HOLDOUT)
    print("  WHAT HAPPENED TO EVERY EVENT")
    print("  " + "-" * 66)
    print(f"  {'at-risk value':<34}Rs.{at_risk / 100:>14,.0f}")
    print(f"  {'events':<34}{total:>17,}")
    print(f"  {'held out (control arm)':<34}{holdout:>17,}")
    print()

    by_action: dict[str, int] = {}
    for h in handled:
        by_action[str(h.executed)] = by_action.get(str(h.executed), 0) + 1
    for action, count in sorted(by_action.items(), key=lambda kv: -kv[1]):
        print(f"  {action:<34}{count:>17,}")

    print()
    by_status: dict[str, int] = {}
    for h in handled:
        by_status[str(h.result.status)] = by_status.get(str(h.result.status), 0) + 1
    for status, count in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"  execution: {status:<23}{count:>17,}")
    print()


def _gates(handled: list[Handled]) -> None:
    print("  WHY ACTIONS WERE BLOCKED")
    print("  " + "-" * 66)
    denials: dict[str, int] = {}
    for h in handled:
        for gate in h.record.denied_by:
            denials[gate] = denials.get(gate, 0) + 1
    if not denials:
        print("  nothing was blocked\n")
        return
    for gate, count in sorted(denials.items(), key=lambda kv: -kv[1]):
        print(f"  {gate:<34}{count:>17,}")
    blocked_value = sum(
        h.intent.event.amount_paise
        for h in handled
        if h.record.denied_by and h.intent.arm is Arm.TREATMENT
    )
    print(f"\n  value we chose NOT to chase   Rs.{blocked_value / 100:>14,.0f}")
    print("  (compliance is a cost. Not measuring it is how it quietly stops being real.)\n")


def _traces(handled: list[Handled]) -> None:
    """Show one decision of each interesting shape, in full."""
    print("  SAMPLE DECISIONS, END TO END")
    print("  " + "=" * 66)
    wanted: list[tuple[str, Callable[[Handled], bool]]] = [
        ("a retry that needed no message", lambda h: h.executed is ActionKind.RETRY_SCHEDULED),
        (
            "an instrument switch, because retrying was futile",
            lambda h: h.executed is ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH,
        ),
        ("blocked by a gate", lambda h: bool(h.record.denied_by)),
        (
            "our own bug, routed away from the customer",
            lambda h: h.executed is ActionKind.ROUTE_TO_OPS,
        ),
        ("parked for a human", lambda h: h.executed is ActionKind.QUEUED_FOR_APPROVAL),
        ("held out, so we did nothing on purpose", lambda h: h.intent.arm is Arm.HOLDOUT),
    ]
    for label, predicate in wanted:
        match = next((h for h in handled if predicate(h)), None)
        if match is None:
            continue
        print(f"\n  # {label}")
        for line in match.explain().splitlines():
            print(f"  {line}")
        print(f"    -> {match.result.status}: {match.result.detail}")
    print()


def _integrity(ledger: Ledger, engine: RecoveryEngine, handled: list[Handled]) -> None:
    print("  " + "=" * 66)
    print("  INTEGRITY")
    print("  " + "-" * 66)
    ledger.verify()
    print(f"  ledger              {len(ledger):,} records, hash chain verified")
    print(f"  head                {ledger.head[:32]}...")

    notifier = engine.executor.notifier
    sent = len(notifier.sent) if isinstance(notifier, RecordingNotifier) else 0
    contacted_in_holdout = sum(
        1 for h in handled if h.intent.arm is Arm.HOLDOUT and h.executed.is_contact
    )
    unresolved = sum(1 for h in handled if h.result.status is ExecutionStatus.UNRESOLVED)

    print(f"  messages sent       {sent:,}")
    print(f"  holdout contacted   {contacted_in_holdout}  (must be 0, or measurement is void)")
    print(
        f"  ops queue           {len(engine.executor.ops_queue):,} items, "
        f"Rs.{engine.executor.ops_queue.total_paise / 100:,.0f}"
    )
    print(
        f"  approval queue      {len(engine.executor.approval_queue):,} items, "
        f"Rs.{engine.executor.approval_queue.total_paise / 100:,.0f}"
    )
    print(f"  unresolved          {unresolved}  (needs a human)")

    assert contacted_in_holdout == 0, "holdout contaminated"
    print("\n  every decision above is reconstructable from the ledger alone.\n")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    n = 300
    for i, arg in enumerate(args):
        if arg in ("-n", "--events") and i + 1 < len(args):
            n = int(args[i + 1])
    return demo(n_events=n)


if __name__ == "__main__":
    raise SystemExit(main())
