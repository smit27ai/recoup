"""Console API tests.

The endpoints that change state are the ones worth testing hard: approving a parked
action, and promoting a mined rule into the permanent table. Both are irreversible
in the sense that matters -- one authorises money movement, the other changes how
every future event is classified.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from recoup.console.api import ConsoleState, create_app
from recoup.diagnosis import taxonomy
from recoup.diagnosis.escalation import EscalationService, StubEscalator
from recoup.domain import AtRiskEvent, Channel, Customer, RiskKind
from recoup.engine import RecoveryEngine
from recoup.execution import Executor, RecordingNotifier
from recoup.ledger import Ledger
from recoup.policy.gates import IST, CustomerState, EventState
from recoup.razorpay.client import RazorpayClient

MIDDAY = datetime(2026, 9, 1, 11, 0, tzinfo=IST)


def _stub_transport(request: httpx.Request) -> httpx.Response:
    body: dict[str, Any] = {"id": "order_x", "short_url": "https://rzp.io/i/x"}
    if request.method == "GET":
        body = {"entity": "collection", "items": []}
    return httpx.Response(200, json=body)


@pytest.fixture
def client() -> TestClient:
    engine = RecoveryEngine(
        Executor(
            RazorpayClient("rzp_test_k", "s", transport=httpx.MockTransport(_stub_transport)),
            RecordingNotifier(),
        ),
        Ledger(),
        holdout_rate=0.0,
        escalation=EscalationService(StubEscalator()),
    )
    customer = Customer(
        customer_id="cust_0001",
        segment="loyal",
        has_consent=True,
        on_dnd_registry=False,
        preferred_channel=Channel.WHATSAPP,
    )
    cs = CustomerState(customer_id="cust_0001", has_consent=True, on_dnd_registry=False)

    for reason, amount in [
        ("insufficient_funds", 99_900),
        ("card_expired", 249_900),
        ("card_expired", 90_000_00),  # trips the approval threshold
        ("invalid_order_id", 50_000),
        ("acct_balance_shortfall_2027", 45_000),  # unknown -> tier 2
    ]:
        event = AtRiskEvent(
            event_id=f"evt_{reason[:8]}_{amount}",
            customer_id="cust_0001",
            kind=RiskKind.FAILED_PAYMENT,
            amount_paise=amount,
            occurred_at=MIDDAY,
            error_reason=reason,
            method="card",
        )
        engine.handle(
            event, customer, cs, EventState(event_id=event.event_id, attempts_so_far=0), MIDDAY
        )

    return TestClient(create_app(ConsoleState(engine=engine, escalation=engine.escalation)))


# --- reads ------------------------------------------------------------------


def test_metrics_reports_the_cost_of_compliance(client: TestClient) -> None:
    m = client.get("/api/metrics").json()
    assert m["decisions"] == 5
    assert "not_chased_paise" in m, "the cost side must always be on the page"
    assert m["approval_queue"] >= 1


def test_integrity_reports_a_verified_chain(client: TestClient) -> None:
    body = client.get("/api/integrity").json()
    assert body["ok"] is True
    assert len(body["head"]) == 64


def test_decisions_list_and_filters(client: TestClient) -> None:
    assert client.get("/api/decisions").json()["total"] == 5
    tier2 = client.get("/api/decisions?tier=2").json()
    assert all(d["diagnosis_tier"] == 2 for d in tier2["items"])


def test_decision_detail_carries_the_full_trace(client: TestClient) -> None:
    first = client.get("/api/decisions").json()["items"][0]
    detail = client.get(f"/api/decisions/{first['event_id']}").json()
    assert detail["items"][0]["gates"], "every gate that ran must be visible"
    assert "saw" in detail["items"][0]["explain"]


def test_missing_event_is_404(client: TestClient) -> None:
    assert client.get("/api/decisions/evt_nope").status_code == 404


def test_customer_history_answers_why_were_they_messaged(client: TestClient) -> None:
    body = client.get("/api/customers/cust_0001").json()
    assert body["decisions"] == 5
    assert "contacts" in body


# --- approvals --------------------------------------------------------------


def test_approving_appends_and_never_edits(client: TestClient) -> None:
    """The original record described a moment when no human had looked. That stays
    true of it, so a review is a new fact rather than a correction."""
    queue = client.get("/api/queues/approval").json()
    assert queue["count"] >= 1
    event_id = queue["items"][0]["event_id"]

    before = client.get("/api/decisions").json()["total"]
    original = client.get(f"/api/decisions/{event_id}").json()["items"][0]

    res = client.post(
        f"/api/queues/approval/{event_id}/decide?approve=true",
        json={"reviewer": "smit", "note": "genuine renewal"},
    )
    assert res.status_code == 200
    assert res.json()["review"]["approved"] == "yes"

    after = client.get("/api/decisions").json()
    assert after["total"] == before + 1, "a review appends a record"
    unchanged = client.get(f"/api/decisions/{event_id}").json()["items"][0]
    assert unchanged["record_hash"] == original["record_hash"]
    assert client.get("/api/integrity").json()["ok"] is True


def test_review_is_attributed(client: TestClient) -> None:
    event_id = client.get("/api/queues/approval").json()["items"][0]["event_id"]
    client.post(
        f"/api/queues/approval/{event_id}/decide?approve=false",
        json={"reviewer": "asha", "note": "looks like a duplicate"},
    )
    latest = client.get("/api/decisions").json()["items"][0]
    assert latest["metadata"]["reviewer"] == "asha"
    assert latest["metadata"]["approved"] == "no"


def test_double_review_is_rejected(client: TestClient) -> None:
    """Two reviewers racing must not both act on the same parked payment."""
    event_id = client.get("/api/queues/approval").json()["items"][0]["event_id"]
    body = {"reviewer": "smit", "note": ""}
    assert client.post(f"/api/queues/approval/{event_id}/decide", json=body).status_code == 200
    assert client.post(f"/api/queues/approval/{event_id}/decide", json=body).status_code == 409


def test_reviewing_an_unqueued_event_is_404(client: TestClient) -> None:
    res = client.post("/api/queues/approval/evt_nope/decide", json={"reviewer": "x"})
    assert res.status_code == 404


def test_reviewer_name_is_required(client: TestClient) -> None:
    event_id = client.get("/api/queues/approval").json()["items"][0]["event_id"]
    res = client.post(f"/api/queues/approval/{event_id}/decide", json={"reviewer": ""})
    assert res.status_code == 422


# --- rule promotion ---------------------------------------------------------


@pytest.fixture
def sandboxed_taxonomy(tmp_path: Path):
    """Promotion writes to the real TSV, so tests get their own copy."""
    original = taxonomy.TAXONOMY_PATH
    copy = tmp_path / "error_taxonomy.tsv"
    shutil.copy(original, copy)
    taxonomy.TAXONOMY_PATH = copy
    taxonomy.load_taxonomy.cache_clear()
    yield copy
    taxonomy.TAXONOMY_PATH = original
    taxonomy.load_taxonomy.cache_clear()


def test_pending_rules_flag_what_approval_unlocks(client: TestClient) -> None:
    rules = client.get("/api/rules/pending").json()
    assert rules["count"] >= 1
    assert "would_unlock_contact" in rules["items"][0]


def test_approving_a_rule_closes_the_loop(client: TestClient, sandboxed_taxonomy: Path) -> None:
    """Tier 2 proposes, a human approves, and the code is tier 1 from then on --
    resolved by table lookup, with contact permitted through the ordinary path."""
    rules = client.get("/api/rules/pending").json()["items"]
    target = next(r for r in rules if r["in_scope"])
    reason = target["reason"]

    before = len(taxonomy.load_taxonomy())
    res = client.post(f"/api/rules/{reason}/approve", json={"reviewer": "smit", "note": "checked"})
    assert res.status_code == 200
    assert res.json()["promoted_to_tier_1"] is True

    assert len(taxonomy.load_taxonomy()) == before + 1
    promoted = taxonomy.diagnose(reason)
    assert promoted is not None
    assert promoted.tier == 1, "must now resolve by table lookup, with no model call"
    assert reason not in [r["reason"] for r in client.get("/api/rules/pending").json()["items"]]


def test_promoting_twice_is_rejected(client: TestClient, sandboxed_taxonomy: Path) -> None:
    reason = client.get("/api/rules/pending").json()["items"][0]["reason"]
    client.post(f"/api/rules/{reason}/approve", json={"reviewer": "smit"})
    assert client.post(f"/api/rules/{reason}/approve", json={"reviewer": "smit"}).status_code == 404


def test_promotion_validates_before_writing(sandboxed_taxonomy: Path) -> None:
    """A rule that would not load must be rejected rather than written -- a taxonomy
    that fails to parse takes the system down at the next restart, long after the
    reviewer who broke it went home."""
    before = sandboxed_taxonomy.read_text(encoding="utf-8")
    with pytest.raises(taxonomy.PromotionError):
        taxonomy.promote_rule("bad_code\tescalated\tNOT_A_CAUSE\tNOW\t0\t0\tcustomer\t1")
    assert sandboxed_taxonomy.read_text(encoding="utf-8") == before, "file must be untouched"


def test_promotion_rejects_wrong_column_count(sandboxed_taxonomy: Path) -> None:
    with pytest.raises(taxonomy.PromotionError, match="columns"):
        taxonomy.promote_rule("too\tfew\tcolumns")


def test_promotion_rejects_a_duplicate(sandboxed_taxonomy: Path) -> None:
    with pytest.raises(taxonomy.PromotionError, match="already in the taxonomy"):
        taxonomy.promote_rule("card_expired\tescalated\tFUNDS\tNOW\t0\t0\tcustomer\t1")


def test_rejecting_a_rule_does_not_touch_the_taxonomy(
    client: TestClient, sandboxed_taxonomy: Path
) -> None:
    before = sandboxed_taxonomy.read_text(encoding="utf-8")
    reason = client.get("/api/rules/pending").json()["items"][0]["reason"]
    assert client.post(f"/api/rules/{reason}/reject", json={"reviewer": "smit"}).status_code == 200
    assert sandboxed_taxonomy.read_text(encoding="utf-8") == before
    assert reason not in [r["reason"] for r in client.get("/api/rules/pending").json()["items"]]
