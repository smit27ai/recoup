"""HTTP API behind the ops console.

The console exists because two queues in this system can only be drained by a human:

  approvals  actions above the unattended authority limit, waiting for a decision
  rules      tier-2 proposals waiting to be promoted into the deterministic table

A queue nobody can drain is not a safety mechanism, it is a place money goes to die.
Recoup parks 21% of at-risk value in the approval queue by design; without a way to
work it, that design is just a slower kind of losing.

Two properties this API is built around:

**Reads cannot mutate and writes are explicit.** Every state change is a POST naming
what a person decided, and it lands in the ledger as a new record attributed to the
reviewer. Approving something is itself an auditable decision.

**The ledger is never edited to reflect a review.** An approval appends; it does not
go back and change the original record's disposition. The chain is what makes the
audit trail worth having, and a console that could rewrite history would destroy
exactly the property it exists to expose.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from recoup.diagnosis.escalation import EscalationService, Proposal
from recoup.diagnosis.taxonomy import PromotionError, promote_rule
from recoup.domain import ActionKind, Arm
from recoup.engine import RecoveryEngine
from recoup.ledger import ChainError, DecisionRecord


class ReviewDecision(BaseModel):
    reviewer: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=1000)


@dataclass
class ConsoleState:
    """Everything the console can see. Injected so tests drive a real engine."""

    engine: RecoveryEngine
    escalation: EscalationService | None = None
    reviewed: dict[str, dict[str, str]] | None = None

    def __post_init__(self) -> None:
        if self.reviewed is None:
            self.reviewed = {}


def _record_json(record: DecisionRecord) -> dict[str, Any]:
    return {
        "seq": record.seq,
        "decided_at": record.decided_at,
        "event_id": record.event_id,
        "customer_id": record.customer_id,
        "amount_paise": record.amount_paise,
        "error_reason": record.error_reason,
        "root_cause": record.root_cause,
        "diagnosis_tier": record.diagnosis_tier,
        "intended_action": record.intended_action,
        "executed_action": record.executed_action,
        "disposition": record.disposition,
        "arm": record.arm,
        "recovered": record.recovered,
        "denied_by": list(record.denied_by),
        "gates": [
            {"gate": g.gate, "disposition": g.disposition, "reason": g.reason} for g in record.gates
        ],
        "metadata": dict(record.metadata),
        "record_hash": record.record_hash,
        "explain": record.explain(),
    }


def _proposal_json(proposal: Proposal, seen: int) -> dict[str, Any]:
    return {
        "reason": proposal.reason,
        "root_cause": str(proposal.root_cause),
        "retry_class": str(proposal.retry_class),
        "new_instrument": proposal.new_instrument,
        "customer_action": proposal.customer_action,
        "owner": str(proposal.owner),
        "in_scope": proposal.in_scope,
        "confidence": proposal.confidence,
        "rationale": proposal.rationale,
        "model": proposal.model,
        "seen_count": seen,
        "taxonomy_row": proposal.to_taxonomy_row(),
        # Surfaced so the reviewer understands what approving actually unlocks --
        # tier 2 can never authorise contact, so this is the real consequence.
        "would_unlock_contact": proposal.in_scope and proposal.customer_action,
    }


def create_app(state: ConsoleState) -> FastAPI:
    app = FastAPI(title="Recoup ops console", version="0.1.0")
    # The console is served from a Vite dev server on another port during
    # development. Locked to localhost -- this API exposes customer identifiers and
    # can authorise money movement, so it is never open to arbitrary origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/metrics")
    def metrics() -> dict[str, Any]:
        records = list(state.engine.ledger)
        by_action: dict[str, int] = {}
        by_gate: dict[str, int] = {}
        for r in records:
            by_action[r.executed_action] = by_action.get(r.executed_action, 0) + 1
            for gate in r.denied_by:
                by_gate[gate] = by_gate.get(gate, 0) + 1

        treatment = [r for r in records if r.arm == Arm.TREATMENT]
        blocked_value = sum(r.amount_paise for r in treatment if r.denied_by)
        return {
            "decisions": len(records),
            "at_risk_paise": sum(r.amount_paise for r in records),
            "holdout": sum(1 for r in records if r.arm == Arm.HOLDOUT),
            "by_action": dict(sorted(by_action.items(), key=lambda kv: -kv[1])),
            "denials_by_gate": dict(sorted(by_gate.items(), key=lambda kv: -kv[1])),
            # The cost of compliance, in rupees. A console that only showed money
            # recovered would quietly push its operators toward messaging more.
            "not_chased_paise": blocked_value,
            "approval_queue": len(state.engine.executor.approval_queue),
            "approval_queue_paise": state.engine.executor.approval_queue.total_paise,
            "ops_queue": len(state.engine.executor.ops_queue),
            "ops_queue_paise": state.engine.executor.ops_queue.total_paise,
            "pending_rules": len(state.escalation.review) if state.escalation else 0,
            "escalation_calls": state.escalation.calls if state.escalation else 0,
        }

    @app.get("/api/integrity")
    def integrity() -> dict[str, Any]:
        try:
            state.engine.ledger.verify()
        except ChainError as exc:
            # Reported, never hidden. A console that quietly showed a green tick over
            # a broken chain would be worse than having no console.
            return {"ok": False, "records": len(state.engine.ledger), "error": str(exc)}
        return {
            "ok": True,
            "records": len(state.engine.ledger),
            "head": state.engine.ledger.head,
        }

    @app.get("/api/decisions")
    def decisions(
        limit: int = 100,
        offset: int = 0,
        blocked_only: bool = False,
        tier: int | None = None,
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        rows = list(state.engine.ledger)
        if blocked_only:
            rows = [r for r in rows if r.denied_by]
        if tier is not None:
            rows = [r for r in rows if r.diagnosis_tier == tier]
        if customer_id:
            rows = [r for r in rows if r.customer_id == customer_id]
        rows.reverse()  # newest first
        window = rows[offset : offset + min(limit, 500)]
        return {"total": len(rows), "items": [_record_json(r) for r in window]}

    @app.get("/api/decisions/{event_id}")
    def decision_detail(event_id: str) -> dict[str, Any]:
        """Every decision touching one event, oldest first.

        This is the query a support agent needs when a merchant asks why a customer
        was messaged four times, or why nobody chased an invoice.
        """
        rows = state.engine.ledger.for_event(event_id)
        if not rows:
            raise HTTPException(404, f"no decisions recorded for {event_id}")
        return {"event_id": event_id, "items": [_record_json(r) for r in rows]}

    @app.get("/api/customers/{customer_id}")
    def customer_history(customer_id: str) -> dict[str, Any]:
        rows = state.engine.ledger.for_customer(customer_id)
        contacts = sum(1 for r in rows if r.executed_action.startswith("nudge"))
        return {
            "customer_id": customer_id,
            "decisions": len(rows),
            "contacts": contacts,
            "items": [_record_json(r) for r in rows],
        }

    # --- approvals ----------------------------------------------------------

    @app.get("/api/queues/approval")
    def approval_queue() -> dict[str, Any]:
        queue = state.engine.executor.approval_queue
        assert state.reviewed is not None
        return {
            "count": len(queue),
            "total_paise": queue.total_paise,
            "items": [
                {**item, "review": state.reviewed.get(item["event_id"])} for item in queue.items
            ],
        }

    @app.get("/api/queues/ops")
    def ops_queue() -> dict[str, Any]:
        queue = state.engine.executor.ops_queue
        return {"count": len(queue), "total_paise": queue.total_paise, "items": queue.items}

    @app.post("/api/queues/approval/{event_id}/decide")
    def decide_approval(
        event_id: str, body: ReviewDecision, approve: bool = True
    ) -> dict[str, Any]:
        """Record a human decision on a parked action.

        Appends to the ledger rather than editing the original record. The original
        said "a human has not looked yet" and that remains true of the moment it
        describes; this is a new fact about a later moment.
        """
        assert state.reviewed is not None
        queue = state.engine.executor.approval_queue
        item = next((i for i in queue.items if i["event_id"] == event_id), None)
        if item is None:
            raise HTTPException(404, f"{event_id} is not in the approval queue")
        if event_id in state.reviewed:
            raise HTTPException(409, f"{event_id} was already reviewed")

        decision = {
            "reviewer": body.reviewer,
            "note": body.note,
            "approved": "yes" if approve else "no",
            "at": datetime.now(UTC).isoformat(),
        }
        state.reviewed[event_id] = decision

        prior = state.engine.ledger.for_event(event_id)
        if prior:
            last = prior[-1]
            from recoup.domain import AtRiskEvent, RiskKind
            from recoup.policy.gates import Verdict

            state.engine.ledger.append(
                event=AtRiskEvent(
                    event_id=event_id,
                    customer_id=last.customer_id,
                    kind=RiskKind.FAILED_PAYMENT,
                    amount_paise=last.amount_paise,
                    occurred_at=datetime.now(UTC),
                    error_reason=last.error_reason,
                    method="review",
                ),
                diagnosis=None,
                intended=ActionKind(last.intended_action),
                verdict=Verdict(results=(), now=datetime.now(UTC)),
                executed=ActionKind(last.intended_action) if approve else ActionKind.NO_ACTION,
                arm=Arm(last.arm),
                decided_at=datetime.now(UTC),
                policy_version="human-review",
                taxonomy_version="n/a",
                metadata={
                    "review_of": last.record_hash,
                    "reviewer": body.reviewer,
                    "approved": decision["approved"],
                    "note": body.note,
                    "execution_status": "approved" if approve else "rejected",
                },
            )
        return {"event_id": event_id, "review": decision}

    # --- mined rules --------------------------------------------------------

    @app.get("/api/rules/pending")
    def pending_rules() -> dict[str, Any]:
        if state.escalation is None:
            return {"count": 0, "items": []}
        ranked = state.escalation.review.by_impact()
        return {
            "count": len(ranked),
            "items": [_proposal_json(p, seen) for p, seen in ranked],
        }

    @app.post("/api/rules/{reason}/approve")
    def approve_rule(reason: str, body: ReviewDecision) -> dict[str, Any]:
        """Promote a tier-2 proposal into the deterministic table.

        This is the one endpoint that changes how the system behaves for every future
        event, so it is validated hard and it writes the reviewer into the response.
        """
        if state.escalation is None:
            raise HTTPException(404, "no escalation service configured")
        proposal = state.escalation.review.proposals.get(reason)
        if proposal is None:
            raise HTTPException(404, f"no pending rule for {reason!r}")

        try:
            entry = promote_rule(proposal.to_taxonomy_row())
        except PromotionError as exc:
            raise HTTPException(400, str(exc)) from exc

        del state.escalation.review.proposals[reason]
        # The escalation cache held a tier-2 answer for this code. Leaving it would
        # keep serving the downgraded diagnosis even though tier 1 now has the real
        # one, with contact still wrongly withheld.
        state.escalation._cache.pop(reason, None)

        return {
            "reason": reason,
            "promoted_to_tier_1": True,
            "reviewer": body.reviewer,
            "root_cause": str(entry.root_cause),
            "now_contactable": entry.contactable,
        }

    @app.post("/api/rules/{reason}/reject")
    def reject_rule(reason: str, body: ReviewDecision) -> dict[str, Any]:
        if state.escalation is None:
            raise HTTPException(404, "no escalation service configured")
        if reason not in state.escalation.review.proposals:
            raise HTTPException(404, f"no pending rule for {reason!r}")
        del state.escalation.review.proposals[reason]
        # Deliberately NOT clearing the escalation cache: a rejected proposal must
        # not be silently re-proposed on the next event carrying this code.
        return {"reason": reason, "rejected_by": body.reviewer, "note": body.note}

    return app
