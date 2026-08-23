"""Append-only, hash-chained decision ledger.

Every decision Recoup makes lands here as one immutable record answering six
questions, in order:

    what did we see        the event and its diagnosis, including which tier
    what did we want       the action the policy engine proposed
    what were we allowed   every gate result, passing and failing alike
    what did we do         the action actually executed, if any
    what happened          the observed outcome
    what would have happened  the arm, so the counterfactual is reconstructable

A recovery system that cannot answer all six for an arbitrary rupee is not
auditable, whatever its dashboard says.

Three properties this file is built around:

**Append-only with a hash chain.** Each record commits to its predecessor, so any
edit to history invalidates every record after it. This is not about defending
against a sophisticated attacker with write access -- it is about making silent
after-the-fact edits impossible, including our own. When a merchant disputes a
message we sent at 19:04, the ledger either shows the decision or it does not, and
nobody can quietly add it later.

**Reason strings are stored verbatim, not re-derived.** The record keeps the gate's
own words ("last contact 2.1h ago, minimum gap is 24h") rather than a code to be
re-interpreted later against a config that has since changed. An audit trail that
re-computes its own explanations is not evidence.

**Replayable.** `replay()` re-decides historical events under a new policy version
WITHOUT re-executing anything, which is how you answer "what would the new rules
have done last month" before shipping them. That question is otherwise unanswerable
except by experimenting on real customers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from recoup.diagnosis.taxonomy import Diagnosis
from recoup.domain import ActionKind, Arm, AtRiskEvent
from recoup.policy.gates import Disposition, Verdict

GENESIS = "0" * 64
"""Hash of the record before the first one. Anchors the chain."""


@dataclass(frozen=True, slots=True)
class GateRecord:
    """One gate's verdict, frozen in the words it used at the time."""

    gate: str
    disposition: str
    reason: str


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One decision. Immutable, self-describing, and chained to its predecessor."""

    seq: int
    decided_at: str
    """ISO-8601 with offset. String, not datetime, so the hash is stable across
    serialisation round-trips and timezone library versions."""

    event_id: str
    customer_id: str
    amount_paise: int

    # what we saw
    error_reason: str | None
    root_cause: str | None
    diagnosis_tier: int | None
    """1 = table lookup, 2 = model escalation, None = nothing to diagnose."""

    # what we wanted, were allowed, and did
    intended_action: str
    gates: tuple[GateRecord, ...]
    disposition: str
    executed_action: str

    # what happened
    arm: str
    recovered: bool | None
    """None while the outcome is still open."""

    # reproducibility
    policy_version: str
    taxonomy_version: str

    prev_hash: str
    record_hash: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        """Everything the hash commits to. Excludes the hash itself."""
        data = asdict(self)
        data.pop("record_hash")
        return data

    def compute_hash(self) -> str:
        # sort_keys + separators makes this canonical: the same logical record must
        # serialise to the same bytes on any machine, or the chain is worthless.
        blob = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def sealed(self) -> DecisionRecord:
        return replace(self, record_hash=self.compute_hash())

    @property
    def denied_by(self) -> tuple[str, ...]:
        return tuple(g.gate for g in self.gates if g.disposition == Disposition.DENY)

    def explain(self) -> str:
        """The human-readable version, for the decision inspector and for support."""
        lines = [
            f"[{self.seq}] {self.decided_at}  event={self.event_id} "
            f"customer={self.customer_id}  Rs.{self.amount_paise / 100:,.2f}",
            f"  saw       {self.error_reason or '(no gateway error)'} "
            f"-> {self.root_cause or 'n/a'} (tier {self.diagnosis_tier or '-'})",
            f"  wanted    {self.intended_action}",
        ]
        for g in self.gates:
            mark = {"allow": "ok  ", "deny": "DENY", "needs_approval": "HOLD"}[g.disposition]
            lines.append(f"    {mark} {g.gate:<20} {g.reason}")
        lines.append(f"  did       {self.executed_action}   [{self.disposition}]")
        outcome = "open" if self.recovered is None else ("recovered" if self.recovered else "lost")
        lines.append(f"  outcome   {outcome}   arm={self.arm}")
        return "\n".join(lines)


class Ledger:
    """In-memory chain with optional JSONL durability.

    Deliberately not a database. The ledger's contract is append-only and
    verifiable; a table anyone can UPDATE does not provide that, and swapping in
    Postgres later means an append-only table plus these same hashes, not a
    different design.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._records: list[DecisionRecord] = []
        self._path = path
        if path is not None and path.exists():
            self._records = list(load(path))

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[DecisionRecord]:
        return iter(self._records)

    @property
    def head(self) -> str:
        return self._records[-1].record_hash if self._records else GENESIS

    def append(
        self,
        *,
        event: AtRiskEvent,
        diagnosis: Diagnosis | None,
        intended: ActionKind,
        verdict: Verdict,
        executed: ActionKind,
        arm: Arm,
        decided_at: datetime,
        recovered: bool | None = None,
        policy_version: str = "unset",
        taxonomy_version: str = "unset",
        metadata: dict[str, str] | None = None,
    ) -> DecisionRecord:
        record = DecisionRecord(
            seq=len(self._records),
            decided_at=decided_at.isoformat(),
            event_id=event.event_id,
            customer_id=event.customer_id,
            amount_paise=event.amount_paise,
            error_reason=event.error_reason,
            root_cause=str(diagnosis.root_cause) if diagnosis else None,
            diagnosis_tier=diagnosis.tier if diagnosis else None,
            intended_action=str(intended),
            gates=tuple(
                GateRecord(gate=str(r.gate), disposition=str(r.disposition), reason=r.reason)
                for r in verdict.results
            ),
            disposition=str(verdict.disposition),
            executed_action=str(executed),
            arm=str(arm),
            recovered=recovered,
            policy_version=policy_version,
            taxonomy_version=taxonomy_version,
            prev_hash=self.head,
            metadata=metadata or {},
        ).sealed()

        self._records.append(record)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(_to_json(record), ensure_ascii=False) + "\n")
        return record

    def verify(self) -> None:
        """Raise if the chain has been tampered with. Cheap enough to run on startup."""
        verify_chain(self._records)

    def for_event(self, event_id: str) -> tuple[DecisionRecord, ...]:
        return tuple(r for r in self._records if r.event_id == event_id)

    def for_customer(self, customer_id: str) -> tuple[DecisionRecord, ...]:
        """Every decision touching one person -- the query a support agent needs
        when someone asks why they were messaged four times."""
        return tuple(r for r in self._records if r.customer_id == customer_id)

    def denied(self) -> tuple[DecisionRecord, ...]:
        return tuple(r for r in self._records if r.denied_by)

    def awaiting_approval(self) -> tuple[DecisionRecord, ...]:
        return tuple(r for r in self._records if r.disposition == Disposition.NEEDS_APPROVAL)


class ChainError(Exception):
    """The ledger does not verify. Treat every number derived from it as suspect."""


def verify_chain(records: Sequence[DecisionRecord]) -> None:
    prev = GENESIS
    for i, rec in enumerate(records):
        if rec.seq != i:
            raise ChainError(f"record {i}: seq is {rec.seq}, records are out of order or missing")
        if rec.prev_hash != prev:
            raise ChainError(
                f"record {i} ({rec.event_id}): prev_hash does not match record {i - 1} -- "
                "history was edited or a record was removed"
            )
        expected = rec.compute_hash()
        if rec.record_hash != expected:
            raise ChainError(
                f"record {i} ({rec.event_id}): contents do not match its hash -- record was edited"
            )
        prev = rec.record_hash


def _to_json(record: DecisionRecord) -> dict[str, Any]:
    data = asdict(record)
    data["gates"] = [asdict(g) if not isinstance(g, dict) else g for g in record.gates]
    return data


def load(path: Path) -> Iterator[DecisionRecord]:
    """Read a JSONL ledger back. Does NOT verify -- call `verify_chain` explicitly,
    so that verification is always a deliberate act with a visible result."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            raw["gates"] = tuple(GateRecord(**g) for g in raw["gates"])
            yield DecisionRecord(**raw)


def replay(
    records: Iterable[DecisionRecord],
    decide: Callable[[DecisionRecord], str],
) -> list[tuple[DecisionRecord, str]]:
    """Re-decide historical events under a new policy, WITHOUT executing anything.

    `decide` takes a DecisionRecord and returns the action the new policy would have
    chosen. Returns (record, new_action) pairs so a caller can diff old against new
    and see exactly which decisions a proposed rule change would have altered --
    before any customer is affected by it.
    """
    out: list[tuple[DecisionRecord, str]] = []
    for rec in records:
        out.append((rec, str(decide(rec))))
    return out


def diff_replay(pairs: Sequence[tuple[DecisionRecord, str]]) -> dict[str, int]:
    """Summarise a replay: how many decisions would change, and in which direction."""
    counts: dict[str, int] = {}
    for rec, new_action in pairs:
        key = (
            "unchanged"
            if new_action == rec.executed_action
            else (f"{rec.executed_action} -> {new_action}")
        )
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
