"""Tier 1 diagnosis: deterministic error_reason -> root cause lookup.

This is the layer that handles ~110 of Razorpay's documented failure reasons with a
table lookup: no model call, no latency, no cost, no variance, and a decision any
reviewer can audit by reading one TSV row.

Tier 2 (an LLM) exists only for reasons this table does NOT contain -- new codes
Razorpay adds, or gateway passthrough strings. See `escalation.py`. The split is
deliberate and is the answer to "where did you choose not to use AI": a lookup table
beats a language model at looking things up.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

TAXONOMY_PATH = Path(__file__).resolve().parents[3] / "data" / "error_taxonomy.tsv"

_FIELDS = (
    "reason",
    "error_class",
    "root_cause",
    "retry_class",
    "new_instrument",
    "customer_action",
    "owner",
    "in_scope",
)


class RootCause(StrEnum):
    """Why the money did not move, at the granularity recovery policy cares about."""

    FUNDS = "FUNDS"
    INSTRUMENT_INVALID = "INSTRUMENT_INVALID"
    INSTRUMENT_BLOCKED = "INSTRUMENT_BLOCKED"
    INSTRUMENT_NOT_ENROLLED = "INSTRUMENT_NOT_ENROLLED"
    AUTH_ABANDONED = "AUTH_ABANDONED"
    AUTH_FAILED = "AUTH_FAILED"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    ISSUER_DOWN = "ISSUER_DOWN"
    PSP_DOWN = "PSP_DOWN"
    GATEWAY_DOWN = "GATEWAY_DOWN"
    RISK_DECLINE = "RISK_DECLINE"
    MANDATE_PROBLEM = "MANDATE_PROBLEM"
    CREDIT_INELIGIBLE = "CREDIT_INELIGIBLE"
    MERCHANT_CONFIG = "MERCHANT_CONFIG"
    INTEGRATION_BUG = "INTEGRATION_BUG"
    COMPLIANCE = "COMPLIANCE"
    OPS = "OPS"
    UNKNOWN = "UNKNOWN"


class RetryClass(StrEnum):
    """When, if ever, another attempt is worth making."""

    NOW = "NOW"
    SOON = "SOON"
    SCHEDULED = "SCHEDULED"
    NEVER = "NEVER"


class Owner(StrEnum):
    CUSTOMER = "customer"
    BUSINESS = "business"
    BANK = "bank"
    RAZORPAY = "razorpay"


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """What tier 1 concluded about a single failure."""

    reason: str
    error_class: str
    root_cause: RootCause
    retry_class: RetryClass
    new_instrument: bool
    """Retrying the SAME instrument is provably futile."""
    customer_action: bool
    """A human must act, so contacting them can actually change the outcome."""
    owner: Owner
    in_scope: bool
    """False means this is a merchant bug or ops ticket, not recoverable revenue."""
    tier: int = 1
    """1 = table lookup. 2 = LLM escalation. Recorded on every decision."""
    confidence: float = 1.0

    @property
    def contactable(self) -> bool:
        """Whether messaging this customer could plausibly recover the money.

        Note this is necessary but nowhere near sufficient -- the compliance gates
        still get an absolute veto downstream. Nothing in diagnosis authorises contact.
        """
        return self.in_scope and self.customer_action

    @property
    def retryable(self) -> bool:
        return self.in_scope and self.retry_class is not RetryClass.NEVER


@lru_cache(maxsize=1)
def load_taxonomy(path: Path | None = None) -> dict[str, Diagnosis]:
    """Load and validate the taxonomy TSV. Cached; the table is immutable at runtime."""
    src = path or TAXONOMY_PATH
    if not src.exists():
        raise FileNotFoundError(f"taxonomy not found at {src}")

    table: dict[str, Diagnosis] = {}
    with src.open(encoding="utf-8", newline="") as fh:
        rows = (line for line in fh if not line.startswith("#") and line.strip())
        for lineno, row in enumerate(csv.reader(rows, delimiter="\t"), start=1):
            if len(row) != len(_FIELDS):
                raise ValueError(
                    f"{src.name} row {lineno}: expected {len(_FIELDS)} columns, got {len(row)}"
                )
            rec = dict(zip(_FIELDS, row, strict=True))
            reason = rec["reason"]
            if reason in table:
                # The Razorpay docs legitimately list payment_method_not_enabled twice
                # (cards and UPI) with identical routing. Identical duplicates are fine;
                # conflicting ones are a data bug we must not paper over.
                if table[reason].root_cause != RootCause(rec["root_cause"]):
                    raise ValueError(f"{src.name}: conflicting duplicate for {reason!r}")
                continue
            table[reason] = Diagnosis(
                reason=reason,
                error_class=rec["error_class"],
                root_cause=RootCause(rec["root_cause"]),
                retry_class=RetryClass(rec["retry_class"]),
                new_instrument=rec["new_instrument"] == "1",
                customer_action=rec["customer_action"] == "1",
                owner=Owner(rec["owner"]),
                in_scope=rec["in_scope"] == "1",
            )
    return table


def diagnose(error_reason: str | None) -> Diagnosis | None:
    """Tier 1 lookup.

    Returns None when the reason is absent from the table, which is the signal to
    escalate to tier 2. Returning None rather than guessing is the point: an
    unrecognised failure must never be silently bucketed as UNKNOWN and dunned.
    """
    if not error_reason:
        return None
    return load_taxonomy().get(error_reason.strip().lower())
