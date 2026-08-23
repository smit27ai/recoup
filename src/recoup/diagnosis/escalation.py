"""Tier 2: model-backed diagnosis, for the codes the table does not contain.

Tier 1 resolves ~110 documented Razorpay error reasons by table lookup. This module
exists for what tier 1 returns `None` on: a code Razorpay adds after we shipped, or
a gateway passthrough string nobody has classified yet. That is a genuinely
ambiguous natural-language problem, which is what a language model is actually good
at -- unlike looking things up in a table, which it is bad at.

Three ideas carry this file.

**Trust is graded by blast radius, not by the model's confidence.** Three tiers of
consequence, and each needs a different kind of authority:

  route to ops    zero customer impact. Accepted at any confidence -- being wrong
                  costs an ops ticket nobody needed.
  silent retry    near-zero impact. Disturbs nobody; a wasted attempt costs an API
                  call. Needs only enough confidence to be worth the attempt.
  contact         high impact, irreversible. Dunning a real person over our own
                  integration bug is the most damaging mistake this system makes.

So tier 2 may route and may retry, and it may NEVER authorise contacting a customer
-- at any confidence, including 0.99. The gate on contact is human review of the
mined rule, not the model's self-assessment, because a model can be confidently
wrong and its confidence is exactly the thing you cannot check at 3am. Once a
reviewer approves the rule it becomes tier 1 and contact unlocks through the
ordinary path.

An earlier version of this used one confidence threshold for everything, which
blocked harmless retries on low-confidence codes while still leaning on confidence
for the one decision where confidence is not good enough evidence. Both halves were
wrong in the same way: they treated a single scalar as if it measured risk.

**Every escalation mines a rule.** A tier-2 answer is not just a decision, it is a
candidate row for `error_taxonomy.tsv`. Proposals land in a review queue; once a
human approves one, that code is tier 1 forever after -- free, instant, deterministic,
and auditable. The model's job is to shrink its own job. A system that calls a model
for the same unknown code on the hundred-thousandth occurrence has not learned
anything.

**The same unknown code costs one call, not N.** Results are cached by reason, so a
new code appearing across ten thousand events produces exactly one model call. This
is a cost argument second and a consistency argument first: two events with an
identical error reason must never receive different diagnoses because the sampler
went a different way.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Protocol, cast

from recoup.diagnosis.taxonomy import (
    Diagnosis,
    Owner,
    RetryClass,
    RootCause,
    load_taxonomy,
)

MIN_CONFIDENCE_TO_RETRY = 0.5
"""Floor for spending a silent retry on a model-classified code.

Low, on purpose: a retry disturbs nobody, so the cost of acting on a mediocre guess
is one wasted API call, while the cost of NOT acting is real money left on the
floor. There is deliberately no corresponding threshold for contact -- see the
module docstring. No confidence value unlocks messaging a customer.
"""

DEFAULT_MODEL = "claude-opus-5"
"""Escalations are rare and each one can authorise money movement or a message to a
real person. This is the wrong place to economise on capability; the caching below
is what keeps the cost negligible."""

# Root causes whose consequence is "do less". Safe to accept without a confidence
# floor, because the failure mode is a wasted ops ticket rather than a wronged
# customer.
DE_ESCALATING = frozenset(
    {
        RootCause.MERCHANT_CONFIG,
        RootCause.INTEGRATION_BUG,
        RootCause.COMPLIANCE,
        RootCause.OPS,
        RootCause.RISK_DECLINE,
    }
)


@dataclass(frozen=True, slots=True)
class Proposal:
    """What tier 2 believes, plus everything needed to audit or approve it."""

    reason: str
    root_cause: RootCause
    retry_class: RetryClass
    new_instrument: bool
    customer_action: bool
    owner: Owner
    in_scope: bool
    confidence: float
    rationale: str
    model: str
    proposed_at: datetime

    def to_diagnosis(self) -> Diagnosis:
        return Diagnosis(
            reason=self.reason,
            error_class="escalated",
            root_cause=self.root_cause,
            retry_class=self.retry_class,
            new_instrument=self.new_instrument,
            customer_action=self.customer_action,
            owner=self.owner,
            in_scope=self.in_scope,
            tier=2,
            confidence=self.confidence,
        )

    def to_taxonomy_row(self) -> str:
        """The exact TSV line a reviewer would paste into `error_taxonomy.tsv`."""
        return "\t".join(
            [
                self.reason,
                "escalated",
                str(self.root_cause),
                str(self.retry_class),
                "1" if self.new_instrument else "0",
                "1" if self.customer_action else "0",
                str(self.owner),
                "1" if self.in_scope else "0",
            ]
        )


class Escalator(Protocol):
    """Anything that can turn an unknown error reason into a Proposal."""

    name: str

    def propose(self, reason: str, context: dict[str, Any]) -> Proposal | None: ...


class InvalidProposal(Exception):
    """The model returned something outside the schema. Never coerced -- rejected."""


# --- schema shared by every backend -----------------------------------------

PROPOSAL_TOOL: dict[str, Any] = {
    "name": "classify_payment_failure",
    "description": (
        "Classify an unrecognised payment failure reason onto the recovery taxonomy. "
        "Answer only from the error reason and the surrounding payment context. If "
        "the reason is ambiguous, say so with a low confidence rather than guessing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "root_cause": {"type": "string", "enum": [str(rc) for rc in RootCause]},
            "retry_class": {"type": "string", "enum": [str(rc) for rc in RetryClass]},
            "new_instrument": {
                "type": "boolean",
                "description": "True if retrying the SAME payment instrument is futile.",
            },
            "customer_action": {
                "type": "boolean",
                "description": "True if a human customer must do something to resolve this.",
            },
            "owner": {"type": "string", "enum": [str(o) for o in Owner]},
            "in_scope": {
                "type": "boolean",
                "description": (
                    "True if this is recoverable customer revenue. False if it is a "
                    "merchant integration bug, a configuration problem, or an ops "
                    "ticket -- those must never result in contacting a customer."
                ),
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string", "description": "One or two sentences."},
        },
        "required": [
            "root_cause",
            "retry_class",
            "new_instrument",
            "customer_action",
            "owner",
            "in_scope",
            "confidence",
            "rationale",
        ],
    },
}

SYSTEM_PROMPT = """You classify payment failures for an Indian payments recovery system.

You are given an error reason that is NOT in our table of ~110 known Razorpay error
reasons. Your job is to place it on our taxonomy so a recovery policy can act.

The four questions that matter:
1. root_cause     -- why the money did not move
2. retry_class    -- NOW (minutes), SOON (hours), SCHEDULED (days), NEVER
3. new_instrument -- is retrying the SAME instrument provably futile
4. customer_action-- must a human customer do something

And the one that matters most: in_scope. If this failure is our own integration bug,
a merchant configuration problem, a compliance block, or an internal ops issue, then
in_scope is FALSE and no customer may be contacted about it. Contacting a customer
over a merchant-side bug is the most damaging error you can make here. When the
reason looks like it concerns the merchant account, the API request, or internal
processing rather than the customer's money or instrument, prefer in_scope=false.

Be calibrated. Reporting confidence 0.5 on a genuinely ambiguous code is correct and
useful; a confident wrong answer is worse than an admitted uncertainty, because low
confidence routes to a human and a high one may authorise a message to a real person.
"""


def build_prompt(reason: str, context: dict[str, Any]) -> str:
    known = sorted(load_taxonomy())[:40]
    return (
        f"Unrecognised error reason: {reason!r}\n\n"
        f"Payment context:\n{json.dumps(context, indent=2, default=str)}\n\n"
        f"For calibration, some reasons already in our table:\n"
        + ", ".join(known)
        + "\n\nClassify the unrecognised reason."
    )


# --- backends ---------------------------------------------------------------


@dataclass(slots=True)
class StubEscalator:
    """Deterministic offline backend.

    Not only a test double: it is the fallback whenever no API key is configured, so
    the system degrades to something predictable rather than to an exception. It
    reasons from substrings, which is crude but transparent, and it errs toward
    `in_scope=False` -- the direction where being wrong costs an ops ticket rather
    than a wrongly-dunned customer.
    """

    name: str = "stub"
    confidence: float = 0.55
    """Above MIN_CONFIDENCE_TO_RETRY, so the stub can still buy a silent retry on a
    plausible match -- but contact is closed to tier 2 entirely, so this number can
    never be what messages a real person."""

    def propose(self, reason: str, context: dict[str, Any]) -> Proposal | None:
        text = reason.lower()
        rules: Sequence[tuple[tuple[str, ...], RootCause, RetryClass, bool, bool, Owner, bool]] = (
            (
                ("expired", "invalid_card", "card_number"),
                RootCause.INSTRUMENT_INVALID,
                RetryClass.NEVER,
                True,
                True,
                Owner.CUSTOMER,
                True,
            ),
            (
                ("blocked", "restricted", "frozen"),
                RootCause.INSTRUMENT_BLOCKED,
                RetryClass.NEVER,
                True,
                True,
                Owner.CUSTOMER,
                True,
            ),
            (
                ("insufficient", "balance", "funds"),
                RootCause.FUNDS,
                RetryClass.SCHEDULED,
                False,
                True,
                Owner.CUSTOMER,
                True,
            ),
            (
                ("limit", "exceeded", "quota"),
                RootCause.LIMIT_EXCEEDED,
                RetryClass.SCHEDULED,
                False,
                True,
                Owner.CUSTOMER,
                True,
            ),
            (
                ("otp", "pin", "authentication", "3ds"),
                RootCause.AUTH_FAILED,
                RetryClass.NOW,
                False,
                True,
                Owner.CUSTOMER,
                True,
            ),
            (
                ("cancel", "abandon", "timed_out", "timeout", "expired_session"),
                RootCause.AUTH_ABANDONED,
                RetryClass.NOW,
                False,
                True,
                Owner.CUSTOMER,
                True,
            ),
            (
                ("downtime", "unavailable", "technical", "gateway"),
                RootCause.GATEWAY_DOWN,
                RetryClass.SOON,
                False,
                False,
                Owner.RAZORPAY,
                True,
            ),
            (
                ("bank", "issuer"),
                RootCause.ISSUER_DOWN,
                RetryClass.SOON,
                False,
                False,
                Owner.BANK,
                True,
            ),
            (
                ("mandate", "autopay", "subscription"),
                RootCause.MANDATE_PROBLEM,
                RetryClass.SOON,
                False,
                True,
                Owner.CUSTOMER,
                True,
            ),
            (
                ("not_enabled", "not_activated", "merchant", "config"),
                RootCause.MERCHANT_CONFIG,
                RetryClass.NEVER,
                False,
                False,
                Owner.BUSINESS,
                False,
            ),
            (
                ("invalid_request", "validation", "duplicate", "mismatch", "order_"),
                RootCause.INTEGRATION_BUG,
                RetryClass.NEVER,
                False,
                False,
                Owner.BUSINESS,
                False,
            ),
        )
        for needles, cause, retry, new_inst, cust_act, owner, in_scope in rules:
            # Report the needle that ACTUALLY matched, not the first in the tuple.
            # This rationale is shown to a human deciding whether to promote the rule
            # into the permanent table, so a plausible-but-wrong explanation is worse
            # than none: it invites approving a rule that fired for a different
            # reason than the one stated.
            matched = next((n for n in needles if n in text), None)
            if matched is not None:
                return Proposal(
                    reason=reason,
                    root_cause=cause,
                    retry_class=retry,
                    new_instrument=new_inst,
                    customer_action=cust_act,
                    owner=owner,
                    in_scope=in_scope,
                    confidence=self.confidence,
                    rationale=f"substring match on {matched!r} (stub backend)",
                    model=self.name,
                    proposed_at=datetime.now(),
                )
        # Nothing matched. Say so rather than inventing a category.
        return Proposal(
            reason=reason,
            root_cause=RootCause.UNKNOWN,
            retry_class=RetryClass.NEVER,
            new_instrument=False,
            customer_action=False,
            owner=Owner.BUSINESS,
            in_scope=False,
            confidence=0.2,
            rationale="no rule matched; refusing to guess (stub backend)",
            model=self.name,
            proposed_at=datetime.now(),
        )


@dataclass(slots=True)
class ClaudeEscalator:
    """Anthropic-backed backend.

    Uses a strict tool schema so the answer parses deterministically -- an escalation
    that needs regex to interpret is one that can be silently misread. The system
    prompt and the sample of known reasons are a stable prefix, so prompt caching
    makes repeat calls cheap.
    """

    api_key: str
    model: str = DEFAULT_MODEL
    name: str = "claude"
    max_tokens: int = 1024

    def propose(self, reason: str, context: dict[str, Any]) -> Proposal | None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise RuntimeError(
                "anthropic SDK not installed; install it or use StubEscalator"
            ) from exc

        client = anthropic.Anthropic(api_key=self.api_key)
        # The SDK's parameter types are TypedDicts. Importing them at module level
        # would make `anthropic` a hard dependency and defeat the optional-extra
        # design, so the shapes are declared as plain dicts above and cast here.
        # The schema itself is still enforced -- by PROPOSAL_TOOL on the way out and
        # by parse_proposal on the way back in.
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=cast(
                "Any",
                [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        # Stable prefix, so repeat escalations pay a fraction of the
                        # input cost.
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            ),
            tools=cast("Any", [PROPOSAL_TOOL]),
            # Force the tool: a prose answer here would be unparseable, and an
            # escalation that needs regex to interpret can be silently misread.
            tool_choice=cast("Any", {"type": "tool", "name": PROPOSAL_TOOL["name"]}),
            messages=[{"role": "user", "content": build_prompt(reason, context)}],
        )
        for block in response.content:
            # Tagged-union narrowing on `.type`, so the SDK's own types confirm the
            # block really carries `.input` rather than us assuming it does.
            if block.type == "tool_use":
                return parse_proposal(reason, dict(block.input), model=self.model)
        raise InvalidProposal("model returned no tool_use block")


def parse_proposal(reason: str, payload: dict[str, Any], *, model: str) -> Proposal:
    """Validate a raw model answer into a Proposal.

    Rejects rather than coerces. A model that returns a root cause we do not have is
    telling us something is wrong, and quietly mapping it onto the nearest valid
    value would hide exactly the signal we need.
    """
    try:
        root_cause = RootCause(payload["root_cause"])
        retry_class = RetryClass(payload["retry_class"])
        owner = Owner(payload["owner"])
        confidence = float(payload["confidence"])
    except (KeyError, ValueError) as exc:
        raise InvalidProposal(f"unparseable proposal for {reason!r}: {exc}") from exc

    if not 0.0 <= confidence <= 1.0:
        raise InvalidProposal(f"confidence {confidence} out of range for {reason!r}")

    for flag in ("new_instrument", "customer_action", "in_scope"):
        if not isinstance(payload.get(flag), bool):
            raise InvalidProposal(f"{flag} must be a boolean for {reason!r}")

    return Proposal(
        reason=reason,
        root_cause=root_cause,
        retry_class=retry_class,
        new_instrument=bool(payload["new_instrument"]),
        customer_action=bool(payload["customer_action"]),
        owner=owner,
        in_scope=bool(payload["in_scope"]),
        confidence=confidence,
        rationale=str(payload.get("rationale", "")),
        model=model,
        proposed_at=datetime.now(),
    )


# --- the service ------------------------------------------------------------


@dataclass(slots=True)
class ReviewQueue:
    """Candidate taxonomy rows awaiting a human.

    This is the rule-mining loop: approve one and the code becomes tier 1 forever.
    """

    proposals: dict[str, Proposal] = field(default_factory=dict)
    seen_count: dict[str, int] = field(default_factory=dict)

    def record(self, proposal: Proposal) -> None:
        self.proposals.setdefault(proposal.reason, proposal)
        self.seen_count[proposal.reason] = self.seen_count.get(proposal.reason, 0) + 1

    def __len__(self) -> int:
        return len(self.proposals)

    def by_impact(self) -> list[tuple[Proposal, int]]:
        """Most-seen first -- the rows worth a reviewer's attention today."""
        return sorted(
            ((p, self.seen_count[r]) for r, p in self.proposals.items()),
            key=lambda pair: -pair[1],
        )

    def as_taxonomy_rows(self) -> str:
        return "\n".join(p.to_taxonomy_row() for p, _ in self.by_impact())


class EscalationService:
    """Tier 2, with the safety policy wrapped around whichever backend is in use."""

    def __init__(
        self,
        backend: Escalator | None = None,
        *,
        review_queue: ReviewQueue | None = None,
        min_confidence_to_retry: float = MIN_CONFIDENCE_TO_RETRY,
    ) -> None:
        self.backend: Escalator = backend if backend is not None else StubEscalator()
        # Explicit None check: ReviewQueue defines __len__, so `or` would discard an
        # empty queue passed by the caller. Same trap as IdempotencyStore.
        self.review = review_queue if review_queue is not None else ReviewQueue()
        self.min_confidence_to_retry = min_confidence_to_retry
        self._cache: dict[str, Diagnosis | None] = {}
        self.calls = 0

    def diagnose(self, reason: str, context: dict[str, Any] | None = None) -> Diagnosis | None:
        """Escalate an unmapped reason. Returns None when nothing safe can be said.

        Cached by reason: the same unknown code costs one model call no matter how
        many events carry it, which matters for consistency before it matters for
        cost -- two identical failures must not get different diagnoses because the
        sampler went a different way.
        """
        if reason in self._cache:
            proposal = self.review.proposals.get(reason)
            if proposal is not None:
                self.review.seen_count[reason] += 1
            return self._cache[reason]

        self.calls += 1
        try:
            proposal = self.backend.propose(reason, context or {})
        except (InvalidProposal, Exception):
            # Tier 2 is an enhancement, never a dependency. If it is down, or
            # rate-limited, or returned nonsense, tier 1 already said None and the
            # caller routes to ops. Failing the whole recovery pipeline because a
            # model call failed would be a far worse outcome than an ops ticket.
            self._cache[reason] = None
            return None

        if proposal is None:
            self._cache[reason] = None
            return None

        self.review.record(proposal)
        diagnosis = self._apply_safety_policy(proposal)
        self._cache[reason] = diagnosis
        return diagnosis

    def _apply_safety_policy(self, proposal: Proposal) -> Diagnosis | None:
        """Graded trust. See the module docstring.

        Three outcomes:
          - routes away from the customer -> accepted as-is, at any confidence
          - in scope and worth a retry     -> accepted, with contact stripped out
          - too unsure to be worth trying  -> None, so the caller routes to a human
        """
        diagnosis = proposal.to_diagnosis()

        if proposal.root_cause in DE_ESCALATING or not proposal.in_scope:
            # Routes work AWAY from the customer. Being wrong costs an ops ticket,
            # so this needs no confidence floor at all.
            return diagnosis

        if proposal.confidence < self.min_confidence_to_retry:
            # Not confident enough to be worth even a silent attempt. A human looks.
            return None

        # Confident enough to retry, and never confident enough to contact. The
        # rule sits in the review queue; approving it promotes the code to tier 1
        # and contact unlocks through the ordinary path, having been seen by a
        # person. `replace` rather than a hand-built Diagnosis so new fields on the
        # dataclass cannot be silently dropped here.
        return replace(diagnosis, customer_action=False)


def build_escalator() -> Escalator:
    """Pick a backend from the environment, preferring the real one."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return ClaudeEscalator(api_key=key)
    return StubEscalator()
