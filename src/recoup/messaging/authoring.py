"""Authoring candidate templates with a model.

This is where the LLM belongs in messaging, and the only place it belongs. It writes
Hindi and Hinglish variants of a recovery message ONCE, as a registrable template
with tagged variable slots; a human reviews them; approved ones go to DLT/Meta for
registration; and from then on the send path fills them with no model involved.

The difference matters practically, not just architecturally. Generating text per
message would mean every message is unregistered and therefore undeliverable, and it
would put a model in the path of something a customer reads, at volume, with no
review. Generating templates means one reviewed artefact serves millions of sends,
and the reviewer sees the exact string that will reach people.

Every proposal is validated before a human ever sees it, and validation failures are
returned rather than corrected. A model that hardcodes an amount into a template has
made a specific, checkable mistake; quietly patching it would hide that the model
needs a better prompt, and would mean the reviewer approves text nobody wrote.

Hinglish is the interesting case and is deliberately not "Hindi with English words".
It is how Indian customers actually read transactional messages -- Latin script,
English for payment nouns because those are the words on the app, Hindi for the
connective tissue. Getting that wrong reads as a translation, which reads as a scam.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol, cast

from recoup.domain import ActionKind, Channel
from recoup.messaging.templates import (
    LENGTH_LIMITS,
    MessageTemplate,
    TemplateStatus,
    Variable,
    VariableKind,
    validate,
)

DEFAULT_MODEL = "claude-opus-5"
"""Authoring happens a handful of times per action, and each artefact is read by
millions of customers. Cheap is the wrong axis to optimise."""

LANGUAGES = ("en", "hi", "hinglish")

SYSTEM_PROMPT = """You write payment-recovery message templates for Indian customers.

These are registered with TRAI's DLT platform before use, so you are writing a
TEMPLATE, not a message. Hard requirements:

- Put every piece of variable data in a {slot}. NEVER write a literal rupee amount
  or a literal URL. A template with a baked-in amount is wrong for every other
  amount and wastes a multi-day registration cycle.
- Keep SMS templates under 160 characters INCLUDING worst-case variable values --
  assume an amount like Rs.10,00,000.00 and a link like https://rzp.io/i/AbCdEfGhIj.
- Never manufacture urgency or imply consequences: no legal action, no penalties, no
  "last chance", no "final warning". These perform worse and breach RBI conduct
  expectations for recovery communication.
- State plainly what happened, what it costs, and how to fix it. Nothing else.

On language:
- "en": plain Indian English. Not American, not florid.
- "hi": Devanagari script, natural spoken Hindi, not a literal translation.
- "hinglish": LATIN script. This is how Indian customers actually read transactional
  messages: English for payment nouns (payment, link, card, UPI) because those are
  the words in the app, Hindi for the connective tissue. Do NOT write Devanagari
  here, and do not translate the payment nouns. Getting this wrong reads as a
  translation, and a translated payment message reads as a scam.
"""

TEMPLATE_TOOL: dict[str, Any] = {
    "name": "propose_template",
    "description": "Propose one registrable message template with tagged variables.",
    "input_schema": {
        "type": "object",
        "properties": {
            "body": {
                "type": "string",
                "description": (
                    "Template text with {slot} placeholders. No literal amounts or URLs."
                ),
            },
            "variables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": [str(k) for k in VariableKind],
                        },
                        "description": {"type": "string"},
                    },
                    "required": ["name", "kind"],
                },
            },
            "rationale": {
                "type": "string",
                "description": "One sentence on the wording choices, for the reviewer.",
            },
        },
        "required": ["body", "variables", "rationale"],
    },
}

INTENT: dict[ActionKind, str] = {
    ActionKind.NUDGE: (
        "Their payment did not go through. Tell them, and give them a link to "
        "complete it whenever convenient. No pressure -- the payment may well have "
        "failed for a reason that was not theirs."
    ),
    ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH: (
        "Their saved payment method cannot complete this payment -- an expired card, "
        "a blocked instrument. Retrying the same method will fail again, so ask them "
        "to pay with a different method using the link."
    ),
    ActionKind.NUDGE_WITH_INCENTIVE: (
        "Their payment did not go through, and we are offering a discount to "
        "complete it. State the discount as a variable, never a literal."
    ),
}


class TemplateAuthor(Protocol):
    name: str

    def author(
        self, action: ActionKind, language: str, channel: Channel
    ) -> MessageTemplate | None: ...


@dataclass(slots=True)
class StubAuthor:
    """Hand-written templates, used offline and as the floor.

    Not a placeholder for the real thing so much as the thing that must always exist:
    if template authoring is unavailable, recovery still needs something registrable
    to send. These are the versions a human wrote, and they are what the model's
    proposals are reviewed against.
    """

    name: str = "stub"

    def author(self, action: ActionKind, language: str, channel: Channel) -> MessageTemplate | None:
        body = _STUB_BODIES.get((action, language))
        if body is None:
            return None
        return MessageTemplate(
            template_id=f"{action}-{language}-{channel}",
            action=action,
            language=language,
            channel=channel,
            body=body,
            variables=_variables_for(body),
            authored_by=self.name,
            rationale="hand-written baseline",
        )


# Kept short deliberately: an SMS template must survive worst-case variable values
# inside 160 characters, and Devanagari costs more per character on the wire.
_STUB_BODIES: dict[tuple[ActionKind, str], str] = {
    (
        ActionKind.NUDGE,
        "en",
    ): "Your payment of {amount} did not go through. Complete it here: {link}",
    (
        ActionKind.NUDGE,
        "hinglish",
    ): "Aapka {amount} ka payment complete nahi hua. Yahan pura karein: {link}",
    (ActionKind.NUDGE, "hi"): "आपका {amount} का भुगतान पूरा नहीं हुआ। यहाँ पूरा करें: {link}",
    (
        ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH,
        "en",
    ): "Your saved method could not complete {amount}. Pay another way: {link}",
    (
        ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH,
        "hinglish",
    ): "Saved method se {amount} ka payment nahi hua. Doosre method se karein: {link}",
    (
        ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH,
        "hi",
    ): "सेव किए गए तरीके से {amount} का भुगतान नहीं हुआ। दूसरे तरीके से करें: {link}",
    (
        ActionKind.NUDGE_WITH_INCENTIVE,
        "en",
    ): "Your payment of {amount} did not go through. Complete it with {discount} off: {link}",
    (
        ActionKind.NUDGE_WITH_INCENTIVE,
        "hinglish",
    ): "Aapka {amount} ka payment nahi hua. {discount} off ke saath karein: {link}",
    (
        ActionKind.NUDGE_WITH_INCENTIVE,
        "hi",
    ): "आपका {amount} का भुगतान नहीं हुआ। {discount} छूट के साथ पूरा करें: {link}",
}

_KINDS = {
    "amount": VariableKind.AMOUNT,
    "link": VariableKind.URL,
    "name": VariableKind.NAME,
    "date": VariableKind.DATE,
    "discount": VariableKind.AMOUNT,
    "order_id": VariableKind.ORDER_ID,
}


def _variables_for(body: str) -> tuple[Variable, ...]:
    from recoup.messaging.templates import SLOT

    return tuple(
        Variable(name=name, kind=_KINDS.get(name, VariableKind.NAME))
        for name in dict.fromkeys(SLOT.findall(body))
    )


@dataclass(slots=True)
class ClaudeAuthor:
    """Model-backed authoring, with a strict tool schema."""

    api_key: str
    model: str = DEFAULT_MODEL
    name: str = "claude"

    def author(self, action: ActionKind, language: str, channel: Channel) -> MessageTemplate | None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("anthropic SDK not installed; use StubAuthor") from exc

        intent = INTENT.get(action)
        if intent is None:
            return None

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=cast(
                "Any",
                [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            ),
            tools=cast("Any", [TEMPLATE_TOOL]),
            tool_choice=cast("Any", {"type": "tool", "name": TEMPLATE_TOOL["name"]}),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Channel: {channel} (limit {LENGTH_LIMITS[channel]} characters, "
                        f"including worst-case variable values)\n"
                        f"Language: {language}\n"
                        f"Situation: {intent}\n\n"
                        "Propose one template."
                    ),
                }
            ],
        )
        for block in response.content:
            if block.type == "tool_use":
                return _from_payload(dict(block.input), action, language, channel, self.model)
        return None


def _from_payload(
    payload: dict[str, Any],
    action: ActionKind,
    language: str,
    channel: Channel,
    model: str,
) -> MessageTemplate:
    variables = tuple(
        Variable(
            name=str(v["name"]),
            kind=VariableKind(v["kind"]),
            description=str(v.get("description", "")),
        )
        for v in payload.get("variables", [])
    )
    return MessageTemplate(
        template_id=f"{action}-{language}-{channel}",
        action=action,
        language=language,
        channel=channel,
        body=str(payload["body"]),
        variables=variables,
        status=TemplateStatus.DRAFT,
        authored_by=model,
        rationale=str(payload.get("rationale", "")),
    )


@dataclass(frozen=True, slots=True)
class Proposal:
    """A candidate template plus everything wrong with it."""

    template: MessageTemplate
    problems: tuple[str, ...]

    @property
    def registrable(self) -> bool:
        return not self.problems

    def report(self) -> str:
        head = (
            f"{self.template.template_id}  [{self.template.authored_by}]\n  {self.template.body}\n"
        )
        if self.registrable:
            return head + "  ok, ready for a reviewer"
        return head + "\n".join(f"  REJECTED: {p}" for p in self.problems)


class TemplateAuthoringService:
    """Author candidates, validate them, and hand a reviewer only what is registrable."""

    def __init__(self, author: TemplateAuthor | None = None) -> None:
        self.author_backend: TemplateAuthor = author if author is not None else StubAuthor()
        self.proposals: list[Proposal] = []

    def propose(
        self, action: ActionKind, channel: Channel, languages: tuple[str, ...] = LANGUAGES
    ) -> list[Proposal]:
        out: list[Proposal] = []
        for language in languages:
            try:
                template = self.author_backend.author(action, language, channel)
            except Exception:
                # Authoring is an offline, human-reviewed activity. If it fails the
                # existing approved templates keep sending; nothing degrades for a
                # customer.
                continue
            if template is None:
                continue
            proposal = Proposal(template=template, problems=tuple(validate(template)))
            out.append(proposal)
            self.proposals.append(proposal)
        return out

    def registrable(self) -> list[Proposal]:
        return [p for p in self.proposals if p.registrable]

    def as_review_payload(self) -> str:
        return json.dumps(
            [
                {
                    "template_id": p.template.template_id,
                    "language": p.template.language,
                    "channel": str(p.template.channel),
                    "body": p.template.body,
                    "variables": [
                        {"name": v.name, "kind": str(v.kind)} for v in p.template.variables
                    ],
                    "authored_by": p.template.authored_by,
                    "rationale": p.template.rationale,
                    "problems": list(p.problems),
                }
                for p in self.proposals
            ],
            ensure_ascii=False,
            indent=2,
        )


def build_author() -> TemplateAuthor:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return ClaudeAuthor(api_key=key) if key else StubAuthor()
