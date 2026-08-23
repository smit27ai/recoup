"""Message templates, and why no model runs in the send path.

The obvious build for "LLM message localisation" is to generate Hindi and Hinglish
copy at send time. In India that produces messages that are never delivered.

Under TRAI's TCCCPR 2018 and the DLT framework, every commercial SMS must match a
pre-registered content template EXACTLY -- punctuation, spacing and variable
positions included -- or the operator's scrubbing engine rejects it before it
reaches a handset. Registration takes days, and since January 2026 each variable
must carry a declared data type. WhatsApp is the same shape of constraint: a
business-initiated message outside the 24-hour service window must use a template
Meta has pre-approved.

A recovery nudge is business-initiated and outside any service window by definition.
So freeform generated text is not a compliance risk here, it is an undeliverable
message -- dropped upstream of us, silently, with the money still unrecovered.

That inverts where the model belongs. It does not write messages; it AUTHORS
CANDIDATE TEMPLATES, a human reviews and submits them for registration, and the send
path fills the approved template deterministically with no model involved. Exactly
the shape of the tier-2 rule-mining loop: the model proposes once, a person approves,
and the system runs deterministically forever after.

The validation below is therefore not stylistic. A template with a hardcoded amount
is not merely bad copy -- it is a template that is wrong for every other amount and
that burns a 3-7 day registration cycle to discover.

Sources for the constraints encoded here:
  TRAI TCCCPR 2018 / DLT content-template registration and scrubbing
  WhatsApp Business Platform template policy (24-hour service window)
Verify current rules with counsel before registering anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from recoup.domain import ActionKind, Channel

SLOT = re.compile(r"\{(\w+)\}")

# Channel limits. SMS is the binding one: past 160 GSM-7 characters a message splits
# into segments, each billed separately and each able to arrive out of order.
LENGTH_LIMITS: dict[Channel, int] = {
    Channel.SMS: 160,
    Channel.WHATSAPP: 1024,
    Channel.EMAIL: 4000,
    Channel.NONE: 4000,
}


class VariableKind(StrEnum):
    """Declared data type for a slot.

    DLT requires variables to be pre-tagged with a type and purpose, and only data
    matching the tag may be substituted at runtime. Modelling that here means a
    mismatch fails locally in milliseconds rather than at the operator days later.
    """

    AMOUNT = "amount"
    URL = "url"
    NAME = "name"
    DATE = "date"
    ORDER_ID = "order_id"


class TemplateStatus(StrEnum):
    DRAFT = "draft"
    """Authored, not yet reviewed. Must never be sent."""
    APPROVED = "approved"
    """Reviewed locally and registered with the operator. Sendable."""
    REJECTED = "rejected"


class TemplateError(Exception):
    """A template is not fit to register or send. Never silently corrected."""


# Language that manufactures pressure or implies consequences the merchant cannot
# deliver. Blocked because it is both worse-performing and, under RBI conduct
# expectations for recovery, a complaint waiting to happen -- and because a model
# asked for "urgent" copy will reach for exactly these by default.
FORBIDDEN = (
    "legal action",
    "court",
    "police",
    "blacklist",
    "blocked permanently",
    "last chance",
    "final warning",
    "immediately or",
    "failure to pay will",
    "penalty",
    "defaulter",
    "recovery agent",
)

# A literal rupee amount or URL inside template text means it was baked in rather
# than passed as a variable -- wrong for every other amount, and unfixable without
# another registration cycle.
LITERAL_AMOUNT = re.compile(r"(?:₹|Rs\.?\s*)\d")
LITERAL_URL = re.compile(r"https?://|www\.|rzp\.io")


@dataclass(frozen=True, slots=True)
class Variable:
    name: str
    kind: VariableKind
    description: str = ""


@dataclass(frozen=True, slots=True)
class MessageTemplate:
    """One registrable template.

    `body` carries `{slot}` placeholders. What gets registered is this exact string;
    what gets sent is this string with slots filled and nothing else changed.
    """

    template_id: str
    action: ActionKind
    language: str
    """en | hi | hinglish."""
    channel: Channel
    body: str
    variables: tuple[Variable, ...]
    status: TemplateStatus = TemplateStatus.DRAFT
    dlt_template_id: str | None = None
    authored_by: str = "human"
    rationale: str = ""

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(SLOT.findall(self.body))

    @property
    def sendable(self) -> bool:
        return self.status is TemplateStatus.APPROVED

    def render(self, values: dict[str, str]) -> str:
        """Fill the slots. No model, no variation, no cleverness.

        Refuses on a missing or unexpected value rather than substituting a blank,
        because a template rendered with an empty slot no longer matches the
        registered one and is rejected at the operator anyway -- better to fail here,
        loudly, than to have a message vanish silently in the network.
        """
        if not self.sendable:
            raise TemplateError(
                f"template {self.template_id} is {self.status}, not approved for sending"
            )
        declared = {v.name for v in self.variables}
        missing = declared - values.keys()
        if missing:
            raise TemplateError(f"missing values for {sorted(missing)}")
        extra = values.keys() - declared
        if extra:
            raise TemplateError(f"unexpected values for {sorted(extra)}")

        rendered = self.body
        for name, value in values.items():
            if not str(value).strip():
                raise TemplateError(f"value for {name!r} is empty")
            rendered = rendered.replace(f"{{{name}}}", str(value))

        limit = LENGTH_LIMITS[self.channel]
        if len(rendered) > limit:
            raise TemplateError(
                f"rendered message is {len(rendered)} chars, over the {self.channel} "
                f"limit of {limit}"
            )
        return rendered

    def matches_rendered(self, rendered: str, values: dict[str, str]) -> bool:
        """Would the operator accept this as an instance of this template?

        Reconstructs the template from the rendered text by putting the slots back.
        This is the check the DLT scrubbing engine performs, done locally first.
        """
        recovered = rendered
        # Longest values first, so a short value that is a substring of a longer one
        # cannot corrupt the reconstruction.
        for name, value in sorted(values.items(), key=lambda kv: -len(str(kv[1]))):
            recovered = recovered.replace(str(value), f"{{{name}}}", 1)
        return recovered == self.body


def validate(template: MessageTemplate) -> list[str]:
    """Everything wrong with a template, all at once.

    Returns every problem rather than the first, because a reviewer waiting 3-7 days
    for registration should not discover faults one round trip at a time.
    """
    problems: list[str] = []
    body = template.body
    declared = {v.name for v in template.variables}
    used = set(template.slots)

    if not body.strip():
        problems.append("body is empty")

    for name in sorted(used - declared):
        problems.append(f"slot {{{name}}} is used but not declared as a variable")
    for name in sorted(declared - used):
        problems.append(f"variable {name!r} is declared but never used in the body")

    if LITERAL_AMOUNT.search(body):
        problems.append(
            "body contains a literal rupee amount; amounts must be a tagged variable "
            "or the template is wrong for every other amount"
        )
    if LITERAL_URL.search(body):
        problems.append("body contains a literal URL; links must be a tagged variable")

    lowered = body.lower()
    for phrase in FORBIDDEN:
        if phrase in lowered:
            problems.append(f"body contains coercive language: {phrase!r}")

    if template.action.is_contact:
        kinds = {v.kind for v in template.variables}
        if VariableKind.URL not in kinds:
            problems.append(
                "a contact template must carry a payment URL variable -- telling "
                "someone to pay without saying how is worse than silence"
            )

    # Worst case: every slot filled with a plausible maximum. A template that fits
    # only for small amounts and short links will split into billed SMS segments in
    # production, months after registration.
    worst = body
    for variable in template.variables:
        worst = worst.replace(f"{{{variable.name}}}", _worst_case(variable.kind))
    limit = LENGTH_LIMITS[template.channel]
    if len(worst) > limit:
        problems.append(
            f"worst-case render is {len(worst)} chars, over the {template.channel} limit of {limit}"
        )

    if template.language not in {"en", "hi", "hinglish"}:
        problems.append(f"unsupported language {template.language!r}")

    return problems


def _worst_case(kind: VariableKind) -> str:
    return {
        VariableKind.AMOUNT: "Rs.10,00,000.00",
        VariableKind.URL: "https://rzp.io/i/AbCdEfGhIj",
        VariableKind.NAME: "Venkataraghavan Subramanian",
        VariableKind.DATE: "31 December 2026",
        VariableKind.ORDER_ID: "order_QpX1mN2oP3qR4sT5u",
    }[kind]


@dataclass(slots=True)
class TemplateRegistry:
    """Approved templates, keyed by what the send path knows.

    Lookup falls back along language before it falls back along channel: an English
    message on the customer's preferred channel beats a Hindi one somewhere they do
    not read. Falling back is always better than not sending, and always worse than
    having the right template registered -- which is what the review queue is for.
    """

    templates: dict[str, MessageTemplate] = field(default_factory=dict)

    def add(self, template: MessageTemplate) -> MessageTemplate:
        problems = validate(template)
        if problems:
            raise TemplateError(
                f"{template.template_id} is not registrable:\n  - " + "\n  - ".join(problems)
            )
        self.templates[template.template_id] = template
        return template

    def approve(self, template_id: str, dlt_template_id: str) -> MessageTemplate:
        """Mark a template as registered with the operator.

        The DLT id is required: without it there is nothing to prove registration
        happened, and an unregistered send is rejected upstream regardless of what
        our own status field claims.
        """
        from dataclasses import replace

        template = self.templates.get(template_id)
        if template is None:
            raise TemplateError(f"no template {template_id!r}")
        if not dlt_template_id.strip():
            raise TemplateError("a DLT/Meta template id is required to approve")
        approved = replace(
            template, status=TemplateStatus.APPROVED, dlt_template_id=dlt_template_id
        )
        self.templates[template_id] = approved
        return approved

    def find(self, action: ActionKind, language: str, channel: Channel) -> MessageTemplate | None:
        """Best approved template, degrading language first, then channel."""
        candidates = [t for t in self.templates.values() if t.sendable and t.action == action]
        for lang in (language, "hinglish" if language == "hi" else "en", "en"):
            for chan in (channel, Channel.SMS, Channel.EMAIL):
                for template in candidates:
                    if template.language == lang and template.channel == chan:
                        return template
        return None

    def pending(self) -> list[MessageTemplate]:
        return [t for t in self.templates.values() if t.status is TemplateStatus.DRAFT]

    def __len__(self) -> int:
        return len(self.templates)
