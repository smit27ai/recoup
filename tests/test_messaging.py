"""Template and authoring tests.

Validation here is not stylistic. Every rule encodes something that would otherwise
fail days later at a telecom operator, or reach a real person as coercion. A
template that fails registration costs a 3-7 day cycle to discover.
"""

from __future__ import annotations

import pytest

from recoup.domain import ActionKind, Channel
from recoup.messaging.authoring import StubAuthor, TemplateAuthoringService, _from_payload
from recoup.messaging.templates import (
    MessageTemplate,
    TemplateError,
    TemplateRegistry,
    TemplateStatus,
    Variable,
    VariableKind,
    validate,
)

AMOUNT = Variable("amount", VariableKind.AMOUNT)
LINK = Variable("link", VariableKind.URL)


def _template(body: str, **kw) -> MessageTemplate:
    base = {
        "template_id": "t1",
        "action": ActionKind.NUDGE,
        "language": "en",
        "channel": Channel.SMS,
        "body": body,
        "variables": (AMOUNT, LINK),
    }
    return MessageTemplate(**{**base, **kw})


GOOD = "Your payment of {amount} did not go through. Complete it here: {link}"


# --- validation -------------------------------------------------------------


def test_a_sound_template_validates() -> None:
    assert validate(_template(GOOD)) == []


def test_literal_amount_is_rejected() -> None:
    """The single most likely model mistake: a template baked to one amount is wrong
    for every other amount, and unfixable without another registration cycle."""
    problems = validate(
        _template("Your payment of Rs.2,499 failed. Pay here: {link}", variables=(LINK,))
    )
    assert any("literal rupee amount" in p for p in problems)


def test_literal_url_is_rejected() -> None:
    problems = validate(
        _template("Your payment of {amount} failed. Pay at https://rzp.io/i/x", variables=(AMOUNT,))
    )
    assert any("literal URL" in p for p in problems)


def test_undeclared_slot_is_rejected() -> None:
    """DLT requires every variable pre-tagged with a type; an untagged slot cannot
    be populated at runtime."""
    problems = validate(_template(GOOD + " Ref {order_id}"))
    assert any("order_id" in p and "not declared" in p for p in problems)


def test_declared_but_unused_variable_is_rejected() -> None:
    problems = validate(
        _template(GOOD, variables=(AMOUNT, LINK, Variable("name", VariableKind.NAME)))
    )
    assert any("never used" in p for p in problems)


@pytest.mark.parametrize(
    "phrase",
    ["legal action", "final warning", "last chance", "penalty", "recovery agent"],
)
def test_coercive_language_is_rejected(phrase: str) -> None:
    """Worse-performing and a breach of RBI conduct expectations. A model asked for
    urgency reaches for exactly these."""
    problems = validate(_template(f"Pay {{amount}} now or face {phrase}: {{link}}"))
    assert any("coercive" in p for p in problems)


def test_contact_template_must_carry_a_link() -> None:
    """Telling someone to pay without saying how is worse than silence."""
    problems = validate(
        _template("Your payment of {amount} did not go through.", variables=(AMOUNT,))
    )
    assert any("payment URL variable" in p for p in problems)


def test_worst_case_length_is_checked_not_typical_length() -> None:
    """A template that fits for Rs.99 and splits into billed segments for Rs.10,00,000
    fails in production months after registration."""
    body = (
        "Dear customer, we noticed your payment of {amount} could not be completed "
        "at this time, please use the following secure link to complete it: {link}"
    )
    problems = validate(_template(body))
    assert any("worst-case" in p for p in problems)


def test_the_same_template_is_fine_on_whatsapp() -> None:
    body = (
        "Dear customer, we noticed your payment of {amount} could not be completed "
        "at this time, please use the following secure link to complete it: {link}"
    )
    assert validate(_template(body, channel=Channel.WHATSAPP)) == []


def test_unsupported_language_is_rejected() -> None:
    assert any("unsupported language" in p for p in validate(_template(GOOD, language="fr")))


def test_all_problems_are_reported_at_once() -> None:
    """A reviewer waiting days per registration round trip should not discover faults
    one at a time."""
    problems = validate(
        _template("Pay Rs.500 at https://x.com or face legal action", variables=(AMOUNT, LINK))
    )
    assert len(problems) >= 4


# --- rendering --------------------------------------------------------------


def test_draft_templates_cannot_be_sent() -> None:
    with pytest.raises(TemplateError, match="not approved"):
        _template(GOOD).render({"amount": "Rs.999.00", "link": "https://rzp.io/i/x"})


def test_approved_template_renders() -> None:
    template = _template(GOOD, status=TemplateStatus.APPROVED)
    out = template.render({"amount": "Rs.999.00", "link": "https://rzp.io/i/x"})
    assert "Rs.999.00" in out
    assert "https://rzp.io/i/x" in out
    assert "{" not in out


def test_missing_value_refuses_rather_than_leaving_a_blank() -> None:
    """A rendered blank no longer matches the registered template and is rejected at
    the operator anyway. Better to fail loudly here."""
    template = _template(GOOD, status=TemplateStatus.APPROVED)
    with pytest.raises(TemplateError, match="missing values"):
        template.render({"amount": "Rs.999.00"})


def test_empty_value_is_refused() -> None:
    template = _template(GOOD, status=TemplateStatus.APPROVED)
    with pytest.raises(TemplateError, match="empty"):
        template.render({"amount": "Rs.999.00", "link": "  "})


def test_unexpected_value_is_refused() -> None:
    template = _template(GOOD, status=TemplateStatus.APPROVED)
    with pytest.raises(TemplateError, match="unexpected"):
        template.render({"amount": "x", "link": "y", "surprise": "z"})


def test_oversized_render_is_refused() -> None:
    template = _template(GOOD, status=TemplateStatus.APPROVED)
    with pytest.raises(TemplateError, match="over the"):
        template.render({"amount": "Rs." + "9" * 200, "link": "https://rzp.io/i/x"})


def test_rendered_message_reconstructs_to_the_registered_template() -> None:
    """This is the check the DLT scrubbing engine performs at the operator. Running
    it locally means a mismatch surfaces in milliseconds instead of as silent
    non-delivery."""
    template = _template(GOOD, status=TemplateStatus.APPROVED)
    values = {"amount": "Rs.2,499.00", "link": "https://rzp.io/i/AbCd"}
    assert template.matches_rendered(template.render(values), values)


# --- registry ---------------------------------------------------------------


def test_registry_refuses_an_invalid_template() -> None:
    registry = TemplateRegistry()
    with pytest.raises(TemplateError, match="not registrable"):
        registry.add(_template("Pay Rs.500 now: {link}", variables=(LINK,)))


def test_approval_requires_a_registration_id() -> None:
    """Without one there is nothing proving registration happened, and an
    unregistered send is rejected upstream regardless of our own status field."""
    registry = TemplateRegistry()
    registry.add(_template(GOOD))
    with pytest.raises(TemplateError, match="DLT/Meta template id is required"):
        registry.approve("t1", "")


def test_only_approved_templates_are_findable() -> None:
    registry = TemplateRegistry()
    registry.add(_template(GOOD))
    assert registry.find(ActionKind.NUDGE, "en", Channel.SMS) is None
    registry.approve("t1", "DLT-12345")
    assert registry.find(ActionKind.NUDGE, "en", Channel.SMS) is not None


def test_language_falls_back_before_channel() -> None:
    """An English message on the channel they read beats a Hindi one somewhere they
    do not."""
    registry = TemplateRegistry()
    for language in ("en",):
        template = _template(
            GOOD, template_id=f"n-{language}", language=language, channel=Channel.WHATSAPP
        )
        registry.add(template)
        registry.approve(template.template_id, "DLT-1")
    found = registry.find(ActionKind.NUDGE, "hi", Channel.WHATSAPP)
    assert found is not None
    assert found.language == "en"


def test_no_template_at_all_returns_none_rather_than_improvising() -> None:
    assert TemplateRegistry().find(ActionKind.NUDGE, "en", Channel.SMS) is None


# --- authoring --------------------------------------------------------------


def test_every_stub_template_is_registrable() -> None:
    """The offline baseline must always be sendable -- it is the floor when
    authoring is unavailable."""
    service = TemplateAuthoringService(StubAuthor())
    for action in (
        ActionKind.NUDGE,
        ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH,
        ActionKind.NUDGE_WITH_INCENTIVE,
    ):
        service.propose(action, Channel.SMS)
    assert len(service.proposals) == 9
    assert len(service.registrable()) == 9


def test_hinglish_is_latin_script_not_devanagari() -> None:
    """Hinglish is how Indian customers actually read transactional messages. Writing
    it in Devanagari makes it Hindi, and a mistranslated payment message reads as a
    scam."""
    template = StubAuthor().author(ActionKind.NUDGE, "hinglish", Channel.SMS)
    assert template is not None
    assert not any("ऀ" <= ch <= "ॿ" for ch in template.body)
    assert "payment" in template.body.lower(), "payment nouns stay in English"


def test_hindi_is_devanagari() -> None:
    template = StubAuthor().author(ActionKind.NUDGE, "hi", Channel.SMS)
    assert template is not None
    assert any("ऀ" <= ch <= "ॿ" for ch in template.body)


def test_a_bad_model_proposal_is_reported_not_corrected() -> None:
    """Quietly patching a hardcoded amount would hide that the prompt needs work, and
    would mean the reviewer approves text nobody wrote."""
    proposal = _from_payload(
        {
            "body": "Aapka Rs.2,499 ka payment fail hua. Pay here: {link}",
            "variables": [{"name": "link", "kind": "url"}],
            "rationale": "test",
        },
        ActionKind.NUDGE,
        "hinglish",
        Channel.SMS,
        "test-model",
    )
    problems = validate(proposal)
    assert any("literal rupee amount" in p for p in problems)
    assert "Rs.2,499" in proposal.body, "the proposal is preserved verbatim for review"


def test_authoring_failure_is_never_load_bearing() -> None:
    """Authoring is offline and human-reviewed. If it breaks, approved templates keep
    sending and nothing degrades for a customer."""

    class Broken:
        name = "broken"

        def author(self, action, language, channel):
            raise RuntimeError("model unavailable")

    service = TemplateAuthoringService(Broken())
    assert service.propose(ActionKind.NUDGE, Channel.SMS) == []


def test_review_payload_is_json_and_carries_the_problems() -> None:
    import json

    service = TemplateAuthoringService(StubAuthor())
    service.propose(ActionKind.NUDGE, Channel.SMS)
    rows = json.loads(service.as_review_payload())
    assert rows
    assert {"template_id", "body", "variables", "problems"} <= rows[0].keys()


# --- the send path ----------------------------------------------------------


def test_default_registry_covers_every_contact_action() -> None:
    from recoup.execution import default_registry

    registry = default_registry()
    for action in (
        ActionKind.NUDGE,
        ActionKind.NUDGE_WITH_INSTRUMENT_SWITCH,
        ActionKind.NUDGE_WITH_INCENTIVE,
    ):
        for language in ("en", "hi", "hinglish"):
            assert registry.find(action, language, Channel.WHATSAPP) is not None


# --- the shipped taxonomy is a curated artefact -----------------------------


def test_shipped_taxonomy_contains_no_runtime_promotions() -> None:
    """The tracked taxonomy is sourced from Razorpay's published error list. Rows with
    error_class 'escalated' got there by a runtime promotion -- someone approving a
    tier-2 proposal through the console -- and must never be committed.

    This exists because it happened. Driving the console demo in a browser approved
    two stub-authored rules straight into the tracked file, and they were committed
    twice before anyone noticed. The only signal was a line-ending warning in the
    commit output, which is not a signal anyone reads.
    """
    from recoup.diagnosis.taxonomy import TAXONOMY_PATH

    rows = [
        line
        for line in TAXONOMY_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    promoted = [r.split("\t")[0] for r in rows if "\tescalated\t" in r]
    assert not promoted, (
        f"runtime-promoted rules were committed to the shipped taxonomy: {promoted}. "
        "Approving rules in the console writes to the real file -- revert it before "
        "committing."
    )


def test_shipped_taxonomy_is_the_expected_size() -> None:
    """A blunt guard against the file drifting in either direction unnoticed."""
    from recoup.diagnosis.taxonomy import load_taxonomy

    load_taxonomy.cache_clear()
    assert len(load_taxonomy()) == 110
