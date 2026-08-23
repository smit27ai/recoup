"""Tests that actually execute the Claude-backed code paths.

These exist because of a bug that shipped: ClaudeAuthor referenced an unimported
name and 268 tests passed anyway, because nothing ever called it. The model-backed
paths are dormant until an API key is present, which means they are exactly the code
most likely to be broken at the moment someone first turns them on.

The Anthropic client is replaced with a double, so no network call happens and no key
is needed -- what is under test is our request construction and response handling,
which is the part we own and the part that was wrong.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from recoup.diagnosis.escalation import ClaudeEscalator
from recoup.domain import ActionKind, Channel
from recoup.messaging.authoring import ClaudeAuthor
from recoup.messaging.templates import validate


class _Block:
    """Stands in for a tool_use content block."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.type = "tool_use"
        self.input = payload


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.content = [_Block(payload)]


class _Messages:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        return _Response(self._payload)


class _Client:
    def __init__(self, payload: dict[str, Any], **_: Any) -> None:
        self.messages = _Messages(payload)


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Install a fake `anthropic` module the lazy imports will pick up."""

    def _install(payload: dict[str, Any]) -> _Client:
        client = _Client(payload)
        module = types.ModuleType("anthropic")
        module.Anthropic = lambda **kw: client  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "anthropic", module)
        return client

    return _install


# --- template authoring -----------------------------------------------------

GOOD_TEMPLATE = {
    "body": "Aapka {amount} ka payment nahi hua. Yahan karein: {link}",
    "variables": [
        {"name": "amount", "kind": "amount"},
        {"name": "link", "kind": "url"},
    ],
    "rationale": "Latin script, payment nouns in English",
}


def test_claude_author_runs_end_to_end(fake_anthropic) -> None:
    """The regression. This would have caught the unimported LENGTH_LIMITS."""
    fake_anthropic(GOOD_TEMPLATE)
    template = ClaudeAuthor(api_key="sk-test").author(ActionKind.NUDGE, "hinglish", Channel.SMS)
    assert template is not None
    assert template.language == "hinglish"
    assert validate(template) == []


def test_author_tells_the_model_the_channel_limit(fake_anthropic) -> None:
    """A template over the SMS limit is useless, so the limit has to be in the
    prompt -- and reading it is what the missing import broke."""
    client = fake_anthropic(GOOD_TEMPLATE)
    ClaudeAuthor(api_key="sk-test").author(ActionKind.NUDGE, "en", Channel.SMS)
    content = client.messages.calls[0]["messages"][0]["content"]
    assert "160" in content, "the SMS limit must reach the model"


def test_author_forces_the_tool_so_output_is_parseable(fake_anthropic) -> None:
    client = fake_anthropic(GOOD_TEMPLATE)
    ClaudeAuthor(api_key="sk-test").author(ActionKind.NUDGE, "en", Channel.SMS)
    call = client.messages.calls[0]
    assert call["tool_choice"]["type"] == "tool"
    assert call["model"] == "claude-opus-5"


def test_author_caches_the_system_prompt(fake_anthropic) -> None:
    client = fake_anthropic(GOOD_TEMPLATE)
    ClaudeAuthor(api_key="sk-test").author(ActionKind.NUDGE, "en", Channel.SMS)
    system = client.messages.calls[0]["system"]
    assert system[0]["cache_control"]["type"] == "ephemeral"


def test_author_returns_none_for_an_action_with_no_message(fake_anthropic) -> None:
    fake_anthropic(GOOD_TEMPLATE)
    assert ClaudeAuthor(api_key="sk-test").author(ActionKind.RETRY_NOW, "en", Channel.SMS) is None


def test_a_bad_model_template_survives_to_be_reviewed(fake_anthropic) -> None:
    """Hardcoded amount. Preserved verbatim and reported, never patched."""
    fake_anthropic(
        {
            "body": "Aapka Rs.2,499 ka payment nahi hua: {link}",
            "variables": [{"name": "link", "kind": "url"}],
            "rationale": "x",
        }
    )
    template = ClaudeAuthor(api_key="sk-test").author(ActionKind.NUDGE, "hinglish", Channel.SMS)
    assert template is not None
    assert "Rs.2,499" in template.body
    assert any("literal rupee amount" in p for p in validate(template))


# --- tier-2 escalation ------------------------------------------------------

GOOD_PROPOSAL = {
    "root_cause": "FUNDS",
    "retry_class": "SCHEDULED",
    "new_instrument": False,
    "customer_action": True,
    "owner": "customer",
    "in_scope": True,
    "confidence": 0.82,
    "rationale": "balance language",
}


def test_claude_escalator_runs_end_to_end(fake_anthropic) -> None:
    fake_anthropic(GOOD_PROPOSAL)
    proposal = ClaudeEscalator(api_key="sk-test").propose("acct_low_balance", {})
    assert proposal is not None
    assert str(proposal.root_cause) == "FUNDS"
    assert proposal.confidence == 0.82


def test_escalator_sends_the_context_it_was_given(fake_anthropic) -> None:
    client = fake_anthropic(GOOD_PROPOSAL)
    ClaudeEscalator(api_key="sk-test").propose(
        "acct_low_balance", {"method": "upi", "amount_paise": 45000}
    )
    content = client.messages.calls[0]["messages"][0]["content"]
    assert "acct_low_balance" in content
    assert "upi" in content


def test_escalator_forces_the_tool(fake_anthropic) -> None:
    client = fake_anthropic(GOOD_PROPOSAL)
    ClaudeEscalator(api_key="sk-test").propose("x", {})
    assert client.messages.calls[0]["tool_choice"]["type"] == "tool"


def test_escalator_rejects_an_out_of_enum_root_cause(fake_anthropic) -> None:
    """Rejected, never coerced onto the nearest valid value."""
    from recoup.diagnosis.escalation import InvalidProposal

    fake_anthropic({**GOOD_PROPOSAL, "root_cause": "VIBES"})
    with pytest.raises(InvalidProposal):
        ClaudeEscalator(api_key="sk-test").propose("x", {})


def test_a_confident_model_still_cannot_unlock_contact(fake_anthropic) -> None:
    """The safety invariant, exercised through the real backend rather than a stub."""
    from recoup.diagnosis.escalation import EscalationService

    fake_anthropic({**GOOD_PROPOSAL, "confidence": 0.99})
    service = EscalationService(ClaudeEscalator(api_key="sk-test"))
    diagnosis = service.diagnose("acct_low_balance")
    assert diagnosis is not None
    assert diagnosis.contactable is False
