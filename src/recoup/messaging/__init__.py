"""Message templates and the authoring loop that produces them."""

from recoup.messaging.authoring import (
    ClaudeAuthor,
    Proposal,
    StubAuthor,
    TemplateAuthoringService,
    build_author,
)
from recoup.messaging.templates import (
    MessageTemplate,
    TemplateError,
    TemplateRegistry,
    TemplateStatus,
    Variable,
    VariableKind,
    validate,
)

__all__ = [
    "ClaudeAuthor",
    "MessageTemplate",
    "Proposal",
    "StubAuthor",
    "TemplateAuthoringService",
    "TemplateError",
    "TemplateRegistry",
    "TemplateStatus",
    "Variable",
    "VariableKind",
    "build_author",
    "validate",
]
