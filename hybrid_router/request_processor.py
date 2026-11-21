"""Request processor: validation, normalization, request IDs, context, output requirements."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Iterable, Optional

from hybrid_router.config import Settings
from hybrid_router.models import Message, Mode, OutputRequirements, Request

log = logging.getLogger("hybrid_router")

ALLOWED_CONTEXT_ROLES = {"user", "assistant"}


class RequestValidationError(ValueError):
    """Raised when a request cannot be processed."""


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


def normalize_prompt(prompt: str) -> str:
    """Trim surrounding whitespace and unify line endings; inner content is untouched."""
    return prompt.replace("\r\n", "\n").replace("\r", "\n").strip()


# --- Output-format detection -------------------------------------------------

_JSON_REQUEST = re.compile(
    r"|".join(
        [
            r"\b(?:in|as|into)\s+(?:a\s+|an\s+)?(?:valid\s+|strict\s+|pure\s+)?json\b",
            r"\b(?:return|output|respond|reply)\b[^.?!\n]{0,30}\bjson\b",
            r"\bjson\s+only\b",
            r"\bonly\s+(?:valid\s+)?json\b",
        ]
    ),
    re.IGNORECASE,
)
_FIELDS_CLAUSE = re.compile(r"\b(?:keys|fields|properties)\b([^.\n]*)", re.IGNORECASE)
_QUOTED_NAME = re.compile(r"[\"'`]([A-Za-z_][\w\-]*)[\"'`]")
_LIST_SPLIT = re.compile(r"\s*(?:,|\band\b)\s*", re.IGNORECASE)
_BARE_NAME = re.compile(r"^[A-Za-z_][\w\-]*$")


def detect_output_requirements(prompt: str) -> OutputRequirements:
    """Detect an explicit JSON request and, if stated, its required top-level fields.

    Supported field syntaxes:  keys "name" and "age"   |   fields: name, age, email
    """
    if not _JSON_REQUEST.search(prompt):
        return OutputRequirements()
    fields: list[str] = []
    for clause in _FIELDS_CLAUSE.finditer(prompt):
        segment = clause.group(1)
        quoted = _QUOTED_NAME.findall(segment)
        if quoted:
            fields.extend(quoted)
        elif segment.lstrip().startswith(":"):
            for part in _LIST_SPLIT.split(segment.lstrip()[1:]):
                part = part.strip()
                if _BARE_NAME.match(part):
                    fields.append(part)
    return OutputRequirements(format="json", required_fields=tuple(dict.fromkeys(fields)))


# --- Validation ----------------------------------------------------------------


def _parse_mode(mode: Any) -> Mode:
    if isinstance(mode, Mode):
        return mode
    try:
        return Mode(str(mode).strip().lower())
    except ValueError:
        valid = ", ".join(m.value for m in Mode)
        raise RequestValidationError(f"Unknown execution mode '{mode}'. Use one of: {valid}.")


def _parse_context(context: Optional[Iterable[Any]], max_messages: int) -> tuple[Message, ...]:
    if context is None:
        return ()
    if isinstance(context, (str, bytes)) or not isinstance(context, Iterable):
        raise RequestValidationError("Context must be a list of messages.")
    messages: list[Message] = []
    for i, item in enumerate(context):
        if isinstance(item, Message):
            role, content = item.role, item.content
        elif isinstance(item, dict):
            role, content = item.get("role"), item.get("content")
        else:
            raise RequestValidationError(f"Context message {i} must have 'role' and 'content'.")
        if role not in ALLOWED_CONTEXT_ROLES:
            raise RequestValidationError(
                f"Context message {i} has invalid role '{role}' (use 'user' or 'assistant')."
            )
        if not isinstance(content, str) or not content.strip():
            raise RequestValidationError(f"Context message {i} has empty or non-text content.")
        messages.append(Message(role=role, content=normalize_prompt(content)))
    # Keep only the most recent turns; older turns are dropped, not summarized.
    if max_messages >= 0:
        messages = messages[-max_messages:] if max_messages else []
    return tuple(messages)


def process_request(
    prompt: Any,
    mode: Any = Mode.AUTO,
    context: Optional[Iterable[Any]] = None,
    settings: Optional[Settings] = None,
    request_id: Optional[str] = None,
) -> Request:
    """Validate and normalize raw input into a Request. Raises RequestValidationError."""
    settings = settings or Settings()
    rid = request_id or new_request_id()

    if not isinstance(prompt, str):
        raise RequestValidationError("Prompt must be text.")
    text = normalize_prompt(prompt)
    if not text:
        raise RequestValidationError("Prompt is empty. Please enter a request.")
    if len(text) > settings.max_prompt_chars:
        raise RequestValidationError(
            f"Prompt is too long ({len(text):,} characters; limit is "
            f"{settings.max_prompt_chars:,})."
        )

    request = Request(
        request_id=rid,
        prompt=text,
        mode=_parse_mode(mode),
        context=_parse_context(context, settings.max_context_messages),
        output=detect_output_requirements(text),
    )
    log.info(
        "[%s] request accepted: mode=%s prompt_chars=%d context_messages=%d output_format=%s",
        rid, request.mode.value, len(text), len(request.context), request.output.format,
    )
    return request
