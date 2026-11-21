"""Shared prompt formatting so both backends receive equivalent input."""

from __future__ import annotations

import re

from hybrid_router.models import Request

SYSTEM_PROMPT = (
    "You are a helpful, accurate assistant. Answer the user's request directly and clearly. "
    "When the user asks for a specific output format such as JSON, respond with only that "
    "format and no extra commentary."
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def build_messages(request: Request) -> list[dict[str, str]]:
    """Chat-format messages: system prompt, prior context, then the current prompt."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend({"role": m.role, "content": m.content} for m in request.context)
    messages.append({"role": "user", "content": request.prompt})
    return messages


def clean_output(text: object) -> str:
    """Strip reasoning blocks (e.g. Qwen3 <think>…</think>) and surrounding whitespace."""
    if not isinstance(text, str):
        return ""
    return _THINK_BLOCK.sub("", text).strip()
