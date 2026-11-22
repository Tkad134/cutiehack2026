"""Shared prompt formatting so both backends receive equivalent input."""

from __future__ import annotations

import re

from hybrid_router.models import Request

SYSTEM_PROMPT = (
    "You are Alice, a helpful, accurate AI assistant powered by a hybrid local/cloud router. "
    "Answer the user's request directly and clearly."
)
# Added only when the prompt explicitly asks for JSON. Mentioning JSON on every
# request makes small local models (e.g. Qwen3 1.7B) answer plain questions in JSON.
JSON_INSTRUCTION = (
    " The user wants JSON: respond with only valid JSON and no extra commentary or code fences."
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def system_prompt_for(request: Request) -> str:
    if request.output.format == "json":
        return SYSTEM_PROMPT + JSON_INSTRUCTION
    return SYSTEM_PROMPT


def build_messages(request: Request) -> list[dict[str, str]]:
    """Chat-format messages: system prompt, prior context, then the current prompt."""
    messages = [{"role": "system", "content": system_prompt_for(request)}]
    messages.extend({"role": m.role, "content": m.content} for m in request.context)
    messages.append({"role": "user", "content": request.prompt})
    return messages


def clean_output(text: object) -> str:
    """Strip reasoning blocks (e.g. Qwen3 <think>…</think>) and surrounding whitespace."""
    if not isinstance(text, str):
        return ""
    return _THINK_BLOCK.sub("", text).strip()
