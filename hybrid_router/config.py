"""Application settings loaded from environment variables / a .env file.

The OpenAI API key is deliberately NOT stored here so it can never leak through
a repr, log line, or UI dump. The cloud engine reads it directly from the
environment when it builds its client.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    # Local inference (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    local_model: str = "qwen3:1.7b"
    local_timeout_s: float = 120.0
    local_num_ctx: int = 4096
    local_max_input_chars: int = 6000  # routing signal: longer inputs go to the cloud

    # Cloud inference (OpenAI)
    cloud_model: str = "gpt-4o-mini"
    cloud_timeout_s: float = 60.0

    # Request processor limits
    max_prompt_chars: int = 20000
    max_context_messages: int = 10

    # Routing heuristics
    debug_code_lines_threshold: int = 20


def _get(name: str, default, cast):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw.strip())
    except ValueError:
        return default


def load_settings() -> Settings:
    load_dotenv(override=True)
    d = Settings()
    return Settings(
        ollama_base_url=_get("OLLAMA_BASE_URL", d.ollama_base_url, str).rstrip("/"),
        local_model=_get("LOCAL_MODEL", d.local_model, str),
        local_timeout_s=_get("LOCAL_TIMEOUT_SECONDS", d.local_timeout_s, float),
        local_num_ctx=_get("LOCAL_NUM_CTX", d.local_num_ctx, int),
        local_max_input_chars=_get("LOCAL_MAX_INPUT_CHARS", d.local_max_input_chars, int),
        cloud_model=_get("OPENAI_MODEL", d.cloud_model, str),
        cloud_timeout_s=_get("CLOUD_TIMEOUT_SECONDS", d.cloud_timeout_s, float),
        max_prompt_chars=_get("MAX_PROMPT_CHARS", d.max_prompt_chars, int),
        max_context_messages=_get("MAX_CONTEXT_MESSAGES", d.max_context_messages, int),
        debug_code_lines_threshold=_get(
            "DEBUG_CODE_LINES_THRESHOLD", d.debug_code_lines_threshold, int
        ),
    )
