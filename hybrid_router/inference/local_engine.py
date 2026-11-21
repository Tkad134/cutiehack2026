"""Local inference engine: Qwen3 via the Ollama HTTP API."""

from __future__ import annotations

import logging
from typing import Optional

import requests

from hybrid_router.config import Settings
from hybrid_router.inference.prompting import build_messages, clean_output
from hybrid_router.models import (
    Destination,
    ErrorCode,
    InferenceResult,
    Request,
    failure,
    success,
)

log = logging.getLogger("hybrid_router")

CONNECT_TIMEOUT_S = 5.0


class LocalEngine:
    destination = Destination.LOCAL

    def __init__(self, settings: Optional[Settings] = None, session: Optional[requests.Session] = None):
        self.settings = settings or Settings()
        self.session = session or requests.Session()

    @property
    def model_name(self) -> str:
        return self.settings.local_model

    def _fail(self, request: Request, code: ErrorCode, message: str) -> InferenceResult:
        log.warning("[%s] local inference failed: %s", request.request_id, code.value)
        return failure(request.request_id, self.destination, self.model_name, code, message)

    def generate(self, request: Request) -> InferenceResult:
        s = self.settings
        url = f"{s.ollama_base_url}/api/chat"
        payload = {
            "model": s.local_model,
            "messages": build_messages(request),
            "stream": False,
            "think": False,  # Qwen3: skip the reasoning trace for faster answers
            "options": {"num_ctx": s.local_num_ctx},
        }
        log.info("[%s] calling local model %s", request.request_id, s.local_model)
        try:
            resp = self.session.post(url, json=payload, timeout=(CONNECT_TIMEOUT_S, s.local_timeout_s))
        except requests.exceptions.ConnectTimeout:
            return self._fail(
                request, ErrorCode.SERVICE_UNAVAILABLE,
                f"Could not connect to Ollama at {s.ollama_base_url}. Is `ollama serve` running?",
            )
        except requests.exceptions.Timeout:
            return self._fail(
                request, ErrorCode.TIMEOUT,
                f"Local model did not respond within {s.local_timeout_s:g} seconds.",
            )
        except requests.exceptions.ConnectionError:
            return self._fail(
                request, ErrorCode.SERVICE_UNAVAILABLE,
                f"Ollama is not reachable at {s.ollama_base_url}. Start it with `ollama serve`.",
            )
        except requests.exceptions.RequestException as exc:
            return self._fail(request, ErrorCode.GENERATION_FAILED, f"Local request failed: {exc}")

        try:
            data = resp.json()
        except ValueError:
            data = None
        server_error = str(data.get("error", "")) if isinstance(data, dict) else ""

        if resp.status_code == 404 or "not found" in server_error.lower():
            return self._fail(
                request, ErrorCode.MODEL_NOT_FOUND,
                f"Local model '{s.local_model}' is not installed. Run `ollama pull {s.local_model}`.",
            )
        if not resp.ok or server_error:
            detail = server_error or f"HTTP {resp.status_code}"
            return self._fail(request, ErrorCode.GENERATION_FAILED, f"Local generation failed: {detail}")
        if not isinstance(data, dict):
            return self._fail(request, ErrorCode.GENERATION_FAILED, "Ollama returned an unreadable response.")

        message = data.get("message") or {}
        text = clean_output(message.get("content") if isinstance(message, dict) else None)
        if not text:
            return self._fail(request, ErrorCode.EMPTY_RESPONSE, "Local model returned no usable text.")
        log.info("[%s] local inference succeeded", request.request_id)
        return success(request.request_id, self.destination, self.model_name, text)
