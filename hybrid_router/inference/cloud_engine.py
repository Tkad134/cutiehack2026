"""Cloud inference engine: one configurable OpenAI chat model.

The API key is read from the OPENAI_API_KEY environment variable (a .env file
is loaded by config.load_settings). Error messages are written by this module
rather than echoed from the SDK so that no credential fragment is ever shown
or logged.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import openai

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


class CloudEngine:
    destination = Destination.CLOUD

    def __init__(self, settings: Optional[Settings] = None, client: Any = None):
        self.settings = settings or Settings()
        self._client = client

    @property
    def model_name(self) -> str:
        return self.settings.cloud_model

    @property
    def configured(self) -> bool:
        return self._client is not None or bool(os.getenv("OPENAI_API_KEY", "").strip())

    def _get_client(self) -> Any:
        if self._client is None:
            key = os.getenv("OPENAI_API_KEY", "").strip()
            if not key:
                return None
            # max_retries=0 keeps each cloud attempt bounded to a single call.
            self._client = openai.OpenAI(
                api_key=key, timeout=self.settings.cloud_timeout_s, max_retries=0
            )
        return self._client

    def _fail(self, request: Request, code: ErrorCode, message: str) -> InferenceResult:
        log.warning("[%s] cloud inference failed: %s", request.request_id, code.value)
        return failure(request.request_id, self.destination, self.model_name, code, message)

    def generate(self, request: Request) -> InferenceResult:
        client = self._get_client()
        if client is None:
            return self._fail(
                request, ErrorCode.NOT_CONFIGURED,
                "Cloud model is not configured. Set OPENAI_API_KEY in your environment or .env file.",
            )
        log.info("[%s] calling cloud model %s", request.request_id, self.model_name)
        try:
            resp = client.chat.completions.create(
                model=self.model_name, messages=build_messages(request)
            )
        except openai.AuthenticationError:
            return self._fail(request, ErrorCode.AUTH_FAILED, "OpenAI rejected the API key (invalid credentials).")
        except openai.PermissionDeniedError:
            return self._fail(request, ErrorCode.AUTH_FAILED, "OpenAI denied access for this API key or model.")
        except openai.RateLimitError as exc:
            quota = "insufficient_quota" in str(getattr(exc, "code", "") or "")
            msg = (
                "OpenAI quota exhausted for this account. Check your billing settings."
                if quota else "OpenAI rate limit reached. Wait a moment and try again."
            )
            return self._fail(request, ErrorCode.RATE_LIMITED, msg)
        except openai.APITimeoutError:
            return self._fail(
                request, ErrorCode.TIMEOUT,
                f"Cloud model did not respond within {self.settings.cloud_timeout_s:g} seconds.",
            )
        except openai.APIConnectionError:
            return self._fail(request, ErrorCode.NETWORK_ERROR, "Could not reach the OpenAI API (network error).")
        except openai.NotFoundError:
            return self._fail(
                request, ErrorCode.MODEL_NOT_FOUND,
                f"Cloud model '{self.model_name}' was not found. Check OPENAI_MODEL.",
            )
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                return self._fail(
                    request, ErrorCode.SERVICE_UNAVAILABLE,
                    f"OpenAI service is unavailable (HTTP {exc.status_code}). Try again later.",
                )
            return self._fail(
                request, ErrorCode.GENERATION_FAILED, f"OpenAI request failed (HTTP {exc.status_code})."
            )
        except openai.OpenAIError as exc:
            return self._fail(request, ErrorCode.GENERATION_FAILED, f"OpenAI request failed ({type(exc).__name__}).")

        try:
            text = clean_output(resp.choices[0].message.content)
        except (AttributeError, IndexError, TypeError):
            text = ""
        if not text:
            return self._fail(request, ErrorCode.EMPTY_RESPONSE, "Cloud model returned no usable text.")
        log.info("[%s] cloud inference succeeded", request.request_id)
        return success(request.request_id, self.destination, self.model_name, text)
