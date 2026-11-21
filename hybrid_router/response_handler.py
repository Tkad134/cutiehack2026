"""Response handler: empty-output detection, basic format validation, bounded fallback."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from typing import Any, Optional

from hybrid_router.models import (
    Destination,
    ErrorCode,
    ErrorInfo,
    InferenceResult,
    Mode,
    Request,
    RoutingDecision,
    Status,
)

log = logging.getLogger("hybrid_router")

_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```$", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse JSON from a response, allowing a single surrounding ``` code fence."""
    stripped = text.strip()
    m = _FENCE.match(stripped)
    if m:
        stripped = m.group(1).strip()
    return json.loads(stripped)


def check_format(text: str, request: Request) -> Optional[str]:
    """Return an error message if the explicitly requested format is not met, else None."""
    if request.output.format != "json":
        return None
    try:
        data = extract_json(text)
    except json.JSONDecodeError as exc:
        return f"JSON was requested but the response is not valid JSON ({exc.msg} at line {exc.lineno})."
    required = request.output.required_fields
    if not required:
        return None
    objects = data if isinstance(data, list) else [data]
    if not objects or not all(isinstance(o, dict) for o in objects):
        return "JSON with fields was requested but the response is not a JSON object."
    missing = sorted({f for o in objects for f in required if f not in o})
    if missing:
        return f"JSON response is missing required field(s): {', '.join(missing)}."
    return None


def validate_result(result: InferenceResult, request: Request) -> InferenceResult:
    """Mark a result failed if it has no usable text or violates a requested format."""
    if result.status is Status.FAILED:
        return result
    if not result.response or not result.response.strip():
        return replace(
            result, status=Status.FAILED,
            error=ErrorInfo(ErrorCode.EMPTY_RESPONSE, "The model returned no usable text."),
        )
    problem = check_format(result.response, request)
    if problem:
        return replace(result, status=Status.FAILED, error=ErrorInfo(ErrorCode.INVALID_FORMAT, problem))
    return result


class ResponseHandler:
    """Validates results and performs at most one local -> cloud fallback in Automatic Mode."""

    def __init__(self, cloud_engine: Any):
        self.cloud_engine = cloud_engine

    def handle(
        self, request: Request, decision: RoutingDecision, result: InferenceResult
    ) -> InferenceResult:
        checked = validate_result(result, request)
        log.info(
            "[%s] %s result: %s%s", request.request_id, decision.destination.value,
            checked.status.value, f" ({checked.error.code.value})" if checked.error else "",
        )
        if checked.ok or not self._fallback_allowed(request, decision, checked):
            return checked

        assert checked.error is not None
        fallback_reason = (
            f"Local model failed ({checked.error.code.value}): {checked.error.message} "
            "Escalated once to the cloud model."
        )
        log.info("[%s] falling back to cloud after local %s", request.request_id, checked.error.code.value)
        try:
            cloud_result = self.cloud_engine.generate(request)
        except Exception as exc:  # never let an engine bug crash the app
            log.exception("[%s] cloud engine raised during fallback", request.request_id)
            cloud_result = InferenceResult(
                request.request_id, Destination.CLOUD, getattr(self.cloud_engine, "model_name", None),
                "", Status.FAILED,
                error=ErrorInfo(ErrorCode.INTERNAL_ERROR, f"Cloud engine error: {type(exc).__name__}"),
            )
        final = validate_result(cloud_result, request)
        log.info(
            "[%s] fallback result: %s%s", request.request_id, final.status.value,
            f" ({final.error.code.value})" if final.error else "",
        )
        # Fallback is terminal: no further retries regardless of the outcome.
        return replace(final, fallback_used=True, fallback_reason=fallback_reason)

    @staticmethod
    def _fallback_allowed(request: Request, decision: RoutingDecision, result: InferenceResult) -> bool:
        return (
            request.mode is Mode.AUTO
            and decision.destination is Destination.LOCAL
            and result.model is Destination.LOCAL
            and not result.fallback_used
        )
