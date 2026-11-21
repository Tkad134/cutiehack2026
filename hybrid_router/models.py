"""Shared data structures passed between components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class Mode(str, Enum):
    AUTO = "auto"
    LOCAL = "local"
    CLOUD = "cloud"


class Destination(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class Status(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class ErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    EMPTY_RESPONSE = "empty_response"
    INVALID_FORMAT = "invalid_format"
    TIMEOUT = "timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"
    MODEL_NOT_FOUND = "model_not_found"
    GENERATION_FAILED = "generation_failed"
    NOT_CONFIGURED = "not_configured"
    AUTH_FAILED = "auth_failed"
    RATE_LIMITED = "rate_limited"
    NETWORK_ERROR = "network_error"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class Message:
    role: str  # "user" or "assistant"
    content: str


@dataclass(frozen=True)
class OutputRequirements:
    """Explicit output-format requirements detected in the prompt."""

    format: Optional[str] = None  # currently only "json" is supported
    required_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt: str
    mode: Mode = Mode.AUTO
    context: tuple[Message, ...] = ()
    output: OutputRequirements = field(default_factory=OutputRequirements)


@dataclass(frozen=True)
class RoutingDecision:
    request_id: str
    destination: Destination
    reason: str
    strategy: str  # "rule_based" or "manual_override"
    category: str = "general"


@dataclass(frozen=True)
class ErrorInfo:
    code: ErrorCode
    message: str


@dataclass(frozen=True)
class InferenceResult:
    request_id: str
    model: Optional[Destination]  # None when no inference ran (e.g. invalid request)
    model_name: Optional[str]
    response: str
    status: Status
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    error: Optional[ErrorInfo] = None

    @property
    def ok(self) -> bool:
        return self.status is Status.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model"] = self.model.value if self.model else None
        d["status"] = self.status.value
        if self.error:
            d["error"] = {"code": self.error.code.value, "message": self.error.message}
        return d


def success(request_id: str, model: Destination, model_name: str, text: str) -> InferenceResult:
    return InferenceResult(request_id, model, model_name, text, Status.SUCCESS)


def failure(
    request_id: str,
    model: Optional[Destination],
    model_name: Optional[str],
    code: ErrorCode,
    message: str,
    response: str = "",
) -> InferenceResult:
    return InferenceResult(
        request_id, model, model_name, response, Status.FAILED, error=ErrorInfo(code, message)
    )
