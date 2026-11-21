"""End-to-end pipeline: request processor -> router -> engine -> response handler."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from hybrid_router.config import Settings, load_settings
from hybrid_router.inference.cloud_engine import CloudEngine
from hybrid_router.inference.local_engine import LocalEngine
from hybrid_router.models import (
    Destination,
    ErrorCode,
    ErrorInfo,
    InferenceResult,
    Request,
    RoutingDecision,
    Status,
    failure,
)
from hybrid_router.request_processor import RequestValidationError, new_request_id, process_request
from hybrid_router.response_handler import ResponseHandler
from hybrid_router.router import Router

log = logging.getLogger("hybrid_router")


@dataclass(frozen=True)
class PipelineOutcome:
    request: Optional[Request]  # None if the input was rejected
    decision: Optional[RoutingDecision]  # None if the input was rejected
    result: InferenceResult


class HybridPipeline:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        router: Optional[Router] = None,
        local_engine: Any = None,
        cloud_engine: Any = None,
        handler: Optional[ResponseHandler] = None,
    ):
        self.settings = settings or load_settings()
        self.router = router or Router(self.settings)
        self.local_engine = local_engine or LocalEngine(self.settings)
        self.cloud_engine = cloud_engine or CloudEngine(self.settings)
        self.handler = handler or ResponseHandler(self.cloud_engine)

    def run(
        self, prompt: Any, mode: Any = "auto", context: Optional[Iterable[Any]] = None
    ) -> PipelineOutcome:
        rid = new_request_id()
        try:
            request = process_request(prompt, mode, context, self.settings, request_id=rid)
        except RequestValidationError as exc:
            log.info("[%s] request rejected: %s", rid, exc)
            return PipelineOutcome(
                None, None, failure(rid, None, None, ErrorCode.INVALID_REQUEST, str(exc))
            )

        decision = self.router.route(request)
        engine = self.local_engine if decision.destination is Destination.LOCAL else self.cloud_engine
        try:
            result = engine.generate(request)
        except Exception as exc:  # engines return structured errors; this is a safety net
            log.exception("[%s] %s engine raised", rid, decision.destination.value)
            result = InferenceResult(
                rid, decision.destination, getattr(engine, "model_name", None), "", Status.FAILED,
                error=ErrorInfo(ErrorCode.INTERNAL_ERROR, f"Inference engine error: {type(exc).__name__}"),
            )

        try:
            final = self.handler.handle(request, decision, result)
        except Exception as exc:
            log.exception("[%s] response handler raised", rid)
            final = failure(
                rid, result.model, result.model_name, ErrorCode.INTERNAL_ERROR,
                f"Response handling error: {type(exc).__name__}",
            )
        return PipelineOutcome(request, decision, final)
