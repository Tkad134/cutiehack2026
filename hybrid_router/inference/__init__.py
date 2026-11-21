"""Inference engines. Both expose `generate(request) -> InferenceResult`."""

from typing import Protocol

from hybrid_router.inference.cloud_engine import CloudEngine
from hybrid_router.inference.local_engine import LocalEngine
from hybrid_router.models import Destination, InferenceResult, Request


class InferenceEngine(Protocol):
    destination: Destination

    @property
    def model_name(self) -> str: ...

    def generate(self, request: Request) -> InferenceResult: ...


__all__ = ["InferenceEngine", "LocalEngine", "CloudEngine"]
