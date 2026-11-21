"""Hybrid LLM router: routes requests between a local Ollama model and the OpenAI API."""

from hybrid_router.pipeline import HybridPipeline, PipelineOutcome

__all__ = ["HybridPipeline", "PipelineOutcome"]
