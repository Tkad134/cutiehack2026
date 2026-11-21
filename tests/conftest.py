import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_router.config import Settings  # noqa: E402
from hybrid_router.models import Destination, ErrorCode, failure, success  # noqa: E402


class FakeEngine:
    """Engine stub that returns a scripted result and records calls."""

    def __init__(self, destination, text="ok", error=None, raises=None, model_name=None):
        self.destination = destination
        self.model_name = model_name or f"fake-{destination.value}"
        self.text, self.error, self.raises = text, error, raises
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        if self.raises:
            raise self.raises
        if self.error:
            return failure(request.request_id, self.destination, self.model_name, self.error, "scripted failure")
        return success(request.request_id, self.destination, self.model_name, self.text)


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def make_local():
    return lambda **kw: FakeEngine(Destination.LOCAL, **kw)


@pytest.fixture
def make_cloud():
    return lambda **kw: FakeEngine(Destination.CLOUD, **kw)


__all__ = ["FakeEngine", "ErrorCode"]
