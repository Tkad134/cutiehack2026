from dataclasses import replace
from unittest.mock import Mock

import pytest
import requests

from hybrid_router.config import Settings
from hybrid_router.model_selection import discover_local_models, select_models


def test_discovery_sorts_and_deduplicates(monkeypatch):
    get = Mock(return_value=Mock(json=lambda: {"models": [
        {"name": "z-model:latest"}, {"name": "a-model:7b"}, {"name": "z-model:latest"},
    ]}))
    monkeypatch.setattr(requests, "get", get)
    result = discover_local_models("http://localhost:11434/")
    assert result.names == ("a-model:7b", "z-model:latest") and result.error is None
    assert get.call_args.args == ("http://localhost:11434/api/tags",)


@pytest.mark.parametrize("payload", [{}, [], {"models": None}, {"models": [None]}])
def test_discovery_handles_malformed_response(monkeypatch, payload):
    monkeypatch.setattr(requests, "get", lambda *a, **kw: Mock(json=lambda: payload))
    result = discover_local_models("http://localhost:11434")
    assert result.error and not result.names


def test_discovery_handles_offline_ollama(monkeypatch):
    monkeypatch.setattr(requests, "get", Mock(side_effect=requests.ConnectionError("offline")))
    assert discover_local_models("http://localhost:11434").error


def test_overrides_preserve_defaults_and_limits():
    defaults = Settings(local_timeout_s=42, max_context_messages=6)
    selected = select_models(defaults, " other-local:7b ", " custom-cloud ")
    assert selected == replace(defaults, local_model="other-local:7b", cloud_model="custom-cloud")
    assert defaults.local_model == "qwen3:1.7b"


@pytest.mark.parametrize("local,cloud", [("", "cloud"), ("local", "  "), ("two words", "cloud")])
def test_bad_model_id_is_rejected(local, cloud):
    with pytest.raises(ValueError):
        select_models(Settings(), local, cloud)
