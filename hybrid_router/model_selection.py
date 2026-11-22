"""Session model overrides and read-only discovery of installed Ollama models."""

from __future__ import annotations

from dataclasses import dataclass, replace

import requests

from hybrid_router.config import Settings


@dataclass(frozen=True)
class LocalModels:
    names: tuple[str, ...] = ()
    error: str | None = None


def discover_local_models(base_url: str) -> LocalModels:
    """List installed models without pulling or running any of them."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=(2.0, 3.0))
        response.raise_for_status()
        data = response.json()
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            raise ValueError("Invalid models list")
        names = set()
        for model in models:
            name = model.get("name") if isinstance(model, dict) else None
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Invalid model name")
            names.add(name.strip())
        return LocalModels(tuple(sorted(names)))
    except requests.RequestException:
        return LocalModels(error="Could not list Ollama models. Check that Ollama is running, then refresh.")
    except ValueError:
        return LocalModels(error="Ollama returned an unreadable model list. Try refreshing.")


def select_models(settings: Settings, local_model: str, cloud_model: str) -> Settings:
    """Return a new settings object, preserving all non-model settings."""
    names = []
    for label, value in (("Local", local_model), ("Cloud", cloud_model)):
        name = value.strip()
        if not name or len(name) > 200 or any(char.isspace() for char in name):
            raise ValueError(f"{label} model must be a model ID without spaces (up to 200 characters).")
        names.append(name)
    return replace(settings, local_model=names[0], cloud_model=names[1])
