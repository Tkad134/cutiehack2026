from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from hybrid_router.config import Settings
from hybrid_router.model_selection import LocalModels
from hybrid_router.models import Destination, ErrorCode, failure, success


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr("hybrid_router.config.load_settings", lambda: Settings())
    monkeypatch.setattr(
        "hybrid_router.model_selection.discover_local_models",
        lambda url: LocalModels(("qwen3.5:35b", "test-local:7b")),
    )
    st.cache_data.clear()
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_selected_models_reach_fallback(app, monkeypatch):
    calls = []

    def local(engine, request):
        calls.append(engine.model_name)
        return failure(request.request_id, Destination.LOCAL, engine.model_name, ErrorCode.TIMEOUT, "timeout")

    def cloud(engine, request):
        calls.append(engine.model_name)
        return success(request.request_id, Destination.CLOUD, engine.model_name, "Fallback worked.")

    monkeypatch.setattr("hybrid_router.inference.local_engine.LocalEngine.generate", local)
    monkeypatch.setattr("hybrid_router.inference.cloud_engine.CloudEngine.generate", cloud)
    app.selectbox[0].select("test-local:7b")
    app.text_input[0].input("custom-cloud")
    button(app, "Apply models").click().run()
    assert not app.exception
    app.chat_input[0].set_value("Explain cats").run()
    assert not app.exception
    assert calls == ["test-local:7b", "custom-cloud"]
    assert app.session_state["total_fallback"] == 1


def test_native_selection_survives_rerun(app):
    app.select_slider[0].set_value(0).run()
    button(app, "Refresh local models").click().run()
    assert not app.exception
    assert app.session_state["mode_index"] == 0


def test_invalid_cloud_id_keeps_active_models(app):
    app.text_input[0].input(" ")
    button(app, "Apply models").click().run()
    assert not app.exception and len(app.error) == 1
    assert app.session_state["active_cloud_model"] == "gpt-4o-mini"
