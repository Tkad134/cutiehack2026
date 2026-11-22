from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from hybrid_router.config import Settings
from hybrid_router.models import Destination, ErrorCode, failure, success


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr("hybrid_router.config.load_settings", lambda: Settings())
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()


def test_configured_models_reach_fallback(app, monkeypatch):
    calls = []

    def local(engine, request):
        calls.append(engine.model_name)
        return failure(request.request_id, Destination.LOCAL, engine.model_name, ErrorCode.TIMEOUT, "timeout")

    def cloud(engine, request):
        calls.append(engine.model_name)
        return success(request.request_id, Destination.CLOUD, engine.model_name, "Fallback worked.")

    monkeypatch.setattr("hybrid_router.inference.local_engine.LocalEngine.generate", local)
    monkeypatch.setattr("hybrid_router.inference.cloud_engine.CloudEngine.generate", cloud)
    app.chat_input[0].set_value("Explain cats").run()
    assert not app.exception
    assert calls == [Settings().local_model, Settings().cloud_model]
    assert app.session_state["total_fallback"] == 1


def test_mode_selection_survives_rerun(app):
    app.select_slider[0].set_value(0).run()
    app.run()
    assert not app.exception
    assert app.session_state["mode_index"] == 0


def test_conversation_is_sent_as_context(app, monkeypatch):
    seen = []

    def local(engine, request):
        seen.append(len(request.context))
        return success(request.request_id, Destination.LOCAL, engine.model_name, "Hi there.")

    monkeypatch.setattr("hybrid_router.inference.local_engine.LocalEngine.generate", local)
    app.chat_input[0].set_value("Explain cats").run()
    app.session_state["typing"] = None
    app.chat_input[0].set_value("Explain dogs").run()
    assert not app.exception
    assert seen == [0, 2]


def test_failed_turns_are_not_sent_as_context(app, monkeypatch):
    seen = []

    def local(engine, request):
        seen.append([(m.role, m.content) for m in request.context])
        if request.prompt == "first":
            return failure(request.request_id, Destination.LOCAL, engine.model_name, ErrorCode.TIMEOUT, "timed out")
        return success(request.request_id, Destination.LOCAL, engine.model_name, "ok")

    monkeypatch.setattr("hybrid_router.inference.local_engine.LocalEngine.generate", local)
    app.select_slider[0].set_value(0).run()  # local only, so the failure is not retried in the cloud
    app.chat_input[0].set_value("first").run()
    app.chat_input[0].set_value("second").run()
    assert not app.exception
    assert seen[1] == []  # the failed exchange is not context
    assert [h["content"] for h in app.session_state["history"]][:2] == ["first", "timed out"]  # still on screen


def test_long_conversation_stays_on_screen(app, monkeypatch):
    seen = []

    def local(engine, request):
        seen.append(len(request.context))
        return success(request.request_id, Destination.LOCAL, engine.model_name, "ok")

    monkeypatch.setattr("hybrid_router.inference.local_engine.LocalEngine.generate", local)
    for i in range(7):
        app.session_state["typing"] = None
        app.chat_input[0].set_value(f"Explain topic {i}").run()
    assert not app.exception
    users = [h["content"] for h in app.session_state["history"] if h["role"] == "user"]
    assert users == [f"Explain topic {i}" for i in range(7)]
    assert max(seen) == Settings().max_context_messages  # context is still capped per request


def test_workspace_section_removed(app):
    assert not app.selectbox and not app.text_input and not app.toggle
    assert not [e for e in app.sidebar.expander if e.label == "Rabbits"]
