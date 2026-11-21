"""Engine tests with the HTTP layer / OpenAI client mocked out."""

from types import SimpleNamespace

import httpx
import openai
import pytest
import requests

from hybrid_router.config import Settings
from hybrid_router.inference.cloud_engine import CloudEngine
from hybrid_router.inference.local_engine import LocalEngine
from hybrid_router.inference.prompting import SYSTEM_PROMPT, build_messages
from hybrid_router.models import Destination, ErrorCode, Status
from hybrid_router.request_processor import process_request

REQ = process_request(
    "Explain cats", context=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey"}]
)


# --- shared prompt format ------------------------------------------------------

def test_build_messages_order():
    msgs = build_messages(REQ)
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"]
    assert msgs[-1]["content"] == "Explain cats"


# --- local engine --------------------------------------------------------------

class FakeResponse:
    def __init__(self, status=200, data=None):
        self.status_code, self._data = status, data

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.payloads = response, exc, []

    def post(self, url, json=None, timeout=None):
        self.payloads.append((url, json, timeout))
        if self.exc:
            raise self.exc
        return self.response


def local(response=None, exc=None):
    session = FakeSession(response, exc)
    return LocalEngine(Settings(), session=session), session


def test_local_success_strips_think():
    eng, sess = local(FakeResponse(200, {"message": {"content": "<think>hmm</think>\nCats are mammals."}}))
    r = eng.generate(REQ)
    assert r.ok and r.response == "Cats are mammals." and r.model is Destination.LOCAL
    url, payload, timeout = sess.payloads[0]
    assert url.endswith("/api/chat") and payload["model"] == "qwen3:1.7b"
    assert payload["messages"] == build_messages(REQ)
    assert timeout[1] == Settings().local_timeout_s


@pytest.mark.parametrize(
    "resp,exc,code",
    [
        (None, requests.exceptions.ConnectionError(), ErrorCode.SERVICE_UNAVAILABLE),
        (None, requests.exceptions.ConnectTimeout(), ErrorCode.SERVICE_UNAVAILABLE),
        (None, requests.exceptions.ReadTimeout(), ErrorCode.TIMEOUT),
        (FakeResponse(404, {"error": "model 'qwen3:1.7b' not found"}), None, ErrorCode.MODEL_NOT_FOUND),
        (FakeResponse(500, {"error": "out of memory"}), None, ErrorCode.GENERATION_FAILED),
        (FakeResponse(200, None), None, ErrorCode.GENERATION_FAILED),
        (FakeResponse(200, {"message": {"content": "  "}}), None, ErrorCode.EMPTY_RESPONSE),
        (FakeResponse(200, {"message": {"content": "<think>only thinking</think>"}}), None, ErrorCode.EMPTY_RESPONSE),
    ],
)
def test_local_errors(resp, exc, code):
    eng, _ = local(resp, exc)
    r = eng.generate(REQ)
    assert r.status is Status.FAILED and r.error.code is code and r.model is Destination.LOCAL


# --- cloud engine --------------------------------------------------------------

_HTTP_REQ = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_error(cls, status, body=None):
    return cls("err sk-secret123", response=httpx.Response(status, request=_HTTP_REQ), body=body)


class FakeClient:
    def __init__(self, content="Cats purr.", exc=None):
        self.calls = []
        self.exc, self.content = exc, content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def test_cloud_success_uses_same_messages():
    client = FakeClient()
    r = CloudEngine(Settings(), client=client).generate(REQ)
    assert r.ok and r.response == "Cats purr." and r.model is Destination.CLOUD
    assert client.calls[0]["messages"] == build_messages(REQ)
    assert client.calls[0]["model"] == Settings().cloud_model


def test_cloud_not_configured(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    eng = CloudEngine(Settings())
    assert not eng.configured
    r = eng.generate(REQ)
    assert r.error.code is ErrorCode.NOT_CONFIGURED


@pytest.mark.parametrize(
    "exc,code",
    [
        (_status_error(openai.AuthenticationError, 401), ErrorCode.AUTH_FAILED),
        (_status_error(openai.PermissionDeniedError, 403), ErrorCode.AUTH_FAILED),
        (_status_error(openai.RateLimitError, 429), ErrorCode.RATE_LIMITED),
        (_status_error(openai.NotFoundError, 404), ErrorCode.MODEL_NOT_FOUND),
        (_status_error(openai.InternalServerError, 503), ErrorCode.SERVICE_UNAVAILABLE),
        (_status_error(openai.BadRequestError, 400), ErrorCode.GENERATION_FAILED),
        (openai.APITimeoutError(request=_HTTP_REQ), ErrorCode.TIMEOUT),
        (openai.APIConnectionError(request=_HTTP_REQ), ErrorCode.NETWORK_ERROR),
    ],
)
def test_cloud_errors_never_leak_key(exc, code):
    r = CloudEngine(Settings(), client=FakeClient(exc=exc)).generate(REQ)
    assert r.status is Status.FAILED and r.error.code is code
    assert "sk-" not in r.error.message


def test_cloud_empty_response():
    r = CloudEngine(Settings(), client=FakeClient(content=None)).generate(REQ)
    assert r.error.code is ErrorCode.EMPTY_RESPONSE
