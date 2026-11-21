import pytest

from hybrid_router.config import Settings
from hybrid_router.models import Destination, ErrorCode, Status
from hybrid_router.pipeline import HybridPipeline


def make_pipeline(local, cloud):
    return HybridPipeline(settings=Settings(), local_engine=local, cloud_engine=cloud)


def test_local_success_no_fallback(make_local, make_cloud):
    local, cloud = make_local(text="Binary search halves..."), make_cloud()
    out = make_pipeline(local, cloud).run("Explain binary search")
    r = out.result
    assert r.ok and r.model is Destination.LOCAL and not r.fallback_used and r.error is None
    assert r.request_id == out.request.request_id == out.decision.request_id
    assert len(cloud.calls) == 0


def test_cloud_route(make_local, make_cloud):
    local, cloud = make_local(), make_cloud(text="Proof: ...")
    r = make_pipeline(local, cloud).run("Prove that sqrt(2) is irrational").result
    assert r.ok and r.model is Destination.CLOUD and not r.fallback_used
    assert len(local.calls) == 0


@pytest.mark.parametrize(
    "code", [ErrorCode.SERVICE_UNAVAILABLE, ErrorCode.TIMEOUT, ErrorCode.MODEL_NOT_FOUND, ErrorCode.EMPTY_RESPONSE]
)
def test_auto_mode_falls_back_once(make_local, make_cloud, code):
    local, cloud = make_local(error=code), make_cloud(text="cloud answer")
    r = make_pipeline(local, cloud).run("Explain binary search").result
    assert r.ok and r.model is Destination.CLOUD and r.fallback_used
    assert code.value in r.fallback_reason
    assert len(local.calls) == 1 and len(cloud.calls) == 1


def test_fallback_that_also_fails_stops(make_local, make_cloud):
    local, cloud = make_local(error=ErrorCode.TIMEOUT), make_cloud(error=ErrorCode.RATE_LIMITED)
    r = make_pipeline(local, cloud).run("Explain binary search").result
    assert r.status is Status.FAILED and r.fallback_used
    assert r.error.code is ErrorCode.RATE_LIMITED
    assert len(cloud.calls) == 1


def test_local_only_never_falls_back(make_local, make_cloud):
    local, cloud = make_local(error=ErrorCode.SERVICE_UNAVAILABLE), make_cloud()
    r = make_pipeline(local, cloud).run("Explain binary search", mode="local").result
    assert r.status is Status.FAILED and r.model is Destination.LOCAL and not r.fallback_used
    assert len(cloud.calls) == 0


def test_cloud_only_failure_no_retry(make_local, make_cloud):
    local, cloud = make_local(), make_cloud(error=ErrorCode.AUTH_FAILED)
    r = make_pipeline(local, cloud).run("Explain binary search", mode="cloud").result
    assert r.status is Status.FAILED and r.error.code is ErrorCode.AUTH_FAILED
    assert len(local.calls) == 0 and len(cloud.calls) == 1


def test_cloud_routed_failure_no_fallback(make_local, make_cloud):
    local, cloud = make_local(), make_cloud(error=ErrorCode.NETWORK_ERROR)
    r = make_pipeline(local, cloud).run("Prove the Pythagorean theorem").result
    assert r.status is Status.FAILED and not r.fallback_used and len(cloud.calls) == 1


def test_empty_text_detected_by_handler(make_local, make_cloud):
    local, cloud = make_local(text="   "), make_cloud(text="real")
    r = make_pipeline(local, cloud).run("Explain cats", mode="local").result
    assert r.status is Status.FAILED and r.error.code is ErrorCode.EMPTY_RESPONSE


def test_invalid_json_triggers_fallback(make_local, make_cloud):
    local = make_local(text="Sure! Here you go: {name: Bob}")
    cloud = make_cloud(text='```json\n{"name": "Bob", "age": 30}\n```')
    r = make_pipeline(local, cloud).run('Return JSON with keys "name" and "age" for Bob, 30').result
    assert r.ok and r.fallback_used and "invalid_format" in r.fallback_reason


def test_missing_json_fields_fail(make_local, make_cloud):
    local, cloud = make_local(text='{"name": "Bob"}'), make_cloud()
    r = make_pipeline(local, cloud).run('Return JSON with keys "name" and "age"', mode="local").result
    assert r.status is Status.FAILED and r.error.code is ErrorCode.INVALID_FORMAT
    assert "age" in r.error.message
    assert r.response == '{"name": "Bob"}'  # raw output preserved


def test_invalid_input_does_not_crash(make_local, make_cloud):
    local, cloud = make_local(), make_cloud()
    out = make_pipeline(local, cloud).run("   ")
    assert out.request is None and out.decision is None
    assert out.result.error.code is ErrorCode.INVALID_REQUEST
    assert len(local.calls) == len(cloud.calls) == 0


def test_engine_exception_is_contained(make_local, make_cloud):
    local, cloud = make_local(raises=RuntimeError("boom")), make_cloud(raises=RuntimeError("boom"))
    r = make_pipeline(local, cloud).run("Explain cats").result
    assert r.status is Status.FAILED and r.fallback_used
    assert r.error.code is ErrorCode.INTERNAL_ERROR


def test_context_passed_identically(make_local, make_cloud):
    local, cloud = make_local(error=ErrorCode.TIMEOUT), make_cloud()
    ctx = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    make_pipeline(local, cloud).run("Explain cats", context=ctx)
    assert local.calls[0] == cloud.calls[0]
    assert len(local.calls[0].context) == 2


def test_result_dict_shape(make_local, make_cloud):
    r = make_pipeline(make_local(text="hi"), make_cloud()).run("Explain cats").result
    d = r.to_dict()
    for key in ("request_id", "model", "response", "status", "fallback_used", "error"):
        assert key in d
    assert d["model"] == "local" and d["status"] == "success"
