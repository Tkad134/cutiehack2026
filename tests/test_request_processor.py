import pytest

from hybrid_router.config import Settings
from hybrid_router.models import Message, Mode
from hybrid_router.request_processor import (
    RequestValidationError,
    detect_output_requirements,
    process_request,
)


def test_normalizes_and_assigns_id():
    r = process_request("  \r\nExplain binary search\r\n  ", "auto")
    assert r.prompt == "Explain binary search"
    assert r.request_id.startswith("req_")
    assert r.mode is Mode.AUTO


def test_unique_ids():
    assert process_request("hi").request_id != process_request("hi").request_id


@pytest.mark.parametrize("bad", ["", "   \n\t ", None, 42])
def test_rejects_empty_or_non_text(bad):
    with pytest.raises(RequestValidationError):
        process_request(bad)


def test_rejects_too_long():
    with pytest.raises(RequestValidationError, match="too long"):
        process_request("x" * 101, settings=Settings(max_prompt_chars=100))


def test_rejects_unknown_mode():
    with pytest.raises(RequestValidationError, match="mode"):
        process_request("hi", mode="turbo")


def test_context_parsed_and_limited():
    ctx = [{"role": "user", "content": f"m{i}"} for i in range(5)]
    r = process_request("hi", context=ctx, settings=Settings(max_context_messages=2))
    assert r.context == (Message("user", "m3"), Message("user", "m4"))


@pytest.mark.parametrize(
    "ctx", [[{"role": "system", "content": "x"}], [{"role": "user", "content": ""}], ["nope"], "str"]
)
def test_rejects_bad_context(ctx):
    with pytest.raises(RequestValidationError):
        process_request("hi", context=ctx)


@pytest.mark.parametrize(
    "prompt,fmt,fields",
    [
        ("Explain what JSON is", None, ()),
        ("List three fruits as JSON", "json", ()),
        ('Return JSON with keys "name" and "age" for Bob', "json", ("name", "age")),
        ("Respond in JSON with fields: title, year and author.", "json", ("title", "year", "author")),
    ],
)
def test_output_requirements(prompt, fmt, fields):
    req = detect_output_requirements(prompt)
    assert req.format == fmt
    assert req.required_fields == fields
