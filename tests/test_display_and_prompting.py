import pytest

from hybrid_router.inference.prompting import JSON_INSTRUCTION, SYSTEM_PROMPT, build_messages
from hybrid_router.request_processor import process_request
from hybrid_router.ui import markdown_math


# --- System prompt only mentions JSON when JSON was requested -----------------

@pytest.mark.parametrize(
    "prompt", ["Explain binary search", "Write a haiku about rain", "Summarize this paragraph"]
)
def test_plain_prompts_do_not_mention_json(prompt):
    system = build_messages(process_request(prompt))[0]["content"]
    assert system == SYSTEM_PROMPT
    assert "json" not in system.lower()


def test_json_prompts_get_json_instruction():
    system = build_messages(process_request('Return JSON with keys "name" and "age"'))[0]["content"]
    assert system == SYSTEM_PROMPT + JSON_INSTRUCTION


# --- Math delimiters are converted for Streamlit, code is untouched -----------

@pytest.mark.parametrize(
    "text,expected",
    [
        (r"Let \( p_1, p_2 \) be primes.", "Let $p_1, p_2$ be primes."),
        (r"So \[ P = p_1 p_2 + 1 \] holds.", "So \n$$\nP = p_1 p_2 + 1\n$$\n holds."),
        ("No math here, it costs $5.", "No math here, it costs $5."),
        (r"Inline `\(x\)` stays as code.", r"Inline `\(x\)` stays as code."),
        ("```latex\n\\( x \\)\n```", "```latex\n\\( x \\)\n```"),
        (r"Unclosed \( stays as is", r"Unclosed \( stays as is"),
    ],
)
def test_markdown_math(text, expected):
    assert markdown_math(text) == expected
