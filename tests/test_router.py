import pytest

from hybrid_router.config import Settings
from hybrid_router.models import Destination, Message, Mode
from hybrid_router.request_processor import process_request
from hybrid_router.router import Router, classify


def route(prompt, mode="auto", context=None, settings=None):
    settings = settings or Settings()
    return Router(settings).route(process_request(prompt, mode, context, settings))


@pytest.mark.parametrize(
    "prompt,category",
    [
        ("Summarize this article about climate policy in three bullet points", "summarization"),
        ("Explain how photosynthesis works", "explanation"),
        ("Write a haiku about autumn rain", "creative_writing"),
        ("Write a Python function that reverses a string", "coding"),
        ("Solve the equation 2x + 3 = 11", "math"),
        ("Give an in-depth analysis of the trade-offs in this system design", "advanced_technical_analysis"),
        ("Rephrase this sentence to sound more formal: hey what's up", "rewriting"),
        ("Who won the 2018 World Cup?", "general"),
    ],
)
def test_classification(prompt, category):
    assert classify(prompt) == category


@pytest.mark.parametrize(
    "prompt",
    [
        "Explain binary search",
        "Summarize the plot of Romeo and Juliet in two sentences",
        "Write a short poem about the ocean",
        "Rewrite this to be more concise: We are going to be having a meeting tomorrow.",
        "Write a Python function that checks whether a number is even",
        "What is 12 * 7?",
        "How can I improve my essay's introduction?",  # 'improve' must not match 'prove'
        "Please proofread this paragraph",  # 'proofread' must not match 'proof'
    ],
)
def test_simple_tasks_go_local(prompt):
    d = route(prompt)
    assert d.destination is Destination.LOCAL, d.reason
    assert d.strategy == "rule_based"


@pytest.mark.parametrize(
    "prompt,signal",
    [
        ("Prove that there are infinitely many primes", "advanced proof"),
        ("Show by induction that 1+2+...+n = n(n+1)/2", "advanced proof"),
        ("Why does my program segfault when freeing this pointer?", "complex debugging"),
        ("Here is my traceback, what's wrong?", "complex debugging"),
        ("Optimize this SQL query for performance on a 50M row table", "advanced optimization"),
        ("What is the time complexity of this algorithm?", "advanced optimization"),
        ("Design a scalable distributed system for chat", "advanced technical analysis"),
        ("Find the eigenvalues of this matrix and derive the general solution", "advanced mathematics"),
    ],
)
def test_complex_tasks_go_cloud(prompt, signal):
    d = route(prompt)
    assert d.destination is Destination.CLOUD
    assert signal in d.reason


def test_long_debug_request_goes_cloud():
    code = "\n".join(f"x{i} = {i}" for i in range(25))
    d = route(f"Please debug this, it's broken:\n{code}")
    assert d.destination is Destination.CLOUD
    assert "25 lines" in d.reason or "26 lines" in d.reason


def test_length_limit_goes_cloud():
    s = Settings(local_max_input_chars=100)
    d = route("Explain " + "very " * 30 + "simply", settings=s)
    assert d.destination is Destination.CLOUD
    assert "over the local limit" in d.reason


def test_context_counts_toward_length():
    s = Settings(local_max_input_chars=100)
    ctx = [{"role": "user", "content": "a" * 90}, {"role": "assistant", "content": "b" * 20}]
    assert route("Explain cats", context=ctx, settings=s).destination is Destination.CLOUD


@pytest.mark.parametrize("mode,dest", [("local", Destination.LOCAL), ("cloud", Destination.CLOUD)])
def test_manual_override(mode, dest):
    for prompt in ["Prove Fermat's last theorem", "Write a poem"]:
        d = route(prompt, mode=mode)
        assert d.destination is dest
        assert d.strategy == "manual_override"
        assert "bypassed" in d.reason


def test_deterministic():
    p = "Optimize this function's runtime and explain the trade-offs"
    assert route(p).reason == route(p).reason
    assert route(p).destination == route(p).destination


def test_decision_has_required_fields():
    d = route("Explain binary search")
    assert d.request_id.startswith("req_") and d.reason and d.strategy and d.destination
