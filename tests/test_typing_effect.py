from hybrid_router.typing_effect import reveal_text


def test_fast_chat_pacing():
    text = "a" * 2000
    partial, complete = reveal_text(text, 1)
    assert len(partial) == 180 and not complete


def test_reveal_completes_without_changing_markdown():
    text = "**Hello**\n```python\nx = 2\n```"
    assert reveal_text(text, 60) == (text, True)


def test_negative_elapsed_and_empty_answer():
    assert reveal_text("Hello", -1) == ("", False)
    assert reveal_text("", 0) == ("", True)
