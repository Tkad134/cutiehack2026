"""Time-based response reveal using the standard five-character typing word."""

WORDS_PER_MINUTE = 2160
CHARACTERS_PER_WORD = 5


def reveal_text(text: str, elapsed_s: float) -> tuple[str, bool]:
    """Reveal at a chat-like 180 characters per second without changing the stored answer or sleeping."""
    characters = int(max(0.0, elapsed_s) * WORDS_PER_MINUTE * CHARACTERS_PER_WORD / 60)
    return text[:characters], characters >= len(text)
