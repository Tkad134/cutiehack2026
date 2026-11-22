"""Small, consistent vector marks used by the chat interface."""

from pathlib import Path

ALICE_LOGO = Path(__file__).resolve().parents[1] / "assets" / "alice-logo.svg"

ICONS = {
    "candle": '<path d="M12 2c-2 2.5-3 3.5-3 5a3 3 0 0 0 6 0c0-1.5-1-2.5-3-5Z" fill="#f4bc6e"/><path d="M12 10v2"/><path d="M8 12h8v8H8Z" fill="#fff1d8"/><path d="M8 14c1.5 1 2.5-1 4 0s2.5 1 4 0M5 21h14"/>',
    "cook": '<path d="M5 12h14v3a7 7 0 0 1-14 0Z" fill="#f5dbab"/><path d="M2 12h20M4 21h16M8 8c-2-2 2-3 0-5M12 8c-2-2 2-3 0-5M16 8c-2-2 2-3 0-5"/>',
    "spell": '<path d="m4 19 9-9 3 3-9 9Z" fill="#edc3d7"/><path d="m11 12 3 3M18 2l.8 2.7L22 6l-3.2 1.3L18 10l-.8-2.7L14 6l3.2-1.3ZM6 3v4M4 5h4M20 15v4M18 17h4"/>',
    "potion": '<path d="M9 5.5v4L5.2 15a4 4 0 0 0 3.3 6h7a4 4 0 0 0 3.3-6L15 9.5v-4" fill="#e4f1ee"/><rect x="8.5" y="2.5" width="7" height="3" rx="1" fill="#e1b477"/><path d="M7 14c2.4-1.6 5.1 1.6 10 0l1.4 2.2a2.5 2.5 0 0 1-2.1 3.8H7.7a2.5 2.5 0 0 1-2.1-3.8Z" fill="#70b7ad" stroke="none"/><path d="M8 16.5v1.8M18.5 4v3M17 5.5h3"/>',
    "rabbit": '<path d="M7.5 11C4.5 7 3.7 2.8 5.5 2.2c1.8-.6 4.2 3.2 4.5 7.6ZM16.5 11c3-4 3.8-8.2 2-8.8-1.8-.6-4.2 3.2-4.5 7.6Z" fill="#fff5e5"/><ellipse cx="12" cy="15" rx="7" ry="6" fill="#fff5e5"/><path d="m7 5 1.5 4M17 5l-1.5 4" stroke="#d8899d"/><circle cx="9.2" cy="14.5" r=".65" fill="#334457" stroke="none"/><circle cx="14.8" cy="14.5" r=".65" fill="#334457" stroke="none"/><path d="m10.5 16 1.5 1.5 1.5-1.5Z" fill="#d8899d" stroke="none"/><path d="M12 17.5v1M10 18.5q2 1.5 4 0"/>',
    "cake": '<path d="M4 10h16v10H4Z" fill="#e9ba7e"/><path d="M4 15h16" stroke="#b16b55"/><path d="M4 10c0-2.7 16-2.7 16 0v2a1.5 1.5 0 0 1-3 0v-1a1.5 1.5 0 0 1-3 0v1a1.5 1.5 0 0 1-3 0v-1a1.5 1.5 0 0 1-3 0v1a2 2 0 0 1-4 0Z" fill="#fff6eb"/><circle cx="12" cy="5.5" r="2" fill="#d9748f"/><path d="M12 3.5c0-1 1-1.5 2-1.5M2.5 21h19"/>',
    "flash": '<path d="m13.5 2-9 12H11l-.5 8 9-12H13l.5-8Z"/>',
    "bot": '<rect x="4" y="7" width="16" height="13" rx="4"/><path d="M12 3v4M10 3h4M1.5 12v4M22.5 12v4M9 16h6"/><circle cx="8.5" cy="11.5" r=".8" fill="currentColor" stroke="none"/><circle cx="15.5" cy="11.5" r=".8" fill="currentColor" stroke="none"/>',
    "brain": '<path d="M12 5a3 3 0 0 0-5.8-1A3.7 3.7 0 0 0 3 9.5a4 4 0 0 0 1 7.3A3.8 3.8 0 0 0 12 19V5Zm0 0a3 3 0 0 1 5.8-1A3.7 3.7 0 0 1 21 9.5a4 4 0 0 1-1 7.3A3.8 3.8 0 0 1 12 19"/><path d="M7 8c0 2 1 3 2.5 3M4 16c1-1 2-1.5 3.5-1M17 8c0 2-1 3-2.5 3M20 16c-1-1-2-1.5-3.5-1"/>',
    "arrow": '<path d="M7 17 17 7M7 7h10v10"/>',
}


def icon(name: str, size: int = 20, color: str = "currentColor") -> str:
    if name == "brand":
        return (ALICE_LOGO.read_text().replace('width="64"', f'width="{size}"')
                .replace('height="64"', f'height="{size}"').replace("#F6CD86", color)
                .replace('<svg ', '<svg aria-hidden="true" ', 1))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.65" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'
    )
