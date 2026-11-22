"""Alice: a focused chat workspace for local and cloud inference.

Run with: python3 -m streamlit run app.py
"""

from __future__ import annotations

import base64
import html
import logging
import time
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from hybrid_router.config import Settings, load_settings
from hybrid_router.models import Destination
from hybrid_router.pipeline import HybridPipeline, PipelineOutcome
from hybrid_router.ui import ALICE_LOGO, icon, markdown_math
from hybrid_router.typing_effect import reveal_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODE_MAP = {"Drink me!": "local", "Follow the rabbit": "auto", "Eat me!": "cloud"}
MODE_KEYS = list(MODE_MAP)
MODE_META = {
    "Drink me!": {"icon": "potion", "desc": "On your device · No API cost"},
    "Follow the rabbit": {"icon": "rabbit", "desc": "The right model, automatically"},
    "Eat me!": {"icon": "cake", "desc": "Cloud inference for demanding tasks"},
}
SUGGESTIONS = (
    ("Light a candle!", "candle", "Explain a concept to me in simple terms: "),
    ("Let’s cook!", "cook", "Help me draft a clear, concise message about "),
    ("Derive a magic spell!", "spell", "Help me understand this code:\n"),
)
STYLE_PATH = Path(__file__).parent / "assets" / "style.css"
FONT_STYLE_PATH = Path(__file__).parent / "assets" / "fonts" / "fonts.css"
BACKGROUND_PATH = Path(__file__).parent / "assets" / "wonderland-alice-background.png"


def get_pipeline(settings: Settings) -> HybridPipeline:
    return HybridPipeline(settings=settings)


def init_state() -> None:
    for key, value in {
        "mode_index": 1, "processing": False, "pending": None, "history": [],
        "input_error": None, "typing": None, "total_local": 0,
        "total_cloud": 0, "total_fallback": 0, "est_saved_tokens": 0,
    }.items():
        st.session_state.setdefault(key, value)


def on_submit() -> None:
    prompt = (st.session_state.get("prompt_input") or "").strip()
    if not prompt:
        st.session_state.input_error = "Write a message to start the conversation."
        return
    st.session_state.input_error = None
    st.session_state.typing = None
    # Failed turns stay on screen but are never sent as context: their "reply" is an
    # app error message, not something the model said. The request processor keeps
    # only the most recent MAX_CONTEXT_MESSAGES of what remains.
    context = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.history if not item.get("failed")
    ]
    st.session_state.pending = {
        "prompt": prompt, "mode": MODE_MAP[MODE_KEYS[st.session_state.mode_index]], "context": context,
    }
    st.session_state.processing = True


def clear_chat() -> None:
    st.session_state.history = []
    st.session_state.typing = None
    st.session_state.input_error = None
    for key in ("total_local", "total_cloud", "total_fallback", "est_saved_tokens"):
        st.session_state[key] = 0


def set_draft(prompt: str) -> None:
    st.session_state.prompt_input = prompt


def _route_chip(outcome: PipelineOutcome) -> str:
    result = outcome.result
    name = html.escape(result.model_name or "")
    if result.fallback_used:
        return f'<span class="route-chip chip-fallback">{icon("cake")}Cloud fallback · {name}</span>'
    if result.model is Destination.LOCAL:
        return f'<span class="route-chip">{icon("potion")}On device · {name}</span>'
    if result.model is Destination.CLOUD:
        return f'<span class="route-chip">{icon("cake")}Cloud · {name}</span>'
    return ""


def render_mode_picker(busy: bool) -> None:
    mode = MODE_KEYS[st.session_state.mode_index]
    image = quote(icon(MODE_META[mode]["icon"], color="#334457"), safe="")
    with st.container(key="mode-picker", width=36):
        st.markdown(f'<style>.st-key-mode-picker {{ --mode-icon: url("data:image/svg+xml,{image}"); }}</style>', unsafe_allow_html=True)
        with st.popover(f"Choose mode: {mode}", disabled=busy):
            with st.container(key="mode-sheet"):
                # One rule per stop so the thumb logo swaps in the browser as soon as the
                # slider moves, without waiting for the server rerun.
                per_stop_icons = "".join(
                    f'.st-key-mode-sheet:has(input[type="range"][value="{index}"]) '
                    f'{{ --selector-icon: url("data:image/svg+xml,'
                    f'{quote(icon(MODE_META[name]["icon"], color="#334457"), safe="")}"); }}'
                    for index, name in enumerate(MODE_KEYS)
                )
                st.markdown(
                    f'<style>.st-key-mode-sheet {{ --selector-icon: url("data:image/svg+xml,{image}"); }}'
                    f'{per_stop_icons}</style>'
                    '<div class="mode-sheet-heading"><span>Wonders</span><span>Slide to choose</span></div>',
                    unsafe_allow_html=True,
                )
                st.select_slider(
                    "Wonders", options=[0, 1, 2], key="mode_index",
                    format_func=lambda value: MODE_KEYS[value],
                    label_visibility="collapsed", disabled=busy,
                )
                labels = "".join(
                    '<span class="%s">%s</span>' % ("active" if index == st.session_state.mode_index else "", name)
                    for index, name in enumerate(MODE_KEYS)
                )
                st.markdown(
                    f'<div class="mode-sheet-stops">{labels}</div>'
                    f'<div class="mode-sheet-detail">{MODE_META[mode]["desc"]}</div>',
                    unsafe_allow_html=True,
                )



@st.fragment(run_every=0.1 if st.session_state.get("typing") else None)
def render_chat() -> None:
    typing = st.session_state.get("typing")
    for index, item in enumerate(st.session_state.history):
        avatar = ":material/person:" if item["role"] == "user" else str(ALICE_LOGO)
        with st.chat_message(item["role"], avatar=avatar):
            if item["role"] == "assistant":
                meta = item.get("meta", {})
                if meta.get("chip_html"):
                    st.markdown(meta["chip_html"], unsafe_allow_html=True)
                if typing and index == typing["message_index"]:
                    partial, complete = reveal_text(item["content"], time.monotonic() - typing["started_at"])
                    if complete:
                        st.session_state.typing = None
                        st.rerun()
                    with st.container(key="typing-response"):
                        st.markdown(markdown_math(partial) or "\u200b")
                    if st.button("Show full response", key="finish_typing"):
                        st.session_state.typing = None
                        st.rerun()
                else:
                    st.markdown(markdown_math(item["content"]))
                if meta.get("reason") and not (typing and index == typing["message_index"]):
                    with st.expander("Routing details"):
                        st.caption(meta["reason"])
            else:
                st.markdown(item["content"])
    if st.session_state.processing and st.session_state.pending:
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(st.session_state.pending["prompt"])
        with st.chat_message("assistant", avatar=str(ALICE_LOGO)):
            st.markdown(
                '<div class="thinking" role="status"><span class="thinking-dots"><i></i><i></i><i></i></span>Dreaming…</div>',
                unsafe_allow_html=True,
            )


def render_sidebar(settings: Settings, busy: bool) -> Settings:
    with st.sidebar:
        st.markdown(
            f'<div class="sidebar-brand"><span class="brand-mark">{icon("brand", 36)}</span>'
            '<span class="brand-name">Alice</span></div>', unsafe_allow_html=True,
        )
        st.button("New dream", icon=":material/add:", on_click=clear_chat, use_container_width=True, disabled=busy)
        st.divider()
        st.markdown('<div class="sidebar-label">This dream</div>', unsafe_allow_html=True)
        total = st.session_state.total_local + st.session_state.total_cloud
        stats = "".join(
            f'<div class="session-stat {kind}"><strong>{value}</strong><span>{label}</span></div>'
            for label, value, kind in (
                ("Requests", total, ""), ("On device", st.session_state.total_local, "local"),
                ("Cloud", st.session_state.total_cloud, ""), ("Fallbacks", st.session_state.total_fallback, ""),
            )
        )
        st.markdown(f'<div class="session-stats">{stats}</div>', unsafe_allow_html=True)
        if st.session_state.total_local:
            st.markdown(f'<div class="saved-tokens">~{st.session_state.est_saved_tokens:,} cloud output tokens avoided</div>', unsafe_allow_html=True)
            st.caption("Estimated from response length; actual savings vary by model.")
        st.divider()
        for model, symbol in ((settings.local_model, "potion"), (settings.cloud_model, "cake")):
            st.markdown(f'<div class="sidebar-model">{icon(symbol)}<span>{html.escape(model)}</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-footer">Built for wonders.</div>', unsafe_allow_html=True)
    return settings


def render_composer(busy: bool) -> None:
    mode = MODE_KEYS[st.session_state.mode_index]
    with st.container(key="composer"):
        st.chat_input("Message Alice…", key="prompt_input", disabled=busy, on_submit=on_submit)
        with st.container(key="composer-toolbar", horizontal=True, vertical_alignment="center", gap="small"):
            render_mode_picker(busy)
            st.markdown(
                f'<div class="mode-summary"><strong>{mode}</strong><span>{MODE_META[mode]["desc"]}</span></div>',
                unsafe_allow_html=True,
            )


def process_pending(settings: Settings) -> None:
    pending = st.session_state.pending
    pipeline = get_pipeline(settings)
    outcome = pipeline.run(pending["prompt"], pending["mode"], pending["context"])
    result, decision = outcome.result, outcome.decision
    if result.ok:
        if result.model is Destination.LOCAL:
            st.session_state.total_local += 1
            st.session_state.est_saved_tokens += max(len(result.response) // 4, 1)
        elif result.model is Destination.CLOUD:
            st.session_state.total_cloud += 1
        if result.fallback_used:
            st.session_state.total_fallback += 1
    message = result.error.message if (not result.ok and result.error) else (result.response or "")
    failed = not result.ok
    st.session_state.history.extend([
        {"role": "user", "content": pending["prompt"], "failed": failed},
        {"role": "assistant", "content": message, "failed": failed, "meta": {
            "chip_html": _route_chip(outcome) if decision else "",
            "reason": result.fallback_reason or (decision.reason if decision else ""),
        }},
    ])
    # The whole conversation stays visible; the context limit is applied per request.
    st.session_state.typing = (
        {"message_index": len(st.session_state.history) - 1, "started_at": time.monotonic()}
        if result.ok and message else None
    )
    st.session_state.pending = None
    st.session_state.processing = False
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Alice — Your AI workspace", page_icon=str(ALICE_LOGO), layout="centered", initial_sidebar_state="auto")
    background_data = base64.b64encode(BACKGROUND_PATH.read_bytes()).decode("ascii")
    suggestion_styles = "".join(
        f'.st-key-suggestion_{symbol} {{ --suggestion-icon: url("data:image/svg+xml,{quote(icon(symbol, color="#f0d09a"), safe="")}"); }}'
        for _, symbol, _ in SUGGESTIONS
    )
    st.markdown(
        f'<style>:root {{ --wonderland-background: url("data:image/png;base64,{background_data}"); }}'
        f'{FONT_STYLE_PATH.read_text()}\n{STYLE_PATH.read_text()}\n{suggestion_styles}</style>',
        unsafe_allow_html=True,
    )
    init_state()
    busy = st.session_state.processing
    settings = render_sidebar(load_settings(), busy)
    st.markdown(
        '<div class="workspace-topbar">'
        f'<div class="workspace-type">{icon("brand")}LOCAL + CLOUD</div></div>', unsafe_allow_html=True,
    )
    empty = not st.session_state.history and not busy
    if empty:
        st.markdown(
            f'<div class="welcome"><div class="welcome-mark">{icon("brand", 72)}</div>'
            '<h1>What shall we dream up?</h1><p>Follow your curiosity. A little wonder starts here.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        with st.container(key="conversation"):
            render_chat()
    if st.session_state.input_error:
        st.warning(st.session_state.input_error)
    render_composer(busy)
    if empty:
        with st.container(key="suggestions"):
            for col, (label, symbol, prompt) in zip(st.columns(3), SUGGESTIONS):
                col.button(label, key=f"suggestion_{symbol}", on_click=set_draft, args=(prompt,), use_container_width=True)
    st.markdown('<div class="composer-footnote">Local when it fits. Cloud when it counts.</div>', unsafe_allow_html=True)
    if busy and st.session_state.pending:
        process_pending(settings)


main()
