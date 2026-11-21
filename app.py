"""Streamlit UI for the Hybrid LLM Router — redesigned.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st

from hybrid_router.models import Destination
from hybrid_router.pipeline import HybridPipeline, PipelineOutcome

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODES = {
    "auto": {
        "label": "Auto",
        "icon": "⚡",
        "desc": "Smart routing — picks local or cloud, falls back if needed",
        "color": "#7C3AED",
    },
    "local": {
        "label": "Native",
        "icon": "🖥️",
        "desc": "Always runs on your machine via Ollama. No cloud calls ever.",
        "color": "#059669",
    },
    "cloud": {
        "label": "Cloud",
        "icon": "☁️",
        "desc": "Always routes to the cloud model. Highest capability.",
        "color": "#0EA5E9",
    },
}

# Rough cost estimate in USD per 1K tokens (just for display — not billed)
COST_PER_1K_CLOUD = 0.00015   # gpt-4o-mini output
COST_PER_1K_LOCAL = 0.0       # running locally = $0

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ---- Global ---- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0F0F12;
    color: #E2E8F0;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #16161D;
    border-right: 1px solid #2A2A3A;
}

/* ---- Header ---- */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 8px 0 20px 0;
}
.app-header .logo {
    font-size: 2.4rem;
    line-height: 1;
}
.app-header h1 {
    margin: 0;
    font-size: 1.7rem;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -0.5px;
}
.app-header p {
    margin: 2px 0 0 0;
    font-size: 0.82rem;
    color: #64748B;
}

/* ---- Mode cards ---- */
.mode-grid {
    display: flex;
    gap: 10px;
    margin-bottom: 24px;
}
.mode-card {
    flex: 1;
    padding: 14px 16px;
    border-radius: 12px;
    border: 2px solid #2A2A3A;
    background: #16161D;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
    text-align: center;
}
.mode-card.selected {
    border-color: var(--mode-color);
    background: color-mix(in srgb, var(--mode-color) 12%, #16161D);
}
.mode-card .icon { font-size: 1.5rem; }
.mode-card .name {
    font-weight: 700;
    font-size: 0.95rem;
    color: #F1F5F9;
    margin-top: 6px;
}
.mode-card .desc {
    font-size: 0.72rem;
    color: #64748B;
    margin-top: 4px;
    line-height: 1.4;
}

/* ---- Chat bubbles ---- */
.chat-wrap { display: flex; flex-direction: column; gap: 16px; margin-bottom: 20px; }

.bubble-user {
    align-self: flex-end;
    background: #7C3AED;
    color: #fff;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    max-width: 80%;
    font-size: 0.9rem;
    line-height: 1.6;
    white-space: pre-wrap;
}
.bubble-assistant {
    align-self: flex-start;
    background: #1E1E2E;
    color: #E2E8F0;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    max-width: 85%;
    font-size: 0.9rem;
    line-height: 1.6;
    border: 1px solid #2A2A3A;
    white-space: pre-wrap;
}

/* ---- Route badge ---- */
.route-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    margin-bottom: 6px;
}
.badge-local  { background: #052E16; color: #34D399; border: 1px solid #059669; }
.badge-cloud  { background: #082F49; color: #38BDF8; border: 1px solid #0EA5E9; }
.badge-fallback { background: #431407; color: #FB923C; border: 1px solid #EA580C; }

/* ---- Routing reason ---- */
.route-reason {
    font-size: 0.75rem;
    color: #94A3B8;
    margin: 0 0 10px 0;
    padding: 8px 12px;
    background: #16161D;
    border-left: 3px solid #334155;
    border-radius: 0 8px 8px 0;
}

/* ---- Savings counter ---- */
.savings-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #052E16;
    border: 1px solid #059669;
    color: #34D399;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* ---- Input area ---- */
textarea { background: #16161D !important; color: #E2E8F0 !important; border-radius: 10px !important; }
div[data-testid="stTextArea"] > div { background: #16161D; border-radius: 10px; }

/* ---- Buttons ---- */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.88rem;
    transition: opacity 0.15s;
}
.stButton > button:hover { opacity: 0.85; }

/* ---- Divider ---- */
hr { border-color: #2A2A3A; margin: 16px 0; }

/* ---- Sidebar stats ---- */
.stat-block {
    background: #0F0F12;
    border: 1px solid #2A2A3A;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.stat-block .label { font-size: 0.72rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-block .value { font-size: 1.3rem; font-weight: 700; color: #F1F5F9; }

/* ---- Error / warning overrides ---- */
[data-testid="stAlert"] { border-radius: 10px; }
</style>
"""

# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------


def get_pipeline() -> HybridPipeline:
    return HybridPipeline()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_state() -> None:
    st.session_state.setdefault("mode", "auto")
    st.session_state.setdefault("processing", False)
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("history", [])      # [{role, content, meta}]
    st.session_state.setdefault("input_error", None)
    st.session_state.setdefault("use_context", True)
    # stats
    st.session_state.setdefault("total_local", 0)
    st.session_state.setdefault("total_cloud", 0)
    st.session_state.setdefault("total_fallback", 0)
    st.session_state.setdefault("est_saved_tokens", 0)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def on_submit() -> None:
    prompt = st.session_state.get("prompt_input", "").strip()
    if not prompt:
        st.session_state.input_error = "Please enter a prompt before submitting."
        return
    st.session_state.input_error = None
    context = []
    if st.session_state.use_context:
        for item in st.session_state.history:
            context.append({"role": item["role"], "content": item["content"]})
    st.session_state.pending = {
        "prompt": prompt,
        "mode": st.session_state.mode,
        "context": context,
    }
    st.session_state.processing = True


def clear_chat() -> None:
    st.session_state.history = []
    st.session_state.total_local = 0
    st.session_state.total_cloud = 0
    st.session_state.total_fallback = 0
    st.session_state.est_saved_tokens = 0


def set_mode(m: str) -> None:
    st.session_state.mode = m


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _route_badge(outcome: PipelineOutcome) -> str:
    result, decision = outcome.result, outcome.decision
    if result.fallback_used:
        return '<span class="route-badge badge-fallback">⚠️ Fallback → Cloud</span>'
    if result.model is Destination.LOCAL:
        return f'<span class="route-badge badge-local">🖥️ Local · {result.model_name}</span>'
    if result.model is Destination.CLOUD:
        return f'<span class="route-badge badge-cloud">☁️ Cloud · {result.model_name}</span>'
    return ""


def render_mode_cards(busy: bool) -> None:
    cols = st.columns(3)
    for col, (key, info) in zip(cols, MODES.items()):
        selected = st.session_state.mode == key
        border = info["color"] if selected else "#2A2A3A"
        bg = f"color-mix(in srgb, {info['color']} 12%, #16161D)" if selected else "#16161D"
        html = f"""
        <div style="
            padding:14px 10px; border-radius:12px;
            border: 2px solid {border};
            background: {bg};
            text-align:center; cursor:pointer;
        ">
          <div style="font-size:1.6rem">{info['icon']}</div>
          <div style="font-weight:700; color:#F1F5F9; margin-top:6px; font-size:0.95rem">{info['label']}</div>
          <div style="font-size:0.70rem; color:#94A3B8; margin-top:4px; line-height:1.35">{info['desc']}</div>
        </div>"""
        with col:
            st.markdown(html, unsafe_allow_html=True)
            if not busy:
                st.button(
                    f"{'✓ ' if selected else ''}{info['label']}",
                    key=f"btn_mode_{key}",
                    on_click=set_mode,
                    args=(key,),
                    use_container_width=True,
                    type="primary" if selected else "secondary",
                )


def render_chat_history() -> None:
    if not st.session_state.history:
        return
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for item in st.session_state.history:
        if item["role"] == "user":
            st.markdown(
                f'<div class="bubble-user">{item["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            meta = item.get("meta", {})
            badge_html = meta.get("badge_html", "")
            reason = meta.get("reason", "")
            reason_html = (
                f'<div class="route-reason">🔀 {reason}</div>' if reason else ""
            )
            st.markdown(
                f'{badge_html}{reason_html}'
                f'<div class="bubble-assistant">{item["content"]}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar(pipeline: HybridPipeline) -> None:
    s = pipeline.settings
    with st.sidebar:
        st.markdown("## ⚙️ Config")
        st.markdown(
            f'<div class="stat-block">'
            f'<div class="label">Local Model</div>'
            f'<div class="value" style="font-size:0.95rem;color:#34D399">🖥️ {s.local_model}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="stat-block">'
            f'<div class="label">Cloud Model</div>'
            f'<div class="value" style="font-size:0.95rem;color:#38BDF8">☁️ {s.cloud_model}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if not pipeline.cloud_engine.configured:
            st.warning("⚠️ OPENAI_API_KEY not set — cloud calls will fail.", icon="⚠️")

        st.divider()
        st.markdown("## 📊 Session Stats")

        total = st.session_state.total_local + st.session_state.total_cloud
        saved_usd = (st.session_state.est_saved_tokens / 1000) * COST_PER_1K_CLOUD

        for label, val, color in [
            ("Total Requests", total, "#F1F5F9"),
            ("Ran Locally", st.session_state.total_local, "#34D399"),
            ("Sent to Cloud", st.session_state.total_cloud, "#38BDF8"),
            ("Fallbacks", st.session_state.total_fallback, "#FB923C"),
        ]:
            st.markdown(
                f'<div class="stat-block">'
                f'<div class="label">{label}</div>'
                f'<div class="value" style="color:{color}">{val}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        # Estimated savings
        if st.session_state.total_local > 0:
            st.markdown(
                f'<div style="margin-top:10px">'
                f'<span class="savings-pill">💰 Est. saved ~\${saved_usd:.4f} vs all-cloud</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("## 🛠 Settings")
        st.session_state.use_context = st.toggle(
            "Send conversation context", value=st.session_state.use_context
        )
        turns = len([h for h in st.session_state.history if h["role"] == "user"])
        st.caption(f"{turns} turn(s) in context")
        st.button("🗑️ Clear conversation", on_click=clear_chat, use_container_width=True)

        st.divider()
        st.markdown(
            '<p style="font-size:0.72rem;color:#334155;text-align:center">'
            "Hybrid LLM Router · CutieHack 2026</p>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Hybrid LLM Router",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_state()

    pipeline = get_pipeline()
    busy = st.session_state.processing

    render_sidebar(pipeline)

    # ---- Header ----
    st.markdown(
        """
        <div class="app-header">
          <div class="logo">⚡</div>
          <div>
            <h1>Hybrid LLM Router</h1>
            <p>Save money by running simple tasks locally, sending complex ones to the cloud.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Mode selector ----
    st.markdown("### Choose execution mode")
    render_mode_cards(busy)
    st.divider()

    # ---- Chat history ----
    render_chat_history()

    # ---- Input ----
    if st.session_state.input_error:
        st.warning(st.session_state.input_error)

    with st.container():
        prompt_col, btn_col = st.columns([9, 1])
        with prompt_col:
            st.text_area(
                "Your prompt",
                key="prompt_input",
                height=120,
                disabled=busy,
                placeholder=(
                    "e.g. Explain binary search in simple terms  —  "
                    f"Mode: {MODES[st.session_state.mode]['label']}"
                ),
                label_visibility="collapsed",
            )
        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True)
            st.button(
                "Send ➤" if not busy else "...",
                type="primary",
                on_click=on_submit,
                disabled=busy,
                use_container_width=True,
            )

    # ---- Processing ----
    if busy and st.session_state.pending:
        pending = st.session_state.pending
        mode_info = MODES[pending["mode"]]
        with st.spinner(f"{mode_info['icon']} Routing & generating…"):
            outcome = pipeline.run(pending["prompt"], pending["mode"], pending["context"])

        result, decision = outcome.result, outcome.decision

        # Update stats
        if result.ok:
            if result.model is Destination.LOCAL:
                st.session_state.total_local += 1
                # Estimate tokens saved: rough 4 chars per token
                token_estimate = max(len(result.response) // 4, 1)
                st.session_state.est_saved_tokens += token_estimate
            elif result.model is Destination.CLOUD:
                st.session_state.total_cloud += 1
            if result.fallback_used:
                st.session_state.total_fallback += 1

        # Build meta for the assistant bubble
        badge_html = _route_badge(outcome) if decision else ""
        reason = decision.reason if decision else ""
        if not result.ok:
            msg = result.error.message if result.error else "Inference failed."
        else:
            msg = result.response or ""

        # Append to history
        st.session_state.history.append({"role": "user", "content": pending["prompt"]})
        st.session_state.history.append({
            "role": "assistant",
            "content": msg,
            "meta": {"badge_html": badge_html, "reason": reason},
        })

        # Trim context
        s = pipeline.settings
        while len(st.session_state.history) > s.max_context_messages:
            st.session_state.history.pop(0)

        st.session_state.pending = None
        st.session_state.processing = False
        st.rerun()


main()
