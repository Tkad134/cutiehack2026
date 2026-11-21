"""Streamlit user interface for the hybrid LLM router.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st

from hybrid_router.models import Destination
from hybrid_router.pipeline import HybridPipeline, PipelineOutcome

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODES = {
    "Automatic": "auto",
    "Local only": "local",
    "Cloud only": "cloud",
}
MODE_HELP = (
    "Automatic: the router picks local or cloud, and a failed local answer is retried once "
    "in the cloud. Local only / Cloud only: skip routing and never fall back."
)


@st.cache_resource
def get_pipeline() -> HybridPipeline:
    return HybridPipeline()


def init_state() -> None:
    st.session_state.setdefault("processing", False)
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("outcome", None)
    st.session_state.setdefault("history", [])  # list of {"role", "content"}
    st.session_state.setdefault("input_error", None)


def on_submit() -> None:
    prompt = st.session_state.get("prompt_input", "")
    if not prompt.strip():
        st.session_state.input_error = "Please enter a prompt before submitting."
        return
    st.session_state.input_error = None
    st.session_state.pending = {
        "prompt": prompt,
        "mode": MODES[st.session_state.mode_label],
        "context": list(st.session_state.history) if st.session_state.use_context else [],
    }
    st.session_state.processing = True


def clear_conversation() -> None:
    st.session_state.history = []
    st.session_state.outcome = None


def render_outcome(outcome: PipelineOutcome) -> None:
    result, decision = outcome.result, outcome.decision

    if decision is None:  # rejected by the request processor
        st.error(result.error.message if result.error else "Request rejected.")
        return

    # Model indicator
    if result.model is Destination.LOCAL:
        badge = f"🖥️ **Local** · `{result.model_name}`"
    elif result.model is Destination.CLOUD:
        badge = f"☁️ **Cloud** · `{result.model_name}`"
    else:
        badge = "No model"
    status = "✅ success" if result.ok else "❌ failed"
    st.markdown(f"{badge} &nbsp;&nbsp; {status} &nbsp;&nbsp; `{result.request_id}`")

    # Routing explanation
    strategy = "Manual override" if decision.strategy == "manual_override" else "Automatic routing"
    st.info(f"**{strategy} → {decision.destination.value}.** {decision.reason}")
    if result.fallback_used:
        st.warning(f"**Fallback: local → cloud.** {result.fallback_reason}")

    # Response / error
    if result.ok:
        st.markdown("#### Response")
        st.markdown(result.response)
    else:
        st.error(result.error.message if result.error else "Inference failed.")
        if result.response:
            with st.expander("Raw model output"):
                st.text(result.response)

    with st.expander("Details (JSON)"):
        st.json({
            "routing": {
                "request_id": decision.request_id,
                "destination": decision.destination.value,
                "reason": decision.reason,
                "strategy": decision.strategy,
                "category": decision.category,
            },
            "result": result.to_dict(),
        })


def main() -> None:
    st.set_page_config(page_title="Hybrid LLM Router", page_icon="🔀", layout="centered")
    init_state()
    pipeline = get_pipeline()
    s = pipeline.settings

    st.title("🔀 Hybrid LLM Router")
    st.caption(f"Local: `{s.local_model}` via Ollama · Cloud: `{s.cloud_model}` via OpenAI")
    if not pipeline.cloud_engine.configured:
        st.caption("⚠️ OPENAI_API_KEY is not set, so cloud requests and fallback will fail.")

    busy = st.session_state.processing
    st.radio("Execution mode", list(MODES), key="mode_label", horizontal=True,
             help=MODE_HELP, disabled=busy)
    st.text_area("Prompt", key="prompt_input", height=160, disabled=busy,
                 placeholder="e.g. Explain binary search in simple terms")

    col1, col2, col3 = st.columns([1, 2, 1])
    col1.button("Submit", type="primary", on_click=on_submit, disabled=busy)
    col2.checkbox(
        f"Use conversation as context ({len(st.session_state.history) // 2} turns)",
        key="use_context", value=True, disabled=busy,
    )
    col3.button("Clear", on_click=clear_conversation, disabled=busy)

    if st.session_state.input_error:
        st.warning(st.session_state.input_error)

    if busy and st.session_state.pending:
        pending = st.session_state.pending
        with st.spinner("Routing and generating…"):
            outcome = pipeline.run(pending["prompt"], pending["mode"], pending["context"])
        st.session_state.outcome = outcome
        if outcome.result.ok and outcome.request is not None:
            st.session_state.history.extend([
                {"role": "user", "content": outcome.request.prompt},
                {"role": "assistant", "content": outcome.result.response},
            ])
            st.session_state.history = st.session_state.history[-s.max_context_messages:]
        st.session_state.pending = None
        st.session_state.processing = False
        st.rerun()

    if st.session_state.outcome is not None:
        st.divider()
        render_outcome(st.session_state.outcome)


main()
