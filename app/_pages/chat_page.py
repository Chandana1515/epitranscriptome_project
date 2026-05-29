"""AI Assistant page — chat with LLM about your results."""

import streamlit as st
from ollama_helper import (
    check_ollama_status,
    get_available_models,
    stream_chat_with_ollama,
    DEFAULT_MODEL,
)
from result_loader import load_all_results, get_results_overview


# Quick-ask suggestions
SUGGESTED_QUESTIONS = [
    "Summarize the key findings from this analysis",
    "Which genes show the strongest modification changes and why?",
    "What do the SHAP results tell us about modification susceptibility?",
    "Which biological pathways are enriched and what does that mean?",
    "How does hypoxia affect RNA modifications in these cancer cells?",
    "What is the role of m6A in the oxidative stress response?",
    "Explain the volcano plot results",
    "How reliable are these results given the sample size?",
]


def render():
    st.header("AI assistant")
    st.caption(
        "Ask questions about your pipeline results. "
        "The AI uses your actual data to provide grounded explanations."
    )

    # ── Ollama status check ─────────────────────────────────
    ollama_ok, available_models = check_ollama_status()

    if not ollama_ok:
        st.error("Ollama is not running")
        st.markdown("""
**To set up the AI assistant:**

1. Install Ollama:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2. Pull a model:
```bash
ollama pull llama3.1:8b
```

3. Start the server (if not auto-started):
```bash
ollama serve
```

Then refresh this page.
        """)
        return

    # Model selector
    col1, col2 = st.columns([3, 1])
    with col2:
        if available_models:
            model = st.selectbox(
                "Model",
                available_models,
                index=(
                    available_models.index(DEFAULT_MODEL)
                    if DEFAULT_MODEL in available_models
                    else 0
                ),
            )
        else:
            st.warning("No models found")
            st.code("ollama pull llama3.1:8b")
            return

    with col1:
        st.success(f"Ollama connected — {len(available_models)} model(s) available")

    # ── Load results for context ────────────────────────────
    pipeline_dir = st.session_state.get("pipeline_dir", "")
    results = {}

    if pipeline_dir:
        results = load_all_results(pipeline_dir)
        st.session_state.results = results

        overview = get_results_overview(results)
        loaded_count = sum(1 for v in overview.values() if v not in ["not found", "empty"])

        if loaded_count > 0:
            with st.expander(f"Loaded data: {loaded_count} result sets", expanded=False):
                for key, status in overview.items():
                    icon = "✅" if status not in ["not found", "empty"] else "⬜"
                    st.markdown(f"{icon} **{key}**: {status}")
        else:
            st.info(
                "No pipeline results loaded. The AI can still answer general "
                "biology questions, but won't reference your specific data."
            )
    else:
        st.info("Set the pipeline directory in **Upload & Configure** to load results.")

    st.divider()

    # ── Chat history ────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🧬"):
                st.markdown(msg["content"])

    # ── Suggested questions (only show if no history) ───────
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        # Show in 2 columns
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS[:6]):
            with cols[i % 2]:
                if st.button(
                    q,
                    key=f"suggest_{i}",
                    use_container_width=True,
                ):
                    st.session_state._pending_question = q
                    st.rerun()

    # ── Handle pending question from suggestion buttons ─────
    pending = st.session_state.pop("_pending_question", None)

    # ── Chat input ──────────────────────────────────────────
    user_input = st.chat_input("Ask about your results...")

    # Use pending question if no direct input
    question = pending or user_input

    if question:
        # Show user message
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        # Stream AI response
        with st.chat_message("assistant", avatar="🧬"):
            response_text = st.write_stream(
                stream_chat_with_ollama(
                    user_message=question,
                    results=results,
                    chat_history=st.session_state.chat_history[:-1],
                    model=model,
                )
            )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response_text}
        )

    # ── Clear chat button ───────────────────────────────────
    if st.session_state.chat_history:
        st.divider()
        if st.button("Clear conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
