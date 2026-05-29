"""
EpiTranscriptome Analyzer — Streamlit Application
===================================================
Interactive dashboard for RNA modification analysis pipeline
with AI-powered result interpretation via Ollama.

Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="EpiTranscriptome Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean up default streamlit styling */
    .stApp > header { background: transparent; }
    section[data-testid="stSidebar"] {
        background: #f8f9fb;
        border-right: 1px solid #e8e8ec;
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.3rem;
        font-weight: 600;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4f46e5;
        margin-bottom: 1rem;
    }
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #f8f9fb;
        border: 1px solid #e8e8ec;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.8rem;
        color: #6b7280;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
    /* Chat styling */
    .chat-user {
        background: #eef2ff;
        border-radius: 12px 12px 4px 12px;
        padding: 10px 14px;
        margin: 6px 0;
        margin-left: 20%;
        color: #3730a3;
        font-size: 0.9rem;
    }
    .chat-ai {
        background: #f8f9fb;
        border: 1px solid #e8e8ec;
        border-radius: 12px 12px 12px 4px;
        padding: 10px 14px;
        margin: 6px 0;
        margin-right: 10%;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    /* Pipeline step indicators */
    .step-done { color: #059669; }
    .step-running { color: #d97706; }
    .step-pending { color: #9ca3af; }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ────────────────────────────
defaults = {
    "pipeline_status": "idle",       # idle, running, complete, error
    "pipeline_step": 0,
    "pipeline_log": [],
    "results_loaded": False,
    "chat_history": [],
    "config": None,
    "results": {},
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Sidebar Navigation ─────────────────────────────────────
with st.sidebar:
    st.markdown("# 🧬 EpiTranscriptome")
    st.caption("Explainable AI for RNA modifications in plasma-treated skin cancer")

    page = st.radio(
        "Navigation",
        ["📁 Upload & Configure", "▶️ Run Pipeline", "📊 Results", "🤖 AI Assistant"],
        label_visibility="collapsed",
    )

    st.divider()

    # Pipeline status indicator
    status = st.session_state.pipeline_status
    if status == "idle":
        st.info("Pipeline not started")
    elif status == "running":
        st.warning(f"Pipeline running... Step {st.session_state.pipeline_step}/8")
    elif status == "complete":
        st.success("Pipeline complete")
    elif status == "error":
        st.error("Pipeline error")

    st.divider()
    st.caption("Master's Thesis — Universität Rostock")
    st.caption("Chandana Nagaraju, 2026")


# ── Page Router ─────────────────────────────────────────────
if page == "📁 Upload & Configure":
    from _pages import upload_page
    upload_page.render()

elif page == "▶️ Run Pipeline":
    from _pages import pipeline_page
    pipeline_page.render()

elif page == "📊 Results":
    from _pages import results_page
    results_page.render()

elif page == "🤖 AI Assistant":
    from _pages import chat_page
    chat_page.render()
