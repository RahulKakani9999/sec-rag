"""Streamlit UI for the SEC 10-K RAG system."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# Bridge Streamlit Cloud secrets into os.environ so config.py picks them up
# via os.getenv().  On Streamlit Cloud, secrets set in the dashboard are
# exposed as st.secrets; locally, os.environ is already populated from .env.
# setdefault means the local env var always wins if both are present.
for _k, _v in st.secrets.items():
    os.environ.setdefault(_k, str(_v))

import config

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SEC 10-K Research Assistant",
    page_icon="📊",
    layout="centered",
)

# ── Warm up models and indexes once at startup ────────────────────────────────

@st.cache_resource(show_spinner="Loading models and index…")
def _load_pipeline():
    """Pre-load BGE embeddings, cross-encoder, Chroma, and BM25.

    Called once at startup via cache_resource.  The Chroma collection and BM25
    pickle are opened from disk — nothing is rebuilt or re-embedded.
    """
    from src.indexing.embed import embed_texts          # loads BAAI/bge-small-en-v1.5
    from src.retrieval.reranker import get_model        # loads ms-marco cross-encoder
    from src.indexing.vectorstore import get_collection, load_bm25

    embed_texts(["warmup"])                             # triggers model download/load
    get_model()                                         # triggers cross-encoder load
    col = get_collection()                              # opens existing Chroma DB
    _, bm25_texts, _ = load_bm25()                     # unpickles chroma_db/bm25.pkl
    return col.count(), len(bm25_texts)


n_vectors, n_bm25 = _load_pipeline()

# ── Header ────────────────────────────────────────────────────────────────────

st.title("📊 SEC 10-K Research Assistant")
st.caption(
    f"Ask questions about Apple's 2025 annual report · "
    f"{n_vectors:,} indexed chunks · "
    f"hybrid retrieval (BM25 + dense) · cross-encoder reranking · Groq generation"
)
st.divider()

if not config.GROQ_API_KEY:
    st.error(
        "**GROQ_API_KEY is not set.** "
        "Go to Space Settings → Secrets and add `GROQ_API_KEY`."
    )
    st.stop()

# ── Example questions ─────────────────────────────────────────────────────────

EXAMPLES = [
    "What were Apple's total assets as of September 27, 2025?",
    "What was Apple's total net sales in fiscal 2025?",
    "How much did Apple spend on R&D in fiscal 2025?",
    "What were Apple's total liabilities as of September 27, 2025?",
]

st.markdown("**Try an example or type your own question:**")
cols = st.columns(2)
chosen_example = None
for i, ex in enumerate(EXAMPLES):
    if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
        chosen_example = ex

st.markdown("")

# ── Question form ─────────────────────────────────────────────────────────────

with st.form("question_form", clear_on_submit=False):
    question = st.text_input(
        "Your question",
        value=chosen_example or "",
        placeholder="What were Apple's total assets as of September 27, 2025?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

# ── Pipeline ──────────────────────────────────────────────────────────────────

if submitted and question.strip():
    from src.pipeline import answer_question

    with st.spinner("Retrieving and generating…"):
        result = answer_question(question.strip())

    answer  = result["answer"]
    sources = result["sources"]

    st.markdown("### Answer")
    st.markdown(answer)

    with st.expander(f"View sources — {len(sources)} chunks retrieved"):
        for i, src in enumerate(sources, 1):
            meta    = src["metadata"]
            section = meta.get("section", "?")
            ticker  = meta.get("ticker", "")
            year    = meta.get("year", "")
            score   = src.get("rerank_score", src.get("score", 0.0))
            # Normalize tab-separated table data for readable display
            text    = src["text"].replace("\t", "  ")

            st.markdown(
                f"**\\[Source {i}\\]** &nbsp; `{ticker} {year}` &nbsp;·&nbsp; "
                f"*{section}* &nbsp; `rerank={score:.3f}`"
            )
            st.code(text[:700] + ("…" if len(text) > 700 else ""), language=None)
            if i < len(sources):
                st.divider()

elif submitted:
    st.warning("Please enter a question before clicking Ask.")
