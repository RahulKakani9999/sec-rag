"""Hybrid retrieval: combine vector search and BM25 via Reciprocal Rank Fusion."""
from __future__ import annotations

import hashlib

import numpy as np

import config
from src.indexing.embed import embed_texts
from src.indexing.vectorstore import get_collection, load_bm25

# RRF constant — standard default; higher values reduce the impact of top ranks.
_RRF_K = 60


def _text_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def retrieve(query: str, top_k: int = None) -> list[dict]:
    """Return up to top_k candidates via hybrid dense+sparse retrieval.

    Each result is a dict with keys: text, metadata, score (RRF).
    """
    top_k = top_k if top_k is not None else config.TOP_K_RETRIEVE
    n_candidates = top_k * 2  # over-fetch from each source before fusion

    rrf: dict[str, float] = {}
    pool: dict[str, dict] = {}  # key -> {text, metadata}

    # ── Dense retrieval (Chroma) ──────────────────────────────────────────────
    qvec = embed_texts([query])
    collection = get_collection()
    n_dense = min(n_candidates, collection.count())
    if n_dense > 0:
        chroma = collection.query(
            query_embeddings=qvec,
            n_results=n_dense,
            include=["documents", "metadatas", "distances"],
        )
        for rank, (doc, meta) in enumerate(
            zip(chroma["documents"][0], chroma["metadatas"][0])
        ):
            key = _text_key(doc)
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            pool.setdefault(key, {"text": doc, "metadata": meta})

    # ── Sparse retrieval (BM25) ───────────────────────────────────────────────
    bm25, bm25_texts, bm25_metas = load_bm25()
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:n_candidates]
    for rank, idx in enumerate(top_indices):
        if scores[idx] == 0:
            break  # no keyword overlap at all — stop early
        doc = bm25_texts[idx]
        key = _text_key(doc)
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
        pool.setdefault(key, {"text": doc, "metadata": bm25_metas[idx]})

    # ── Fuse and return top_k ─────────────────────────────────────────────────
    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"score": score, **pool[key]} for key, score in ranked]
