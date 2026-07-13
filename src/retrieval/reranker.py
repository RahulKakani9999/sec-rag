"""Cross-encoder reranking of retrieved candidates."""
from __future__ import annotations

from sentence_transformers import CrossEncoder

import config

_model: CrossEncoder | None = None


def get_model() -> CrossEncoder:
    """Return the singleton cross-encoder, loading it on first call."""
    global _model
    if _model is None:
        _model = CrossEncoder(config.RERANK_MODEL)
    return _model


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = None,
) -> list[dict]:
    """Score each (query, candidate) pair with the cross-encoder.

    Returns top_k results sorted by rerank_score descending.
    Each result carries the original keys plus a new 'rerank_score' float.
    """
    top_k = top_k if top_k is not None else config.TOP_K_RERANK
    if not candidates:
        return []

    # Normalize tabs to spaces so the cross-encoder (trained on prose) can parse
    # financial tables. The stored text and BM25 index keep the original tabs.
    pairs = [(query, c["text"].replace("\t", "  ")) for c in candidates]
    scores = get_model().predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [
        {**candidate, "rerank_score": float(score)}
        for score, candidate in ranked[:top_k]
    ]


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from src.retrieval.retriever import retrieve

    QUERY = "What was Apple's total net sales in 2025?"
    print(f"Query: {QUERY!r}")
    print()

    print(f"[1/2] Hybrid retrieval (top_k={config.TOP_K_RETRIEVE}) ...")
    candidates = retrieve(QUERY)
    print(f"      {len(candidates)} candidates returned")

    print(f"[2/2] Reranking (top_k={config.TOP_K_RERANK}) ...")
    results = rerank(QUERY, candidates)

    print()
    print("=" * 70)
    print(f"TOP {len(results)} RESULTS")
    print("=" * 70)
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        snippet = r["text"].replace("\n", " ").replace("\t", "  ")[:200]
        print(
            f"\nRank {i}  rerank={r['rerank_score']:+.4f}  rrf={r['score']:.5f}\n"
            f"  section : {meta.get('section')}\n"
            f"  ticker  : {meta.get('ticker')}  year={meta.get('year')}\n"
            f"  snippet : {snippet} ..."
        )
