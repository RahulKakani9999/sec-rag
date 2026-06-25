"""End-to-end pipeline: question in, cited answer out."""
from __future__ import annotations

import config
from src.retrieval.retriever import retrieve
from src.retrieval.reranker import rerank
from src.generation.generator import generate


def answer_question(question: str) -> dict:
    """Run retrieve → rerank → generate and return the result.

    Returns:
        {
            "answer":  str,           # LLM response with [Source N] citations
            "sources": list[dict],    # reranked chunks (text + metadata + scores)
        }
    """
    candidates = retrieve(question, top_k=config.TOP_K_RETRIEVE)
    top_chunks = rerank(question, candidates, top_k=config.TOP_K_RERANK)
    answer = generate(question, top_chunks)
    return {"answer": answer, "sources": top_chunks}


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    questions = [
        "What was Apple's total net sales in fiscal 2025?",
        "What is Apple's CEO's home address?",
    ]

    for question in questions:
        print("=" * 70)
        print(f"Q: {question}")
        print("=" * 70)

        result = answer_question(question)

        print(f"\nANSWER:\n{result['answer']}")

        print(f"\nSOURCES ({len(result['sources'])}):")
        for i, src in enumerate(result["sources"], 1):
            meta = src["metadata"]
            snippet = src["text"].replace("\n", " ").replace("\t", "  ")[:120]
            print(
                f"  [{i}] rerank={src['rerank_score']:+.3f} | "
                f"{meta.get('ticker')} {meta.get('year')} — "
                f"{meta.get('section')}\n"
                f"      {snippet} ..."
            )
        print()
