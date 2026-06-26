"""Run the test set through the pipeline and score with RAGAS.

Usage:
  python eval/run_eval.py                  # compare full vs baseline (default)
  python eval/run_eval.py --mode full      # full pipeline only
  python eval/run_eval.py --mode baseline  # dense-only, no reranking
  python eval/run_eval.py --mode compare   # explicit compare

Comparing baseline vs improved system is the single most important artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Allow running as a plain script from any directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.generation.generator import generate
from src.indexing.embed import embed_texts
from src.indexing.vectorstore import get_collection
from src.pipeline import answer_question

TEST_SET_PATH = Path(__file__).parent / "test_set.json"
METRICS = ["faithfulness", "answer_relevancy", "context_precision"]


# ── Test-set loader ────────────────────────────────────────────────────────────

def load_test_set(limit: int | None = None) -> list[dict]:
    """Load test_set.json, skipping placeholder entries, optionally capped at limit."""
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        items = json.load(f)
    items = [
        item for item in items
        if not str(item.get("question", "")).startswith("FILL IN")
    ]
    if limit is not None:
        items = items[:limit]
    return items


# ── Baseline pipeline (dense-only, no BM25, no reranking) ─────────────────────

def answer_question_baseline(question: str) -> dict:
    """Vector-only retrieval with no BM25 fusion and no cross-encoder reranking."""
    qvec = embed_texts([question])
    col  = get_collection()
    n    = min(config.TOP_K_RERANK, col.count())
    res  = col.query(
        query_embeddings=qvec,
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    sources = [
        {
            "text":         doc,
            "metadata":     meta,
            "score":        float(1.0 - dist),
            "rerank_score": float(1.0 - dist),
        }
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        )
    ]
    return {"answer": generate(question, sources), "sources": sources}


# ── Row collection ─────────────────────────────────────────────────────────────

def collect_rows(items: list[dict], pipeline_fn) -> dict:
    """Run each question through pipeline_fn; return RAGAS-ready column dict."""
    rows: dict[str, list] = {
        "question":     [],
        "answer":       [],
        "contexts":     [],
        "ground_truth": [],
    }
    for item in items:
        q = item["question"]
        print(f"    {q[:68]}...")
        result = pipeline_fn(q)
        rows["question"].append(q)
        rows["answer"].append(result["answer"])
        rows["contexts"].append([s["text"] for s in result["sources"]])
        rows["ground_truth"].append(item.get("answer", item.get("ground_truth", "")))
    return rows


# ── RAGAS scoring ──────────────────────────────────────────────────────────────

def run_ragas(rows: dict) -> dict:
    """Score rows with faithfulness, answer_relevancy, context_precision.

    Uses Groq (via OpenAI-compatible endpoint) as the judge LLM and
    the project's own embedding model for answer_relevancy similarity.
    Imports are deferred so the rest of the module can load without RAGAS.
    """
    # Import directly from submodules to avoid the broken ragas.__init__
    from ragas.evaluation import evaluate, RunConfig             # noqa: PLC0415
    from ragas.metrics import (                                   # noqa: PLC0415
        faithfulness, answer_relevancy, context_precision,
    )
    from ragas.llms import LangchainLLMWrapper                   # noqa: PLC0415
    from ragas.embeddings import LangchainEmbeddingsWrapper      # noqa: PLC0415
    from langchain_openai import ChatOpenAI                      # noqa: PLC0415
    from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: PLC0415
    from datasets import Dataset                                  # noqa: PLC0415

    llm = LangchainLLMWrapper(ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        temperature=0,
    ))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    ))

    # strictness=1 → RAGAS generates one question variant instead of the default
    # 3, avoiding the n>1 completion requests that Groq rejects.
    answer_relevancy.strictness = 1

    result = evaluate(
        Dataset.from_dict(rows),
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=emb,
        raise_exceptions=False,
        run_config=RunConfig(timeout=120, max_retries=3, max_workers=1),
    )
    return {m: _safe(result[m]) for m in METRICS}


def _safe(v) -> float:
    try:
        f = float(v)
        return f if not math.isnan(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


# ── Result printing ────────────────────────────────────────────────────────────

def _print_single(scores: dict, label: str) -> None:
    print(f"\n  {label}")
    print(f"  {'Metric':<26} {'Score':>7}")
    print(f"  {'-'*35}")
    for m in METRICS:
        print(f"  {m:<26} {scores[m]:>7.3f}")
    avg = sum(scores[m] for m in METRICS) / len(METRICS)
    print(f"  {'-'*35}")
    print(f"  {'Average':<26} {avg:>7.3f}")


def _print_compare(full: dict, base: dict) -> None:
    print(f"\n  {'Metric':<26} {'Full':>7} {'Baseline':>9} {'Delta':>8}")
    print(f"  {'-'*53}")
    for m in METRICS:
        f, b = full[m], base[m]
        delta = f - b
        sign  = "+" if delta >= 0 else ""
        print(f"  {m:<26} {f:>7.3f} {b:>9.3f} {sign}{delta:>7.3f}")
    avg_f = sum(full[m] for m in METRICS) / len(METRICS)
    avg_b = sum(base[m] for m in METRICS) / len(METRICS)
    delta = avg_f - avg_b
    sign  = "+" if delta >= 0 else ""
    print(f"  {'-'*53}")
    print(f"  {'Average':<26} {avg_f:>7.3f} {avg_b:>9.3f} {sign}{delta:>7.3f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation harness for sec-rag"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "baseline", "compare"],
        default="compare",
        help="Pipeline variant(s) to evaluate (default: compare)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cap evaluation at the first N questions (default: all)",
    )
    args = parser.parse_args()

    items = load_test_set(limit=args.limit)
    if not items:
        print("No runnable questions in test_set.json (all entries are placeholders).")
        sys.exit(1)
    print(f"Loaded {len(items)} question(s) from {TEST_SET_PATH.name}\n")

    full_scores = base_scores = None

    if args.mode in ("full", "compare"):
        print(f"[Full pipeline]  {len(items)} question(s) ...")
        full_rows   = collect_rows(items, answer_question)
        print("  Scoring with RAGAS ...")
        full_scores = run_ragas(full_rows)

    if args.mode in ("baseline", "compare"):
        print(f"\n[Baseline]       {len(items)} question(s) ...")
        base_rows   = collect_rows(items, answer_question_baseline)
        print("  Scoring with RAGAS ...")
        base_scores = run_ragas(base_rows)

    print("\n" + "=" * 57)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 57)

    if args.mode == "full":
        _print_single(full_scores, "Full pipeline  (hybrid retrieval + cross-encoder reranking)")
    elif args.mode == "baseline":
        _print_single(base_scores, "Baseline  (dense-only, no BM25, no reranking)")
    else:
        _print_compare(full_scores, base_scores)

    print()


if __name__ == "__main__":
    main()
