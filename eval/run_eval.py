"""Run the test set through the pipeline and score with RAGAS.

Usage:
  python eval/run_eval.py                            # compare full vs baseline, all questions
  python eval/run_eval.py --mode full --limit 20     # first 20 questions, full pipeline
  python eval/run_eval.py --mode full --category unanswerable --ids calc_01 calc_02
  python eval/run_eval.py --mode full --ids lookup_01 lookup_02 reasoning_03
  python eval/run_eval.py --mode full --category direct_lookup --delay 2

Flags:
  --mode          full | baseline | compare  (default: compare)
  --limit N       cap at first N questions after other filters
  --ids           one or more question IDs to include
  --category      one or more category names to include
                  (--ids and --category are combined with OR)
  --delay N       seconds to sleep between generation calls (default: 1)
  --judge-model   Groq model ID for RAGAS judge calls (default: config.LLM_JUDGE_MODEL)
                  Generation always uses config.LLM_MODEL regardless of this flag.
  --summary       print averages from eval/results.jsonl and exit (no scoring)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.generation.generator import generate
from src.indexing.embed import embed_texts
from src.indexing.vectorstore import get_collection
from src.pipeline import answer_question

TEST_SET_PATH = Path(__file__).parent / "test_set.json"
RESULTS_PATH  = Path(__file__).parent / "results.jsonl"
METRICS = ["faithfulness", "answer_relevancy", "context_precision"]


# ── Test-set loader ────────────────────────────────────────────────────────────

def load_test_set(
    limit: int | None = None,
    ids: list[str] | None = None,
    categories: list[str] | None = None,
) -> list[dict]:
    """Load and filter test_set.json.

    ids and categories are combined with OR: a question is included if its id
    is in ids OR its category is in categories.  If neither is given, all
    non-placeholder questions are returned.  limit is applied last.
    """
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        items = json.load(f)

    items = [
        item for item in items
        if not str(item.get("question", "")).startswith("FILL IN")
    ]

    if ids or categories:
        ids_set  = set(ids or [])
        cats_set = set(categories or [])
        items = [
            item for item in items
            if item.get("id") in ids_set or item.get("category") in cats_set
        ]

    if limit is not None:
        items = items[:limit]

    return items


# ── Baseline pipeline ──────────────────────────────────────────────────────────

def answer_question_baseline(question: str) -> dict:
    """Dense-only retrieval — no BM25 fusion, no cross-encoder reranking."""
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

def collect_rows(
    items: list[dict],
    pipeline_fn,
    delay: float = 1.0,
) -> dict:
    """Run each question through pipeline_fn; return RAGAS-ready column dict.

    delay (seconds) is inserted between calls to avoid triggering per-minute
    token rate limits on the generation model.
    """
    rows: dict[str, list] = {
        "question":     [],
        "answer":       [],
        "contexts":     [],
        "ground_truth": [],
    }
    for i, item in enumerate(items):
        if i > 0 and delay > 0:
            time.sleep(delay)
        q = item["question"]
        print(f"  [{i+1:>2}/{len(items)}] {q[:70]}...")
        result = pipeline_fn(q)
        rows["question"].append(q)
        rows["answer"].append(result["answer"])
        rows["contexts"].append([s["text"] for s in result["sources"]])
        rows["ground_truth"].append(item.get("answer", item.get("ground_truth", "")))
    return rows


# ── RAGAS scoring ──────────────────────────────────────────────────────────────

def run_ragas(rows: dict, judge_model: str | None = None) -> tuple[dict, object]:
    """Score rows; return (aggregate_scores_dict, per_sample_dataframe).

    judge_model overrides config.LLM_JUDGE_MODEL for this call.
    """
    from ragas.evaluation import evaluate, RunConfig             # noqa: PLC0415
    from ragas.metrics import (                                   # noqa: PLC0415
        faithfulness, answer_relevancy, context_precision,
    )
    from ragas.llms import LangchainLLMWrapper                   # noqa: PLC0415
    from ragas.embeddings import LangchainEmbeddingsWrapper      # noqa: PLC0415
    from langchain_openai import ChatOpenAI                      # noqa: PLC0415
    from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: PLC0415
    from datasets import Dataset                                  # noqa: PLC0415

    model_name = judge_model or config.LLM_JUDGE_MODEL
    print(f"  Judge model: {model_name}")
    llm = LangchainLLMWrapper(ChatOpenAI(
        model=model_name,
        api_key=config.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        temperature=0,
    ))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    ))

    answer_relevancy.strictness = 1  # avoid n>1 requests that Groq rejects

    result = evaluate(
        Dataset.from_dict(rows),
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=emb,
        raise_exceptions=False,
        run_config=RunConfig(timeout=120, max_retries=3, max_workers=1),
    )
    agg = {m: _safe(result[m]) for m in METRICS}
    return agg, result.to_pandas()


def _safe(v) -> float:
    try:
        f = float(v)
        return f if not math.isnan(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


# ── Persistent results (JSONL) ─────────────────────────────────────────────────

def load_results() -> dict[str, dict]:
    """Return {question_id: record} from results.jsonl (empty dict if file absent)."""
    if not RESULTS_PATH.exists():
        return {}
    records: dict[str, dict] = {}
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records[rec["id"]] = rec
    return records


def build_records(items: list[dict], df) -> list[dict]:
    """Zip scored items with their RAGAS DataFrame rows into saveable dicts."""
    records = []
    for item, (_, row) in zip(items, df.iterrows()):
        records.append({
            "id":                item.get("id", ""),
            "category":          item.get("category", ""),
            "question":          item["question"],
            "answer":            str(row.get("answer", "")),
            "faithfulness":      _safe(row.get("faithfulness",      float("nan"))),
            "answer_relevancy":  _safe(row.get("answer_relevancy",  float("nan"))),
            "context_precision": _safe(row.get("context_precision", float("nan"))),
        })
    return records


def append_results(records: list[dict]) -> None:
    """Append records to results.jsonl (one JSON object per line)."""
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ── Printing ───────────────────────────────────────────────────────────────────

_COL = {m: min(m, key=lambda _: 0) for m in METRICS}   # placeholder

def _metric_abbrev(m: str) -> str:
    return {"faithfulness": "Faith", "answer_relevancy": "Relev", "context_precision": "Prec"}[m]


def _print_per_question(df, items: list[dict]) -> None:
    """Print a row per question showing metric scores and a generated-answer snippet."""
    abbrevs = [_metric_abbrev(m) for m in METRICS]
    hdr = f"  {'#':>3}  {'ID':<17} " + "  ".join(f"{a:>6}" for a in abbrevs) + "  Generated answer"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2 + 30))

    for i, (item, (_, row)) in enumerate(zip(items, df.iterrows()), 1):
        scores = [_safe(row.get(m, float("nan"))) for m in METRICS]
        qid    = item.get("id", f"q{i:02d}")
        cat    = item.get("category", "")

        # Collapse whitespace in the generated answer for a clean one-liner
        ans = str(row.get("answer", "")).replace("\n", " ").replace("\t", " ")
        # Truncate cleanly
        ans_display = ans[:65] + ("…" if len(ans) > 65 else "")

        score_str = "  ".join(f"{s:>6.3f}" for s in scores)
        print(f"  {i:>3}. {qid:<17} {score_str}  {ans_display!r}")

    print()


def _print_aggregate(scores: dict, label: str) -> None:
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


def _print_summary(records: dict[str, dict]) -> None:
    """Print overall and per-category averages from the accumulated results file."""
    if not records:
        print(f"No results found in {RESULTS_PATH}. Run --mode full first.")
        return

    rows = list(records.values())
    abbrevs = [_metric_abbrev(m) for m in METRICS]

    print(f"\nResults file : {RESULTS_PATH}")
    print(f"Questions    : {len(rows)}\n")

    from collections import defaultdict
    by_cat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cat[r.get("category", "?")].append(r)

    hdr = f"  {'Category':<22} {'N':>4}  " + "  ".join(f"{a:>6}" for a in abbrevs) + "  Avg"
    print(hdr)
    print("  " + "-" * (len(hdr) + 2))

    overall_vals: dict[str, list] = {m: [] for m in METRICS}
    for cat in sorted(by_cat.keys()):
        cat_rows = by_cat[cat]
        means = {m: sum(r[m] for r in cat_rows) / len(cat_rows) for m in METRICS}
        for m in METRICS:
            overall_vals[m].extend(r[m] for r in cat_rows)
        row_avg = sum(means[m] for m in METRICS) / len(METRICS)
        score_str = "  ".join(f"{means[m]:>6.3f}" for m in METRICS)
        print(f"  {cat:<22} {len(cat_rows):>4}  {score_str}  {row_avg:.3f}")

    print("  " + "-" * (len(hdr) + 2))
    overall_means = {
        m: sum(overall_vals[m]) / len(overall_vals[m]) if overall_vals[m] else 0.0
        for m in METRICS
    }
    overall_avg = sum(overall_means[m] for m in METRICS) / len(METRICS)
    score_str = "  ".join(f"{overall_means[m]:>6.3f}" for m in METRICS)
    print(f"  {'OVERALL':<22} {len(rows):>4}  {score_str}  {overall_avg:.3f}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation harness for sec-rag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["full", "baseline", "compare"],
                        default="compare",
                        help="Pipeline variant(s) to evaluate (default: compare)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Cap at first N questions after other filters")
    parser.add_argument("--ids", nargs="+", default=None, metavar="ID",
                        help="Include specific question IDs (OR-combined with --category)")
    parser.add_argument("--category", nargs="+", default=None, metavar="CAT",
                        help="Include questions whose category matches (OR with --ids)")
    parser.add_argument("--delay", type=float, default=1.0, metavar="SEC",
                        help="Seconds between generation calls (default: 1)")
    parser.add_argument("--judge-model", default=None, metavar="MODEL",
                        help="Groq model for RAGAS judge calls (default: config.LLM_JUDGE_MODEL)")
    parser.add_argument("--summary", action="store_true",
                        help="Print averages from results.jsonl and exit (no scoring)")
    args = parser.parse_args()

    # ── Summary-only mode ──
    if args.summary:
        _print_summary(load_results())
        return

    items = load_test_set(limit=args.limit, ids=args.ids, categories=args.category)
    if not items:
        print("No matching questions found. Check --ids / --category values.")
        sys.exit(1)

    cats = sorted({item.get("category", "?") for item in items})
    print(f"Loaded {len(items)} question(s) from test_set.json  "
          f"[categories: {', '.join(cats)}]")

    # ── Skip already-scored questions (full pipeline only) ──
    already_scored: set[str] = set()
    if args.mode in ("full", "compare"):
        already_scored = set(load_results().keys())
        if already_scored:
            before = len(items)
            items = [it for it in items if it.get("id") not in already_scored]
            skipped = before - len(items)
            if skipped:
                print(f"Skipping {skipped} already-scored question(s) "
                      f"(found in {RESULTS_PATH.name}).")
            if not items:
                print("All selected questions are already scored. "
                      "Use --summary to view results.")
                sys.exit(0)

    print()
    full_scores = full_df = base_scores = base_df = None

    if args.mode in ("full", "compare"):
        print(f"[Full pipeline]  generating answers (delay={args.delay}s) ...")
        full_rows = collect_rows(items, answer_question, delay=args.delay)
        print("\n  Scoring with RAGAS ...")
        full_scores, full_df = run_ragas(full_rows, judge_model=args.judge_model)
        new_records = build_records(items, full_df)
        append_results(new_records)
        print(f"  Saved {len(new_records)} record(s) → {RESULTS_PATH.name}")

    if args.mode in ("baseline", "compare"):
        print(f"\n[Baseline]  generating answers (delay={args.delay}s) ...")
        base_rows = collect_rows(items, answer_question_baseline, delay=args.delay)
        print("\n  Scoring with RAGAS ...")
        base_scores, base_df = run_ragas(base_rows, judge_model=args.judge_model)

    # ── Per-question results ──
    print("\n" + "=" * 70)
    print("  PER-QUESTION RESULTS")
    print("=" * 70)

    if args.mode in ("full", "compare") and full_df is not None:
        print("\n  Full pipeline:")
        _print_per_question(full_df, items)

    if args.mode in ("baseline", "compare") and base_df is not None:
        print("\n  Baseline:")
        _print_per_question(base_df, items)

    # ── Aggregate summary ──
    print("=" * 70)
    print("  AGGREGATE SCORES")
    print("=" * 70)

    if args.mode == "full":
        _print_aggregate(full_scores, "Full pipeline  (hybrid retrieval + cross-encoder reranking)")
    elif args.mode == "baseline":
        _print_aggregate(base_scores, "Baseline  (dense-only, no BM25, no reranking)")
    else:
        _print_compare(full_scores, base_scores)

    print()


if __name__ == "__main__":
    main()
