"""Generation-only spot check — no RAGAS scoring."""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from src.pipeline import answer_question

IDS = ["lookup_14", "calc_06", "calc_07"]

test_set = json.loads(
    (__import__("pathlib").Path(__file__).parent / "test_set.json").read_text(encoding="utf-8")
)
items = [q for q in test_set if q.get("id") in IDS]
items.sort(key=lambda q: IDS.index(q["id"]))

for item in items:
    qid      = item["id"]
    question = item["question"]
    ref      = item.get("answer", item.get("ground_truth", ""))

    print("=" * 78)
    print(f"  {qid}  [{item.get('category','')}]")
    print(f"  Q: {question}")
    print(f"  Ref: {ref}")
    print("=" * 78)

    result = answer_question(question)
    answer = result["answer"]

    print(f"\n  ANSWER:\n")
    for line in answer.splitlines():
        print(f"    {line}")

    print(f"\n  --- top-5 chunks used ---")
    for i, src in enumerate(result["sources"], 1):
        meta    = src["metadata"]
        section = meta.get("section", "?")
        snippet = src["text"][:120].replace("\n", " ")
        print(f"  [{i}] {section}  |  {snippet} ...")

    print()
