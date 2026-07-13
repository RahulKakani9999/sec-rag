"""Compare baseline (results.jsonl) vs option-3 (results_after.jsonl)."""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

BASE  = Path(__file__).parent / "results.jsonl"
OPT3  = Path(__file__).parent / "results_after.jsonl"
METRICS = ["faithfulness", "answer_relevancy", "context_precision"]
THRESHOLD = 0.05


def load(p):
    recs = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            recs[r["id"]] = r
    return recs


baseline = load(BASE)
opt3     = load(OPT3)
ids = sorted(opt3.keys(), key=lambda x: (x.split("_")[0], int(x.split("_")[1])))

improved, regressed = [], []

W = 94
print("=" * W)
print("  OPTION 3 vs BASELINE  |  faithfulness / relevancy / precision")
print("  Baseline = results.jsonl  |  Option 3 = results_after.jsonl")
print("=" * W)
print()
print("  {:<17}  {:6}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}".format(
    "ID", "", "Faith", "Relev", "Prec", "Avg", "D-Avg"))
print("  " + "-" * 72)

for qid in ids:
    b = baseline.get(qid)
    a = opt3[qid]
    a_s = [a[m] for m in METRICS]
    a_avg = sum(a_s) / 3

    if b:
        b_s   = [b[m] for m in METRICS]
        b_avg = sum(b_s) / 3
        delta = a_avg - b_avg

        def fmt(av, bv):
            d = av - bv
            s = "{:7.3f}".format(av)
            if   d >=  THRESHOLD: return s + "(+)"
            elif d <= -THRESHOLD: return s + "(-)"
            return s + "   "

        b_line = "  ".join("{:7.3f}".format(v) for v in b_s)
        a_line = "  ".join(fmt(a_s[i], b_s[i]) for i in range(3))
        d_str  = "{:+.3f}".format(delta)
        print("  {:<17}  BEFORE  {}  {:7.3f}".format(qid, b_line, b_avg))
        print("  {:17}   OPT3   {}  {:7.3f}  {}".format("", a_line, a_avg, d_str))
        if   delta >=  THRESHOLD: improved.append((qid, delta))
        elif delta <= -THRESHOLD: regressed.append((qid, delta))
    else:
        a_line = "  ".join("{:7.3f}".format(v) for v in a_s)
        print("  {:<17}   OPT3   {}  {:7.3f}  (no baseline)".format(qid, a_line, a_avg))
    print()

# Aggregate
print("=" * W)
print("  AGGREGATE (these 17 questions)")
print("=" * W)
b_rows = [baseline[q] for q in ids if q in baseline]
a_rows = [opt3[q]     for q in ids]
print("  {:<22}  {:>9}  {:>9}  {:>7}".format("Metric", "Baseline", "Option-3", "Delta"))
print("  " + "-" * 52)
for m in METRICS:
    bv = sum(r[m] for r in b_rows) / len(b_rows)
    av = sum(r[m] for r in a_rows) / len(a_rows)
    print("  {:<22}  {:9.3f}  {:9.3f}  {:+7.3f}".format(m, bv, av, av - bv))
b_all = sum(sum(r[m] for m in METRICS) / 3 for r in b_rows) / len(b_rows)
a_all = sum(sum(r[m] for m in METRICS) / 3 for r in a_rows) / len(a_rows)
print("  " + "-" * 52)
print("  {:<22}  {:9.3f}  {:9.3f}  {:+7.3f}".format("Average", b_all, a_all, a_all - b_all))

print()
print("  Improved  (delta-avg >= +{:.2f}): {}".format(THRESHOLD, len(improved)))
for qid, d in improved:
    print("    {}  {:+.3f}".format(qid, d))
print("  Regressed (delta-avg <= -{:.2f}): {}".format(THRESHOLD, len(regressed)))
for qid, d in regressed:
    print("    {}  {:+.3f}".format(qid, d))

# lookup_14 spotlight
print()
print("=" * W)
print("  LOOKUP_14 — 'What were Apple's total assets as of September 27, 2025?'")
print("=" * W)
if "lookup_14" in opt3:
    rec = opt3["lookup_14"]
    ans = rec["answer"]
    found = "359,241" in ans or "359241" in ans.replace(",", "")
    print("  Option-3 answer:")
    for line in ans[:400].splitlines():
        print("    " + line)
    print()
    print("  Scores: Faith={:.3f}  Relev={:.3f}  Prec={:.3f}".format(
        rec["faithfulness"], rec["answer_relevancy"], rec["context_precision"]))
    print()
    print("  >>> Contains '$359,241 million'? {}".format("YES — FIXED" if found else "NO — still wrong"))
if "lookup_14" in baseline:
    b = baseline["lookup_14"]
    print()
    print("  Baseline answer (first 150 chars):")
    print("    " + b["answer"][:150])
    print("  Baseline scores: Faith={:.3f}  Relev={:.3f}  Prec={:.3f}".format(
        b["faithfulness"], b["answer_relevancy"], b["context_precision"]))
