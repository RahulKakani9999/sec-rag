---
title: SEC 10-K Research Assistant
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.31.0
app_file: app/streamlit_app.py
pinned: false
---

# SEC 10-K Research Assistant

A production RAG pipeline for querying SEC 10-K annual filings. Ask a question in plain English, get a cited answer grounded in the actual filing, or an honest refusal when the answer isn't there.

🔗 **[Live demo on Streamlit Cloud](https://your-app-url.streamlit.app)** ← replace with your URL after deploy

---

## The problem

A 10-K filing is 100–300 pages of dense prose, legal boilerplate, and financial tables. A question like *"What were Apple's total assets?"* requires knowing exactly which table to look in, and the answer sits beside dozens of similar-looking numbers. The iXBRL format SEC requires adds an extra layer: the HTML embeds hidden XBRL metadata elements that must be stripped before text extraction. Doing this accurately at scale — across multiple companies and filing years — rules out simple keyword search.

---

## What it does

Ask a plain-English question → get a grounded answer with inline source citations (`[Source N]`) pointing to the exact filing passages used. If the answer cannot be supported by the retrieved context, the system refuses rather than hallucinating: *"I don't know based on the provided filings."* Arithmetic on figures present in the context (totals, differences, percentages) is permitted and cited as derived.

---

## Architecture

```
SEC EDGAR
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Ingestion                                                   │
│  sec-edgar-downloader → full-submission.txt (SGML)         │
│  BeautifulSoup + lxml: strip display:none iXBRL metadata,  │
│  convert <table> → tab-separated rows, extract body text   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Structure-aware chunking                                    │
│  Split at "Item N." section headers (regex + \xa0 aware)   │
│  Within each section: tables always emitted whole;         │
│  short text buffers (section labels, date rows) fused      │
│  into the following table so data is never orphaned        │
│  from its header.  CHUNK_SIZE=800 tok, OVERLAP=100 tok     │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌─────────────────────┐  ┌──────────────────────────────────┐
│ Dense index         │  │ Sparse index                     │
│ BAAI/bge-small-     │  │ BM25Okapi (rank-bm25)            │
│ en-v1.5 (384-dim,   │  │ pickled to chroma_db/bm25.pkl   │
│ L2-normalised)      │  │                                  │
│ ChromaDB cosine     │  │                                  │
└──────────┬──────────┘  └──────────────────┬───────────────┘
           │                                │
           └──────────────┬─────────────────┘
                          │   QUERY TIME
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Hybrid retrieval                                            │
│  Dense:  embed query → Chroma top-40 by cosine             │
│  Sparse: tokenise query → BM25 top-40 by TF-IDF score      │
│  Fuse:   Reciprocal Rank Fusion (k=60) → top-20 candidates │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Cross-encoder reranking                                     │
│  cross-encoder/ms-marco-MiniLM-L-6-v2                      │
│  Scores all 20 (query, chunk) pairs; returns top 5         │
│  Tabs normalised to spaces before scoring (table fix)      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Grounded generation                                         │
│  Groq API: llama-3.3-70b-versatile                         │
│  System prompt: cite every claim as [Source N];            │
│  refuse if answer not derivable from context;              │
│  arithmetic on explicit context figures is permitted       │
└─────────────────────────────────────────────────────────────┘
```

**Why hybrid retrieval?** Dense embeddings generalise well (synonym matching, paraphrase) but can miss exact financial terms. BM25 is precise on exact strings like ticker symbols, line-item names, and fiscal dates. RRF fusion gets both benefits without requiring a learned combination weight.

**Why reranking?** The cross-encoder sees the full (query, passage) pair and is significantly more accurate than cosine similarity at deciding relevance — but too slow to run over the whole corpus. The two-stage pipeline keeps latency manageable: cheap approximate retrieval narrows to 20 candidates, then the cross-encoder scores only those 20.

---

## Evaluation

Evaluated on a hand-built test set of 50 question–answer pairs across five categories, scored with [RAGAS](https://github.com/explodinggradients/ragas) (faithfulness, answer relevancy, context precision) using `llama-3.1-8b-instant` as the judge model.

### Scores by category (baseline, 50 questions)

| Category | N | Faithfulness | Answer Relevancy | Context Precision | Avg |
|---|---|---|---|---|---|
| reasoning | 10 | 0.799 | 0.837 | 0.912 | **0.849** |
| comparison / trend | 5 | 0.750 | 0.746 | 0.936 | **0.811** |
| calculation | 10 | 0.713 | 0.735 | 0.880 | **0.776** |
| direct lookup | 15 | 0.565 | 0.907 | 0.666 | **0.713** |
| unanswerable | 10 | — | — | — | *see below* |
| **OVERALL** | **50** | **0.687** | **0.702** | **0.652** | **0.680** |

### Unanswerable / refusal accuracy

All 10 unanswerable questions were correctly refused: **10/10 (100%)**.

RAGAS metrics are not reported for this category because the framework is designed around answers that ground claims against retrieved context. A correct refusal — "I don't know based on the provided filings." — contains no extractable claims, so faithfulness is undefined, answer relevancy is near zero (the LLM judge generates a synthetic question from the refusal text, which matches nothing), and context precision is 0 by design. The right metric here is simply: did the system refuse when it should? It did, every time.

### Notes on the direct-lookup faithfulness score

Direct lookups score the lowest on faithfulness (0.565) despite nearly all answers being factually correct. The root cause is a judge-model artifact: the 8B judge model decomposes a one-sentence answer like *"Apple's total net sales were \$416,161 million [Source 1]."* into two sub-claims — the numerical fact and the citation tag — then fails to ground the citation string against the retrieved passage. This consistently produces faithfulness = 0.5 rather than 1.0 for correct single-fact answers. Reasoning and calculation questions score higher because their multi-sentence answers generate more genuine verifiable claims, diluting the effect.

### Retrieval bug: lookup\_14 (total assets)

One lookup question — *"What were Apple's total assets as of September 27, 2025?"* — produced a refusal in the baseline despite the answer ($359,241 million) being in the filing. Tracing the retrieval pipeline revealed the cause:

The greedy packer in `chunker.py` emitted the balance-sheet section header ("CONSOLIDATED BALANCE SHEETS / September 27, 2025") as a standalone text chunk immediately before emitting the data table as a separate chunk. The data chunk then had no descriptive text to anchor its embedding — it was just a grid of numbers. Without "balance sheet" or "September 27, 2025" in its text, the cross-encoder scored it below the top-5 cut-off (rank 15/20, score 4.29) when the query mentioned the date.

**Fix:** Modified `_pack()` in `chunker.py` to fold short text buffers (≤120 chars — section labels and date-header rows) into the following table chunk rather than emitting them separately. After re-indexing, the fused chunk ranked 1st (score 8.49) for the total-assets query. Confirmed via generation-only test: lookup\_14 now correctly answers *"Apple's total assets as of September 27, 2025, were \$359,241 million [Source 1]."*

---

## Known limitations

**Single filing indexed.** The deployed app indexes only the Apple 2025 10-K (163 chunks). The pipeline is designed for multi-company, multi-year indexing — config lists AAPL, MSFT, GOOGL, NVDA, AMZN — but those filings have not been ingested into the deployed instance.

**One calculation question still fails (calc\_06: R&D dollar increase).** The question asks for the dollar change in R&D expense from 2024 to 2025. The income-statement table chunk containing `Research and development: $34,550M (2025) / $31,370M (2024)` does not reach the top-5 reranked results. The query token `"r&d"` is a single BM25 token that never matches `"Research"` or `"development"` separately, and the dense embedding isn't specific enough to elevate it above narrative MD&A chunks. The cross-encoder would score it correctly if it made the candidate pool — it doesn't. Increasing `TOP_K_RETRIEVE` from 20 to 40 brings the chunk into the pool at position 33, but it still doesn't survive the top-5 cutoff. A query expansion or synonym injection step would address this.

**Free-tier API quota limits the eval throughput.** Groq's free tier allows 100k tokens/day on `llama-3.3-70b-versatile`. A full 50-question eval requires ~2 days of quota. The RAGAS judge calls use a separate model (`llama-3.1-8b-instant`) to avoid exhausting the generation quota, but the day-limit still constrains how quickly before/after comparisons can be run.

**RAGAS scores for the chunker fix are not yet available.** The option-3 improvement was confirmed functionally (generation-only test), but the full 50-question re-eval ran into the daily quota limit and was not completed. The scores in the table above reflect the pre-fix baseline.

---

## Tech stack

| Layer | Tool |
|---|---|
| Filing download | `sec-edgar-downloader` |
| HTML parsing | `BeautifulSoup` + `lxml` |
| Embedding model | `BAAI/bge-small-en-v1.5` (sentence-transformers, 384-dim) |
| Vector store | `ChromaDB` (cosine space, PersistentClient) |
| Keyword index | `BM25Okapi` (rank-bm25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `llama-3.3-70b-versatile` via Groq API |
| Eval framework | RAGAS 0.4.x |
| UI | Streamlit |

---

## Run locally

```bash
git clone https://github.com/your-username/sec-rag.git
cd sec-rag

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements_dev.txt

# Set your Groq API key
cp .env.example .env   # then edit .env and add GROQ_API_KEY=gsk_...
# or just:
echo "GROQ_API_KEY=gsk_your_key" > .env

# The pre-built index is included in the repo (chroma_db/).
# To re-ingest and re-index from scratch:
python -m src.ingestion.download     # downloads filings to data/raw/
python -m src.indexing.vectorstore   # parses, chunks, embeds, indexes

# Run the app
streamlit run app/streamlit_app.py
```

The pre-built index is committed to the repo (`chroma_db/`), so `streamlit run` works immediately without re-ingesting.

To reproduce the evaluation:

```bash
# Full 50-question eval (requires ~100k Groq tokens — may span 2 days on free tier)
python eval/run_eval.py --mode full --delay 13

# Summary of accumulated results
python eval/run_eval.py --summary
```
