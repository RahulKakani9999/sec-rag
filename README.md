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

# SEC 10-K RAG — Ask Questions Across Annual Reports

Ask a plain-English question about public companies' annual filings and get a
**correct, sourced answer in seconds** — instead of reading hundreds of pages.

> 🔗 **Live demo:** _add your Hugging Face Spaces / Railway link here_

---

## The problem

A company's annual report (the **10-K**) is 100–300 pages of dense financial and
legal text full of tables and footnotes. Answering a simple question — *"How much
did Apple spend on R&D in 2023 vs 2022?"* — means hunting through giant PDFs and
risking misreading a number. Across ten companies, it's hours of error-prone work.
The information is public, but practically out of reach.

## What this does

Ask a question in natural language → get an answer in seconds, with citations
pointing to the exact passages it came from, and an honest "I don't know" when the
answer isn't in the filings.

## How it works

```
EDGAR filings → parse (keep tables) → structure-aware chunks (+metadata)
   → embed + index (Chroma) + keyword index (BM25)
   → hybrid retrieval → cross-encoder rerank
   → LLM answer with enforced citations + refusal
```

_(Replace this block with a proper architecture diagram before submitting.)_

## Results

| Metric | Baseline | Final |
|---|---|---|
| Faithfulness | _fill_ | _fill_ |
| Answer relevance | _fill_ | _fill_ |
| Retrieval precision | _fill_ | _fill_ |

Measured with RAGAS on a hand-built test set of 30–50 Q/A pairs. See `eval/`.

## Run it locally

```bash
git clone https://github.com/RahulKakani9999/sec-rag.git
cd sec-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your keys
python -m src.ingestion.download
streamlit run app/streamlit_app.py
```

## Tech stack

Python · SEC EDGAR · unstructured · sentence-transformers (BGE) · ChromaDB ·
BM25 · cross-encoder reranker · Groq (Llama 3.3) · RAGAS · Streamlit · Docker

## Project structure

```
src/ingestion   download + parse filings
src/chunking    structure-aware splitting
src/indexing    embeddings + vector/keyword stores
src/retrieval   hybrid search + reranking
src/generation  prompt, LLM call, citations
eval/           test set + RAGAS scoring
app/            Streamlit UI
config.py       all settings in one place
```
