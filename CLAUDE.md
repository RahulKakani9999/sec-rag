# Project guidance for Claude Code

## Attribution
- Do NOT add Claude as a co-author on commits.
- Do NOT include "Generated with Claude Code" or any Claude/Anthropic
  attribution line in commit messages.
- Keep commit messages plain and conventional (e.g. "feat: add hybrid retriever").

## Project context
This is a production RAG system over SEC 10-K filings. Pipeline stages:
ingestion -> chunking -> indexing -> retrieval -> reranking -> generation,
with an evaluation harness (RAGAS) and a Streamlit UI.

## Code conventions
- All tunable settings live in config.py — do not hardcode model names,
  chunk sizes, or top-k values elsewhere.
- Each pipeline stage stays in its own module under src/.
- Write small, testable functions. Prefer clarity over cleverness.
- Never commit data/ contents, .env, or API keys.

## When writing commits
- One logical change per commit.
- Use conventional prefixes: feat, fix, refactor, docs, test, chore.
