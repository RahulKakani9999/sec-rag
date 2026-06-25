"""Build the prompt, call the LLM, enforce citations and refusal."""
from __future__ import annotations

import groq

import config

_client: groq.Groq | None = None


def _get_client() -> groq.Groq:
    global _client
    if _client is None:
        _client = groq.Groq(api_key=config.GROQ_API_KEY)
    return _client


# ── Prompt construction ───────────────────────────────────────────────────────

_SYSTEM = """\
You are a financial analyst assistant. You answer questions exclusively from \
the SEC 10-K filing excerpts supplied in the CONTEXT block below.

Rules you must follow without exception:
1. Base every claim on the provided excerpts only. Do not use outside knowledge.
2. After each factual claim, cite the source like this: [Source N].
3. If the context does not contain enough information to answer the question, \
respond with exactly: "I don't know based on the provided filings."
4. Never speculate, infer, or guess beyond what is explicitly stated.
"""


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        header = (
            f"[Source {i}] "
            f"{meta.get('ticker', '?')} {meta.get('year', '?')} — "
            f"{meta.get('section', '?')}"
        )
        context_parts.append(f"{header}\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)
    return f"CONTEXT:\n\n{context}\n\nQUESTION: {question}"


# ── Public API ────────────────────────────────────────────────────────────────

def generate(question: str, chunks: list[dict]) -> str:
    """Call the LLM and return a cited answer grounded in chunks.

    If chunks is empty the model is told there is no context, which triggers
    the refusal clause in the system prompt.
    """
    user_prompt = _build_prompt(question, chunks)

    response = _get_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,  # deterministic — citations must be precise
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()
