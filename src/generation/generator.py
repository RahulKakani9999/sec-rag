"""Build the prompt, call the LLM, enforce citations and refusal."""
from __future__ import annotations

import re
import time

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
your entire response must be exactly: "I don't know based on the provided filings." \
Do not use this phrase when you have already computed or stated a correct answer — \
it is a replacement for an answer, not a suffix to one.
4. Never speculate, infer, or guess about facts that are not present in the context.
5. You MAY perform arithmetic (subtraction, addition, percentages, differences) on \
numbers that appear explicitly in the context. A result derived by arithmetic from \
context figures is grounded — do not follow it with a refusal.
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

# Parses "try again in 17m28.896s" (minutes+seconds) or "try again in 28.896s"
_RETRY_RE_MS = re.compile(r"try again in (\d+)m([\d.]+)s", re.IGNORECASE)
_RETRY_RE_S  = re.compile(r"try again in ([\d.]+)s",        re.IGNORECASE)
_MAX_RETRIES = 6   # more retries to ride out TPD rolling-window waits


def _parse_wait(exc_str: str) -> float:
    """Extract wait seconds from a Groq rate-limit error message."""
    m = _RETRY_RE_MS.search(exc_str)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2)) + 5.0
    m = _RETRY_RE_S.search(exc_str)
    if m:
        return float(m.group(1)) + 2.0
    return 20.0


def generate(question: str, chunks: list[dict]) -> str:
    """Call the LLM and return a cited answer grounded in chunks.

    If chunks is empty the model is told there is no context, which triggers
    the refusal clause in the system prompt.  Retries automatically on
    transient TPM/TPD 429s, sleeping for the retry duration stated in the error.
    """
    user_prompt = _build_prompt(question, chunks)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user_prompt},
    ]

    for attempt in range(_MAX_RETRIES):
        try:
            response = _get_client().chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except groq.RateLimitError as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _parse_wait(str(exc))
            print(f"    [rate limit] waiting {wait:.0f}s before retry {attempt + 2}/{_MAX_RETRIES} ...")
            time.sleep(wait)

    raise RuntimeError("generate() exhausted retries")
