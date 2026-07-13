"""Structure-aware chunking with company/year/section metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import config

# ── Section detection ─────────────────────────────────────────────────────────

# Matches "Item 1." / "Item 1A." with \xa0 or regular whitespace before title.
# Excludes tab-containing lines so TOC rows are never treated as headers.
_ITEM_RE = re.compile(
    r"^Item\s+(?P<num>\d+[A-C]?)\.[\s\xa0]*(?P<title>[^\t\n]*?)\s*$",
    re.IGNORECASE,
)

_ITEM_NAMES: dict[str, str] = {
    "1":  "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2":  "Properties",
    "3":  "Legal Proceedings",
    "4":  "Mine Safety Disclosures",
    "5":  "Market for Common Equity",
    "6":  "Selected Financial Data",
    "7":  "MD&A",
    "7A": "Market Risk",
    "8":  "Financial Statements",
    "9":  "Changes in Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Foreign Jurisdictions",
    "10": "Directors and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Related Transactions",
    "14": "Principal Accountant Fees",
    "15": "Exhibits and Financial Statements",
    "16": "Form 10-K Summary",
}


def _section_label(num: str, inline_title: str) -> str:
    num = num.upper()
    name = _ITEM_NAMES.get(num) or inline_title.strip() or f"Item {num}"
    return f"Item {num} – {name}"


# ── Core data type ────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tok(text: str) -> int:
    """Approximate token count at ~4 chars/token."""
    return max(1, len(text) // 4)


def _is_table_line(line: str) -> bool:
    return "\t" in line


# Character-length thresholds for the statement-structure fusion fix.
# Financial statement date-header rows ("September 27, 2025  September 28, 2024")
# and section labels ("ASSETS:", "LIABILITIES AND SHAREHOLDERS' EQUITY:") are
# always short.  Keeping these thresholds conservative avoids touching regular
# text paragraphs or multi-row tables.
_SMALL_BUF_CHARS = 120   # text buffers ≤ this are fused into the next table
_TINY_TABLE_CHARS = 120  # table chunks ≤ this don't reset overlap_tail


# ── Section splitting ─────────────────────────────────────────────────────────

def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return [(section_name, section_text), ...] split on Item N. headers."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_name = "Preamble"
    buf: list[str] = []

    for line in lines:
        # Only test lines that contain no tabs (filters out TOC and table rows)
        if "\t" not in line:
            m = _ITEM_RE.match(line.strip())
            if m:
                sections.append((current_name, buf))
                current_name = _section_label(m.group("num"), m.group("title"))
                buf = [line]
                continue
        buf.append(line)

    sections.append((current_name, buf))
    return [(name, "\n".join(ls)) for name, ls in sections if ls]


# ── Unit grouping (table blocks vs. text paragraphs) ─────────────────────────

def _group_units(section_text: str) -> list[tuple[bool, str]]:
    """Split section text into (is_table, text) units.

    Consecutive tab-containing lines are kept as one table block so they are
    never broken across chunk boundaries.  Non-table text is split at blank
    lines so individual paragraphs can be packed independently.
    """
    lines = section_text.splitlines()
    units: list[tuple[bool, str]] = []
    buf: list[str] = []
    in_table = False

    def _flush() -> None:
        if buf:
            units.append((in_table, "\n".join(buf)))
        buf.clear()

    for line in lines:
        is_tab = _is_table_line(line)

        if is_tab != in_table:
            _flush()
            in_table = is_tab

        if not in_table and not line.strip():
            # Blank line in text: paragraph boundary — emit what we have
            _flush()
        else:
            buf.append(line)

    _flush()
    return [(is_tbl, txt) for is_tbl, txt in units if txt.strip()]


# ── Text splitter (fallback for single-newline-separated blobs) ───────────────

def _split_text(text: str, max_chars: int) -> list[str]:
    """Break a long text at newlines so each piece is ≤ max_chars.

    BeautifulSoup's get_text(separator='\\n') never produces blank lines, so
    large sections arrive as one continuous blob.  We split at the last newline
    before the limit; if no newline exists we fall back to a hard character cut.
    """
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    while len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return [p for p in parts if p]


# ── Greedy packer ─────────────────────────────────────────────────────────────

def _pack(
    units: list[tuple[bool, str]],
    base_meta: dict,
    size: int,
    overlap: int,
) -> list[Chunk]:
    """Greedy-pack units into chunks; tables are always kept whole."""
    size_chars = size * 4
    overlap_chars = overlap * 4

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_chars = 0
    overlap_tail = ""

    def _emit() -> None:
        nonlocal buf, buf_chars, overlap_tail
        combined = "\n\n".join(p for p in buf if p.strip()).strip()
        if combined:
            chunks.append(Chunk(text=combined, metadata=dict(base_meta)))
            overlap_tail = combined[-overlap_chars:]
        buf = []
        buf_chars = 0

    for is_table, text in units:
        text_chars = len(text)

        if is_table:
            buf_combined = "\n\n".join(p for p in buf if p.strip()).strip()
            if buf and buf_chars <= _SMALL_BUF_CHARS:
                # Small text buffer (e.g. "ASSETS:", "LIABILITIES:") — fuse it
                # with the following table instead of emitting an orphaned chunk.
                context = (overlap_tail + "\n\n" + buf_combined).strip() if overlap_tail else buf_combined
                table_text = (context + "\n\n" + text).strip() if context else text
                buf = []
                buf_chars = 0
                chunks.append(Chunk(text=table_text, metadata=dict(base_meta)))
                # After fusing, reset overlap so the next fused section label
                # doesn't inherit raw numeric table lines as its preamble.
                # Tiny tables (date-header rows) keep overlap so the statement
                # name still carries forward to the next data table.
                if len(text) > _TINY_TABLE_CHARS:
                    overlap_tail = ""
            else:
                if buf:
                    _emit()
                table_text = (overlap_tail + "\n\n" + text).strip() if overlap_tail else text
                chunks.append(Chunk(text=table_text, metadata=dict(base_meta)))
                # Tiny tables (date-header rows ≤ _TINY_TABLE_CHARS, e.g.
                # "September 27, 2025  September 28, 2024") must NOT reset
                # overlap_tail — the statement name from the preceding text chunk
                # must carry forward to the next data table.  Fold the tiny
                # table text into overlap_tail so the data table's preamble
                # includes both the statement name AND the date row.
                if len(text) > _TINY_TABLE_CHARS:
                    overlap_tail = ""
                else:
                    overlap_tail = (overlap_tail + "\n\n" + text).strip()[-overlap_chars:]
        else:
            # Split the unit first if it's larger than one chunk on its own.
            sub_units = _split_text(text, size_chars) if text_chars > size_chars else [text]
            for sub in sub_units:
                sub_chars = len(sub)
                if buf_chars + sub_chars <= size_chars:
                    buf.append(sub)
                    buf_chars += sub_chars
                else:
                    _emit()
                    seed = (overlap_tail + "\n\n" + sub).strip() if overlap_tail else sub
                    buf = [seed]
                    buf_chars = len(seed)
                    if buf_chars > size_chars:
                        _emit()

    if buf:
        _emit()

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    ticker: str,
    year: int,
    size: int = None,
    overlap: int = None,
) -> list[Chunk]:
    """Chunk a parsed 10-K text string; attach ticker/year/section metadata."""
    size = size if size is not None else config.CHUNK_SIZE
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP

    all_chunks: list[Chunk] = []
    for section_name, section_text in _split_sections(text):
        if not section_text.strip():
            continue
        meta = {"ticker": ticker, "year": year, "section": section_name}
        units = _group_units(section_text)
        all_chunks.extend(_pack(units, meta, size, overlap))

    return all_chunks


def _meta_from_path(path: Path) -> tuple[str, int]:
    """Extract (ticker, filing_year) from a sec-edgar-downloader path."""
    parts = path.parts
    try:
        idx = next(i for i, p in enumerate(parts) if p == "sec-edgar-filings")
        ticker = parts[idx + 1].upper()
    except (StopIteration, IndexError):
        ticker = "UNKNOWN"

    # Accession number format: XXXXXXXXXX-YY-NNNNNN
    m = re.match(r"\d{10}-(\d{2})-\d{6}", path.parent.name)
    if m:
        yy = int(m.group(1))
        year = 2000 + yy if yy < 50 else 1900 + yy
    else:
        year = 0

    return ticker, year


def chunk_file(path: Path) -> list[Chunk]:
    """Chunk a parsed .txt filing; derive ticker and year from the path."""
    ticker, year = _meta_from_path(path)
    text = path.read_text(encoding="utf-8")
    return chunk_text(text, ticker, year)


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # Parse on-the-fly from the raw submission file
    raw_path = (
        config.RAW_DIR
        / "sec-edgar-filings" / "AAPL" / "10-K"
        / "0000320193-25-000079" / "full-submission.txt"
    )
    if not raw_path.exists():
        print(f"File not found: {raw_path}")
        sys.exit(1)

    from src.ingestion.parse import parse_filing
    print(f"Parsing {raw_path.name} ...")
    text = parse_filing(raw_path)

    ticker, year = _meta_from_path(raw_path)
    print(f"Chunking (ticker={ticker}, year={year}, "
          f"chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}) ...")
    chunks = chunk_text(text, ticker, year)

    # ── Summary ──
    section_counts: dict[str, int] = {}
    table_chunks = 0
    for c in chunks:
        section_counts[c.metadata["section"]] = (
            section_counts.get(c.metadata["section"], 0) + 1
        )
        if "\t" in c.text:
            table_chunks += 1

    print(f"\n{'='*60}")
    print(f"Total chunks : {len(chunks)}")
    print(f"Table chunks : {table_chunks}")
    print(f"Sections     : {len(section_counts)}")
    print(f"\nChunks per section:")
    for sec, count in section_counts.items():
        print(f"  {count:3d}  {sec}")

    # ── Sample chunks ──
    samples = [
        chunks[0],                        # first chunk (preamble/cover)
        next((c for c in chunks if "MD&A" in c.metadata["section"]), chunks[len(chunks)//2]),
        next((c for c in chunks if "\t" in c.text), chunks[-1]),  # first table chunk
    ]

    print(f"\n{'='*60}")
    for i, chunk in enumerate(samples, 1):
        tok = _tok(chunk.text)
        has_table = "\t" in chunk.text
        print(f"\n--- Sample {i} | ~{tok} tokens | table={has_table} ---")
        print(f"    metadata: {chunk.metadata}")
        preview = chunk.text[:400].replace("\n", "\n    ")
        print(f"    {preview}")
        if len(chunk.text) > 400:
            print("    [...]")
