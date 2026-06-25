"""Parse filing HTML into clean text, preserving tables."""
import re
from pathlib import Path
from bs4 import BeautifulSoup
import config

_PRIMARY_DOC_RE = re.compile(
    r"<DOCUMENT>\s*<TYPE>10-K\b.*?<TEXT>(.*?)</TEXT>",
    re.DOTALL,
)


def _extract_html_from_submission(text: str) -> str:
    """Pull the primary 10-K HTML out of a full-submission.txt SGML container."""
    m = _PRIMARY_DOC_RE.search(text)
    if not m:
        raise ValueError("Could not find <TYPE>10-K ... <TEXT> block in submission file")
    html = m.group(1).strip()
    # Strip the optional <XBRL>...</XBRL> wrapper the SEC adds around iXBRL docs.
    if html.startswith("<XBRL>"):
        html = html[len("<XBRL>"):]
    if html.endswith("</XBRL>"):
        html = html[: -len("</XBRL>")]
    return html


def _table_to_text(table) -> str:
    """Convert an HTML table to tab-separated rows."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append("\t".join(cells))
    return "\n".join(rows)


def parse_filing(path: Path) -> str:
    """Return clean text from a 10-K filing; tables become TSV blocks.

    Accepts either a standalone .htm/.html file or a full-submission.txt
    SGML container (as downloaded by sec-edgar-downloader v5+).
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    html = _extract_html_from_submission(raw) if path.name == "full-submission.txt" else raw
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "head", "meta", "noscript"]):
        tag.decompose()

    # iXBRL documents hide XBRL metadata in display:none elements; remove them.
    for tag in soup.find_all(style=lambda s: s and "display:none" in s.replace(" ", "")):
        tag.decompose()

    # Convert each table to plain text before extracting the document text.
    for table in soup.find_all("table"):
        table_text = _table_to_text(table)
        replacement = soup.new_tag("div")
        replacement.string = table_text
        table.replace_with(replacement)

    body = soup.body or soup
    return body.get_text(separator="\n", strip=True)


def parse_and_save(html_path: Path, out_dir: Path = None) -> Path:
    """Parse one filing and write plain text to out_dir; return the output path."""
    out_dir = out_dir or config.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    text = parse_filing(html_path)
    rel = html_path.relative_to(config.RAW_DIR)
    out_path = out_dir / rel.with_suffix(".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def parse_all(raw_dir: Path = None, out_dir: Path = None) -> list[Path]:
    """Parse every HTML filing under raw_dir; return list of written text files."""
    raw_dir = raw_dir or config.RAW_DIR
    out_dir = out_dir or config.PROCESSED_DIR

    html_paths = (
        sorted(raw_dir.rglob("*.htm"))
        + sorted(raw_dir.rglob("*.html"))
        + sorted(raw_dir.rglob("full-submission.txt"))
    )
    return [parse_and_save(p, out_dir) for p in html_paths]
