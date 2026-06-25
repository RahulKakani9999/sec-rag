"""Download 10-K filings from SEC EDGAR."""
from pathlib import Path
from sec_edgar_downloader import Downloader
import config


def _make_downloader(dest: Path) -> Downloader:
    """Build a Downloader from SEC_USER_AGENT (expected format: 'Name email@host')."""
    agent = config.SEC_USER_AGENT or ""
    parts = agent.rsplit(" ", 1)
    name = parts[0] if len(parts) == 2 else "SecRAG"
    email = parts[1] if len(parts) == 2 else agent
    return Downloader(name, email, dest)


def download_filings(
    companies: list = None,
    limit: int = None,
    dest: Path = None,
) -> list[Path]:
    """Download 10-K filings for each ticker; return paths to HTML filing documents.

    Files are saved under dest/<ticker>/10-K/<accession>/ by the library.
    """
    companies = companies if companies is not None else config.COMPANIES
    limit = limit if limit is not None else config.FILINGS_PER_COMPANY
    dest = dest or config.RAW_DIR
    dest.mkdir(parents=True, exist_ok=True)

    dl = _make_downloader(dest)
    for ticker in companies:
        dl.get("10-K", ticker, limit=limit)

    return sorted(dest.rglob("*.htm")) + sorted(dest.rglob("*.html"))
