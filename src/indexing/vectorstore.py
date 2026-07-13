"""Set up and populate the Chroma vector store and BM25 index."""
from __future__ import annotations

import pickle
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

import config
from src.chunking.chunker import Chunk
from src.indexing.embed import embed_texts

COLLECTION_NAME = "sec_10k"
BM25_PATH = config.CHROMA_DIR / "bm25.pkl"

# Chroma batch size — keeps individual requests well under any size limit.
_UPSERT_BATCH = 500


# ── Chroma helpers ─────────────────────────────────────────────────────────────

def _client() -> chromadb.PersistentClient:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(reset: bool = False) -> chromadb.Collection:
    """Return the Chroma collection, creating it if necessary.

    Pass reset=True to wipe and recreate (useful when re-indexing a filing).
    """
    client = _client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_input(chunk: Chunk) -> str:
    """Prepend section/ticker/year so orphaned data chunks embed with document context.

    The prefix is used only for embedding — the original text is stored in Chroma
    unchanged so retrieved snippets stay clean.
    """
    section = chunk.metadata.get("section", "")
    ticker  = chunk.metadata.get("ticker", "")
    year    = chunk.metadata.get("year", "")
    prefix  = f"[{section} -- {ticker} {year}]\n" if section else ""
    return f"{prefix}{chunk.text}"


def add_chunks(chunks: list[Chunk], reset: bool = False) -> tuple[int, int]:
    """Embed chunks and upsert into Chroma.

    Returns (vectors_stored, embedding_dim).
    """
    if not chunks:
        return 0, 0

    embed_inputs = [_embed_input(c) for c in chunks]
    embeddings: np.ndarray = embed_texts(embed_inputs, show_progress=True)
    dim = int(embeddings.shape[1])

    collection = get_collection(reset=reset)

    ids, docs, metas = [], [], []
    for i, chunk in enumerate(chunks):
        ticker = chunk.metadata.get("ticker", "UNK")
        year   = chunk.metadata.get("year", 0)
        ids.append(f"{ticker}_{year}_{i:05d}")
        docs.append(chunk.text)
        metas.append(chunk.metadata)

    for start in range(0, len(ids), _UPSERT_BATCH):
        sl = slice(start, start + _UPSERT_BATCH)
        collection.upsert(
            ids=ids[sl],
            embeddings=embeddings[sl],
            documents=docs[sl],
            metadatas=metas[sl],
        )

    return len(ids), dim


# ── BM25 helpers ──────────────────────────────────────────────────────────────

def save_bm25(chunks: list[Chunk], path: Path = None) -> Path:
    """Build a BM25 index from chunks and pickle it to disk.

    The pickle stores the BM25Okapi object plus the parallel text/metadata
    lists needed to retrieve the original chunks after scoring.
    """
    path = Path(path or BM25_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    texts     = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]
    tokenized = [t.lower().split() for t in texts]

    payload = {
        "bm25":      BM25Okapi(tokenized),
        "texts":     texts,
        "metadatas": metadatas,
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    return path


def load_bm25(path: Path = None) -> tuple[BM25Okapi, list[str], list[dict]]:
    """Load a pickled BM25 index; returns (bm25, texts, metadatas)."""
    path = Path(path or BM25_PATH)
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    return obj["bm25"], obj["texts"], obj["metadatas"]


# ── CLI: end-to-end index of the AAPL filing ─────────────────────────────────

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from src.ingestion.parse import parse_filing
    from src.chunking.chunker import chunk_text, _meta_from_path

    raw = (
        config.RAW_DIR
        / "sec-edgar-filings" / "AAPL" / "10-K"
        / "0000320193-25-000079" / "full-submission.txt"
    )

    # ── Parse ──
    print(f"[1/4] Parsing  {raw.name} ...")
    text = parse_filing(raw)

    # ── Chunk ──
    ticker, year = _meta_from_path(raw)
    print(f"[2/4] Chunking (ticker={ticker}, year={year}) ...")
    chunks = chunk_text(text, ticker, year)
    print(f"      {len(chunks)} chunks")

    # ── Embed + Chroma ──
    print(f"[3/4] Embedding and storing in Chroma (reset=True) ...")
    n_stored, dim = add_chunks(chunks, reset=True)
    print(f"      {n_stored} vectors stored, embedding dim={dim}")

    # ── BM25 ──
    print(f"[4/4] Building BM25 index ...")
    bm25_path = save_bm25(chunks)
    print(f"      Saved to {bm25_path}")

    # ── Smoke-test: query Chroma ──
    print()
    print("Smoke-test — querying Chroma for 'Apple revenue 2025' ...")
    from src.indexing.embed import embed_texts as _embed
    qvec = _embed(["Apple revenue 2025"])
    col  = get_collection()
    results = col.query(
        query_embeddings=qvec,
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"  dist={dist:.4f}  {meta}  | {doc[:120].replace(chr(10), ' ')} ...")
