"""Embed chunks using the configured sentence-transformers model."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

import config

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the singleton embedding model, loading it on first call."""
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed_texts(
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Embed a list of strings; returns float32 array of shape (N, dim).

    Embeddings are L2-normalised so dot product == cosine similarity.
    """
    return get_model().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
