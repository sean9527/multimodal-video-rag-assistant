"""Text embeddings via sentence-transformers (BGE-m3, multilingual).

BGE-m3 handles English + 中文, so transcript retrieval works regardless of the
video's language. We L2-normalize the vectors so inner product == cosine
similarity — cheaper to compare and what the vector index expects.

The model is loaded once (lru_cache) and reused; on this box it runs on the GPU
(settings.device="cuda"), which is where the 3090 finally earns its keep.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from mvrag.config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        settings.text_embed_model,
        device=settings.device,
        cache_folder=str(settings.model_cache_dir),
    )


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return an (N, dim) float32 array of L2-normalized embeddings."""
    return _model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False,
    )


def embedding_dim() -> int:
    m = _model()
    # method was renamed in newer sentence-transformers; support both
    fn = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
    return fn()
