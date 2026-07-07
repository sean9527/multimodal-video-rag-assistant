"""Cross-encoder reranking (Phase 3.5).

Dense retrieval uses a BI-encoder: query and chunk are embedded separately, then
compared by cosine — fast, but coarse (the model never sees the pair together).
A CROSS-encoder (bge-reranker-v2-m3) jointly encodes (query, chunk) and scores
relevance directly — much more accurate, but too slow to run over the whole
corpus. So we use the classic two-stage pattern: dense-retrieve N candidates
cheaply, then rerank them with the cross-encoder and keep the top-k. This is the
main lever for retrieval *precision*, and it matters more as the corpus grows.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from mvrag.config import settings
from mvrag.schemas import RetrievedChunk


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.rerank_model, device=settings.device)


def rerank_chunks(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Re-score (query, chunk) pairs with the cross-encoder and return the top_k,
    with each chunk's `score` replaced by the reranker's relevance (0..1)."""
    if not chunks:
        return []
    # CrossEncoder.predict already applies the model's activation (sigmoid for
    # this 1-label reranker) -> scores already in [0, 1]. Do NOT sigmoid again.
    scores = np.asarray(_reranker().predict([(query, c.text) for c in chunks]), dtype=float)
    order = np.argsort(-scores)[:top_k]
    return [chunks[int(i)].model_copy(update={"score": float(scores[int(i)])}) for i in order]
