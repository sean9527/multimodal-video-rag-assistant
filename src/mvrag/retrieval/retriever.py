"""Unified cross-modal retriever, scoped by course / video, with optional rerank.

One query -> two encoders -> two indexes:
- BGE encodes the query to search the transcript index (answer content)
- CLIP's text encoder encodes the SAME query to search the keyframe index

Two-stage text retrieval: dense-retrieve `rerank_top_n` candidates, then (if
enabled) a cross-encoder reranks them down to `top_k_text`. `course_id` /
`video_id` scope the search via metadata filtering. Each stream's top-k is
returned separately (the two spaces' scores aren't comparable; a single fused
ranking would use RRF).
"""
from __future__ import annotations

from mvrag.config import settings
from mvrag.embedding.image import embed_query_text
from mvrag.embedding.text import embed_texts
from mvrag.indexing.store import search_images, search_text
from mvrag.retrieval.rerank import rerank_chunks
from mvrag.schemas import RetrievalBundle, RetrievedChunk, RetrievedFrame


def retrieve(
    query: str,
    course_id: str | None = None,
    video_id: str | None = None,
    k_text: int | None = None,
    k_image: int | None = None,
    rerank: bool | None = None,
) -> RetrievalBundle:
    k_text = k_text or settings.top_k_text
    k_image = k_image or settings.top_k_image
    rerank = settings.use_reranker if rerank is None else rerank

    # stage 1: dense retrieve (more candidates when we'll rerank)
    n_text = settings.rerank_top_n if rerank else k_text
    tq = embed_texts([query])[0].tolist()
    chunks = [
        RetrievedChunk(video_id=h["video_id"], start=h["start"], end=h["end"],
                       text=h["text"], score=1.0 - h["_distance"])
        for h in search_text(tq, n_text, course_id=course_id, video_id=video_id)
    ]
    # stage 2: cross-encoder rerank down to top_k (or just truncate)
    chunks = rerank_chunks(query, chunks, k_text) if rerank else chunks[:k_text]

    iq = embed_query_text([query])[0].tolist()
    frames = [
        RetrievedFrame(video_id=h["video_id"], timestamp=h["timestamp"],
                       frame_path=h["frame_path"], score=1.0 - h["_distance"])
        for h in search_images(iq, k_image, course_id=course_id, video_id=video_id)
    ]

    return RetrievalBundle(query=query, chunks=chunks, frames=frames)
