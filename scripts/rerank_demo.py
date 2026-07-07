"""Before/after: dense retrieval vs cross-encoder reranking (Phase 3.5).

Usage:  python scripts/rerank_demo.py "your question" [--course C] [--video V]
Shows the same candidates ranked by bi-encoder cosine vs by the cross-encoder.
"""
from __future__ import annotations

import argparse

from mvrag.config import settings
from mvrag.embedding.text import embed_texts
from mvrag.indexing.store import search_text
from mvrag.retrieval.rerank import rerank_chunks
from mvrag.schemas import RetrievedChunk, format_timestamp


def _show(chunks, n) -> None:
    for c in chunks[:n]:
        print(f"  score={c.score:.3f}  [{c.video_id} @ "
              f"{format_timestamp(c.start)}-{format_timestamp(c.end)}]  {c.text[:70]}...")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--course", default=None)
    p.add_argument("--video", default=None)
    args = p.parse_args()

    tq = embed_texts([args.question])[0].tolist()
    hits = search_text(tq, settings.rerank_top_n, course_id=args.course, video_id=args.video)
    cand = [
        RetrievedChunk(video_id=h["video_id"], start=h["start"], end=h["end"],
                       text=h["text"], score=1.0 - h["_distance"])
        for h in hits
    ]

    print(f"query: {args.question!r}   (dense candidates: {len(cand)})\n")
    print("── DENSE (bi-encoder cosine) top-6 ──")
    _show(cand, 6)
    print("\n── RERANKED (cross-encoder bge-reranker-v2-m3) top-6 ──")
    _show(rerank_chunks(args.question, cand, settings.top_k_text), 6)


if __name__ == "__main__":
    main()
