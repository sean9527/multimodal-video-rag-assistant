"""Phase 2a smoke test: chunk + embed a video's transcript, then demo semantic search.

Usage:  python scripts/embed_test.py data/your_video.mp4 ["your question"]
Requires that `python scripts/ingest.py <video>` has been run first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from mvrag.config import settings
from mvrag.embedding.text import embed_texts, embedding_dim
from mvrag.indexing.chunking import chunk_segments
from mvrag.schemas import IngestionResult, format_timestamp


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python scripts/embed_test.py <video_path> ["your question"]')
        raise SystemExit(1)

    video = Path(sys.argv[1])
    query = sys.argv[2] if len(sys.argv) > 2 else "how do you reshape a tensor"

    ing_path = settings.data_dir / video.stem / "ingestion.json"
    if not ing_path.exists():
        print(f"not found: {ing_path}\n-> run  python scripts/ingest.py {video}  first")
        raise SystemExit(1)

    result = IngestionResult.model_validate_json(ing_path.read_text(encoding="utf-8"))
    chunks = chunk_segments(result.segments)
    print(f"segments={len(result.segments)} -> chunks={len(chunks)}")

    # first call loads BGE-m3 on the GPU (downloads it once, ~2GB)
    vecs = embed_texts([c.text for c in chunks])
    print(f"text embeddings: shape={vecs.shape}  dim={embedding_dim()}")

    # semantic search: vectors are L2-normalized, so a dot product is cosine sim
    q = embed_texts([query])[0]
    sims = vecs @ q
    top = np.argsort(-sims)[:3]

    print(f"\nquery: {query!r}\ntop-3 transcript chunks by semantic similarity:")
    for rank, i in enumerate(top, 1):
        c = chunks[int(i)]
        print(f"  #{rank}  sim={sims[int(i)]:.3f}  "
              f"[{format_timestamp(c.start)}-{format_timestamp(c.end)}]  {c.text[:110]}...")


if __name__ == "__main__":
    main()
