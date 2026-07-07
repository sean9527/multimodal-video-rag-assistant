"""Query the persisted LanceDB store — both modalities (Phase 2c verification).

Unlike embed_test/clip_test (which embed on the fly), this reads vectors back
from disk, proving the store persists and retrieves correctly.

Usage:  python scripts/search_test.py "your question"
"""
from __future__ import annotations

import sys

from mvrag.config import settings
from mvrag.embedding.image import embed_query_text
from mvrag.embedding.text import embed_texts
from mvrag.indexing.store import search_images, search_text
from mvrag.schemas import format_timestamp


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "how do you reshape a tensor"
    print(f"query: {query!r}")

    # transcript index: query encoded with BGE (same space as the chunks)
    tq = embed_texts([query])[0].tolist()
    print("\n[transcript matches]")
    for r in search_text(tq, k=settings.top_k_text):
        sim = 1.0 - r["_distance"]  # cosine distance -> similarity
        print(f"  sim={sim:.3f}  [{format_timestamp(r['start'])}-{format_timestamp(r['end'])}]  {r['text'][:90]}...")

    # keyframe index: SAME query encoded with CLIP's text encoder (CLIP space)
    iq = embed_query_text([query])[0].tolist()
    print("\n[keyframe matches]")
    for r in search_images(iq, k=settings.top_k_image):
        sim = 1.0 - r["_distance"]
        print(f"  sim={sim:.3f}  [{format_timestamp(r['timestamp'])}]  {r['frame_path']}")


if __name__ == "__main__":
    main()
