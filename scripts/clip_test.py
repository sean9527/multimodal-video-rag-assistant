"""Phase 2b smoke test: embed keyframes with CLIP, then do text->frame retrieval.

Usage:  python scripts/clip_test.py data/your_video.mp4 ["what to look for"]
Requires that `python scripts/ingest.py <video>` has been run first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from mvrag.config import settings
from mvrag.embedding.image import embed_images, embed_query_text
from mvrag.schemas import IngestionResult, format_timestamp


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python scripts/clip_test.py <video_path> ["what to look for"]')
        raise SystemExit(1)

    video = Path(sys.argv[1])
    query = sys.argv[2] if len(sys.argv) > 2 else "python code on screen"

    ing_path = settings.data_dir / video.stem / "ingestion.json"
    if not ing_path.exists():
        print(f"not found: {ing_path}\n-> run  python scripts/ingest.py {video}  first")
        raise SystemExit(1)

    result = IngestionResult.model_validate_json(ing_path.read_text(encoding="utf-8"))
    if not result.keyframes:
        print("no keyframes in ingestion.json")
        raise SystemExit(1)

    paths = [kf.frame_path for kf in result.keyframes]
    ivecs = embed_images(paths)  # first call loads OpenCLIP on the GPU (downloads once)
    print(f"keyframes={len(paths)}  image embeddings: shape={ivecs.shape}")

    # cross-modal: encode the text query into the SAME CLIP space, then cosine
    q = embed_query_text([query])[0]
    sims = ivecs @ q
    top = np.argsort(-sims)[:3]

    print(f"\nquery: {query!r}\ntop-3 keyframes by cross-modal (text->image) similarity:")
    for rank, i in enumerate(top, 1):
        kf = result.keyframes[int(i)]
        print(f"  #{rank}  sim={sims[int(i)]:.3f}  [{format_timestamp(kf.timestamp)}]  {kf.frame_path}")
    print("\n(open those .jpg files to eyeball whether CLIP found relevant frames)")


if __name__ == "__main__":
    main()
