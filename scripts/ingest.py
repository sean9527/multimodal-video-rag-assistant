"""Run the full Phase 1 ingestion (transcript + keyframes) on a video.

Usage:  python scripts/ingest.py <video_path> [course_id]
Example: python scripts/ingest.py data/CSCE2230_lec01.mp4 CSCE2230
"""
from __future__ import annotations

import sys
from pathlib import Path

from mvrag.ingestion.pipeline import ingest_video
from mvrag.schemas import format_timestamp


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/ingest.py <video_path> [course_id]")
        raise SystemExit(1)

    video = Path(sys.argv[1])
    course_id = sys.argv[2] if len(sys.argv) > 2 else "uncategorized"
    if not video.exists():
        print(f"file not found: {video}")
        raise SystemExit(1)

    result = ingest_video(video, course_id=course_id)

    print("\n=== ingestion complete ===")
    print(f"video_id   : {result.video_id}")
    print(f"course_id  : {result.course_id}")
    print(f"duration   : {format_timestamp(result.duration)}")
    print(f"segments   : {len(result.segments)}")
    print(f"keyframes  : {len(result.keyframes)}")
    print(f"saved      : data/{result.video_id}/ingestion.json  (+ frames/)")


if __name__ == "__main__":
    main()
