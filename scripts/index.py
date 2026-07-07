"""Index a video's ingestion output into LanceDB (Phase 2c).

Usage:  python scripts/index.py <video_path> <course_id>
Pass the SAME course_id you used for ingest.py (it locates the artifacts at
data/<course__stem>/). Requires `python scripts/ingest.py <video> <course>` first.
"""
from __future__ import annotations

import sys
from pathlib import Path

from mvrag.config import settings
from mvrag.indexing.store import index_video
from mvrag.ingestion.pipeline import make_video_id
from mvrag.schemas import IngestionResult


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/index.py <video_path> <course_id>")
        raise SystemExit(1)

    video = Path(sys.argv[1])
    course = sys.argv[2] if len(sys.argv) > 2 else None
    video_id = make_video_id(course, video) if course else video.stem

    ing_path = settings.data_dir / video_id / "ingestion.json"
    if not ing_path.exists():
        print(f"not found: {ing_path}\n-> run  python scripts/ingest.py {video} {course or ''}  first")
        raise SystemExit(1)

    result = IngestionResult.model_validate_json(ing_path.read_text(encoding="utf-8"))
    n_text, n_img = index_video(result)

    print(f"indexed '{result.video_id}' (course '{result.course_id}'):  "
          f"{n_text} text chunks + {n_img} keyframes")


if __name__ == "__main__":
    main()
