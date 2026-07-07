"""Full Phase 1 ingestion: video -> IngestionResult (transcript + keyframes).

The result (including its course_id scope tag) is persisted to
data/<video_id>/ingestion.json so later phases don't re-run the extraction.
"""
from __future__ import annotations

from pathlib import Path

from mvrag.config import settings
from mvrag.ingestion.audio import extract_audio, transcribe
from mvrag.ingestion.frames import extract_keyframes
from mvrag.schemas import IngestionResult


def make_video_id(course_id: str, video_path: str | Path) -> str:
    """Course-prefixed, filesystem-safe id — so two courses that happen to have a
    video with the same filename (e.g. `lecture01.mp4`) don't collide."""
    raw = f"{course_id}__{Path(video_path).stem}"
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in raw)


def ingest_video(video_path: str | Path, course_id: str = "uncategorized") -> IngestionResult:
    video_path = Path(video_path)
    settings.ensure_dirs()

    video_id = make_video_id(course_id, video_path)
    work = settings.data_dir / video_id
    work.mkdir(parents=True, exist_ok=True)

    # 1) audio -> timestamped transcript
    wav = extract_audio(video_path, work / "audio.wav")
    segments, language, duration = transcribe(wav)

    # 2) frames -> timestamped keyframes
    keyframes = extract_keyframes(video_path, work / "frames", duration)

    result = IngestionResult(
        video_id=video_id,
        course_id=course_id,
        video_path=str(video_path),
        duration=duration,
        language=language,
        segments=segments,
        keyframes=keyframes,
    )
    (work / "ingestion.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    return result
