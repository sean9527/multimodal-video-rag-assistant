"""Phase 1a smoke test: transcribe a video and print timestamped segments.

Usage:  python scripts/transcribe_test.py data/your_video.mp4
"""
from __future__ import annotations

import sys
from pathlib import Path

from mvrag.config import settings
from mvrag.ingestion.audio import extract_audio, transcribe
from mvrag.schemas import format_timestamp


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/transcribe_test.py <video_path>")
        raise SystemExit(1)

    video = Path(sys.argv[1])
    if not video.exists():
        print(f"file not found: {video}")
        raise SystemExit(1)

    settings.ensure_dirs()
    work = settings.data_dir / video.stem

    print(f"[1/2] extracting audio from {video.name} ...")
    wav = extract_audio(video, work / "audio.wav")

    print(f"[2/2] transcribing with faster-whisper "
          f"(model={settings.whisper_model}, device={settings.device}) ...")
    segments, language, duration = transcribe(wav)

    print(f"\nlanguage={language}  duration={format_timestamp(duration)}  "
          f"segments={len(segments)}\n")
    for s in segments[:12]:
        print(f"[{format_timestamp(s.start)}-{format_timestamp(s.end)}] {s.text}")
    if len(segments) > 12:
        print(f"... (+{len(segments) - 12} more segments)")


if __name__ == "__main__":
    main()
