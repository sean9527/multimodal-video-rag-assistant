"""Video frames -> timestamped keyframes.

Two strategies, chosen automatically:
- **scene detection** (PySceneDetect ContentDetector): one keyframe at the
  midpoint of each detected shot. Great for edited videos / slide decks.
- **uniform sampling** (fallback): for low-cut videos (a talking-head lecture
  has almost no cuts) scene detection finds ~nothing, so we sample one frame
  every N seconds to guarantee visual coverage.

Every keyframe carries a timestamp, so it lands on the *same timeline* as the
transcript segments — the anchor that later lets us fuse text + visual retrieval
and cite a source time.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scenedetect import ContentDetector, detect

from mvrag.config import settings
from mvrag.schemas import Keyframe


def _grab_frame(cap: cv2.VideoCapture, timestamp: float, out_path: Path) -> bool:
    """Seek to `timestamp` (seconds) and save that frame as JPEG."""
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    return True


def extract_keyframes(video_path: str | Path, out_dir: str | Path, duration: float) -> list[Keyframe]:
    out_dir = Path(out_dir)
    scenes = detect(str(video_path), ContentDetector(threshold=settings.scene_threshold))

    step = settings.frame_sample_interval
    uniform_n = max(1, int(duration / step))

    # Use scene detection only when it's at least as dense as uniform sampling;
    # otherwise (talking-head lectures, screencasts with few hard cuts) fall back
    # to uniform sampling so we never under-cover the video.
    if len(scenes) >= uniform_n:
        timestamps = [(s.get_seconds() + e.get_seconds()) / 2.0 for s, e in scenes]
        mode = "scene-detection"
    else:
        timestamps = [min(i * step + step / 2.0, duration) for i in range(uniform_n)]
        mode = "uniform-sampling"

    # Cap frame count but keep coverage across the whole video (even subsample,
    # not tail truncation).
    if len(timestamps) > settings.max_keyframes:
        idx = np.linspace(0, len(timestamps) - 1, settings.max_keyframes).astype(int)
        timestamps = [timestamps[int(i)] for i in idx]

    print(f"[keyframes] {mode}: {len(timestamps)} frames")

    cap = cv2.VideoCapture(str(video_path))
    keyframes: list[Keyframe] = []
    for i, ts in enumerate(timestamps):
        path = out_dir / f"frame_{i:04d}.jpg"
        if _grab_frame(cap, ts, path):
            keyframes.append(Keyframe(timestamp=float(ts), frame_path=str(path), scene_index=i))
    cap.release()
    return keyframes
