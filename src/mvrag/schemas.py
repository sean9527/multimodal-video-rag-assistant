"""Core data models shared across the pipeline.

Everything is anchored to time (seconds) and scoped by course_id / video_id —
the time anchor powers timestamp citations, the scope metadata keeps many
courses/videos isolated inside one shared vector store.
"""
from __future__ import annotations

from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    start: float          # seconds
    end: float            # seconds
    text: str


class Keyframe(BaseModel):
    timestamp: float      # seconds
    frame_path: str
    scene_index: int


class TextChunk(BaseModel):
    """A retrieval-sized span of transcript, merged from consecutive segments."""
    chunk_id: int
    start: float
    end: float
    text: str


class IngestionResult(BaseModel):
    """Everything we extract from one video."""
    video_id: str
    course_id: str = "uncategorized"      # scope tag for multi-course isolation
    video_path: str
    duration: float
    language: str | None = None
    segments: list[TranscriptSegment] = []
    keyframes: list[Keyframe] = []


class RetrievedChunk(BaseModel):
    video_id: str
    start: float
    end: float
    text: str
    score: float


class RetrievedFrame(BaseModel):
    video_id: str
    timestamp: float
    frame_path: str
    score: float


class RetrievalBundle(BaseModel):
    """What the retriever hands to the generator: two modality streams, each
    time-stamped and tagged with its source video."""
    query: str
    chunks: list[RetrievedChunk] = []
    frames: list[RetrievedFrame] = []


def format_timestamp(seconds: float) -> str:
    """3.5 -> '00:03', 125.0 -> '02:05'. Used for display and citations."""
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"
