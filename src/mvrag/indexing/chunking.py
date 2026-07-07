"""Group transcript segments into overlapping, timestamped chunks for retrieval.

Whisper segments are short (~5-10s). Embedding each tiny segment hurts retrieval
(too little context per vector), while chunks that are too large dilute the
semantics. We greedily merge consecutive segments up to ~chunk_target_chars,
keep the combined (start, end) span, and step back a little each time so
consecutive chunks share ~chunk_overlap_chars — a fact near a boundary then
isn't split away from its context.
"""
from __future__ import annotations

from mvrag.config import settings
from mvrag.schemas import TextChunk, TranscriptSegment


def chunk_segments(segments: list[TranscriptSegment]) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    n = len(segments)
    target = settings.chunk_target_chars
    overlap = settings.chunk_overlap_chars

    i = 0
    while i < n:
        # grow a group forward until it reaches the target size
        j, length = i, 0
        while j < n and length < target:
            length += len(segments[j].text) + 1
            j += 1

        group = segments[i:j]
        chunks.append(
            TextChunk(
                chunk_id=len(chunks),
                start=group[0].start,
                end=group[-1].end,
                text=" ".join(s.text for s in group).strip(),
            )
        )
        if j >= n:
            break

        # step back a few tail segments so the next chunk overlaps by ~overlap chars
        back, acc = 0, 0
        while back < len(group) - 1 and acc < overlap:
            acc += len(group[-1 - back].text)
            back += 1
        i = j - back  # always advances by >= 1

    return chunks
