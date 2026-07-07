"""Answer a question from retrieved context via a local LLM (Ollama).

Grounding (to curb hallucination): the system prompt restricts the model to the
provided excerpts, each labeled with its [timestamp]; when the scope spans
several videos, each excerpt is also tagged with its video_id so the model can
cite which video. If the answer isn't in the excerpts, the model says so.

Scope: pass `course_id` (search a whole course) and/or `video_id` (one video).
Keyframes are returned alongside for the UI; the text prompt is transcript-only
(that's where the answer lives) — see the design doc's "multimodal value depends
on video type" note.
"""
from __future__ import annotations

import ollama

from mvrag.config import settings
from mvrag.retrieval.retriever import retrieve
from mvrag.schemas import RetrievalBundle, format_timestamp

SYSTEM_PROMPT = (
    "You are a video Q&A assistant. Answer the user's question using ONLY the "
    "transcript excerpts provided. Each excerpt is labeled with its "
    "[timestamp] (and a video id when several videos are shown).\n"
    "- Cite the timestamp(s) you used, e.g. (02:05); if multiple videos are "
    "shown, say which video.\n"
    "- If the excerpts do not contain the answer, say you could not find it.\n"
    "- Be concise."
)


def _format_context(bundle: RetrievalBundle) -> str:
    multi_video = len({c.video_id for c in bundle.chunks}) > 1
    lines = []
    for c in bundle.chunks:
        tag = f"{c.video_id} @ " if multi_video else ""
        lines.append(f"[{tag}{format_timestamp(c.start)}-{format_timestamp(c.end)}] {c.text}")
    return "\n".join(lines)


def answer(query: str, course_id: str | None = None, video_id: str | None = None) -> dict:
    bundle = retrieve(query, course_id=course_id, video_id=video_id)
    user_msg = f"Transcript excerpts:\n{_format_context(bundle)}\n\nQuestion: {query}"

    client = ollama.Client(host=settings.ollama_host)
    try:
        resp = client.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            options={"temperature": 0.2},
        )
    except Exception as e:
        raise RuntimeError(
            f"Ollama call failed ({e}). Is `ollama serve` running and is the model "
            f"'{settings.ollama_model}' pulled?  Try:  ollama pull {settings.ollama_model}"
        ) from e

    return {
        "answer": resp["message"]["content"],
        "chunks": bundle.chunks,
        "frames": bundle.frames,
    }
