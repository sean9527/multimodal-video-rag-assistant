"""Answer a question from retrieved context, via a pluggable LLM (see llm.py).

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

from mvrag.config import settings
from mvrag.generation.llm import get_llm
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

    try:
        text = get_llm().chat(SYSTEM_PROMPT, user_msg)
    except Exception as e:
        raise RuntimeError(
            f"LLM call failed via provider '{settings.llm_provider}' ({e}). "
            f"ollama → is `ollama serve` running and the model pulled? "
            f"openai/anthropic → run `pip install -e '.[api]'` and set "
            f"OPENAI_API_KEY / ANTHROPIC_API_KEY."
        ) from e

    return {"answer": text, "chunks": bundle.chunks, "frames": bundle.frames}
