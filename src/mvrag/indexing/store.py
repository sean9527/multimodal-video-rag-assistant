"""LanceDB multimodal vector store — with course/video scoping.

Two tables (the modalities live in different embedding spaces & dims):
- `text_chunks` : BGE-m3 transcript vectors (1024-d)
- `keyframes`   : OpenCLIP image vectors (512-d)

Every row carries `course_id` + `video_id`. Retrieval filters on them
(`WHERE course_id = ... [AND video_id = ...]`) — that's how ONE shared store
isolates many courses/videos without ever mixing them (multi-tenancy via
metadata filtering; no separate DB per course needed). At scale you'd add an ANN
index (HNSW/IVF) via `table.create_index()`.
"""
from __future__ import annotations

import lancedb
from lancedb.pydantic import LanceModel, Vector

from mvrag.config import settings
from mvrag.embedding.image import embed_images
from mvrag.embedding.text import embed_texts
from mvrag.indexing.chunking import chunk_segments
from mvrag.schemas import IngestionResult

TEXT_DIM = 1024   # BGE-m3
IMAGE_DIM = 512   # OpenCLIP ViT-B/32


class TextRow(LanceModel):
    course_id: str
    video_id: str
    chunk_id: int
    start: float
    end: float
    text: str
    vector: Vector(TEXT_DIM)


class ImageRow(LanceModel):
    course_id: str
    video_id: str
    kf_index: int
    timestamp: float
    frame_path: str
    vector: Vector(IMAGE_DIM)


def _db():
    settings.lancedb_path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(settings.lancedb_path))


def _table(db, name: str, schema):
    if name in db.table_names():   # returns list[str]; list_tables() differs by version
        return db.open_table(name)
    return db.create_table(name, schema=schema)


def index_video(result: IngestionResult, course_id: str | None = None) -> tuple[int, int]:
    """Embed + upsert one video's chunks and keyframes.

    `course_id` overrides result.course_id when given. Idempotent: re-indexing
    the same video_id replaces its rows. Returns (n_text_chunks, n_keyframes).
    """
    cid = course_id or result.course_id
    db = _db()
    text_tbl = _table(db, "text_chunks", TextRow)
    image_tbl = _table(db, "keyframes", ImageRow)

    text_tbl.delete(f"video_id = '{result.video_id}'")
    image_tbl.delete(f"video_id = '{result.video_id}'")

    chunks = chunk_segments(result.segments)
    if chunks:
        tvecs = embed_texts([c.text for c in chunks])
        text_tbl.add([
            {"course_id": cid, "video_id": result.video_id, "chunk_id": c.chunk_id,
             "start": c.start, "end": c.end, "text": c.text, "vector": tvecs[i].tolist()}
            for i, c in enumerate(chunks)
        ])

    if result.keyframes:
        ivecs = embed_images([kf.frame_path for kf in result.keyframes])
        image_tbl.add([
            {"course_id": cid, "video_id": result.video_id, "kf_index": i,
             "timestamp": kf.timestamp, "frame_path": kf.frame_path, "vector": ivecs[i].tolist()}
            for i, kf in enumerate(result.keyframes)
        ])

    return len(chunks), len(result.keyframes)


def _where(course_id: str | None, video_id: str | None) -> str | None:
    clauses = []
    if course_id:
        clauses.append(f"course_id = '{course_id}'")
    if video_id:
        clauses.append(f"video_id = '{video_id}'")
    return " AND ".join(clauses) if clauses else None


def _search(table_name: str, query_vector, k: int, course_id: str | None, video_id: str | None):
    tbl = _db().open_table(table_name)
    q = tbl.search(query_vector).metric("cosine")
    clause = _where(course_id, video_id)
    if clause:
        q = q.where(clause, prefilter=True)
    return q.limit(k).to_list()


def search_text(query_vector, k: int, course_id: str | None = None, video_id: str | None = None) -> list[dict]:
    return _search("text_chunks", query_vector, k, course_id, video_id)


def search_images(query_vector, k: int, course_id: str | None = None, video_id: str | None = None) -> list[dict]:
    return _search("keyframes", query_vector, k, course_id, video_id)


def list_scope() -> dict[str, list[str]]:
    """course_id -> [video_id, ...] currently in the store (for scope pickers /
    'did you mean' hints). Scope tags are exact-match, so this is the source of
    truth for what's queryable."""
    db = _db()
    if "text_chunks" not in db.table_names():
        return {}
    df = db.open_table("text_chunks").to_pandas()[["course_id", "video_id"]].drop_duplicates()
    out: dict[str, list[str]] = {}
    for cid, vid in df.itertuples(index=False):
        out.setdefault(cid, []).append(vid)
    return out
