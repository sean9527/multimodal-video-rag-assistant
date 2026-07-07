"""Build a synthetic retrieval eval set.

Samples transcript chunks and has the local LLM write ONE question per chunk; the
source chunk is the ground truth. This gives labeled (query -> correct chunk)
pairs with no manual labeling. Saves eval/eval_set.jsonl.

Usage:  python scripts/build_eval.py [N]        (default 40)

Caveat: synthetic questions can be lexically close to their source chunk, so
absolute recall is optimistic — but the *relative* dense-vs-rerank comparison is
still valid. Mix in a few hand-written / paraphrased hard queries for rigor.
"""
from __future__ import annotations

import json
import random
import sys

import lancedb
import ollama

from mvrag.config import PROJECT_ROOT, settings

PROMPT = (
    "Read this lecture transcript excerpt and write ONE specific, self-contained "
    "question that it answers. The question must be answerable from this excerpt "
    "alone; do NOT refer to 'the excerpt/video/transcript'. Output only the question.\n\n"
    "Excerpt:\n{text}"
)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    db = lancedb.connect(str(settings.lancedb_path))
    df = db.open_table("text_chunks").to_pandas()
    df = df[df["text"].str.len() > 150].reset_index(drop=True)  # skip tiny/low-info chunks
    if df.empty:
        print("no usable chunks in store — ingest + index some videos first")
        raise SystemExit(1)

    random.seed(0)
    idx = random.sample(range(len(df)), min(n, len(df)))
    client = ollama.Client(host=settings.ollama_host)

    out = []
    for j, i in enumerate(idx, 1):
        row = df.iloc[i]
        resp = client.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": PROMPT.format(text=row["text"])}],
            options={"temperature": 0.3},
        )
        q = resp["message"]["content"].strip().splitlines()[-1].strip()
        out.append({
            "question": q, "course_id": row["course_id"], "video_id": row["video_id"],
            "start": float(row["start"]), "end": float(row["end"]),
        })
        print(f"[{j}/{len(idx)}] {q[:80]}")

    path = PROJECT_ROOT / "eval" / "eval_set.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out), encoding="utf-8")
    print(f"\nsaved {len(out)} eval items -> {path}")


if __name__ == "__main__":
    main()
