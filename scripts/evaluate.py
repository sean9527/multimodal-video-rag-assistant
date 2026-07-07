"""Evaluate retrieval quality: Recall@k and MRR, dense vs dense+rerank.

Usage:  python scripts/evaluate.py
Reads eval/eval_set.jsonl (build it with build_eval.py first). Retrieval is
scoped to each query's course (how the app actually runs).

- stage-1 Recall@N : is the gold chunk retrievable at all (in the N dense
  candidates)? This is the ceiling reranking cannot exceed.
- Recall@k / MRR    : does the gold chunk reach the final top-k the LLM sees,
  and how highly is it ranked — compared for dense vs dense+rerank.
"""
from __future__ import annotations

import json

from mvrag.config import PROJECT_ROOT, settings
from mvrag.embedding.text import embed_texts
from mvrag.indexing.store import search_text
from mvrag.retrieval.rerank import rerank_chunks
from mvrag.schemas import RetrievedChunk


def _key(video_id: str, start: float) -> tuple[str, float]:
    return (video_id, round(float(start), 1))


def _rank(gold: tuple[str, float], ordered: list[RetrievedChunk]) -> int | None:
    keys = [_key(c.video_id, c.start) for c in ordered]
    return keys.index(gold) + 1 if gold in keys else None


def main() -> None:
    path = PROJECT_ROOT / "eval" / "eval_set.jsonl"
    if not path.exists():
        print(f"not found: {path}\n-> run  python scripts/build_eval.py  first")
        raise SystemExit(1)
    items = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    k, n_cand = settings.top_k_text, settings.rerank_top_n
    stage1 = d_rec = r_rec = 0
    d_mrr = r_mrr = 0.0

    for item in items:
        gold = _key(item["video_id"], item["start"])
        qv = embed_texts([item["question"]])[0].tolist()
        cands = [
            RetrievedChunk(video_id=h["video_id"], start=h["start"], end=h["end"],
                           text=h["text"], score=1.0 - h["_distance"])
            for h in search_text(qv, n_cand, course_id=item["course_id"])
        ]
        rd = _rank(gold, cands)                                          # dense rank
        rr = _rank(gold, rerank_chunks(item["question"], cands, len(cands)))  # reranked rank

        if rd and rd <= n_cand:
            stage1 += 1
        if rd and rd <= k:
            d_rec += 1
        if rd:
            d_mrr += 1.0 / rd
        if rr and rr <= k:
            r_rec += 1
        if rr:
            r_mrr += 1.0 / rr

    n = len(items)
    print(f"eval items: {n}    top_k={k}    rerank candidates={n_cand}\n")
    print(f"stage-1 dense Recall@{n_cand} (retrievable ceiling): {stage1 / n:.3f}\n")
    print(f"{'method':<16}{'Recall@' + str(k):<12}{'MRR':<8}")
    print(f"{'dense':<16}{d_rec / n:<12.3f}{d_mrr / n:<8.3f}")
    print(f"{'dense+rerank':<16}{r_rec / n:<12.3f}{r_mrr / n:<8.3f}")


if __name__ == "__main__":
    main()
