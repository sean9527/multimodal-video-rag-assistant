"""Ask a question about indexed video(s) — end to end, scoped by course/video.

Usage:
  python scripts/ask.py "your question" --course CSCE2230           # whole course
  python scripts/ask.py "your question" --video <video_id>          # one video
  python scripts/ask.py "your question" --course CSCE2230 --video X # narrow to one
Requires: video(s) ingested + indexed, and Ollama running with a model.
"""
from __future__ import annotations

import argparse

from mvrag.generation.answerer import answer
from mvrag.schemas import format_timestamp


def main() -> None:
    p = argparse.ArgumentParser(description="Ask a question about indexed video(s).")
    p.add_argument("question")
    p.add_argument("--course", default=None, help="scope to a course_id")
    p.add_argument("--video", default=None, help="scope to a single video_id (filename stem)")
    args = p.parse_args()

    scope = ", ".join(s for s in (f"course={args.course}" if args.course else "",
                                  f"video={args.video}" if args.video else "") if s)
    print(f"scope: {scope or 'ALL (no filter)'}\n")

    result = answer(args.question, course_id=args.course, video_id=args.video)

    print(f"Q: {args.question}\n")
    print(f"A: {result['answer']}\n")
    if not result["chunks"]:
        from mvrag.indexing.store import list_scope
        print("no transcript matched this scope. available scopes in the store:")
        for cid, vids in list_scope().items():
            print(f"  course '{cid}': {vids}")
        return

    print("retrieved transcript sources (all fed to the LLM):")
    for c in result["chunks"]:
        print(f"  [{c.video_id} @ {format_timestamp(c.start)}-{format_timestamp(c.end)}] {c.text[:60]}...")


if __name__ == "__main__":
    main()
