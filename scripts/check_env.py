"""Environment sanity check.

Run:  python scripts/check_env.py

Verifies the external tools and (once installed) the ML stack the pipeline
needs. Safe to run at any phase — it degrades gracefully when a dependency
isn't installed yet.
"""
from __future__ import annotations

import json
import shutil
import sys
import urllib.request


def mark(passed: bool) -> str:
    return "✅" if passed else "❌"


def main() -> None:
    print("== Multimodal Video RAG - environment check ==\n")

    # Python
    py_ok = sys.version_info >= (3, 10)
    print(f"{mark(py_ok)} Python {sys.version.split()[0]}  (need >= 3.10)")

    # ffmpeg (used to extract audio + frames)
    ff = shutil.which("ffmpeg")
    print(f"{mark(ff is not None)} ffmpeg  {('-> ' + ff) if ff else 'NOT FOUND (sudo apt install ffmpeg)'}")

    # torch + CUDA (installed in Phase 1)
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"{mark(True)} torch {torch.__version__}")
        gpu = f" ({torch.cuda.get_device_name(0)})" if cuda else ""
        print(f"{mark(cuda)} CUDA available: {cuda}{gpu}")
    except ImportError:
        print('⏳ torch not installed yet  ->  pip install -e ".[ml]"  (Phase 1)')

    # Ollama (local LLM server)
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            models = [m["name"] for m in json.load(r).get("models", [])]
        print(f"✅ Ollama reachable @ :11434, models: {models or '(none pulled yet)'}")
    except Exception:
        print("❌ Ollama not reachable @ :11434  (install from https://ollama.com, then run `ollama serve`)")

    print("\nDone.")


if __name__ == "__main__":
    main()
