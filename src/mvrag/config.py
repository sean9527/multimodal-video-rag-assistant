"""Central configuration for the Multimodal Video RAG Assistant (mvrag).

Every tunable lives here and can be overridden via environment variables or a
.env file (prefix: MVRAG_). Using pydantic-settings keeps config typed,
validated, and self-documenting — nicer than scattering os.getenv() calls, and
a good thing to point at in a code review.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# repo root = two levels up from src/mvrag/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MVRAG_",
        protected_namespaces=(),  # allow field names like `model_cache_dir`
        extra="ignore",
    )

    # --- Storage paths (gitignored; regenerable) ---
    data_dir: Path = PROJECT_ROOT / "data"                # uploads + extracted artifacts
    storage_dir: Path = PROJECT_ROOT / "storage"          # vector db + model cache
    model_cache_dir: Path = PROJECT_ROOT / "storage" / "models"
    lancedb_path: Path = PROJECT_ROOT / "storage" / "lancedb"

    # --- Compute ---
    device: str = "cuda"                    # cuda | cpu | mps

    # --- ASR: Whisper ---
    whisper_model: str = "large-v3"         # tiny|base|small|medium|large-v3
    whisper_compute_type: str = "float16"   # float16 (GPU) | int8 (CPU)

    # --- Keyframe extraction ---
    scene_threshold: float = 27.0           # PySceneDetect content threshold
    max_keyframes: int = 200                # safety cap per video (MVP)
    frame_sample_interval: float = 5.0      # uniform-sampling fallback: 1 frame / N sec

    # --- Embeddings ---
    text_embed_model: str = "BAAI/bge-m3"   # multilingual (En + 中文), 1024-dim
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

    # --- Transcript chunking ---
    chunk_target_chars: int = 350       # smaller chunk = finer retrieval granularity
    chunk_overlap_chars: int = 80        # (tuned down from 800/150 after observing dilution)

    # --- Retrieval ---
    top_k_text: int = 6
    top_k_image: int = 4

    # --- Reranking (Phase 3.5) ---
    use_reranker: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"   # multilingual cross-encoder
    rerank_top_n: int = 20                          # dense candidates to rerank

    # --- Generation: Ollama (local LLM) ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"          # Ollama's 7b tag is already instruct-tuned
    ollama_vision_model: str = "qwen2.5vl:7b"   # optional keyframe verification

    def ensure_dirs(self) -> None:
        """Create all storage dirs if missing (safe to call on startup)."""
        for p in (self.data_dir, self.storage_dir, self.model_cache_dir, self.lancedb_path):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
