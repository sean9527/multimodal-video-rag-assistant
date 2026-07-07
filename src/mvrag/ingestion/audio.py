"""Audio track -> timestamped transcript, using faster-whisper.

Flow:  video --ffmpeg--> 16 kHz mono wav --faster-whisper--> TranscriptSegment[]

Why faster-whisper (not openai-whisper): it runs on the CTranslate2 backend,
~4x faster and lighter on VRAM (int8/float16 quantization) for the same Whisper
weights — the standard choice for production transcription.
"""
from __future__ import annotations

import ctypes
import glob
import os
import subprocess
from pathlib import Path

from mvrag.config import settings
from mvrag.schemas import TranscriptSegment


def extract_audio(video_path: str | Path, out_wav: str | Path) -> Path:
    """Extract a 16 kHz mono WAV (the format Whisper expects) via ffmpeg.

    Doing this explicitly (rather than handing the .mp4 straight to Whisper)
    keeps the step transparent and robust across container/codec quirks.
    """
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(out_wav)],
        check=True, capture_output=True,
    )
    return out_wav


def _preload_cuda_libs() -> None:
    """Best-effort: expose pip-installed CUDA-12 cuBLAS/cuDNN .so files to
    CTranslate2 (faster-whisper's GPU backend). CTranslate2 dlopen's these by
    soname and does NOT look inside site-packages/nvidia/*, so we preload them
    with RTLD_GLOBAL first. No-op if the CUDA-12 libs aren't present (e.g. the
    installed torch is a CUDA-13 build) — transcribe() then falls back to CPU.
    """
    import sysconfig

    site = sysconfig.get_paths().get("purelib", "")
    if not site:
        return
    # order matters: cudart -> cublas -> cudnn (later libs depend on earlier ones)
    patterns = [
        f"{site}/nvidia/cuda_runtime/lib/libcudart.so*",
        f"{site}/nvidia/cublas/lib/libcublas*.so*",
        f"{site}/nvidia/cudnn/lib/libcudnn*.so*",
    ]
    for pat in patterns:
        for so in sorted(glob.glob(pat)):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def transcribe(audio_path: str | Path) -> tuple[list[TranscriptSegment], str, float]:
    """Transcribe an audio (or video) file into timestamped segments.

    Returns (segments, detected_language, duration_seconds).
    Tries the configured device (GPU) first, then degrades to CPU int8 on any
    failure, so Phase 1 always produces output even if CUDA/cuDNN misbehaves.
    """
    from faster_whisper import WhisperModel

    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)

    def _run(device: str, compute_type: str):
        model = WhisperModel(
            settings.whisper_model,
            device=device,
            compute_type=compute_type,
            download_root=str(settings.model_cache_dir),
        )
        # vad_filter=True: Silero VAD skips silence, so long pauses in a lecture
        # don't produce empty/garbled segments and timestamps stay accurate.
        seg_iter, info = model.transcribe(str(audio_path), vad_filter=True, beam_size=5)
        segments = [
            TranscriptSegment(start=float(s.start), end=float(s.end), text=s.text.strip())
            for s in seg_iter
        ]
        return segments, info.language, float(info.duration)

    if settings.device == "cuda":
        _preload_cuda_libs()
        try:
            return _run("cuda", settings.whisper_compute_type)
        except Exception as e:  # cuDNN / driver / OOM -> degrade gracefully
            print(f"[whisper] GPU failed ({type(e).__name__}: {e}); falling back to CPU int8")
    return _run("cpu", "int8")
