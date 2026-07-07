"""Visual + cross-modal embeddings via OpenCLIP.

CLIP maps images and text into ONE shared space (contrastive training on
image-caption pairs), so a text query can retrieve visually-relevant keyframes.
- embed_images():    keyframes -> vectors   (used at index time)
- embed_query_text(): a text query -> vector (used at query time, SAME space)
Both are L2-normalized so a dot product is cosine similarity.

Limitation (documented): this OpenCLIP checkpoint is English-oriented and trained
on natural images, so it's strongest on real footage (demos, experiments) and
weaker on code/slide screencasts or non-English text-in-image. Future upgrade:
multilingual / Chinese-CLIP.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from mvrag.config import settings


@lru_cache(maxsize=1)
def _model():
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        settings.clip_model,
        pretrained=settings.clip_pretrained,
        cache_dir=str(settings.model_cache_dir),
    )
    tokenizer = open_clip.get_tokenizer(settings.clip_model)
    return model.to(settings.device).eval(), preprocess, tokenizer


def embed_images(paths: list[str], batch_size: int = 32) -> np.ndarray:
    """Return an (N, dim) float32 array of L2-normalized image embeddings."""
    model, preprocess, _ = _model()
    out: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        batch = [preprocess(Image.open(p).convert("RGB")) for p in paths[i : i + batch_size]]
        x = torch.stack(batch).to(settings.device)
        with torch.no_grad():
            feats = model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0, 512), dtype="float32")


def embed_query_text(texts: list[str]) -> np.ndarray:
    """Embed text into the CLIP space (for text->image retrieval)."""
    model, _, tokenizer = _model()
    tokens = tokenizer(texts).to(settings.device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()
