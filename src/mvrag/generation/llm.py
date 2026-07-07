"""Pluggable LLM backend for generation.

The whole system touches an LLM in exactly ONE place (generation), so swapping
the local model for a cloud API is a config change + a small adapter here — no
changes to ingestion, embedding, retrieval, or the UI (dependency inversion:
the high-level answer logic depends on the `LLMClient` interface, not on any
specific provider).

Provider is chosen by `settings.llm_provider`. Cloud API keys come from the
standard env vars (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`), which `config.py`
also loads from `.env`. Cloud SDKs are optional — `pip install -e ".[api]"`.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from mvrag.config import settings


class LLMClient(Protocol):
    def chat(self, system: str, user: str) -> str: ...


class OllamaLLM:
    """Local model via Ollama (default). No API key, runs offline."""

    def __init__(self) -> None:
        import ollama

        self._client = ollama.Client(host=settings.ollama_host)
        self._model = settings.ollama_model

    def chat(self, system: str, user: str) -> str:
        resp = self._client.chat(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={"temperature": 0.2},
        )
        return resp["message"]["content"]


class OpenAILLM:
    """OpenAI (or any OpenAI-compatible endpoint). Reads OPENAI_API_KEY from env."""

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI()  # picks up OPENAI_API_KEY (+ optional OPENAI_BASE_URL)
        self._model = settings.openai_model

    def chat(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""


class ClaudeLLM:
    """Anthropic Claude. Reads ANTHROPIC_API_KEY from env."""

    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY
        self._model = settings.anthropic_model

    def chat(self, system: str, user: str) -> str:
        # Anthropic takes `system` as a top-level parameter, not a message role.
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


_REGISTRY: dict[str, type] = {
    "ollama": OllamaLLM,
    "openai": OpenAILLM,
    "anthropic": ClaudeLLM,
}


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    provider = settings.llm_provider.lower()
    if provider not in _REGISTRY:
        raise ValueError(
            f"unknown MVRAG_LLM_PROVIDER={provider!r}; expected one of {list(_REGISTRY)}"
        )
    return _REGISTRY[provider]()
