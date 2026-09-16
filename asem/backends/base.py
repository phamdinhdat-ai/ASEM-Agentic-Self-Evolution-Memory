"""Backend interface definition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


def content_to_text(content: Any) -> str:
    """Normalize a chat response's ``content`` field to a plain string.

    Some models/proxies return ``content`` as a list of blocks —
    ``[{"type": "text", "text": ...}]`` — instead of a plain string.
    Concatenating the text parts keeps the backend contract
    (``generate() -> str``) intact so downstream JSON parsers see raw text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    return str(content)


def build_thinking_body(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Request-body fields that control the model's thinking / reasoning.

    The API uses the OpenAI-compatible format::

        thinking:          {"type": "enabled" | "disabled"}    # toggle
        reasoning_effort:  "low" | "high" | "max"              # effort

    ``enable_reasoning: true`` is also honoured, for the older vLLM/Qwen
    ``chat_template_kwargs.enable_thinking`` convention. ``extra_body`` is merged
    last, so a config can always hand-write a raw override.

    These are not native openai-SDK parameters, so backends pass them through
    ``extra_body`` — which also works on SDK versions that don't know them.
    """
    body: Dict[str, Any] = {}
    thinking = cfg.get("thinking")
    if thinking is not None:
        body["thinking"] = thinking
    effort = cfg.get("reasoning_effort")
    if effort is not None:
        body["reasoning_effort"] = effort
    if cfg.get("enable_reasoning", cfg.get("reasoning", False)):
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = True
    body.update(cfg.get("extra_body") or {})
    return body


class InferenceBackend(ABC):
    """Interface for inference backends used by ASEM stages."""

    _EMBED_CACHE_MAX = 4096

    def __init__(self) -> None:
        self._token_count: int = 0
        self._embed_cache: Dict[str, np.ndarray] = {}
        self._embed_cache_order: List[str] = []

    @property
    def token_count(self) -> int:
        """Total tokens consumed by all generate() calls so far."""
        return self._token_count

    def reset_token_count(self) -> None:
        """Reset the token counter to zero."""
        self._token_count = 0

    def reset_embed_cache(self) -> None:
        """Clear the internal embedding cache (e.g. between conversations)."""
        self._embed_cache.clear()
        self._embed_cache_order.clear()

    def embed(self, text: str) -> np.ndarray:
        """Return a dense vector for the given text, with LRU caching.

        Identical inputs (e.g. repeated history turns or queries) are
        embedded once and served from cache — a large win on API embedders
        and repeated-ingestion paths.
        """
        cached = self._embed_cache.get(text)
        if cached is not None:
            return cached
        vector = self._embed(text)
        self._cache_embed(text, vector)
        return vector

    @abstractmethod
    def _embed(self, text: str) -> np.ndarray:
        """Backend-specific embedding. Do not call directly; use embed()."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Single-turn text generation."""

    def _cache_embed(self, text: str, vector: np.ndarray) -> None:
        if text in self._embed_cache:
            return
        self._embed_cache[text] = vector
        self._embed_cache_order.append(text)
        if len(self._embed_cache_order) > self._EMBED_CACHE_MAX:
            oldest = self._embed_cache_order.pop(0)
            self._embed_cache.pop(oldest, None)
