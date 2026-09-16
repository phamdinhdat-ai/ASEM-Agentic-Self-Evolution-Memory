"""Backend factory and exports."""

from __future__ import annotations

from typing import Any, Dict

from .base import InferenceBackend
from .huggingface_backend import HuggingFaceBackend
from .langchain_backend import LangChainBackend
from .openai_backend import OpenAIBackend


def build_backend(config: Dict[str, Any]) -> InferenceBackend:
    backend = config.get("backend")
    if backend == "huggingface":
        hf_cfg = config.get("huggingface", {})
        return HuggingFaceBackend.from_config(hf_cfg)
    if backend == "langchain":
        lc_cfg = config.get("langchain", {})
        return LangChainBackend.from_config(lc_cfg)
    if backend == "openai":
        # Native openai SDK client — no LangChain in the request path.
        api_cfg = config.get("openai", {}) or {}
        return OpenAIBackend.from_config(api_cfg)
    raise ValueError(f"Unknown backend: {backend}")


__all__ = [
    "InferenceBackend",
    "HuggingFaceBackend",
    "LangChainBackend",
    "OpenAIBackend",
    "build_backend",
]
