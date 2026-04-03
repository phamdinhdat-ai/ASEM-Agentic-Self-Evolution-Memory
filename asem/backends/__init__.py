"""Backend factory and exports."""

from __future__ import annotations

from typing import Any, Dict

from .huggingface_backend import HuggingFaceBackend
from .langchain_backend import LangChainBackend
from .base import InferenceBackend


def build_backend(config: Dict[str, Any]) -> InferenceBackend:
    backend = config.get("backend")
    if backend == "huggingface":
        hf_cfg = config.get("huggingface", {})
        return HuggingFaceBackend.from_config(hf_cfg)
    if backend == "langchain":
        lc_cfg = config.get("langchain", {})
        return LangChainBackend.from_config(lc_cfg)
    raise ValueError(f"Unknown backend: {backend}")


__all__ = ["InferenceBackend", "HuggingFaceBackend", "LangChainBackend", "build_backend"]
