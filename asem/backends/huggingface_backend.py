"""HuggingFace backend implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import InferenceBackend


class HuggingFaceBackend(InferenceBackend):
    """HuggingFace inference backend using transformers + sentence-transformers."""

    def __init__(self, text_generator: Any, embedder: Any) -> None:
        super().__init__()
        self._text_generator = text_generator
        self._embedder = embedder
        # Try to grab the tokenizer for accurate token counting
        self._tokenizer = None
        try:
            if hasattr(text_generator, "tokenizer"):
                self._tokenizer = text_generator.tokenizer
            elif hasattr(text_generator, "model"):
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    text_generator.model.config._name_or_path
                )
        except Exception:
            pass

    def generate(self, prompt: str, **kwargs) -> str:
        # Count prompt tokens before generation
        if self._tokenizer is not None:
            try:
                prompt_tokens = len(self._tokenizer.encode(prompt))
            except Exception:
                prompt_tokens = len(prompt) // 4
        else:
            prompt_tokens = len(prompt) // 4

        outputs = self._text_generator(prompt, **kwargs)
        if not outputs:
            return ""

        first = outputs[0]
        if isinstance(first, dict):
            if "generated_text" in first:
                result = str(first["generated_text"])
            elif "text" in first:
                result = str(first["text"])
            else:
                result = str(first)
        else:
            result = str(first)

        # Count completion tokens
        if self._tokenizer is not None:
            try:
                completion_tokens = len(self._tokenizer.encode(result))
            except Exception:
                completion_tokens = len(result) // 4
        else:
            completion_tokens = len(result) // 4

        self._token_count += prompt_tokens + completion_tokens
        return result

    def _embed(self, text: str) -> np.ndarray:
        vector = self._embedder.encode(text, convert_to_numpy=True)
        vector = np.asarray(vector)
        if vector.ndim > 1:
            vector = vector.reshape(-1)
        return vector

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "HuggingFaceBackend":
        from transformers import pipeline as hf_pipeline
        from sentence_transformers import SentenceTransformer

        model_name = cfg.get("model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
        pipeline_task = cfg.get("pipeline_task", "text-generation")
        device_map = cfg.get("device_map", "auto")
        max_new_tokens = cfg.get("max_new_tokens", 512)
        temperature = cfg.get("temperature", 0.0)
        load_in_4bit = bool(cfg.get("load_in_4bit", False))

        model_kwargs: Dict[str, Any] = {}
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        text_generator = hf_pipeline(
            pipeline_task,
            model=model_name,
            device_map=device_map,
            model_kwargs=model_kwargs or None,
        )

        embedder_name = cfg.get(
            "embedder_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        embedder = SentenceTransformer(embedder_name)

        return cls(
            text_generator=text_generator,
            embedder=embedder,
        )
