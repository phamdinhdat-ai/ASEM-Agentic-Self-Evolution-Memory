"""HuggingFace backend implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import InferenceBackend


class HuggingFaceBackend(InferenceBackend):
    """HuggingFace inference backend using transformers + sentence-transformers."""

    def __init__(
        self,
        text_generator: Any,
        embedder: Any,
        generation_defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._text_generator = text_generator
        self._embedder = embedder
        # Generation kwargs applied on every ``generate`` call that does not
        # override them (e.g. max_new_tokens, do_sample, return_full_text).
        self._generation_defaults: Dict[str, Any] = dict(generation_defaults or {})
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

        call_kwargs = {**self._generation_defaults, **kwargs}
        outputs = self._text_generator(prompt, **call_kwargs)
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
        max_new_tokens = int(cfg.get("max_new_tokens", 512))
        temperature = float(cfg.get("temperature", 0.0) or 0.0)
        load_in_4bit = bool(cfg.get("load_in_4bit", False))

        model_kwargs: Dict[str, Any] = {}
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        torch_dtype = cfg.get("torch_dtype")
        if torch_dtype:
            model_kwargs["torch_dtype"] = _resolve_torch_dtype(torch_dtype)

        # Generation defaults applied by ``generate`` on every call. Without
        # these the pipeline falls back to a tiny default (max_new_tokens=20),
        # which truncates JSON/answers for the small local models.
        generation_defaults: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "return_full_text": bool(cfg.get("return_full_text", False)),
        }
        if temperature > 0:
            generation_defaults["temperature"] = temperature
            generation_defaults["do_sample"] = True
        else:
            generation_defaults["do_sample"] = False

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
            generation_defaults=generation_defaults,
        )


def _resolve_torch_dtype(value: Any) -> Any:
    """Map a config string (e.g. "float32", "bfloat16", "auto") to torch dtype."""
    import torch

    if not isinstance(value, str):
        return value
    mapping = {
        "auto": "auto",
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(value.lower(), value)
