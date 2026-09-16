"""Native OpenAI-SDK inference backend — no LangChain in the request path.

Talks straight to any OpenAI-compatible ``/v1/chat/completions`` endpoint using
the official ``openai`` client, so a call is ``client.chat.completions.create``
with no ``ChatOpenAI`` indirection. Select it with ``backend: "openai"``:

    inference:
      backend: "openai"
      openai:
        model: "deepseek-v4-flash"
        temperature: 0.0
        max_tokens: 4096
        timeout: 120
        # base_url / api_key default to OPENAI_BASE_URL / OPENAI_API_KEY from .env
        #
        # Thinking control (OpenAI-compatible format used by this endpoint):
        #   thinking:         {"type": "enabled" | "disabled"}
        #   reasoning_effort: "low" | "high" | "max"
        thinking:
          type: "disabled"
        # Anything else can be hand-written as a raw body override:
        # extra_body:
        #   some_server_flag: 1
        embedder_provider: "huggingface"
        embedder_name: "sentence-transformers/all-MiniLM-L6-v2"

Embeddings deliberately reuse the SAME builder as the LangChain backend
(``_build_embedder``). The memory bank stores embedder output, so swapping the
LLM path must not change the vectors: banks built with ``backend: "langchain"``
stay byte-compatible with this backend, and the ingest/retrieve phases (or two
different backends) can share one bank.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from .base import InferenceBackend, build_thinking_body, content_to_text
from .langchain_backend import _build_embedder


class OpenAIBackend(InferenceBackend):
    """Chat-completions backend built on the official ``openai`` client."""

    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        embedder: Any = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra_body = dict(extra_body or {})
        self._embedder = embedder
        # Extra create() kwargs merged into every request (escape hatch for
        # server-specific parameters without editing this module).
        self._request_kwargs = dict(request_kwargs or {})

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "OpenAIBackend":
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The 'openai' package is required for backend: 'openai' "
                "(pip install openai)"
            ) from exc

        model = cfg.get("model")
        if not model:
            raise ValueError("OpenAI backend config requires a 'model'")

        base_url = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL")
        api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key for the OpenAI backend: set OPENAI_API_KEY in .env "
                "or 'api_key' in the config's openai block."
            )

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(cfg.get("timeout", 120.0)),
        )

        max_tokens = cfg.get("max_tokens")
        return cls(
            client=client,
            model=str(model),
            temperature=float(cfg.get("temperature", 0.0)),
            max_tokens=int(max_tokens) if max_tokens else None,
            # Thinking controls: thinking={"type": "enabled"|"disabled"},
            # reasoning_effort="low"|"high"|"max", plus raw extra_body.
            extra_body=build_thinking_body(cfg) or None,
            embedder=_build_embedder(cfg),
            request_kwargs=cfg.get("request_kwargs"),
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Single-turn generation via ``chat.completions.create``.

        Per-call overrides (``model``, ``temperature``, ``max_tokens``, or any
        other ``create`` kwarg) are accepted and take precedence over the
        config defaults.
        """
        params: Dict[str, Any] = {
            "model": kwargs.pop("model", self._model),
            "messages": [{"role": "user", "content": prompt}],
        }

        temperature = kwargs.pop("temperature", self._temperature)
        if temperature is not None:
            params["temperature"] = float(temperature)

        max_tokens = kwargs.pop("max_tokens", self._max_tokens)
        if max_tokens:
            params["max_tokens"] = int(max_tokens)

        if self._extra_body:
            params["extra_body"] = self._extra_body

        params.update(self._request_kwargs)
        params.update(kwargs)

        response = self._client.chat.completions.create(**params)

        usage = getattr(response, "usage", None)
        total = getattr(usage, "total_tokens", None) if usage is not None else None
        if total:
            self._token_count += int(total)

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError(
                f"Empty response from model {self._model!r} "
                "(no choices — check max_tokens and the endpoint)"
            )
        return content_to_text(choices[0].message.content)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        vector = self._embedder.embed_query(text)
        return np.asarray(vector, dtype=float)


__all__ = ["OpenAIBackend"]
