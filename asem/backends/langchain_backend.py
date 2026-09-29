"""LangChain backend implementation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import InferenceBackend, build_thinking_body, content_to_text as _content_to_text


class LangChainBackend(InferenceBackend):
    """LangChain inference backend using BaseChatModel and Embeddings."""

    def __init__(
        self,
        llm: Any,
        embedder: Any,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self._llm = llm
        self._embedder = embedder
        # Config-level completion cap, kept so a caller budgeting a prompt
        # against a context window can reserve exactly what will be requested.
        self._max_tokens = int(max_tokens) if max_tokens else None

    @property
    def default_max_tokens(self) -> int | None:
        """The config-level ``max_tokens`` applied when a call passes none."""
        return self._max_tokens

    def generate(self, prompt: str, **kwargs) -> str:
        # Forward per-call overrides (e.g. ``max_tokens``): dropping them here
        # silently ignored the answer agent's cap and sent the client-wide
        # default instead, overflowing small context windows.
        response = self._llm.invoke(prompt, **kwargs)
        # Extract token usage from LangChain response metadata when available
        if hasattr(response, "response_metadata"):
            metadata = response.response_metadata or {}
            usage = metadata.get("token_usage", {})
            if usage:
                self._token_count += usage.get("total_tokens", 0)
            # vLLM / OpenAI-compatible servers report "length" when the
            # completion hit the budget, i.e. the JSON was never finished.
            self.last_finish_reason = (
                metadata.get("finish_reason") or metadata.get("stop_reason")
            )
        if hasattr(response, "content"):
            return _content_to_text(response.content)
        return str(response)

    async def agenerate(self, prompt: str, **kwargs) -> str:
        response = await self._llm.ainvoke(prompt, **kwargs)
        if hasattr(response, "content"):
            return _content_to_text(response.content)
        return str(response)

    def _embed(self, text: str) -> np.ndarray:
        vector = self._embedder.embed_query(text)
        return np.asarray(vector, dtype=float)

    async def aembed(self, text: str) -> np.ndarray:
        vector = await self._embedder.aembed_query(text)
        return np.asarray(vector, dtype=float)
    
    async def astream(self, prompt: str, **kwargs) -> Any:
        async for response in self._llm.astream(prompt):
            if hasattr(response, "content"):
                yield _content_to_text(response.content)
            else:
                yield str(response)


    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LangChainBackend":
        try:
            from langchain_core.messages import HumanMessage
        except ImportError as exc:
            raise ImportError("langchain-core is required for LangChain backend") from exc

        provider = cfg.get("provider", "openai")
        model_name = cfg.get("model")
        temperature = cfg.get("temperature", 0.0)

        llm = _build_llm(provider, model_name, temperature, cfg)
        embedder = _build_embedder(cfg)

        class _Wrapper:
            def __init__(self, inner):
                self._inner = inner

            def invoke(self, prompt: str, **kwargs):
                # Pass through per-call overrides (e.g. ``max_tokens``) so a
                # caller can budget output against a small context window
                # instead of being stuck with the client-wide default.
                return self._inner.invoke([HumanMessage(content=prompt)], **kwargs)

            async def ainvoke(self, prompt: str, **kwargs):
                return await self._inner.ainvoke([HumanMessage(content=prompt)], **kwargs)

            async def astream(self, prompt: str):
                async for chunk in self._inner.astream([HumanMessage(content=prompt)]):
                    yield chunk

        # Only the OpenAI provider applies `max_tokens` to the model, so it is
        # the only provider whose completion cap can be advertised (and thus
        # reserved when a caller budgets a prompt).
        cap = cfg.get("max_tokens") if provider == "openai" else None
        return cls(
            llm=_Wrapper(llm),
            embedder=embedder,
            max_tokens=int(cap) if cap else None,
        )


def _build_llm(provider: str, model_name: str, temperature: float, cfg: Dict[str, Any]) -> Any:
    if provider == "openai":
        from langchain_openai import ChatOpenAI 

        import os
        kwargs: Dict[str, Any] = {"model": model_name, "temperature": temperature}
        if cfg.get("max_tokens"):
            kwargs["max_tokens"] = int(cfg["max_tokens"])
        base_url = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL")
        api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        # Thinking / reasoning control: the OpenAI-compatible
        # `thinking: {type: enabled|disabled}` toggle and `reasoning_effort`,
        # plus the legacy chat_template_kwargs.enable_thinking form.
        extra_body = build_thinking_body(cfg)
        if extra_body:
            kwargs["extra_body"] = extra_body
        return ChatOpenAI(**kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, temperature=temperature)
    if provider in {"huggingface_hub", "huggingface"}:
        from langchain_huggingface import ChatHuggingFace

        return ChatHuggingFace(model_id=model_name, temperature=temperature)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model_name, temperature=temperature)
    raise ValueError(f"Unsupported LangChain provider: {provider}")


def _build_embedder(cfg: Dict[str, Any]) -> Any:
    provider = cfg.get("embedder_provider", cfg.get("provider", "openai"))
    model_name = cfg.get("embedder_name") or cfg.get("embedder_model")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model_name)
    if provider in {"huggingface_hub", "huggingface"}:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)
        except ImportError:
            # Fallback: wrap sentence-transformers directly
            return _SentenceTransformerEmbedder(model_name=model_name)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model_name)
    raise ValueError(f"Unsupported embedding provider: {provider}")


class _SentenceTransformerEmbedder:
    """Lightweight wrapper around sentence-transformers that exposes
    the same ``embed_query`` / ``embed_documents`` interface that
    LangChain embedders provide — used as a fallback when
    ``langchain-huggingface`` is not installed.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Neither langchain-huggingface nor sentence-transformers is "
                "installed. Install one of them:\n"
                "  pip install langchain-huggingface\n"
                "  pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed_query(self, text: str):
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts):
        return self._model.encode(texts, normalize_embeddings=True).tolist()

