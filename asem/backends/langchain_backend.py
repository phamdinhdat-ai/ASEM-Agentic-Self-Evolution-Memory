"""LangChain backend implementation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import InferenceBackend


def _content_to_text(content: Any) -> str:
    """Normalize a chat response's ``content`` field to a plain string.

    Some models/proxies (e.g. OpenAI-format content parts) return
    ``content`` as a list of blocks — ``[{"type": "text", "text": ...}]`` —
    instead of a plain string. Concatenating the text parts keeps the
    backend contract (``generate() -> str``) intact so downstream JSON
    parsers see the raw response text.
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


class LangChainBackend(InferenceBackend):
    """LangChain inference backend using BaseChatModel and Embeddings."""

    def __init__(self, llm: Any, embedder: Any) -> None:
        super().__init__()
        self._llm = llm
        self._embedder = embedder

    def generate(self, prompt: str, **kwargs) -> str:
        response = self._llm.invoke(prompt)
        # Extract token usage from LangChain response metadata when available
        if hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("token_usage", {})
            if usage:
                self._token_count += usage.get("total_tokens", 0)
        if hasattr(response, "content"):
            return _content_to_text(response.content)
        return str(response)

    async def agenerate(self, prompt: str, **kwargs) -> str:
        response = await self._llm.ainvoke(prompt)
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

            def invoke(self, prompt: str):
                return self._inner.invoke([HumanMessage(content=prompt)])

            async def ainvoke(self, prompt: str):
                return await self._inner.ainvoke([HumanMessage(content=prompt)])

            async def astream(self, prompt: str):
                async for chunk in self._inner.astream([HumanMessage(content=prompt)]):
                    yield chunk

        return cls(llm=_Wrapper(llm), embedder=embedder)


def _build_llm(provider: str, model_name: str, temperature: float, cfg: Dict[str, Any]) -> Any:
    if provider == "openai":
        from langchain_openai import ChatOpenAI 

        import os
        kwargs: Dict[str, Any] = {"model": model_name, "temperature": temperature}
        if cfg.get("max_tokens"):
            kwargs["max_tokens"] = int(cfg["max_tokens"])
        base_url = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL")
        enable_reasoning = cfg.get("enable_reasoning", False)
        if enable_reasoning:
            kwargs["reasoning"] = {
                "effort": "high",
                "summary" : None,
            }
            kwargs["extra_body"] = {
                
                "chat_template_kwargs": {
                    "enable_thinking": True
                }
            }
        else:
            kwargs["reasoning"] = {
                "effort": "none",
                "summary" : None,
            }
            kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
        if base_url:
            kwargs["base_url"] = base_url
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
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_name)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model_name)
    raise ValueError(f"Unsupported embedding provider: {provider}")
