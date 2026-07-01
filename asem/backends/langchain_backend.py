"""LangChain backend implementation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import InferenceBackend
from ..logging_utils import get_logger

_logger = get_logger(__name__)


class LangChainBackend(InferenceBackend):
    """LangChain inference backend using BaseChatModel and Embeddings."""

    def __init__(self, llm: Any, embedder: Any) -> None:
        self._llm = llm
        self._embedder = embedder
        _logger.info("LangChainBackend initialized (llm={}, embedder={})",
                     type(llm).__name__, type(embedder).__name__)

    def generate(self, prompt: str, **kwargs) -> str:
        _logger.debug("LLM generate | prompt_len={} | prompt_preview={!r}",
                      len(prompt), prompt[:120])
        try:
            response = self._llm.invoke(prompt)
        except Exception as exc:
            _logger.opt(exception=exc).error("LLM generate failed")
            raise

        if hasattr(response, "content"):
            result = str(response.content)
        else:
            result = str(response)

        _logger.debug("LLM response | len={} | preview={!r}",
                      len(result), result[:200])
        return result

    async def agenerate(self, prompt: str, **kwargs) -> str:
        _logger.debug("LLM agenerate | prompt_len={}", len(prompt))
        try:
            response = await self._llm.ainvoke(prompt)
        except Exception as exc:
            _logger.opt(exception=exc).error("LLM agenerate failed")
            raise

        if hasattr(response, "content"):
            result = str(response.content)
        else:
            result = str(response)

        _logger.debug("LLM async response | len={} | preview={!r}",
                      len(result), result[:200])
        return result

    def embed(self, text: str) -> np.ndarray:
        _logger.debug("Embed text | len={} | preview={!r}",
                      len(text), text[:100])
        try:
            vector = self._embedder.embed_query(text)
        except Exception as exc:
            _logger.opt(exception=exc).error("Embedding failed")
            raise

        result = np.asarray(vector, dtype=float)
        _logger.debug("Embed result | shape={} | norm={:.4f}",
                      result.shape, float(np.linalg.norm(result)))
        return result

    async def aembed(self, text: str) -> np.ndarray:
        _logger.debug("Embed async | text_len={}", len(text))
        try:
            vector = await self._embedder.aembed_query(text)
        except Exception as exc:
            _logger.opt(exception=exc).error("Async embedding failed")
            raise

        result = np.asarray(vector, dtype=float)
        _logger.debug("Embed async result | shape={}", result.shape)
        return result
    
    async def stream(self, prompt: str, **kwargs) -> Any:
        _logger.debug("LLM stream | prompt_len={}", len(prompt))
        try:
            async for response in self._llm.astream(prompt):
                if hasattr(response, "content"):
                    yield str(response.content)
                else:
                    yield str(response)
        except Exception as exc:
            _logger.opt(exception=exc).error("LLM stream failed")
            raise

    async def astream(self, prompt: str, **kwargs) -> Any:
        _logger.debug("LLM astream | prompt_len={}", len(prompt))
        try:
            async for response in self._llm.astream(prompt):
                if hasattr(response, "content"):
                    yield str(response.content)
                else:
                    yield str(response)
        except Exception as exc:
            _logger.opt(exception=exc).error("LLM astream failed")
            raise
    


    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LangChainBackend":
        try:
            from langchain_core.messages import HumanMessage
        except ImportError as exc:
            _logger.error("langchain-core not installed")
            raise ImportError("langchain-core is required for LangChain backend") from exc

        provider = cfg.get("provider", "openai")
        model_name = cfg.get("model")
        temperature = cfg.get("temperature", 0.0)

        _logger.info("Building LangChain LLM: provider={} model={} temperature={}",
                     provider, model_name, temperature)

        llm = _build_llm(provider, model_name, temperature, cfg)
        embedder = _build_embedder(cfg)

        _logger.info("LangChainBackend built successfully")

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
        kwargs['extra_body'] = {
            "chat_template_kwargs": {"enable_thinking": False},
        }
        base_url = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL")
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
    device = cfg.get("embedder_device", "cpu")  # default cpu since LLM is API-based

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model_name)
    if provider in {"huggingface_hub", "huggingface"}:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model_name)
    raise ValueError(f"Unsupported embedding provider: {provider}")
