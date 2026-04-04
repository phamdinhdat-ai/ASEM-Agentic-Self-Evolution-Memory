"""LangChain backend implementation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import InferenceBackend


class LangChainBackend(InferenceBackend):
    """LangChain inference backend using BaseChatModel and Embeddings."""

    def __init__(self, llm: Any, embedder: Any) -> None:
        self._llm = llm
        self._embedder = embedder

    def generate(self, prompt: str, **kwargs) -> str:
        response = self._llm.invoke(prompt)
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    async def agenerate(self, prompt: str, **kwargs) -> str:
        response = await self._llm.ainvoke(prompt)
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    def embed(self, text: str) -> np.ndarray:
        vector = self._embedder.embed_query(text)
        return np.asarray(vector, dtype=float)

    async def aembed(self, text: str) -> np.ndarray:
        vector = await self._embedder.aembed_query(text)
        return np.asarray(vector, dtype=float)
    
    async def  stream(self, prompt: str, **kwargs) -> Any:
        async for response in self._llm.astream(prompt):
            if hasattr(response, "content"):
                yield str(response.content)
            else:
                yield str(response)
    async def astream(self, prompt: str, **kwargs) -> Any:
        async for response in self._llm.astream(prompt):
            if hasattr(response, "content"):
                yield str(response.content)
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

        return ChatOpenAI(model=model_name, temperature=temperature)
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
