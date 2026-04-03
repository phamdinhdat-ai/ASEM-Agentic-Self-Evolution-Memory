"""Backend contract compliance tests."""

import numpy as np
import pytest

from asem.backends import HuggingFaceBackend, LangChainBackend, build_backend


def _skip_if_missing_deps() -> None:
    pytest.importorskip("transformers")
    pytest.importorskip("sentence_transformers")


def test_huggingface_backend_contract() -> None:
    _skip_if_missing_deps()

    cfg = {
        "model_name_or_path": "sshleifer/tiny-gpt2",
        "pipeline_task": "text-generation",
        "max_new_tokens": 8,
        "temperature": 0.0,
        "device_map": "cpu",
        "embedder_name": "sentence-transformers/paraphrase-MiniLM-L3-v2",
    }
    backend = HuggingFaceBackend.from_config(cfg)

    text = backend.generate("Hello", max_new_tokens=4)
    assert isinstance(text, str)

    vec = backend.embed("hello world")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.size > 0


def test_langchain_backend_contract() -> None:
    pytest.importorskip("langchain_core")

    class _MockLLM:
        def invoke(self, prompt: str):
            class _Resp:
                def __init__(self, content: str):
                    self.content = content

            return _Resp("ok")

    class _MockEmbedder:
        def embed_query(self, text: str):
            return [0.1, 0.2, 0.3]

    backend = LangChainBackend(llm=_MockLLM(), embedder=_MockEmbedder())
    text = backend.generate("hi")
    assert isinstance(text, str)
    vec = backend.embed("hi")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1


def test_build_backend_factory_langchain() -> None:
    pytest.importorskip("langchain_core")

    config = {
        "backend": "langchain",
        "langchain": {
            "provider": "openai",
            "model": "gpt-4o",
        },
    }
    with pytest.raises(Exception):
        build_backend(config)


def test_build_backend_factory_huggingface() -> None:
    _skip_if_missing_deps()

    config = {
        "backend": "huggingface",
        "huggingface": {
            "model_name_or_path": "sshleifer/tiny-gpt2",
            "pipeline_task": "text-generation",
            "max_new_tokens": 8,
            "temperature": 0.0,
            "device_map": "cpu",
            "embedder_name": "sentence-transformers/paraphrase-MiniLM-L3-v2",
        },
    }
    backend = build_backend(config)
    assert isinstance(backend, HuggingFaceBackend)
