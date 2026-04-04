"""Backend contract compliance tests."""

import os
import shutil
import subprocess
import numpy as np
import pytest

from asem.backends import HuggingFaceBackend, LangChainBackend, build_backend


def _skip_if_missing_deps() -> None:
    pytest.importorskip("transformers")
    pytest.importorskip("sentence_transformers")


def _get_ollama_model_or_skip() -> str:
    pytest.importorskip("langchain_ollama")

    if shutil.which("ollama") is None:
        pytest.skip("ollama CLI not found")

    configured = os.getenv("ASEM_TEST_OLLAMA_MODEL")
    if configured:
        return configured

    proc = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip("ollama server is not available")

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        pytest.skip("no local ollama models found")

    # Expected format includes header on first line; first column is the model name.
    models = [line.split()[0] for line in lines[1:] if line.split()]
    if not models:
        pytest.skip("unable to determine ollama model from `ollama list`")

    # Prefer chat-capable models over embedding-only models.
    for model in models:
        lowered = model.lower()
        if "embed" in lowered or "embedding" in lowered:
            continue
        return model

    pytest.skip("only embedding-only ollama models found")


def test_huggingface_backend_contract() -> None:
    _skip_if_missing_deps()

    cfg = {
        "model_name_or_path": "gemma3:1b",
        "pipeline_task": "text-generation",
        "max_new_tokens": 8,
        "temperature": 0.0,
        "device_map": "cpu",
        "embedder_name": "granite-embedding:latest",
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


def test_langchain_backend_with_ollama_smoke() -> None:
    model = _get_ollama_model_or_skip()

    config = {
        "backend": "langchain",
        "langchain": {
            "provider": "ollama",
            "model": model,
            "temperature": 0.0,
            "embedder_provider": "ollama",
            "embedder_model": model,
        },
    }

    backend = build_backend(config)
    assert isinstance(backend, LangChainBackend)

    text = backend.generate("Reply with exactly: OK")
    assert isinstance(text, str)
    assert text.strip() != ""

    vec = backend.embed("hello from asem")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.size > 0
