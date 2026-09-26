"""Backend contract compliance tests."""

import os
import shutil
import subprocess
from types import SimpleNamespace

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


def test_langchain_backend_normalizes_content_parts() -> None:
    """generate() must return a plain string even when the provider returns
    OpenAI-style content parts ([{"type": "text", "text": ...}]) instead of
    a plain string — otherwise downstream JSON parsers see the repr of the
    list and extraction silently yields zero notes."""
    pytest.importorskip("langchain_core")

    class _PartsLLM:
        def invoke(self, prompt: str):
            class _Resp:
                content = [
                    {"type": "text", "text": '```json\n[{"content": "hi"}]```'},
                    {"type": "text", "text": "trailer"},
                ]

            return _Resp()

    class _MockEmbedder:
        def embed_query(self, text: str):
            return [0.1, 0.2, 0.3]

    backend = LangChainBackend(llm=_PartsLLM(), embedder=_MockEmbedder())
    text = backend.generate("hi")
    assert isinstance(text, str)
    assert text == '```json\n[{"content": "hi"}]```\ntrailer'


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


# ---------------------------------------------------------------------------
# Native OpenAI-SDK backend (no LangChain in the request path)
# ---------------------------------------------------------------------------

class _StubEmbedder:
    def embed_query(self, text: str):
        return [0.0, 0.0, 0.0, 1.0]


class _FakeCompletions:
    """Stands in for ``client.chat.completions`` and records every request."""

    def __init__(self, content, total_tokens: int = 42, choices: bool = True) -> None:
        self.calls: list[dict] = []
        self._content = content
        self._total = total_tokens
        self._choices = choices

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._choices:
            return SimpleNamespace(choices=[], usage=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))],
            usage=SimpleNamespace(total_tokens=self._total),
        )


def _openai_backend(completions: _FakeCompletions):
    from asem.backends.openai_backend import OpenAIBackend

    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAIBackend(
        client=client,
        model="test-model",
        temperature=0.0,
        max_tokens=128,
        embedder=_StubEmbedder(),
    )


def test_openai_backend_request_shape_and_token_accounting() -> None:
    completions = _FakeCompletions("hello world", total_tokens=17)
    backend = _openai_backend(completions)

    assert backend.generate("prompt") == "hello world"
    sent = completions.calls[0]
    assert sent["model"] == "test-model"
    assert sent["messages"] == [{"role": "user", "content": "prompt"}]
    assert sent["temperature"] == 0.0
    assert sent["max_tokens"] == 128
    assert backend.token_count == 17

    # Per-call overrides win over the config defaults.
    backend.generate("p", model="other", temperature=0.7, max_tokens=8)
    sent = completions.calls[1]
    assert sent["model"] == "other"
    assert sent["temperature"] == 0.7
    assert sent["max_tokens"] == 8


def test_openai_backend_normalizes_content_parts() -> None:
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    backend = _openai_backend(_FakeCompletions(parts))
    assert backend.generate("p") == "a\nb"


def test_openai_backend_empty_choices_raises() -> None:
    backend = _openai_backend(_FakeCompletions("", choices=False))
    with pytest.raises(RuntimeError):
        backend.generate("p")


def test_openai_backend_passes_extra_body_through() -> None:
    from asem.backends.openai_backend import OpenAIBackend

    completions = _FakeCompletions("ok")
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = OpenAIBackend(
        client=client,
        model="m",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        embedder=_StubEmbedder(),
    )
    backend.generate("p")
    assert completions.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_openai_backend_embed_returns_array() -> None:
    backend = _openai_backend(_FakeCompletions("x"))
    vector = backend.embed("hello")
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (4,)


def test_openai_backend_shares_the_embedder_factory() -> None:
    """Vectors must stay identical, so banks built by either backend mix."""
    from asem.backends import langchain_backend, openai_backend

    assert openai_backend._build_embedder is langchain_backend._build_embedder


def test_build_backend_factory_openai(monkeypatch) -> None:
    from asem.backends import openai_backend

    seen: dict = {}

    def _fake_from_config(cfg):
        seen.update(cfg)
        return "sentinel"

    monkeypatch.setattr(
        openai_backend.OpenAIBackend, "from_config", staticmethod(_fake_from_config)
    )

    config = {"backend": "openai", "openai": {"model": "deepseek-v4-flash"}}
    assert build_backend(config) == "sentinel"
    assert seen["model"] == "deepseek-v4-flash"


def test_build_thinking_body_variants() -> None:
    from asem.backends.base import build_thinking_body

    assert build_thinking_body({}) == {}
    # Documented OpenAI-compatible toggle: {"thinking": {"type": ...}}
    assert build_thinking_body({"thinking": {"type": "disabled"}}) == {
        "thinking": {"type": "disabled"}
    }
    # Documented effort control.
    assert build_thinking_body({"reasoning_effort": "low"}) == {"reasoning_effort": "low"}
    # Legacy vLLM/Qwen form is still supported.
    assert build_thinking_body({"enable_reasoning": True}) == {
        "chat_template_kwargs": {"enable_thinking": True}
    }
    # An explicit extra_body override wins.
    assert build_thinking_body(
        {"thinking": {"type": "enabled"}, "extra_body": {"thinking": {"type": "disabled"}}}
    ) == {"thinking": {"type": "disabled"}}


def test_openai_backend_from_config_sends_thinking_toggle() -> None:
    """`thinking: {type: disabled}` must reach the request body."""
    from unittest.mock import patch

    from asem.backends import openai_backend

    completions = _FakeCompletions("ok")
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with patch("openai.OpenAI", return_value=fake_client), patch.object(
        openai_backend, "_build_embedder", return_value=_StubEmbedder()
    ):
        backend = openai_backend.OpenAIBackend.from_config(
            {
                "model": "m",
                "api_key": "test-key",
                "thinking": {"type": "disabled"},
                "reasoning_effort": "low",
            }
        )
        backend.generate("p")

    body = completions.calls[0]["extra_body"]
    assert body["thinking"] == {"type": "disabled"}
    assert body["reasoning_effort"] == "low"


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
