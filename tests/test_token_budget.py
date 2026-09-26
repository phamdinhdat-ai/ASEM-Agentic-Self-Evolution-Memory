"""Token-budget tests: every system must answer inside the SAME window.

The rule under test is that whatever assembles the answer context — the whole
history for ``FullContext``, retrieved notes for the baselines, the JSON payload
of ``AnswerAgent`` — is trimmed so that::

    prompt <= context_window - max_tokens - safety

and that the per-call completion cap actually reaches the backend (reserving a
cap the request does not send would still overflow the window).
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import numpy as np
import pytest

from asem.answer_agent import AnswerAgent
from asem.backends.base import InferenceBackend
from asem.note import Note
from asem.token_budget import (
    CHARS_PER_TOKEN,
    SAFETY_MARGIN_TOKENS,
    clip_block,
    estimate_tokens,
    fit_items,
    generate_with_cap,
    prompt_budget,
    resolve_budget,
)

from eval.baselines import FullContext, NoMemory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG = "configs/models/qwen3_4b_openai.yaml"   # declares answer.context_window: 8192
WINDOW = 1200
CAP = 200
BUDGET = WINDOW - CAP - SAFETY_MARGIN_TOKENS     # 936 tokens


class _RecordingBackend(InferenceBackend):
    """Records every prompt and the per-call kwargs it was generated with."""

    def __init__(self, answer: str = "ok", max_tokens: int | None = None) -> None:
        super().__init__()
        self.answer = answer
        self._cap = max_tokens
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    @property
    def default_max_tokens(self) -> int | None:
        return self._cap

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        self.kwargs.append(dict(kwargs))
        return self.answer

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(8, dtype="float32")
        for i, ch in enumerate(str(text)[:8]):
            vec[i] = float(ord(ch) % 7) + 1.0
        return vec


def _turn(i: int, chars: int = 200) -> str:
    """A deterministic history turn, padded to `chars` characters."""
    body = f"[Caroline] turn {i:03d} " + ("x" * 64)
    return body.ljust(chars, ".")


def _fat_note(nid: str, chars: int) -> Note:
    """A note whose rendering alone can exceed a small prompt budget."""
    body = (f"note {nid} " + ("y" * 40))[:chars].ljust(chars, ".")
    return Note(
        id=nid,
        c=body,
        t=datetime(2023, 5, 8),
        K=["k"],
        G=["personal"],
        X=body,
        e=np.asarray([1.0, 0.0]),
        L=[],
        z=np.asarray([1.0, 0.0]),
        q=0.5,
        session_date="8 May 2023",
        entities=["Caroline"],
        speaker="Caroline",
    )


def _kept_positions(blocks: list, kept: list) -> list[int]:
    """Positions of `kept` inside `blocks` (identity, not equality)."""
    pos, result = 0, []
    for block in kept:
        while pos < len(blocks) and blocks[pos] is not block:
            pos += 1
        assert pos < len(blocks), "kept blocks must come from the original list"
        result.append(pos)
        pos += 1
    return result


def _middle_is_dropped(blocks: list, kept: list) -> bool:
    """True when everything dropped sits strictly inside the two ends."""
    positions = set(_kept_positions(blocks, kept))
    dropped = [i for i in range(len(blocks)) if i not in positions]
    return bool(dropped) and all(0 < i < len(blocks) - 1 for i in dropped)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

def test_estimate_tokens_is_conservative() -> None:
    assert estimate_tokens("x" * (CHARS_PER_TOKEN * 10)) == 10
    assert estimate_tokens("") == 1


def test_prompt_budget_needs_a_window() -> None:
    assert prompt_budget(None, 512) is None
    assert prompt_budget(0, 512) is None


def test_prompt_budget_reserves_the_completion_and_the_margin() -> None:
    assert prompt_budget(8192, 512) == 8192 - 512 - SAFETY_MARGIN_TOKENS
    # No cap declared: only the safety margin is reserved.
    assert prompt_budget(8192, None) == 8192 - SAFETY_MARGIN_TOKENS


def test_resolve_budget_falls_back_to_the_backend_cap() -> None:
    backend = _RecordingBackend(max_tokens=2048)

    assert resolve_budget(backend, 8192, None) == 8192 - 2048 - SAFETY_MARGIN_TOKENS
    # An explicit per-call cap wins over the backend default.
    assert resolve_budget(backend, 8192, 512) == 8192 - 512 - SAFETY_MARGIN_TOKENS
    # A backend without a cap reserves nothing beyond the margin.
    assert resolve_budget(_RecordingBackend(), 8192, None) == 8192 - SAFETY_MARGIN_TOKENS


def test_no_window_means_no_budget() -> None:
    assert resolve_budget(_RecordingBackend(max_tokens=1024), None, 512) is None


def test_generate_with_cap_forwards_the_cap() -> None:
    backend = _RecordingBackend()

    generate_with_cap(backend, "prompt", 256)

    assert backend.kwargs[0] == {"max_tokens": 256}


def test_generate_with_cap_tolerates_a_backend_without_kwargs() -> None:
    class _OldBackend:
        def generate(self, prompt: str) -> str:      # no **kwargs
            return "legacy"

    assert generate_with_cap(_OldBackend(), "prompt", 256) == "legacy"


def test_clip_block_halves_a_block_and_stops() -> None:
    clipped = clip_block("z" * 400)

    assert clipped is not None and len(clipped) < 400
    assert clip_block("tiny") is None


# ---------------------------------------------------------------------------
# fit_items
# ---------------------------------------------------------------------------

def _render_join(blocks) -> str:
    return "\n".join(blocks)


def test_fit_items_without_a_budget_keeps_everything() -> None:
    blocks = [_turn(i) for i in range(30)]

    prompt, kept = fit_items(blocks, _render_join, budget=None)

    assert kept == blocks and prompt == "\n".join(blocks)


def test_fit_items_drops_the_lowest_ranked_first() -> None:
    # Relevance order: the best notes are at the FRONT.
    blocks = [f"note {i} " + "d" * 400 for i in range(10)]

    _, kept = fit_items(blocks, _render_join, budget=200, drop="tail")

    assert kept == blocks[: len(kept)] and 0 < len(kept) < len(blocks)


def test_fit_items_keeps_the_newest_history_and_drops_the_middle() -> None:
    blocks = [_turn(i) for i in range(40)]

    _, kept = fit_items(blocks, _render_join, budget=300, drop="oldest", head_keep=3)

    assert kept[:3] == blocks[:3], "the opening (setup) must survive"
    assert kept[-1] == blocks[-1], "the most recent turn must survive"
    assert len(kept) < len(blocks)
    assert _middle_is_dropped(blocks, kept)


def test_fit_items_shrinks_a_single_oversized_block() -> None:
    blocks = ["q" * 4000]

    prompt, kept = fit_items(
        blocks, _render_join, budget=100, drop="tail",
        min_keep=1, shrink=clip_block,
    )

    assert len(kept) == 1
    assert estimate_tokens(prompt) <= 100
    assert kept[0].endswith("…")


# ---------------------------------------------------------------------------
# FullContext
# ---------------------------------------------------------------------------

def _full_context(backend, **kwargs) -> FullContext:
    return FullContext(
        backend=backend,
        prompt_template="Question: {query}\nHistory:\n{context}",
        **kwargs,
    )


def test_full_context_trims_to_the_token_budget() -> None:
    backend = _RecordingBackend()
    system = _full_context(backend, context_window=WINDOW, max_tokens=CAP)
    history = [_turn(i) for i in range(60)]

    system.answer("where did Caroline live?", history)

    prompt = backend.prompts[0]
    assert estimate_tokens(prompt) <= BUDGET
    assert history[0] in prompt, "the opening turns must survive"
    assert history[-1] in prompt, "the most recent turns must survive"
    assert _middle_is_dropped(history, [t for t in history if t in prompt])


def test_full_context_reserves_the_backend_completion_cap() -> None:
    """Regression: with no `answer.max_tokens` the client-wide cap was ignored.

    Reserving 0 tokens for a 2000-token completion overflowed the window even
    though the prompt itself was trimmed to `context_window - safety`.
    """
    backend = _RecordingBackend(max_tokens=CAP)
    system = _full_context(backend, context_window=WINDOW)   # no explicit cap
    history = [_turn(i) for i in range(60)]

    system.answer("where did Caroline live?", history)

    prompt = backend.prompts[0]
    assert system.answer_budget() == BUDGET
    assert estimate_tokens(prompt) <= BUDGET
    assert backend.kwargs[0] == {"max_tokens": CAP}


def test_full_context_without_a_window_is_untouched() -> None:
    backend = _RecordingBackend()
    system = _full_context(backend)
    history = [_turn(i) for i in range(60)]

    system.answer("where did Caroline live?", history)

    prompt = backend.prompts[0]
    assert all(turn in prompt for turn in history)
    assert backend.kwargs[0] == {}, "no cap declared, so no per-call cap is sent"


def test_full_context_answers_an_empty_history() -> None:
    backend = _RecordingBackend()
    system = _full_context(backend, context_window=WINDOW, max_tokens=CAP)

    system.answer("anything?", [])

    assert "(no prior conversation)" in backend.prompts[0]


def test_no_memory_sends_the_configured_cap() -> None:
    backend = _RecordingBackend()
    system = NoMemory(
        backend=backend,
        prompt_template="Question: {query}",
        max_tokens=CAP,
    )

    system.answer("hello", ["irrelevant " * 50])

    assert backend.kwargs[0] == {"max_tokens": CAP}
    assert backend.prompts[0] == "Question: hello"


# ---------------------------------------------------------------------------
# AnswerAgent
# ---------------------------------------------------------------------------

def test_answer_agent_reserves_the_backend_completion_cap() -> None:
    backend = _RecordingBackend(answer='{"selected_ids": [], "answer": "ok"}', max_tokens=CAP)
    agent = AnswerAgent(
        backend=backend,
        prompt_template="AA:{query} {candidates}",
        baseline_prompt_template="BASE:{query} {context}",
        context_window=WINDOW,
    )
    notes = [_fat_note(f"n{i}", 900) for i in range(60)]

    agent.distil_and_answer("what happened?", notes)

    prompt = backend.prompts[0]
    assert agent.prompt_budget() == BUDGET
    assert estimate_tokens(prompt) <= BUDGET
    assert "note n0" in prompt, "the top-ranked note must survive"
    assert "note n59" not in prompt, "low-ranked notes must be dropped"


def test_answer_agent_keeps_at_least_one_note() -> None:
    backend = _RecordingBackend(answer='{"selected_ids": [], "answer": "ok"}')
    agent = AnswerAgent(
        backend=backend,
        prompt_template="AA:{query} {candidates}",
        baseline_prompt_template="BASE:{query} {context}",
        context_window=300,
    )

    agent.distil_and_answer("what happened?", [_fat_note("n0", 3000)])

    assert "note n0" in backend.prompts[0]


# ---------------------------------------------------------------------------
# Wiring: the budget comes from the config's `answer` block
# ---------------------------------------------------------------------------

def test_answer_budget_from_config_reads_the_answer_block() -> None:
    from eval.systems import answer_budget_from_config

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cfg.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(
                "answer:\n  max_tokens: 512\n  context_window: 8192\n"
            )
        assert answer_budget_from_config(path) == (512, 8192)

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("answer:\n  max_tokens: 512\n")
        assert answer_budget_from_config(path) == (512, None)


def test_build_baselines_shares_one_budget_across_systems(tmp_path) -> None:
    """Every arm of a comparison must answer under the same token budget."""
    pytest.importorskip("faiss")
    from eval.phase_runner import close_system
    from eval.systems import answer_budget_from_config, build_baselines

    max_tokens, window = answer_budget_from_config(CONFIG)
    assert window, "the test config must declare answer.context_window"

    systems = build_baselines(CONFIG, str(tmp_path), backend=_RecordingBackend())
    try:
        for name in ("NoMemory", "FullContext", "SimRetrieval", "AtomicLinking",
                     "RLManagerOnly"):
            system = systems[name]
            assert (system.max_tokens, system.context_window) == (max_tokens, window), name

        # ValueRetrievalOnly answers through the shared AnswerAgent.
        agent = systems["ValueRetrievalOnly"].answer_agent
        assert (agent.max_tokens, agent.context_window) == (max_tokens, window)
    finally:
        # Windows keeps the SQLite files locked until the banks are closed.
        for system in systems.values():
            close_system(system)
