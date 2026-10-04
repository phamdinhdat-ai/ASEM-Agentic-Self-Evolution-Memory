"""Unit tests for G3 multi-hop query decomposition.

The planner's contract, stated as tests: `off` never fires; `heuristic` fires
only on a real conjunction and always keeps the original query; `llm` degrades
to the heuristic on any parse or backend failure so a bad generation cannot lose
a retrieval. The retriever's contract: with no planner attached, retrieval is
byte-for-byte the pre-G3 path; with one, a sub-question's evidence can reach the
answer window even when the original query already filled it.
"""
from __future__ import annotations

import tempfile

import numpy as np
import pytest

from asem.memory_bank import MemoryBank
from asem.note import Note
from asem.query_planner import QueryPlanner


class _EchoBackend:
    """Returns a canned response, and records the prompts it was given."""

    def __init__(self, response: str):
        self.response = response
        self.prompts = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.response

    def embed(self, text: str) -> np.ndarray:
        return np.zeros(2)


class _RaisingBackend:
    def generate(self, prompt: str, **kwargs) -> str:
        raise RuntimeError("backend down")

    def embed(self, text: str) -> np.ndarray:
        return np.zeros(2)


# ---------------------------------------------------------------------------
# Planner: mode gating
# ---------------------------------------------------------------------------

def test_g3_off_mode_never_fires() -> None:
    p = QueryPlanner(backend=_EchoBackend("{}"), mode="off")
    assert p.plan("What subject have Caroline and Melanie both painted?") == []
    assert p.stats["planner_fired"] is False


def test_g3_heuristic_fires_on_conjunction_and_keeps_the_original_first() -> None:
    p = QueryPlanner(mode="heuristic", max_subquestions=3)
    subs = p.plan("What subject have Caroline and Melanie both painted?")
    assert subs, "a conjunction must fire the heuristic"
    assert subs[0] == "What subject have Caroline and Melanie both painted?"


def test_g3_heuristic_skips_single_hop_questions() -> None:
    """The classifier reads these as factual, and there is no conjunction."""
    p = QueryPlanner(mode="heuristic", max_subquestions=3)
    for q in ("What did Caroline research?",
              "What activities does Melanie partake in?",
              "Who supports Caroline when she has a negative experience?"):
        assert p.plan(q) == [], q


def test_g3_heuristic_caps_at_max_subquestions() -> None:
    p = QueryPlanner(mode="heuristic", max_subquestions=2)
    subs = p.plan("A and B and C and D and E")
    assert len(subs) <= 2


# ---------------------------------------------------------------------------
# Planner: LLM mode and its fallbacks
# ---------------------------------------------------------------------------

def test_g3_llm_parses_sub_questions() -> None:
    backend = _EchoBackend('{"sub_questions": ["Who is Caroline?", "Who is Melanie?"]}')
    p = QueryPlanner(backend=backend, mode="llm", max_subquestions=3)
    subs = p.plan("What do Caroline and Melanie both like?")
    assert subs[0] == "What do Caroline and Melanie both like?"  # original kept
    assert "Who is Caroline?" in subs


def test_g3_llm_parses_through_fences_and_prose() -> None:
    backend = _EchoBackend('Sure!\n```json\n{"sub_questions": ["Where did A go?"]}\n```')
    p = QueryPlanner(backend=backend, mode="llm", max_subquestions=3)
    subs = p.plan("What do A and B like?")
    assert "Where did A go?" in subs


def test_g3_llm_falls_back_to_heuristic_on_parse_failure() -> None:
    """A bad generation must degrade to the free split, never lose the query."""
    backend = _EchoBackend("I cannot help with that.")
    p = QueryPlanner(backend=backend, mode="llm", max_subquestions=3)
    subs = p.plan("What subject have Caroline and Melanie both painted?")
    assert subs, "must fall back, not return nothing"
    assert subs[0] == "What subject have Caroline and Melanie both painted?"


def test_g3_llm_falls_back_when_backend_raises() -> None:
    p = QueryPlanner(backend=_RaisingBackend(), mode="llm", max_subquestions=3)
    subs = p.plan("What subject have Caroline and Melanie both painted?")
    assert subs and subs[0].startswith("What subject")


def test_g3_llm_without_backend_uses_heuristic() -> None:
    p = QueryPlanner(backend=None, mode="llm", max_subquestions=3)
    subs = p.plan("What subject have Caroline and Melanie both painted?")
    assert subs and subs[0].startswith("What subject")


def test_g3_llm_does_not_gate_on_the_classifier() -> None:
    """`llm` asks the model for EVERY query — the classifier cannot see multi-hop.

    LoCoMo's "What subject have Caroline and Melanie both painted?" contains
    "what", so `_classify_query_type` calls it factual. Gating the LLM on that
    would skip exactly the questions decomposition exists for.
    """
    backend = _EchoBackend('{"sub_questions": ["Q1", "Q2"]}')
    p = QueryPlanner(backend=backend, mode="llm", max_subquestions=3)
    assert p.plan("What subject have Caroline and Melanie both painted?")
    assert backend.prompts, "the LLM must have been asked"


# ---------------------------------------------------------------------------
# Retriever integration and fusion
# ---------------------------------------------------------------------------

class _StubPlanner:
    """A planner that returns a fixed split, so fusion is tested in isolation."""

    def __init__(self, subs):
        self._subs = subs
        self.stats = {"planner_mode": "stub"}

    def plan(self, query):
        return list(self._subs)


class _FixedEmbed:
    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=float)

    def generate(self, prompt, **kwargs):
        return ""

    def embed(self, text):
        # The sub-question text selects a distinct direction, so the stub can
        # make one sub-question retrieve a note the original query cannot.
        if "SUB" in text:
            return np.asarray([0.0, 1.0])
        return np.asarray([1.0, 0.0])


def _note(nid, vec):
    return Note(id=nid, c=f"content {nid}",
                t=__import__("datetime").datetime(2024, 1, 1),
                K=[nid], G=[], X=f"desc {nid}",
                e=np.asarray(vec, dtype=float), L=[], z=np.asarray(vec, dtype=float), q=0.5)


def _retriever(**kw):
    from asem.enhanced_retriever import EnhancedHybridRetriever
    return EnhancedHybridRetriever(backend=_FixedEmbed([1.0, 0.0]),
                                   k1=5, k2=5, delta=0.0, lambda_weight=0.5,
                                   enable_global_semantics=False, **kw)


def test_g3_no_planner_is_the_pre_g3_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/b.sqlite")
        bank.add(_note("a", [1.0, 0.0]))
        bank.add(_note("b", [0.0, 1.0]))
        r = _retriever(query_planner=None)
        out = r.retrieve("query", bank)
        assert "planner_fired" not in r.stats
        assert [n.id for n in out][:1] == ["a"]
        bank.close()


def test_g3_fusion_lifts_a_subquestion_note_into_the_window() -> None:
    """The mechanism: a note only the SUB-question finds must reach the window.

    Note `b` is orthogonal to the original query, so the base pass ranks it
    last; the stub sub-question "SUB" points straight at it. With fusion the two
    interleave, so `b` is in the top-2 — which is the whole point of G3.
    """
    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/b.sqlite")
        bank.add(_note("a", [1.0, 0.0]))
        bank.add(_note("b", [0.0, 1.0]))
        r = _retriever(query_planner=_StubPlanner(["original", "SUB"]))
        out = r.retrieve("original", bank)
        assert r.stats.get("planner_fired") is True
        assert "b" in [n.id for n in out[:2]], [n.id for n in out]
        bank.close()


def test_g3_fusion_weight_zero_leaves_subquestions_ranked_below() -> None:
    """With weight 0 the sub-question contributes nothing — the control."""
    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/b.sqlite")
        bank.add(_note("a", [1.0, 0.0]))
        bank.add(_note("b", [0.0, 1.0]))
        r = _retriever(query_planner=_StubPlanner(["original", "SUB"]),
                       planner_fuse_weight=0.0)
        out = r.retrieve("original", bank)
        # `b` is still present (the pool is the union) but no longer lifted by
        # the sub-question, so it sorts after `a`.
        assert [n.id for n in out][0] == "a"
        bank.close()
