"""CareTwoStage: lexical repair must equal CareBridge exactly; hybrid only adds candidates."""

import hashlib
import json
from pathlib import Path
import re

import pytest

from asem.care import CareBridge
from asem.care.hybrid import CareTwoStage

ROOT = Path(__file__).resolve().parents[1]
LOCOMO = ROOT / "build/care/locomo10.json"


class DeterministicReader:
    """Reply depends only on the prompt, so two code paths that send the same
    prompts must produce identical traces. Covers every gate branch."""

    def __init__(self):
        self.prompts = []

    def generate(self, prompt: str, **_kwargs) -> str:
        self.prompts.append(prompt)
        ids = re.findall(r"^\[([^\]]+)\] \(", prompt, re.M)
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        kind = h % 4
        if kind == 0 and ids:
            return json.dumps({"answer": "answer-" + str(h % 97), "source_ids": [ids[h % len(ids)]],
                               "search_queries": []})
        if kind == 1:
            return json.dumps({"answer": "No information available", "source_ids": [],
                               "search_queries": ["search " + str(h % 13), "another " + str(h % 7)]})
        if kind == 2:
            return json.dumps({"answer": "No information available", "source_ids": [], "search_queries": []})
        return "not json at all"


def _questions(limit_per_conv=40):
    from scripts.run_care_locomo import turns_for
    data = json.loads(LOCOMO.read_text(encoding="utf-8"))
    for conv in data[:4]:
        turns = turns_for(conv["conversation"])
        for qa in conv["qa"][:limit_per_conv]:
            yield turns, str(qa["question"])


@pytest.mark.skipif(not LOCOMO.exists(), reason="LoCoMo file not available")
def test_lexical_repair_equals_care_bridge():
    keys = ("answer", "source_ids", "bundle_ids", "escalated", "candidate_ids", "invalid_reason")
    n = escalated = 0
    for turns, question in _questions():
        a, b = DeterministicReader(), DeterministicReader()
        original = CareBridge(turns)
        initial = original.rank(question)[:8]
        expected = original.answer_with_initial(question, a, initial)
        two = CareTwoStage(turns)
        fp = two.first_pass(question, b, [two.by_id[t.source_id] for t in initial])
        got = two.repair(question, b, fp, "lexical")
        assert a.prompts == b.prompts                      # identical model calls
        assert {k: expected[k] for k in keys} == {k: got[k] for k in keys}
        assert expected["first_pass"] == got["first_pass"]
        n += 1
        escalated += bool(got["escalated"])
    assert n == 160 and 0 < escalated < n                 # both gate outcomes exercised


@pytest.mark.skipif(not LOCOMO.exists(), reason="LoCoMo file not available")
def test_hybrid_repair_only_adds_candidates():
    checked = 0
    for turns, question in _questions(15):
        two = CareTwoStage(turns, dense_rank=lambda text, k, _two=None: [])
        # a stand-in dense ranker: reverse source order (deterministic, independent of BM25)
        two.dense_rank = lambda text, k, t=two: list(reversed(t.turns))[:k]
        fp = two.first_pass(question, DeterministicReader(), two.rank(question)[:8])
        if not fp["needs_repair"]:
            continue
        lexical = {t.source_id for t in two.expansion(question, fp["first"], "lexical")}
        hybrid = {t.source_id for t in two.expansion(question, fp["first"], "hybrid")}
        assert lexical <= hybrid
        assert hybrid - lexical <= {t.source_id for t in list(reversed(two.turns))[:6]}
        checked += 1
    assert checked > 0


def test_hybrid_requires_a_dense_ranker():
    from asem.care.sourcefirst import SourceTurn
    turns = [SourceTurn("D1:1", "session_1", "1:00 pm on 1 May, 2023", "A", "hello there", 0),
             SourceTurn("D1:2", "session_1", "1:00 pm on 1 May, 2023", "B", "hi", 1)]
    two = CareTwoStage(turns)
    fp = {"first": {"answer": "No information available", "search_queries": [], "invalid_reason": None,
                    "source_ids": [], "bundle_ids": []}, "risk": [], "needs_repair": True, "initial_ids": []}
    with pytest.raises(ValueError, match="dense ranker"):
        two.expansion("where?", fp["first"], "hybrid")


@pytest.mark.skipif(not LOCOMO.exists(), reason="LoCoMo file not available")
def test_graph_repair_only_adds_graph_candidates():
    checked = 0
    for turns, question in _questions(10):
        two = CareTwoStage(turns)
        two.graph_rank = lambda text, k, t=two: t.turns[:k]          # stand-in graph ranker
        fp = two.first_pass(question, DeterministicReader(), two.rank(question)[:8])
        if not fp["needs_repair"]:
            continue
        lexical = {t.source_id for t in two.expansion(question, fp["first"], "lexical")}
        graph = {t.source_id for t in two.expansion(question, fp["first"], "graph")}
        assert lexical <= graph and graph - lexical <= {t.source_id for t in two.turns[:6]}
        checked += 1
    assert checked > 0
