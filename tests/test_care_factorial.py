"""Factorial worker: arms, call accounting, shared first pass, resume. Fake model and embedder."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest

from scripts.run_care_factorial import CARE_ARMS, CONTEXT_ARMS, RAG_ARM, run

ROOT = Path(__file__).resolve().parents[1]
LOCOMO = ROOT / "build/care/locomo10.json"
LME = ROOT / "datasets/longmemeval/longmemeval_oracle.json"


class Reader:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **_):
        self.calls.append({"prompt_tokens": len(prompt) // 4, "completion_tokens": 9, "finish_reason": "stop"})
        if "ORIGINAL TURNS" not in prompt:            # RAG and context baselines
            return "plain answer"
        ids = re.findall(r"^\[([^\]]+)\] \(", prompt, re.M)
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        if h % 3 == 0 and ids:
            return json.dumps({"answer": "x", "source_ids": [ids[0]], "search_queries": []})
        return json.dumps({"answer": "No information available", "source_ids": [],
                           "search_queries": ["find " + str(h % 5)]})


class Tok:
    def apply_chat_template(self, messages, **_):
        return list(range(len(messages[0]["content"]) // 4))


def _vec(text):
    rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
    v = rng.standard_normal(16)
    return (v / np.linalg.norm(v)).astype(np.float16)


def _run(benchmark, path, tmp, limit):
    reader = Reader()
    report = run(benchmark, path, tmp, reader, embed_docs=lambda ts: np.stack([_vec(t) for t in ts]),
                 embed_query=_vec, tokenizer=Tok(), shard_index=0, question_limit=limit, max_new_calls=10_000)
    return reader, report


def _check(report, reader, arms):
    assert set(report["aggregate"]) == set(arms)
    for row in report["rows"]:
        s = row["systems"]
        assert set(s) == set(arms)
        for prefix, modes in (("bm25", ("lex", "hyb")), ("dense", ("lex", "hyb", "graph"))):
            assert s[f"{prefix}_first"]["calls"] == 1
            for mode in modes:
                assert s[f"{prefix}_{mode}"]["calls"] == 1 + int(s[f"{prefix}_{mode}"]["escalated"])
        # the three dense repair arms share one first pass and one gate decision
        assert s["dense_lex"]["escalated"] == s["dense_hyb"]["escalated"] == s["dense_graph"]["escalated"]
        assert s[RAG_ARM]["calls"] == 1
    # physical calls: two shared first passes + repairs + single-call arms
    repairs = (("bm25", "lex"), ("bm25", "hyb"), ("dense", "lex"), ("dense", "hyb"), ("dense", "graph"))
    physical = sum(2 + sum(int(r["systems"][f"{p}_{m}"]["escalated"]) for p, m in repairs)
                   + (len(arms) - len(CARE_ARMS)) for r in report["rows"])
    assert physical == len(reader.calls)


@pytest.mark.skipif(not LOCOMO.exists(), reason="LoCoMo file not available")
def test_locomo_arms_and_calls(tmp_path):
    reader, report = _run("locomo", LOCOMO, tmp_path, 6)
    _check(report, reader, list(CARE_ARMS) + [RAG_ARM])
    assert report["manifest"]["proposed_arm"] == "dense_hyb"
    assert any(r["systems"]["bm25_lex"]["escalated"] for r in report["rows"])
    assert all(r["graph_stats"]["communities"] >= 1 for r in report["rows"])


@pytest.mark.skipif(not LME.exists(), reason="LongMemEval file not available")
def test_longmemeval_arms_calls_and_resume(tmp_path):
    reader, report = _run("longmemeval", LME, tmp_path, 4)
    _check(report, reader, list(CARE_ARMS) + [RAG_ARM] + list(CONTEXT_ARMS))
    assert all(r["systems"]["no_memory"]["evidence_recall"] is None for r in report["rows"])
    # resume: nothing new is called, same report
    reader2, report2 = _run("longmemeval", LME, tmp_path, 4)
    assert reader2.calls == [] and report2["aggregate"] == report["aggregate"]
