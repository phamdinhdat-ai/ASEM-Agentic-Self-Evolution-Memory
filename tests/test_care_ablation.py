"""CARE ablation: default flags equal CARE, each flag changes only its mechanism, memo accounting, worker."""

import hashlib
import json
from pathlib import Path
import re

import pytest

from asem.care.ablation import ABLATION_ARMS, ARMS, AblationFlags, CareAblation
from asem.care.hybrid import CareTwoStage
from asem.care.memo_backend import MemoBackend
from asem.care.sourcefirst import SourceTurn

ROOT = Path(__file__).resolve().parents[1]
LOCOMO = ROOT / "build/care/locomo10.json"
NO_INFO = "No information available"


class DeterministicReader:
    """Reply depends only on the prompt; covers every gate branch (same idea as test_care_hybrid)."""

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
            return json.dumps({"answer": NO_INFO, "source_ids": [],
                               "search_queries": ["search " + str(h % 13), "another " + str(h % 7)]})
        if kind == 2:
            return json.dumps({"answer": NO_INFO, "source_ids": [], "search_queries": []})
        return "not json at all"


class Scripted:
    """Returns the queued replies in order and records prompts."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt: str, **_kwargs) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.replies[min(len(self.prompts), len(self.replies)) - 1])


def _turn(i, speaker, text, session="session_1"):
    return SourceTurn(f"D1:{i}", session, "1:00 pm on 1 May, 2023", speaker, text, i - 1)


def _small():
    return [_turn(1, "Melanie", "I had a painting class on Tuesday."),
            _turn(2, "Caroline", "Sounds fun, Melanie."),
            _turn(3, "Melanie", "Thanks for asking.")]


def _questions(per_conv):
    from scripts.run_care_locomo import turns_for
    data = json.loads(LOCOMO.read_text(encoding="utf-8"))
    for conv in data[:4]:
        turns = turns_for(conv["conversation"])
        for qa in conv["qa"][:per_conv]:
            yield turns, str(qa["question"])


# ---- default flags reproduce CARE ---------------------------------------------------------
@pytest.mark.skipif(not LOCOMO.exists(), reason="LoCoMo file not available")
@pytest.mark.parametrize("force", [False, True])
def test_default_flags_equal_care_two_stage(force):
    keys = ("answer", "source_ids", "bundle_ids", "escalated", "candidate_ids", "invalid_reason")
    n = escalated = 0
    for turns, question in _questions(15):
        a, b = DeterministicReader(), DeterministicReader()
        base = CareTwoStage(turns)
        initial = base.rank(question)[:8]
        fp_a = base.first_pass(question, a, initial)
        expected = base.repair(question, a, fp_a, "lexical")
        abl = CareAblation(turns)
        abl._force_revalidate = force            # force=True runs the re-validation code on every reply
        fp_b = abl.first_pass(question, b, initial)
        got = abl.repair(question, b, fp_b, "lexical")
        assert a.prompts == b.prompts
        assert {k: expected[k] for k in keys} == {k: got[k] for k in keys}
        assert fp_a["first"] == fp_b["first"] and fp_a["needs_repair"] == fp_b["needs_repair"]
        n += 1
        escalated += bool(got["escalated"])
    assert n == 60 and 0 < escalated < n


# ---- each flag changes only its own mechanism ---------------------------------------------
def _first(flags, question, reply, turns=None):
    turns = turns or _small()
    bridge = CareAblation(turns, flags=flags)
    return bridge._reader(question, turns, Scripted(reply))


def test_citation_requirement():
    reply = {"answer": "painting", "source_ids": [], "search_queries": []}
    assert _first(AblationFlags(), "What class was on Tuesday?", reply)["invalid_reason"] == "missing_citation"
    off = _first(AblationFlags(require_citation=False), "What class was on Tuesday?", reply)
    assert off["answer"] == "painting" and off["invalid_reason"] is None and off["source_ids"] == []


def test_speaker_binding_check():
    reply = {"answer": "painting", "source_ids": ["D1:1"], "search_queries": []}
    q = "What class did Caroline have on Tuesday?"
    assert _first(AblationFlags(), q, reply)["invalid_reason"] == "speaker_mismatch"
    assert _first(AblationFlags(check_binding=False), q, reply)["answer"] == "painting"
    # event checks stay on when only binding is off
    q2 = "What class did Caroline have at Halloween?"
    assert _first(AblationFlags(check_binding=False), q2, reply)["invalid_reason"] == "event_anchor_mismatch"


def test_event_check():
    reply = {"answer": "painting", "source_ids": ["D1:1"], "search_queries": []}
    q = "What did Melanie do at Halloween?"
    assert _first(AblationFlags(), q, reply)["invalid_reason"] == "event_anchor_mismatch"
    assert _first(AblationFlags(check_event=False), q, reply)["answer"] == "painting"
    # binding stays on when only the event checks are off
    bound = {"answer": "painting", "source_ids": ["D1:1"], "search_queries": []}
    assert _first(AblationFlags(check_event=False), "What class did Caroline have?", bound)["invalid_reason"] \
        == "speaker_mismatch"


def test_gate_always():
    turns = _small()
    ok = {"answer": "painting", "source_ids": ["D1:1"], "search_queries": []}
    care, always = CareAblation(turns), CareAblation(turns, flags=AblationFlags(gate_always=True))
    q = "What class was on Tuesday?"
    initial = care.rank(q)[:3]
    assert care.first_pass(q, Scripted(ok), initial)["needs_repair"] is False
    reader = Scripted(ok, {**ok, "answer": "second"})
    fp = always.first_pass(q, reader, initial)
    res = always.repair(q, reader, fp, "lexical")
    assert fp["needs_repair"] and res["escalated"] and res["answer"] == "second" and len(reader.prompts) == 2


def _refuse_then_answer():
    return (Scripted({"answer": NO_INFO, "source_ids": [], "search_queries": ["zebra"]},
                     {"answer": "painting", "source_ids": ["D1:1"], "search_queries": []}))


def test_correction_text_only_changes_second_prompt():
    turns = _small()
    for flag, expected in ((True, True), (False, False)):
        bridge = CareAblation(turns, flags=AblationFlags(correction_text=flag))
        reader = _refuse_then_answer()
        fp = bridge.first_pass("What class was on Tuesday?", reader, bridge.rank("class")[:3])
        bridge.repair("What class was on Tuesday?", reader, fp, "lexical")
        assert "CORRECTION:" not in reader.prompts[0]
        assert ("CORRECTION:" in reader.prompts[1]) is expected


def _big():
    turns = [_turn(i + 1, "Melanie" if i % 2 else "Caroline", f"alpha{i} beta{i}", f"session_{i // 10 + 1}")
             for i in range(60)]
    turns[40] = _turn(41, "Melanie", "I saw a zebra at the park.", "session_5")
    return turns


def test_expand_off_rereads_the_first_bundle():
    turns = _big()
    q = "Where did the lake trip happen?"
    for expand in (True, False):
        bridge = CareAblation(turns, flags=AblationFlags(expand=expand))
        reader = _refuse_then_answer()
        initial = bridge.rank(q)[:3]
        fp = bridge.first_pass(q, reader, initial)
        res = bridge.repair(q, reader, fp, "lexical")
        ids = res["candidate_ids"]
        if expand:
            assert set(fp["initial_ids"]) < set(ids)
        else:
            assert ids == fp["initial_ids"]


def test_search_off_ignores_proposed_searches():
    turns = _big()
    q = "Where did the lake trip happen?"
    pools = {}
    for use in (True, False):
        bridge = CareAblation(turns, flags=AblationFlags(use_search=use))
        first = {"search_queries": ["zebra"], "answer": NO_INFO, "invalid_reason": None}
        pools[use] = {t.source_id for t in bridge.expansion(q, first, "lexical")}
    assert "D1:41" in pools[True] and "D1:41" not in pools[False]
    assert pools[False] <= pools[True]


def test_arm_table_changes_one_thing_at_a_time():
    default = AblationFlags().__dict__
    for name in ABLATION_ARMS:
        changed = [k for k, v in ARMS[name].__dict__.items() if v != default[k]]
        assert len(changed) == (2 if name == "no_checks" else 1), name
    assert ARMS["care"] == AblationFlags()


# ---- memo backend ---------------------------------------------------------------------------
def test_memo_dedupes_physical_calls_but_counts_logical_ones(tmp_path):
    seen = []

    def complete(prompt, temperature, max_tokens):
        seen.append(prompt)
        return "reply:" + prompt, {"prompt_tokens": 10, "completion_tokens": 2, "finish_reason": "stop"}

    path = tmp_path / "r.jsonl"
    memo = MemoBackend(complete, path, max_new_calls=2)
    assert memo.generate("a") == memo.generate("a") == "reply:a"
    memo.generate("b")
    assert memo.physical == 2 and len(memo.calls) == 3 and [c["cached"] for c in memo.calls] == [False, True, False]
    assert sum(c["prompt_tokens"] for c in memo.calls) == 30      # per-arm cost as if run alone
    with pytest.raises(RuntimeError, match="cap"):
        memo.generate("c")
    again = MemoBackend(complete, path)                            # resume: persisted replies are reused
    assert again.generate("a") == "reply:a" and again.physical == 0 and len(seen) == 2
    again.generate("a", temperature=0.5)                           # different decoding => different key
    assert again.physical == 1 and len(seen) == 3


# ---- worker smoke ---------------------------------------------------------------------------
@pytest.mark.skipif(not LOCOMO.exists(), reason="LoCoMo file not available")
def test_worker_arms_accounting_and_resume(tmp_path):
    from scripts.run_care_ablation import ARM_NAMES, FIRST, run
    reader = DeterministicReader()

    def complete(prompt, temperature, max_tokens):
        return reader.generate(prompt), {"prompt_tokens": len(prompt) // 4, "completion_tokens": 9,
                                         "finish_reason": "stop"}

    memo = MemoBackend(complete, tmp_path / "responses.jsonl")
    report = run("locomo", LOCOMO, tmp_path, memo, shard_index=0, question_limit=8)
    assert set(report["aggregate"]) == set(ARM_NAMES)
    logical = 0
    for row in report["rows"]:
        s = row["systems"]
        assert set(s) == set(ARM_NAMES) and s[FIRST]["calls"] == 1
        for name in ARMS:
            assert s[name]["calls"] == 1 + int(s[name]["escalated"])
            logical += s[name]["calls"]
        assert s["gate_always"]["escalated"] is True
        assert s["repair_retry_only"]["n_candidates"] <= 8
        assert row["physical_calls"] <= sum(s[n]["calls"] for n in ARMS)
    assert memo.physical == report["physical_calls_total"] < logical      # sharing really happened
    assert report["manifest"]["gold_in_model_context"] is False
    memo2 = MemoBackend(complete, tmp_path / "responses.jsonl")
    report2 = run("locomo", LOCOMO, tmp_path, memo2, shard_index=0, question_limit=8)
    assert memo2.physical == 0 and report2["aggregate"] == report["aggregate"]
