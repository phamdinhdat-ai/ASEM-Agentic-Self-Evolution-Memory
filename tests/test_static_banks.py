"""Tests for the static (ingest-once) bank pipeline.

Covers the two new modules end to end with a deterministic stub backend, so no
model is downloaded and no API is called:

* ``eval.static_banks``  — bank layout, manifest/index, idempotent re-runs,
  ``--force`` rebuilds, and error isolation (one failing system must not stop
  the others or lose the manifest).
* ``eval.static_eval``   — resume from a partial prediction file, per-item error
  recording, and presence of every requested metric (including the LLM judge).
"""

from __future__ import annotations

import json
import os

import pytest

from eval.static_banks import build_static_banks, count_notes, load_manifest
from eval.static_eval import run_static_eval

try:  # pytest inserts the repo root (tests/ is a package), so either form works
    from tests.test_phase_benchmark import StubBackend, _load_groups, _write_dataset
except ImportError:  # pragma: no cover
    from test_phase_benchmark import StubBackend, _load_groups, _write_dataset

CONFIG = "configs/models/qwen2.5_1.5b_hf.yaml"
SYSTEMS = ["FastASEM", "SimRetrieval"]


class JudgeStubBackend(StubBackend):
    """StubBackend that also answers the LLM-judge prompt with valid JSON."""

    def generate(self, prompt: str, **kwargs) -> str:
        if "impartial evaluator" in prompt:
            self.prompts.append(prompt)
            return json.dumps(
                {"is_correct": True, "reasoning": "stub verdict", "error": None}
            )
        return super().generate(prompt, **kwargs)


class AnswerFailBackend(StubBackend):
    """Raises only on the retrieval-answering prompt (SimRetrieval's prompt)."""

    def generate(self, prompt: str, **kwargs) -> str:
        if "Use the retrieved memory notes" in prompt:
            self.prompts.append(prompt)
            raise RuntimeError("simulated backend outage")
        return super().generate(prompt, **kwargs)


class FailAfterNBackend(StubBackend):
    """Succeeds for the first N generate() calls, then raises.

    Models an interruption *mid-conversation*: earlier sessions are already
    persisted in the bank, later ones never run.
    """

    def __init__(self, fail_after: int) -> None:
        super().__init__()
        self.fail_after = fail_after
        self.calls = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        if self.calls > self.fail_after:
            raise RuntimeError("simulated interruption")
        return super().generate(prompt, **kwargs)


def _prepare(tmp_path, systems=SYSTEMS, tag="test"):
    """Write the mini dataset and build the requested banks."""
    dataset_path = tmp_path / "locomo_mini.json"
    _write_dataset(dataset_path)
    raw, groups = _load_groups(dataset_path)
    bank_root = str(tmp_path / "static" / "memory_banks" / "locomo_mini")
    manifest = build_static_banks(
        raw_dataset=raw,
        groups=groups,
        systems=systems,
        ingest_config=CONFIG,
        bank_root=bank_root,
        tag=tag,
        dataset="locomo_mini",
        backend=StubBackend(),
        input_path=str(dataset_path),
    )
    return dataset_path, raw, groups, bank_root, manifest


# ---------------------------------------------------------------------------
# Phase A — eval/static_banks.py
# ---------------------------------------------------------------------------

def test_bank_layout_manifest_and_index(tmp_path):
    dataset_path, raw, groups, bank_root, manifest = _prepare(tmp_path)
    conv = str(groups[0][0]["session_id"])

    tag_dir = os.path.join(bank_root, "test")
    # One bank per system/conversation, at the documented path.
    assert os.path.exists(
        os.path.join(tag_dir, "FastASEM", conv, "fast_asem.sqlite")
    )
    assert os.path.exists(
        os.path.join(tag_dir, "SimRetrieval", conv, "simretrieval.sqlite")
    )
    assert os.path.exists(os.path.join(tag_dir, "manifest.json"))
    assert os.path.exists(os.path.join(tag_dir, "banks.json"))

    assert manifest["run"]["built"] == 2
    assert manifest["run"]["errors"] == 0
    assert manifest["totals"]["banks_ok"] == 2
    assert manifest["totals"]["notes"]["FastASEM"] > 0
    assert manifest["totals"]["notes"]["SimRetrieval"] > 0
    assert manifest["input_sha256"]
    assert manifest["embedder_name"] == "sentence-transformers/all-MiniLM-L6-v2"

    with open(os.path.join(tag_dir, "banks.json"), encoding="utf-8") as fh:
        index = json.load(fh)
    assert index["FastASEM"][conv]["notes"] > 0
    assert index["FastASEM"][conv]["status"] == "ok"


def test_rerun_skips_and_force_rebuilds(tmp_path):
    dataset_path, raw, groups, bank_root, first = _prepare(tmp_path)
    conv = str(groups[0][0]["session_id"])
    notes_first = first["totals"]["notes"]["FastASEM"]

    # Same command again: nothing is rebuilt.
    second = build_static_banks(
        raw_dataset=raw,
        groups=groups,
        systems=SYSTEMS,
        ingest_config=CONFIG,
        bank_root=bank_root,
        tag="test",
        dataset="locomo_mini",
        backend=StubBackend(),
        input_path=str(dataset_path),
    )
    assert second["run"]["built"] == 0
    assert second["run"]["skipped"] == 2
    assert second["totals"]["notes"]["FastASEM"] == notes_first
    assert second["totals"]["banks_skipped"] == 2

    # --force rebuilds from scratch.
    forced = build_static_banks(
        raw_dataset=raw,
        groups=groups,
        systems=SYSTEMS,
        ingest_config=CONFIG,
        bank_root=bank_root,
        tag="test",
        dataset="locomo_mini",
        backend=StubBackend(),
        force=True,
        input_path=str(dataset_path),
    )
    assert forced["run"]["built"] == 2
    assert forced["run"]["skipped"] == 0
    assert forced["totals"]["banks_ok"] == 2
    assert forced["totals"]["notes"]["FastASEM"] == notes_first


def test_failing_system_is_isolated(tmp_path, monkeypatch):
    """One broken system must not stop the others nor lose the manifest."""
    import eval.static_banks as static_banks

    real_ingest = static_banks._ingest_one

    def _flaky(name, *args, **kwargs):
        if name == "FastASEM":
            raise RuntimeError("boom")
        return real_ingest(name, *args, **kwargs)

    monkeypatch.setattr(static_banks, "_ingest_one", _flaky)

    dataset_path, raw, groups, bank_root, manifest = _prepare(tmp_path)
    conv = str(groups[0][0]["session_id"])

    assert manifest["run"]["errors"] == 1
    assert manifest["run"]["built"] == 1

    systems = manifest["conversations"][0]["systems"]
    assert systems["FastASEM"]["status"] == "error"
    assert "boom" in systems["FastASEM"]["error"]
    assert systems["FastASEM"]["traceback"]
    # The healthy system still produced a real bank.
    assert systems["SimRetrieval"]["status"] == "ok"
    assert systems["SimRetrieval"]["notes"] > 0
    assert os.path.exists(
        os.path.join(bank_root, "test", "SimRetrieval", conv, "simretrieval.sqlite")
    )
    # The manifest on disk is valid and reflects the failure.
    reloaded = load_manifest(os.path.join(bank_root, "test", "manifest.json"))
    assert reloaded is not None
    assert reloaded["totals"]["banks_error"] == 1


def test_interrupted_conversation_is_resumed_not_skipped(tmp_path):
    """A bank left half-written by an interruption must be rebuilt on retry.

    Ingestion takes ~26 min per conversation, so a kill mid-conversation is
    likely. The partial bank already contains notes, so a naive
    "skip when the bank is non-empty" check would silently accept an
    INCOMPLETE memory bank as finished.
    """
    dataset_path = tmp_path / "locomo_mini.json"
    _write_dataset(dataset_path)
    raw, groups = _load_groups(dataset_path)
    bank_root = str(tmp_path / "banks")
    conv = str(groups[0][0]["session_id"])

    # 1. Interrupted run: session 1 persists, session 2 raises.
    partial = build_static_banks(
        raw_dataset=raw,
        groups=groups,
        systems=["FastASEM"],
        ingest_config=CONFIG,
        bank_root=bank_root,
        tag="test",
        dataset="locomo_mini",
        backend=FailAfterNBackend(fail_after=1),
        input_path=str(dataset_path),
    )
    bfile = os.path.join(bank_root, "test", "FastASEM", conv, "fast_asem.sqlite")
    assert partial["run"]["errors"] == 1
    assert count_notes(bfile) > 0, "expected a partially written bank on disk"
    assert partial["conversations"][0]["systems"]["FastASEM"]["status"] == "error"

    # 2. Retry with a healthy backend: redo the conversation, do not skip it.
    retry = build_static_banks(
        raw_dataset=raw,
        groups=groups,
        systems=["FastASEM"],
        ingest_config=CONFIG,
        bank_root=bank_root,
        tag="test",
        dataset="locomo_mini",
        backend=StubBackend(),
        input_path=str(dataset_path),
    )
    assert retry["run"]["skipped"] == 0, "partial bank was wrongly treated as complete"
    assert retry["conversations"][0]["systems"]["FastASEM"]["status"] == "ok"

    # 3. A pristine build must yield exactly the same note count (no leftovers).
    clean = build_static_banks(
        raw_dataset=raw,
        groups=groups,
        systems=["FastASEM"],
        ingest_config=CONFIG,
        bank_root=str(tmp_path / "clean"),
        tag="test",
        dataset="locomo_mini",
        backend=StubBackend(),
        input_path=str(dataset_path),
    )
    assert retry["totals"]["notes"]["FastASEM"] == clean["totals"]["notes"]["FastASEM"]


def test_fail_fast_raises(tmp_path, monkeypatch):
    import eval.static_banks as static_banks

    def _always_boom(name, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(static_banks, "_ingest_one", _always_boom)

    dataset_path = tmp_path / "locomo_mini.json"
    _write_dataset(dataset_path)
    raw, groups = _load_groups(dataset_path)

    with pytest.raises(RuntimeError):
        build_static_banks(
            raw_dataset=raw,
            groups=groups,
            systems=["SimRetrieval"],
            ingest_config=CONFIG,
            bank_root=str(tmp_path / "banks"),
            tag="test",
            dataset="locomo_mini",
            backend=StubBackend(),
            fail_fast=True,
            input_path=str(dataset_path),
        )


# ---------------------------------------------------------------------------
# Phase B — eval/static_eval.py
# ---------------------------------------------------------------------------

def _run_eval(tmp_path, groups, bank_root, systems, metrics, out_name="results.json",
              backend=None, **kwargs):
    out_path = str(tmp_path / out_name)
    results = run_static_eval(
        groups=groups,
        systems=systems,
        config_path=CONFIG,
        bank_root=bank_root,
        tag="test",
        dataset="locomo_mini",
        backend=backend or JudgeStubBackend(),
        metric_names=metrics,
        judge_backend=JudgeStubBackend(),
        preds_dir=str(tmp_path / "preds"),
        scores_dir=str(tmp_path / "scores"),
        out_path=out_path,
        per_category=True,
        work_root=str(tmp_path / "work"),
        model_tag="stub",
        **kwargs,
    )
    assert os.path.exists(out_path)
    return results, out_path


def test_eval_metrics_per_category_and_baselines(tmp_path):
    dataset_path, raw, groups, bank_root, _ = _prepare(tmp_path, systems=["SimRetrieval"])
    systems = ["SimRetrieval", "FullContext", "NoMemory"]

    results, _ = _run_eval(
        tmp_path, groups, bank_root, systems,
        metrics=["em", "f1", "rougeL", "judge"],
    )

    assert results["n_qa"] == 3
    assert set(results["systems"]) == set(systems)
    for name in systems:
        entry = results["systems"][name]
        assert entry["n"] == 3
        for metric in ("em", "f1", "rougeL", "judge"):
            assert metric in entry["overall"]
        assert entry["n_answer_errors"] == 0
        assert entry["per_category"], name

    # The stub judge always says "correct".
    assert results["systems"]["SimRetrieval"]["overall"]["judge"] == 1.0
    # Bankless baselines were answered from the item history (no bank needed).
    assert results["systems"]["FullContext"]["missing_conversations"] == 0

    # Predictions were streamed to disk, one line per QA pair per system.
    for name in systems:
        preds_file = os.path.join(str(tmp_path / "preds"), f"test__stub__{name}.jsonl")
        assert os.path.exists(preds_file)
        with open(preds_file, encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        assert len(lines) == 3
        assert {"idx", "pred", "ref", "em", "f1", "rougeL"} <= set(lines[0])


def test_eval_resumes_without_duplicating(tmp_path):
    dataset_path, raw, groups, bank_root, _ = _prepare(tmp_path, systems=["SimRetrieval"])
    systems = ["SimRetrieval"]

    first, _ = _run_eval(tmp_path, groups, bank_root, systems, metrics=["em"], out_name="a.json")
    assert first["systems"]["SimRetrieval"]["n"] == 3

    # Simulate an interrupted run: keep only the first answered QA pair.
    preds_file = os.path.join(str(tmp_path / "preds"), "test__stub__SimRetrieval.jsonl")
    with open(preds_file, encoding="utf-8") as fh:
        lines = [line for line in fh if line.strip()]
    with open(preds_file, "w", encoding="utf-8") as fh:
        fh.writelines(lines[:1])

    second, _ = _run_eval(tmp_path, groups, bank_root, systems, metrics=["em"], out_name="b.json")
    assert second["resumed_from"] >= 1
    assert second["systems"]["SimRetrieval"]["n"] == 3

    with open(preds_file, encoding="utf-8") as fh:
        final = [json.loads(line) for line in fh if line.strip()]
    assert len(final) == 3
    assert len({rec["idx"] for rec in final}) == 3  # no duplicate indices


def test_eval_records_answer_errors_and_continues(tmp_path):
    """A backend outage on one system must be recorded, not fatal."""
    dataset_path, raw, groups, bank_root, _ = _prepare(tmp_path, systems=["SimRetrieval"])
    systems = ["SimRetrieval", "FullContext"]

    results, _ = _run_eval(
        tmp_path, groups, bank_root, systems, metrics=["em"],
        backend=AnswerFailBackend(), out_name="err.json",
    )

    failing = results["systems"]["SimRetrieval"]
    assert failing["n"] == 3
    assert failing["n_answer_errors"] == 3
    assert failing["overall"]["em"] == 0.0

    # The other system still answered normally, and the run completed.
    healthy = results["systems"]["FullContext"]
    assert healthy["n"] == 3
    assert healthy["n_answer_errors"] == 0


def test_eval_missing_bank_raises_unless_allowed(tmp_path):
    """A banked system with no bank fails loudly, or is skipped when allowed."""
    dataset_path = tmp_path / "locomo_mini.json"
    _write_dataset(dataset_path)
    raw, groups = _load_groups(dataset_path)
    empty_root = str(tmp_path / "no_banks")

    with pytest.raises(FileNotFoundError):
        run_static_eval(
            groups=groups,
            systems=["SimRetrieval"],
            config_path=CONFIG,
            bank_root=empty_root,
            tag="test",
            dataset="locomo_mini",
            backend=JudgeStubBackend(),
            metric_names=["em"],
            preds_dir=str(tmp_path / "preds_raise"),
            require_banks=True,
        )

    results = run_static_eval(
        groups=groups,
        systems=["SimRetrieval"],
        config_path=CONFIG,
        bank_root=empty_root,
        tag="test",
        dataset="locomo_mini",
        backend=JudgeStubBackend(),
        metric_names=["em"],
        preds_dir=str(tmp_path / "preds_skip"),
        require_banks=False,
    )
    assert results["systems"]["SimRetrieval"]["n"] == 0
    assert results["systems"]["SimRetrieval"]["missing_conversations"] == 1


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def test_metric_primitives():
    from eval.metrics import (
        canonical_metrics,
        compute_metrics,
        exact_match,
        em_loose,
        normalize_text,
        rouge_l,
        token_f1,
    )

    # Normalisation drops articles, punctuation and case.
    assert normalize_text("The Buddy, the dog!") == "buddy dog"
    assert normalize_text(None) == ""
    assert normalize_text(2023) == "2023"

    assert exact_match("Buddy", "the buddy") == 1.0       # articles dropped
    assert exact_match("Buddy", "Buddy the dog") == 0.0   # strict, no substring
    assert em_loose("Buddy", "Buddy the dog") == 1.0      # legacy substring EM
    assert exact_match("", "Buddy") == 0.0

    assert token_f1("Buddy the dog", "Buddy the dog") == 1.0
    assert token_f1("", "Buddy") == 0.0
    assert token_f1("cat", "dog") == 0.0
    partial = token_f1("Buddy and Alice", "Buddy and Bob")
    assert 0.0 < partial < 1.0
    assert rouge_l("Buddy the dog", "Buddy the dog") == 1.0
    assert rouge_l("cat", "dog") == 0.0

    # Aliases and aggregation.
    assert canonical_metrics(["rouge-l", "token_f1", "bert"]) == [
        "rougeL", "f1", "bertscore_f1"
    ]
    agg = compute_metrics(
        ["Buddy", "Seattle"], ["Buddy", "Seattle"], ["em", "f1", "rougeL"]
    )
    assert agg["n"] == 2
    assert agg["em"] == 1.0 and agg["f1"] == 1.0 and agg["rougeL"] == 1.0

    # Judge accounting excludes failed verdicts from the denominator.
    judged = compute_metrics(
        ["a", "b", "c"], ["a", "b", "c"], ["judge"],
        judge_flags=[True, None, False],
    )
    assert judged["judge"] == 0.5
    assert judged["judge_coverage"] == pytest.approx(2 / 3)
