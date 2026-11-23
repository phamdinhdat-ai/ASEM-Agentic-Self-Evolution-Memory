"""Remote-run path: notebook -> standalone job script, result ingestion, scheduler absorption."""

import json
from pathlib import Path
import shutil

import pytest

import scripts.ingest_remote_results as ingest_mod
import scripts.run_care_on_gpu_box as box
import scripts.schedule_care_ablation as ab
import scripts.schedule_care_v3 as v3

ROOT = Path(__file__).resolve().parents[1]
LOCOMO = ROOT / "build/care/locomo10.json"
NOTEBOOKS = ROOT / "build/kaggle/care-ab"
V3_NOTEBOOKS = ROOT / "build/kaggle/care-v3"


@pytest.mark.skipif(not (NOTEBOOKS / "pilot-locomo/notebook.ipynb").exists(), reason="run prepare_care_ablation_kaggle")
@pytest.mark.parametrize("job", ["pilot-locomo", "locomo-s3", "lme-s1"])
def test_job_source_is_kaggle_free_and_valid_python(tmp_path, job):
    text = box.job_source(NOTEBOOKS / job / "notebook.ipynb", tmp_path)
    assert "/kaggle/" not in text.replace(str(tmp_path).replace("\\", "/"), "")
    assert f"cache={str(tmp_path / 'cache')!r}" in text and "worker_module='scripts.run_care_ablation'" in text
    assert box.output_name(text) == ab.out_name(job)
    assert box.bundle_sha(text) == "5699cf7f86bdba5a71c5d1e8d4f51896a4bc5348d38e05d0475f5415ff62a67b"
    compile(text, job, "exec")


@pytest.mark.skipif(not (V3_NOTEBOOKS / "s0/notebook.ipynb").exists(), reason="run prepare_care_v3_kaggle")
def test_v3_job_source(tmp_path):
    text = box.job_source(V3_NOTEBOOKS / "s2/notebook.ipynb", tmp_path)
    assert box.output_name(text) == "care_v3_lmes_s2" and "run_care_factorial" in text
    assert box.bundle_sha(text).startswith("1459b45a")


def test_dry_run_lists_jobs(tmp_path, capsys):
    if not (NOTEBOOKS / "pilot-locomo/notebook.ipynb").exists():
        pytest.skip("notebooks not built")
    import sys
    argv = sys.argv
    sys.argv = ["x", "--notebooks", str(NOTEBOOKS), "--workdir", str(tmp_path), "--all", "--dry-run"]
    try:
        assert box.main() == 0
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert out.index("pilot-locomo") < out.index("locomo-s0") and out.count("[run ]") == 12


@pytest.mark.skipif(not LOCOMO.exists(), reason="LoCoMo file not available")
def test_ingest_validates_copies_and_scheduler_absorbs(tmp_path, monkeypatch):
    import hashlib, json as _json, re
    from asem.care.memo_backend import MemoBackend
    from scripts.run_care_ablation import run

    class Reader:
        def generate(self, prompt, **_):
            ids = re.findall(r"^\[([^\]]+)\] \(", prompt, re.M)
            h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
            if h % 3 == 0 and ids:
                return _json.dumps({"answer": "x", "source_ids": [ids[0]], "search_queries": []})
            return _json.dumps({"answer": "No information available", "source_ids": [], "search_queries": ["q"]})

    folder = tmp_path / "remote" / ab.out_name("pilot-locomo")
    memo = MemoBackend(lambda p, t, m: (Reader().generate(p), {"prompt_tokens": 5, "completion_tokens": 3,
                                                               "finish_reason": "stop"}),
                       folder / "results/responses.jsonl")
    run("locomo", LOCOMO, folder / "results", memo, shard_index=0, question_limit=4, code_sha=ab.code_sha())
    (folder / "status.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")

    monkeypatch.setattr(ab, "OUTPUT", tmp_path / "out")
    monkeypatch.setattr(ab, "INGEST", tmp_path / "marks")
    report = ingest_mod.ingest("ab", tmp_path / "remote")
    assert report[folder.name].startswith("INGESTED"), report
    assert (tmp_path / "marks/pilot-locomo").exists()
    assert (tmp_path / "out/pilot-locomo" / folder.name / "results/report.json").exists()
    assert ingest_mod.ingest("ab", tmp_path / "remote")[folder.name].startswith("SKIPPED")

    state = {"completed": [], "pending": ["pilot-locomo", "pilot-lme"], "notes": {}}
    ab.absorb_ingested(state)
    assert state["completed"] == ["pilot-locomo"] and state["pending"] == ["pilot-lme"]

    # a result produced by different code is rejected and leaves no marker
    bad = tmp_path / "remote2" / folder.name
    shutil.copytree(folder, bad)
    mpath = bad / "results/report.json"
    data = json.loads(mpath.read_text(encoding="utf-8"))
    data["manifest"]["code_sha256"] = "0" * 64
    mpath.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(ab, "INGEST", tmp_path / "marks2")
    rej = ingest_mod.ingest("ab", tmp_path / "remote2")
    assert rej[folder.name].startswith("REJECTED") and "code sha" in rej[folder.name]
    assert not (tmp_path / "marks2").exists()


def test_v3_absorb(tmp_path, monkeypatch):
    monkeypatch.setattr(v3, "INGEST", tmp_path)
    (tmp_path / "s2").write_text("ok", encoding="utf-8")
    state = {"completed": [0], "pending": [2, 3, 4], "notes": {}}
    v3.absorb_ingested(state)
    assert state["completed"] == [0, 2] and state["pending"] == [3, 4]
