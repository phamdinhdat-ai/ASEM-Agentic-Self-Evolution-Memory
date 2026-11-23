"""Overnight Kaggle submission state-machine contracts."""

from scripts import schedule_care_full_kaggle as scheduler


def test_scheduler_releases_one_gpu_slot_and_submits_one_shard(monkeypatch):
    state = scheduler._initial()
    monkeypatch.setattr(scheduler, "_status",
                        lambda index: ("COMPLETE" if index == 0 else "RUNNING", "ok"))
    monkeypatch.setattr(scheduler, "_download", lambda index: (True, "complete"))
    calls = []
    def command(args, **kwargs):
        calls.append(args)
        return 0, "Kernel version 1 successfully pushed."
    monkeypatch.setattr(scheduler, "_command", command)
    monkeypatch.setattr(scheduler, "_save", lambda value: None)
    result = scheduler.step(state)
    assert result["completed"] == [0]
    assert result["submitted"] == [0, 1, 2]
    assert result["pending"] == [3, 4]
    assert len(calls) == 1


def test_scheduler_retries_a_delayed_output_download(monkeypatch):
    state = scheduler._initial()
    monkeypatch.setattr(scheduler, "_status", lambda index: ("COMPLETE", "ok"))
    monkeypatch.setattr(scheduler, "_download", lambda index: (False, "not ready"))
    monkeypatch.setattr(scheduler, "_command",
                        lambda args, **kwargs: (1, "Maximum batch GPU session count of 2 reached"))
    monkeypatch.setattr(scheduler, "_save", lambda value: None)
    result = scheduler.step(state)
    assert result["completed"] == []
    assert result["failed"] == {}
    assert result["pending"] == [2, 3, 4]
    assert result["last_download_error"] == {"0": "not ready", "1": "not ready"}


def test_scheduler_records_a_completed_notebook_with_failed_worker(monkeypatch, tmp_path):
    state = scheduler._initial()
    output = tmp_path / "shards"
    failed = output / "shard-0/care_relation_full_s0"
    failed.mkdir(parents=True)
    (failed / "status.json").write_text('{"state":"failed"}', encoding="utf-8")
    monkeypatch.setattr(scheduler, "OUTPUT", output)
    monkeypatch.setattr(scheduler, "_status",
                        lambda index: ("COMPLETE" if index == 0 else "RUNNING", "ok"))
    monkeypatch.setattr(scheduler, "_download", lambda index: (False, "worker failed"))
    monkeypatch.setattr(scheduler, "_command", lambda args, **kwargs: (0, "successfully pushed"))
    monkeypatch.setattr(scheduler, "_save", lambda value: None)
    result = scheduler.step(state)
    assert result["failed"]["0"] == "worker failed"
    assert result["state"] == "shard_failed"
