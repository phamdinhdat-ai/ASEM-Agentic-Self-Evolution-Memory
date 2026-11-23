"""Finish SimRetrieval BERTScore and audited four-way CARE report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "build/care/sim_retrieval_scheduler_status.json"
STATE = ROOT / "build/care/sim_retrieval_finalizer_status.json"
PACKAGE = ROOT / "build/kaggle/care-sim-retrieval-bertscore-full"
DOWNLOAD = ROOT / "build/care/sim_retrieval_bertscore_output"
SCORES = ROOT / "build/care/sim_retrieval_bertscore.json"
KERNEL = "phamdinhdatai/asem-care-simretrieval-bertscore-full"


def save(value):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(STATE)


def command(args, timeout=300):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                            errors="replace", timeout=timeout)
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode:
        raise RuntimeError(f"{args[:4]} exited {result.returncode}: {output[-1200:]}")
    return output


def main():
    deadline = datetime.now(timezone.utc) + timedelta(hours=48)
    state = {"protocol": "care-sim-retrieval-finalizer-v1", "state": "waiting_shards",
             "started_at": datetime.now(timezone.utc).isoformat()}
    save(state)
    try:
        while datetime.now(timezone.utc) < deadline:
            scheduler = json.loads(SCHEDULER.read_text(encoding="utf-8"))
            if scheduler["state"] == "complete":
                break
            if scheduler["state"] in ("merge_failed", "shard_failed", "scheduler_timeout"):
                raise RuntimeError(f"SimRetrieval scheduler stopped: {scheduler['state']}")
            time.sleep(90)
        else:
            raise TimeoutError("SimRetrieval shards did not finish within 48 hours")
        state["state"] = "preparing_bertscore"
        save(state)
        command([sys.executable, "-m", "scripts.prepare_care_sim_retrieval_bertscore_kaggle"])
        state["state"] = "submitting_bertscore"
        save(state)
        command(["kaggle", "kernels", "push", "-p", str(PACKAGE)], timeout=180)
        state["state"] = "running_bertscore"
        save(state)
        while datetime.now(timezone.utc) < deadline:
            output = command(["kaggle", "kernels", "status", KERNEL], timeout=90)
            match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output)
            current = match.group(1) if match else "UNKNOWN"
            state["kernel_status"] = current
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            save(state)
            if current == "COMPLETE":
                break
            if current in ("ERROR", "FAILED", "CANCELLED", "CANCELED"):
                raise RuntimeError(f"BERTScore notebook status: {current}")
            time.sleep(90)
        else:
            raise TimeoutError("BERTScore notebook did not finish within 48 hours")
        state["state"] = "downloading_bertscore"
        save(state)
        command(["kaggle", "kernels", "output", KERNEL, "-p", str(DOWNLOAD)], timeout=300)
        downloaded = DOWNLOAD / "care_sim_retrieval_bertscore_full.json"
        if not downloaded.exists():
            raise FileNotFoundError("Missing SimRetrieval BERTScore output")
        shutil.copyfile(downloaded, SCORES)
        state["state"] = "verifying_report"
        save(state)
        command([sys.executable, "-m", "scripts.summarize_care_four_way"], timeout=300)
        state["state"] = "complete"
        state["report"] = str(ROOT / "analysis/ASEM_CARE_four_way_20261112.md")
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        save(state)
    except Exception as exc:
        state["state"] = "failed"
        state["error"] = type(exc).__name__ + ": " + str(exc)[:1300]
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        save(state)
        raise


if __name__ == "__main__":
    main()
