"""Finish baseline BERTScore and paired CARE report after the shard scheduler."""

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
SCHEDULER = ROOT / "build/care/context_baselines_scheduler_status.json"
STATE = ROOT / "build/care/context_baselines_finalizer_status.json"
PACKAGE = ROOT / "build/kaggle/care-context-bertscore-full"
DOWNLOAD = ROOT / "build/care/context_baselines_bertscore_output"
SCORES = ROOT / "build/care/context_baselines_bertscore.json"
KERNEL = "phamdinhdatai/asem-care-context-bertscore-full"


def save(value: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(STATE)


def command(args: list[str], timeout: int = 300) -> str:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                            errors="replace", timeout=timeout)
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode:
        raise RuntimeError(f"{args[:4]} exited {result.returncode}: {output[-1200:]}")
    return output


def main() -> None:
    deadline = datetime.now(timezone.utc) + timedelta(hours=30)
    state = {"protocol": "care-context-baselines-finalizer-v1",
             "state": "waiting_shards",
             "started_at": datetime.now(timezone.utc).isoformat()}
    save(state)
    try:
        while datetime.now(timezone.utc) < deadline:
            scheduler = json.loads(SCHEDULER.read_text(encoding="utf-8"))
            if scheduler["state"] == "complete":
                break
            if scheduler["state"] in ("merge_failed", "shard_failed", "scheduler_timeout"):
                raise RuntimeError(f"Baseline scheduler stopped: {scheduler['state']}")
            time.sleep(90)
        else:
            raise TimeoutError("Baseline shards did not finish within 30 hours")
        state["state"] = "preparing_bertscore"
        save(state)
        command([sys.executable, "-m", "scripts.prepare_care_context_bertscore_kaggle"])
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
            raise TimeoutError("BERTScore notebook did not finish within 30 hours")
        state["state"] = "downloading_bertscore"
        save(state)
        command(["kaggle", "kernels", "output", KERNEL, "-p", str(DOWNLOAD)],
                timeout=300)
        downloaded = DOWNLOAD / "care_context_bertscore_full.json"
        if not downloaded.exists():
            raise FileNotFoundError("Missing Kaggle BERTScore output JSON")
        shutil.copyfile(downloaded, SCORES)
        state["state"] = "verifying_report"
        save(state)
        command([sys.executable, "-m", "scripts.summarize_care_context_baselines"],
                timeout=300)
        state["state"] = "complete"
        state["report"] = str(ROOT / "analysis/ASEM_CARE_vs_context_baselines_20261112.md")
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
