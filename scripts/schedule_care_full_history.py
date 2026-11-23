"""Wait for the 40K pilot, then run five full-history shards one at a time."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from scripts.run_care_full_locomo import SHARDS


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "build/care/full_history_40k_scheduler_status.json"
PACKAGE = ROOT / "build/kaggle/care-full-history-40k"
OUTPUT = ROOT / "build/care/full_history_40k"


def command(args: list[str], *, timeout: int = 300) -> tuple[int, str]:
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                                errors="replace", timeout=timeout)
        return result.returncode, (result.stdout + "\n" + result.stderr).strip()
    except Exception as exc:
        return -1, f"{type(exc).__name__}: {str(exc)[:300]}"


def reference(shard: int | None) -> str:
    suffix = "pilot" if shard is None else f"shard-{shard}"
    return f"phamdinhdatai/asem-care-full-history-40k-{suffix}"


def status(shard: int | None) -> str:
    code, detail = command(["kaggle", "kernels", "status", reference(shard)], timeout=90)
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", detail)
    return match.group(1) if code == 0 and match else "UNKNOWN"


def fetch(shard: int | None) -> tuple[bool, str]:
    pilot = shard is None
    destination = OUTPUT / ("pilot" if pilot else f"shard-{shard}")
    name = f"care_full_history_40k_{'pilot_' if pilot else ''}s{0 if pilot else shard}"
    pattern = rf"{name}/(?:status\.json|results/report\.json|pilot\.log|vllm\.log)"
    code, detail = command(["kaggle", "kernels", "output", reference(shard),
                            "-p", str(destination), "--file-pattern", pattern,
                            "--quiet"], timeout=300)
    base = destination / name
    status_path, report_path = base / "status.json", base / "results/report.json"
    if code or not status_path.exists() or not report_path.exists():
        return False, detail[:500] or "Missing pilot or shard output"
    run_status = json.loads(status_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = 2 if pilot else 10
    if run_status.get("state") != "complete" or len(report["rows"]) != expected:
        return False, f"Incomplete run: {run_status.get('state')}, {len(report['rows'])} rows"
    if report["manifest"]["shard_index"] != (0 if pilot else shard):
        return False, "Wrong shard index"
    if not pilot and {row["sample_id"] for row in report["rows"]} != set(SHARDS[shard]):
        return False, "Wrong conversation IDs"
    if any(row["prompt_tokens_server"] + row["completion_tokens"] > 40960
           for row in report["rows"]):
        return False, "40K model-window violation"
    return True, f"{expected} questions complete"


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE)


def step(state: dict) -> dict:
    if not state["pilot_complete"]:
        state["pilot_status"] = status(None)
        if state["pilot_status"] == "COMPLETE":
            okay, detail = fetch(None)
            if okay:
                state["pilot_complete"] = True
            else:
                state["failed"]["pilot"] = detail
        elif state["pilot_status"] in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
            state["failed"]["pilot"] = state["pilot_status"]

    if state["pilot_complete"] and not state["failed"]:
        for shard in list(state["submitted"]):
            if shard in state["completed"]:
                continue
            current = status(shard)
            state["last_status"][str(shard)] = current
            if current == "COMPLETE":
                okay, detail = fetch(shard)
                if okay:
                    state["completed"].append(shard)
                    state["completed"].sort()
                else:
                    state["failed"][str(shard)] = detail
            elif current in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
                state["failed"][str(shard)] = current

        active = [shard for shard in state["submitted"] if shard not in state["completed"]]
        if not active and state["pending"] and not state["failed"]:
            shard = state["pending"][0]
            code, detail = command(["kaggle", "kernels", "push", "-p",
                                    str(PACKAGE / f"shard-{shard}")], timeout=180)
            if code or "successfully pushed" not in detail:
                state["failed"][str(shard)] = detail[:500]
            else:
                state["pending"].pop(0)
                state["submitted"].append(shard)
                state["last_status"][str(shard)] = "SUBMITTED"

    if state["failed"]:
        state["state"] = "failed"
    elif len(state["completed"]) == 5:
        code, detail = command([sys.executable, "-m", "scripts.summarize_care_full_history"])
        state["state"] = "complete" if code == 0 else "merge_failed"
        if code:
            state["merge_error"] = detail[-1000:]
    else:
        state["state"] = "running"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    save(state)
    print(json.dumps({key: state[key] for key in
                      ("state", "pilot_status", "pilot_complete", "submitted",
                       "completed", "pending", "failed", "last_status")}), flush=True)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
    else:
        state = {"protocol": "care-full-history-40k-scheduler-v1",
                 "state": "running", "pilot_status": "UNKNOWN",
                 "pilot_complete": False, "submitted": [], "completed": [],
                 "pending": [0, 1, 2, 3, 4], "failed": {}, "last_status": {},
                 "started_at": datetime.now(timezone.utc).isoformat()}
    while True:
        state = step(state)
        if args.once or state["state"] in {"failed", "complete", "merge_failed"}:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
