"""Run SimRetrieval pilot then five full shards after context baselines finish."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "build/kaggle/care-sim-retrieval"
OUTPUT = ROOT / "build/care/sim_retrieval"
STATE = ROOT / "build/care/sim_retrieval_scheduler_status.json"
CONTEXT_STATE = ROOT / "build/care/context_baselines_finalizer_status.json"
CONTEXT_SHARDS = ROOT / "build/care/context_baselines_scheduler_status.json"
OWNER = "phamdinhdatai"


def save(value):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(STATE)


def command(args, timeout=120):
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                                errors="replace", timeout=timeout)
        return result.returncode, (result.stdout + "\n" + result.stderr).strip()
    except Exception as exc:
        return -1, type(exc).__name__ + ": " + str(exc)[:300]


def ref(index):
    return f"{OWNER}/asem-care-simretrieval-{'pilot-s0' if index == 'pilot' else 'shard-' + str(index)}"


def status(index):
    code, output = command(["kaggle", "kernels", "status", ref(index)], timeout=90)
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output)
    return (match.group(1) if code == 0 and match else None), output[:500]


def download(index):
    pilot = index == "pilot"
    destination = OUTPUT / ("pilot" if pilot else f"shard-{index}")
    base_name = f"care_sim_retrieval_{'pilot_' if pilot else ''}s{0 if pilot else index}"
    pattern = rf"{base_name}/(?:status\.json|results/report\.json|pilot\.log)"
    code, detail = command(["kaggle", "kernels", "output", ref(index), "-p",
                            str(destination), "--file-pattern", pattern, "--quiet"], timeout=300)
    base = destination / base_name
    status_path, report_path = base / "status.json", base / "results/report.json"
    if code != 0 or not status_path.exists() or not report_path.exists():
        return False, detail[:500] or "Missing retrieval output"
    local_status = json.loads(status_path.read_text(encoding="utf-8"))
    if local_status.get("state") != "complete":
        return False, json.dumps(local_status)[:500]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = 4 if pilot else None
    if expected is not None and (len(report["rows"]) != expected or
                                 report["aggregate"]["sim_retrieval"]["errors"]):
        return False, "Pilot incomplete or contains errors"
    return True, json.dumps(local_status)[:500]


def initial():
    return {"protocol": "care-sim-retrieval-scheduler-v1",
            "pilot": "pending", "submitted": [], "completed": [],
            "pending": list(range(5)), "failed": {}, "last_status": {},
            "state": "waiting_context", "started_at": datetime.now(timezone.utc).isoformat()}


def step(state):
    if state["state"] == "waiting_context":
        if not CONTEXT_STATE.exists() or not CONTEXT_SHARDS.exists():
            save(state)
            return state
        context = json.loads(CONTEXT_STATE.read_text(encoding="utf-8"))
        shards = json.loads(CONTEXT_SHARDS.read_text(encoding="utf-8"))
        if shards.get("state") != "complete":
            save(state)
            return state
        if context.get("state") not in {"complete", "failed"}:
            save(state)
            return state
        if context.get("state") == "failed":
            state["context_report_warning"] = context.get("error", "Context BERTScore failed")
        state["state"] = "running"
    if state["pilot"] == "pending":
        code, output = command(["kaggle", "kernels", "push", "-p",
                                str(ROOT / "build/kaggle/care-sim-retrieval-pilot")], timeout=180)
        if code == 0 and "successfully pushed" in output:
            state["pilot"] = "submitted"
        else:
            state["last_push_error"] = output[:500]
    elif state["pilot"] == "submitted":
        current, detail = status("pilot")
        state["last_status"]["pilot"] = current or "UNKNOWN"
        if current == "COMPLETE":
            ok, output = download("pilot")
            if ok:
                state["pilot"] = "complete"
            else:
                state["failed"]["pilot"] = output
        elif current in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
            state["failed"]["pilot"] = current + ": " + detail
    if state["pilot"] == "complete" and not state["failed"]:
        active = 0
        for index in list(state["submitted"]):
            if index in state["completed"] or str(index) in state["failed"]:
                continue
            current, detail = status(index)
            state["last_status"][str(index)] = current or "UNKNOWN"
            if current == "COMPLETE":
                ok, output = download(index)
                if ok:
                    state["completed"].append(index)
                    state["completed"].sort()
                else:
                    state["failed"][str(index)] = output
            elif current in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
                state["failed"][str(index)] = current + ": " + detail
            else:
                active += 1
        while active < 2 and state["pending"] and not state["failed"]:
            index = state["pending"][0]
            code, output = command(["kaggle", "kernels", "push", "-p",
                                    str(PACKAGE / f"shard-{index}")], timeout=180)
            if code == 0 and "successfully pushed" in output:
                state["pending"].pop(0)
                state["submitted"].append(index)
                state["submitted"].sort()
                state["last_status"][str(index)] = "SUBMITTED"
                active += 1
            else:
                state["last_push_error"] = output[:500]
                break
        if len(state["completed"]) == 5:
            code, output = command([sys.executable, "-m", "scripts.merge_care_sim_retrieval"],
                                   timeout=300)
            if code == 0:
                state["state"] = "complete"
                state["merged_report"] = str(ROOT / "build/care/sim_retrieval_merged.json")
                state.pop("merge_error", None)
            else:
                state["state"] = "merge_failed"
                state["merge_error"] = output[-1500:]
    if state["failed"]:
        state["state"] = "shard_failed"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    save(state)
    print(json.dumps({k: state[k] for k in
                      ("state", "pilot", "submitted", "completed", "pending", "failed")}), flush=True)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=90)
    parser.add_argument("--max-hours", type=int, default=48)
    args = parser.parse_args()
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else initial()
    deadline = datetime.now(timezone.utc) + timedelta(hours=args.max_hours)
    while True:
        state = step(state)
        if args.once or state["state"] in {"complete", "merge_failed", "shard_failed"}:
            break
        if datetime.now(timezone.utc) >= deadline:
            state["state"] = "scheduler_timeout"
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            save(state)
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
