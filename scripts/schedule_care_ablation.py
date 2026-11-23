"""Queue, fetch and validate the CARE ablation (docs/CARE_ablation_protocol_20261121.md).

Chained behind protocol v3: nothing is pushed while the v3 scheduler still has shards to submit, so v3 keeps
priority for the two-session Kaggle limit; the slot v3 leaves free afterwards is used here. Order: the two pilots must pass, then
LoCoMo shards 0-4 and LongMemEval-oracle shards 0-4, at most two kernels at a time. Every fetched kernel is
checked (rows, distinct ids, manifest protocol / dataset / worker-code hashes, gold_in_model_context False, the
reference arm present, physical calls within the cap). A failure stops the queue and is reported, not retried.
When all ten shards pass, scripts.summarize_care_ablation runs for both benchmarks.

    python -m scripts.schedule_care_ablation --poll-seconds 300 [--no-wait-for-v3]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from scripts.run_care_ablation import ARM_NAMES, CODE_FILES

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "build/care/ablation_scheduler_status.json"
V3_STATE = ROOT / "build/care/v3_scheduler_status.json"
INGEST = ROOT / "build/care/ingested_ab"
PACKAGE = ROOT / "build/kaggle/care-ab"
OUTPUT = ROOT / "kaggle_outputs/care_ab"
SHARD_SIZES = {"locomo": [365, 400, 429, 397, 395], "lme": [98, 110, 100, 97, 95]}
DATASET_SHA = {"locomo": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
               "lme": "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c"}
BENCH = {"locomo": "locomo", "lme": "longmemeval"}
MAX_CALLS = {"locomo": 2600, "lme": 1300}
MAX_ACTIVE = 2
TERMINAL_BAD = {"ERROR", "FAILED", "CANCELLED", "CANCELED"}
PILOTS = ["pilot-locomo", "pilot-lme"]
SHARDS = [f"locomo-s{i}" for i in range(5)] + [f"lme-s{i}" for i in range(5)]


def code_sha() -> str:
    return hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in CODE_FILES)).hexdigest()


def command(args: list[str], *, timeout: int = 300) -> tuple[int, str]:
    try:
        r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, errors="replace", timeout=timeout)
        return r.returncode, (r.stdout + "\n" + r.stderr).strip()
    except Exception as exc:
        return -1, f"{type(exc).__name__}: {str(exc)[:300]}"


def slug(name: str) -> str:            # "locomo-s2" -> asem-care-ab-locomo-s2 ; "pilot-lme" -> asem-care-ab-lme-pilot
    if name.startswith("pilot-"):
        return f"asem-care-ab-{name.split('-', 1)[1]}-pilot"
    return f"asem-care-ab-{name}"


def out_name(name: str) -> str:
    return "care_ab_" + slug(name).replace("asem-care-ab-", "").replace("-", "_")


def kernel_status(name: str) -> str:
    code, detail = command(["kaggle", "kernels", "status", f"phamdinhdatai/{slug(name)}"], timeout=90)
    m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", detail)
    return m.group(1) if code == 0 and m else "UNKNOWN"


def fetch_and_validate(name: str) -> tuple[bool, str]:
    dest = OUTPUT / name
    base_name = out_name(name)
    pattern = rf"{base_name}/(?:status\.json|results/.*|pilot\.log|vllm\.log)"
    code, detail = command(["kaggle", "kernels", "output", f"phamdinhdatai/{slug(name)}", "-p", str(dest),
                            "--file-pattern", pattern, "--quiet"], timeout=900)
    if code:
        return False, detail[:400] or "missing output"
    return validate_dir(name, dest / base_name)


def validate_dir(name: str, base: Path) -> tuple[bool, str]:
    """Gate checks for one finished kernel directory (``base`` holds status.json and results/)."""
    bench = "locomo" if "locomo" in name else "lme"
    pilot = name.startswith("pilot-")
    report_path, status_path, resp_path = base / "results/report.json", base / "status.json", base / "results/responses.jsonl"
    if not report_path.exists() or not status_path.exists():
        return False, "missing output"
    if json.loads(status_path.read_text(encoding="utf-8")).get("state") != "complete":
        return False, "worker state is not complete"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    man, rows = report["manifest"], report["rows"]
    shard = 0 if pilot else int(name.rsplit("-s", 1)[1])
    expected = 4 if pilot else SHARD_SIZES[bench][shard]
    problems = []
    if len(rows) != expected:
        problems.append(f"{len(rows)} rows, expected {expected}")
    if len({r["id"] for r in rows}) != len(rows):
        problems.append("duplicate ids")
    if man.get("protocol") != "care-ablation-v1" or man.get("benchmark") != BENCH[bench] or man.get("shard_index") != shard:
        problems.append("wrong protocol, benchmark or shard")
    if man.get("dataset_sha256") != DATASET_SHA[bench]:
        problems.append("dataset sha mismatch")
    if man.get("code_sha256") != code_sha():
        problems.append("code sha mismatch")
    if man.get("gold_in_model_context") is not False:
        problems.append("gold_in_model_context not False")
    if list(man.get("arms", [])) != list(ARM_NAMES) or any(set(r["systems"]) != set(ARM_NAMES) for r in rows):
        problems.append("arms missing or changed")
    n_phys = sum(1 for _ in resp_path.open(encoding="utf-8")) if resp_path.exists() else -1
    if not 0 < n_phys <= MAX_CALLS[bench]:
        problems.append(f"physical calls {n_phys}")
    return (not problems), ("; ".join(problems) if problems else f"{len(rows)} questions, {n_phys} physical calls, hashes match")


def absorb_ingested(state: dict) -> None:
    """Jobs run elsewhere and validated by scripts.ingest_remote_results leave a marker; count them as done."""
    for marker in sorted(INGEST.glob("*")) if INGEST.exists() else []:
        name = marker.name
        if name in PILOTS + SHARDS and name not in state["completed"]:
            state["completed"].append(name)
            state["notes"][name] = "ingested from a remote run"
        if name in state["pending"]:
            state["pending"].remove(name)


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATE)


def step(state: dict, wait_for_v3: bool) -> dict:
    absorb_ingested(state)
    if wait_for_v3:
        v3_full = json.loads(V3_STATE.read_text(encoding="utf-8")) if V3_STATE.exists() else {}
        v3 = v3_full.get("state", "missing")
        state["v3_state"] = v3
        # v3 keeps priority while it still has shards to submit; once every v3 shard is submitted (or v3 has
        # ended) the free Kaggle slot is ours. Kaggle itself refuses a third session, which is not a failure.
        v3_has_pending = v3 in {"running", "missing"} and bool(v3_full.get("pending", [0]))
        if v3_has_pending:
            state["state"] = "waiting_for_v3"
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            save(state)
            print(json.dumps({"state": state["state"], "v3_state": v3}), flush=True)
            return state
    for name in [n for n in state["submitted"] if n not in state["completed"] and n not in state["failed"]]:
        cur = kernel_status(name)
        state["last_status"][name] = cur
        if cur == "COMPLETE":
            ok, detail = fetch_and_validate(name)
            state["notes"][name] = detail
            if ok:
                state["completed"].append(name)
            else:
                state["failed"][name] = detail
        elif cur in TERMINAL_BAD:
            state["failed"][name] = cur
    active = [n for n in state["submitted"] if n not in state["completed"] and n not in state["failed"]]
    pilots_done = all(p in state["completed"] for p in PILOTS)
    while state["pending"] and len(active) < MAX_ACTIVE and not state["failed"]:
        name = state["pending"][0]
        if name not in PILOTS and not pilots_done:
            break                                  # shards wait for both pilots to pass
        code, detail = command(["kaggle", "kernels", "push", "-p", str(PACKAGE / name)], timeout=240)
        if "successfully pushed" in detail:
            state["pending"].pop(0)
            state["submitted"].append(name)
            state["last_status"][name] = "SUBMITTED"
            active.append(name)
        elif "Maximum batch GPU session count" in detail or code == -1:
            state["notes"]["push"] = detail[-200:]
            break
        else:
            state["failed"][f"push-{name}"] = detail[-400:]
            break
    if state["failed"]:
        state["state"] = "failed"
    elif len(state["completed"]) == len(PILOTS) + len(SHARDS):
        codes = [command([sys.executable, "-m", "scripts.summarize_care_ablation", "--benchmark", b], timeout=1800)
                 for b in ("locomo", "lme")]
        state["state"] = "complete" if all(c == 0 for c, _ in codes) else "summary_failed"
        if state["state"] != "complete":
            state["summary_error"] = " | ".join(d[-400:] for c, d in codes if c)
    else:
        state["state"] = "running"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    save(state)
    print(json.dumps({k: state[k] for k in ("state", "submitted", "completed", "pending", "failed")}), flush=True)
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-wait-for-v3", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="job names to take out of the queue (run elsewhere)")
    args = ap.parse_args()
    state = (json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else
             {"protocol": "care-ablation-scheduler-v1", "submitted": [], "completed": [], "failed": {},
              "pending": [*PILOTS, *SHARDS], "last_status": {}, "notes": {},
              "started_at": datetime.now(timezone.utc).isoformat()})
    for name in args.skip:
        if name in state["pending"]:
            state["pending"].remove(name)
    while True:
        state = step(state, not args.no_wait_for_v3)
        if args.once or state["state"] in {"complete", "failed", "summary_failed"}:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
