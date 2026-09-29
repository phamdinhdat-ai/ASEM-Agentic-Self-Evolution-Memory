"""Resume an interrupted static-bank build for ONE conversation.

`build_static_banks.py` treats a failed conversation as disposable: on the next
run it calls `remove_bank_files()` and re-ingests from scratch, so a single bad
LLM response discards hours of work. This script does the opposite — it opens
the existing bank, keeps every session already present, and ingests only the
missing ones.

    conda activate memory-r1
    $env:PYTHONPATH="."
    python scratch_diag/_ckey_resume_bank.py \
        --tag gpt56_luna --system ASEM-THG --conversation locomo_0002 \
        --config configs/models/gpt56_luna.yaml

Safety: refuses to run unless the on-disk note count is non-zero and the session
set is a strict prefix of the conversation's sessions. Pass --force-resume to
override. Use --dry-run to preview without spending a single LLM call.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from asem.logging_utils import get_logger  # noqa: E402

log = get_logger("resume_bank")


def _existing_sessions(bank_file: str) -> Set[str]:
    """Session IDs already persisted in a bank (read-only SQLite)."""
    import sqlite3

    if not os.path.exists(bank_file) or os.path.getsize(bank_file) == 0:
        return set()
    conn = sqlite3.connect(f"file:{bank_file}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM notes WHERE session_id IS NOT NULL"
        ).fetchall()
    except sqlite3.Error:
        return set()
    finally:
        conn.close()
    return {str(r[0]) for r in rows if r[0]}


def _note_count(bank_file: str) -> int:
    from eval.static_banks import count_notes

    return count_notes(bank_file)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--conversation", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="datasets/locomo/locomo10.json")
    ap.add_argument("--bank-root", default="static/memory_banks/locomo10")
    ap.add_argument("--max-retries", type=int, default=None,
                    help="Override llm_retry.max_retries for this run.")
    ap.add_argument("--retries-per-session", type=int, default=3,
                    help="Extra attempts for a session that fails outright.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-resume", action="store_true",
                    help="Proceed even if the safety checks fail.")
    args = ap.parse_args()

    # The API backends read OPENAI_BASE_URL / OPENAI_API_KEY from the process
    # environment; build_static_banks.py loads .env before building the backend.
    # Do the same here, otherwise the backend has no credentials.
    try:
        from dotenv import load_dotenv

        load_dotenv(".env")
    except ImportError:
        pass

    from eval.phase_runner import (
        bank_file, build_backend_from_config, close_system, conversation_index,
        extract_sessions, load_raw_dataset, sessions_to_fast,
    )
    from eval.static_banks import conversation_entry, load_manifest, new_manifest
    from eval.systems import build_asem_thg_system

    # ---- locate the conversation + its sessions -------------------------
    raw = load_raw_dataset(args.dataset)
    idx = conversation_index(args.conversation)
    if idx < 0 or idx >= len(raw):
        log.error("Unknown conversation {!r}", args.conversation)
        return 2
    sessions = extract_sessions(raw[idx].get("conversation", {}))
    if not sessions:
        log.error("Conversation {} has no sessions", args.conversation)
        return 2

    fast = sessions_to_fast(sessions)
    bfile = bank_file(args.bank_root, args.tag, args.system, args.conversation)

    done = _existing_sessions(bfile)
    have = _note_count(bfile)
    missing = [s for s in fast if s["session_id"] not in done]

    log.info("conversation={} | sessions={} | bank={}", args.conversation, len(fast), bfile)
    log.info("already ingested: {} sessions / {} notes", len(done), have)
    log.info("missing          : {} sessions", len(missing))

    if not missing:
        log.info("Nothing to do — every session is already in the bank.")
        return 0

    if args.dry_run:
        for s in missing:
            log.info("  would ingest {} ({} turns, {})",
                     s["session_id"], len(s["turns"]), s.get("date"))
        return 0

    # ---- safety: never silently corrupt a bank we don't understand -------
    if not args.force_resume:
        if have == 0:
            log.error(
                "Bank holds 0 notes. A fresh build would be cheaper and safer — "
                "run build_static_banks.py for this conversation instead. "
                "(override with --force-resume)"
            )
            return 3
        expected = {s["session_id"] for s in fast}
        if not done <= expected:
            log.error(
                "Bank contains session IDs absent from the dataset "
                "(unknown={}). Refusing to resume — the bank may belong to a "
                "different conversation/config. (override with --force-resume)",
                sorted(done - expected)[:5],
            )
            return 3

    # ---- build the system on the EXISTING bank --------------------------
    bdir = os.path.dirname(bfile)
    backend = build_backend_from_config(args.config)
    system = build_asem_thg_system(args.config, bdir, backend=backend)
    bank = system.pipeline.memory_bank

    # Rehydrate the in-memory hyper-graph from SQLite. Without this the new
    # notes would only ever link to each other, leaving every cross-session
    # edge between the old and new halves missing.
    ingestor = system.single_pass_ingestor
    hydrated = ingestor.hyper_graph.hydrate(bank.list_notes())
    log.info("hydrated {} existing notes into the hyper-graph", hydrated)

    t_start = time.perf_counter()
    ok_sessions: List[str] = []
    failed: List[Dict[str, Any]] = []

    try:
        for s in missing:
            sid = s["session_id"]
            t0 = time.perf_counter()
            for attempt in range(1, args.retries_per_session + 1):
                try:
                    notes = ingestor.ingest_session(
                        dialogue_turns=s["turns"],
                        memory_bank=bank,
                        session_date=s.get("date"),
                        session_id=sid,
                    )
                    log.info("[{}] +{} notes ({:.1f}s)", sid, len(notes),
                             time.perf_counter() - t0)
                    ok_sessions.append(sid)
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log.warning("[{}] attempt {}/{} failed: {}: {}",
                                sid, attempt, args.retries_per_session,
                                type(exc).__name__, exc)
                    if attempt < args.retries_per_session:
                        time.sleep(min(2 ** attempt, 15))
            else:
                failed.append({"session_id": sid, "error": "exhausted retries"})
                log.error("[{}] giving up after {} attempts", sid,
                          args.retries_per_session)
    finally:
        total = bank.size()
        close_system(system)
        log.info("bank now holds {} notes (+{} this run) in {:.1f}s",
                 total, total - have, time.perf_counter() - t_start)

    # ---- update the manifest so the index stops claiming 0 notes ---------
    _patch_manifest(args, bfile, total, ok_sessions, failed, have)

    if failed:
        log.error("finished with {} failed session(s): {}",
                  len(failed), [f["session_id"] for f in failed])
        return 1
    log.info("resume complete — {} sessions ingested", len(ok_sessions))
    return 0


def _patch_manifest(args, bfile, total, ok_sessions, failed, had_before) -> None:
    """Record the true state of the bank in manifest.json + banks.json.

    The builder's error path hardcodes notes=0/bytes=0, which hides a partial
    bank from `run_static_eval.py`. Here we write what is actually on disk and
    only mark the conversation complete when no session is missing.
    """
    from eval.static_banks import (
        atomic_write_json, bank_index, count_links, load_manifest, model_name_from_config,
        new_manifest, recompute_totals, sha256_file, thinking_from_config,
    )
    import traceback

    tag_dir = os.path.join(args.bank_root, args.tag)
    manifest_path = os.path.join(tag_dir, "manifest.json")
    manifest = load_manifest(manifest_path) or new_manifest(
        "locomo10", args.tag, args.bank_root, args.dataset, args.config
    )
    entry = next(
        (c for c in manifest["conversations"]
         if c.get("conversation_id") == args.conversation), None
    )
    if entry is None:
        entry = {"conversation_id": args.conversation, "sessions": 0, "turns": 0,
                 "systems": {}}
        manifest["conversations"].append(entry)

    complete = not failed
    entry["systems"][args.system] = {
        "ingest_config": args.config,
        "config_sha256": sha256_file(args.config),
        "model": model_name_from_config(args.config) if os.path.exists(args.config) else "",
        "thinking": thinking_from_config(args.config) if os.path.exists(args.config) else None,
        "notes": int(total),
        "link_edges": count_links(bfile),
        "new_edges": 0,
        "status": "ok" if complete else "error",
        "bank": bfile,
        "bytes": os.path.getsize(bfile) if os.path.exists(bfile) else 0,
        "elapsed_sec": 0.0,
        "resumed": True,
        "sessions_ingested_this_run": ok_sessions,
    }
    if failed:
        entry["systems"][args.system]["failed_sessions"] = failed
    manifest["systems"] = sorted(set(manifest.get("systems", [])) | {args.system})
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    recompute_totals(manifest)
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(os.path.join(tag_dir, "banks.json"), bank_index(manifest))
    log.info("manifest updated: notes={} status={}", total,
             entry["systems"][args.system]["status"])


if __name__ == "__main__":
    raise SystemExit(main())
