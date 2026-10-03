"""In-process per-question bank snapshot hook for LongMemEval (WS2 / bank-runner).

Problem
-------
`ASEMTHGSystem.reset()` clears the ASEM-THG bank before every LongMemEval
question, and the runner flushes the prediction row immediately before that
reset. An EXTERNAL watcher races the reset and usually loses (verified: 7 of 10
first captures read an empty bank). The graph for a question only reliably
exists from *inside* the process.

Approach
--------
Wrap `MemoryBank.clear`. At the instant `clear()` is called, the bank still
holds the graph of the question that was just answered — so snapshot there, keyed
by the question that is about to be finished. `clear()` is called exactly once
per question by `ASEMTHGSystem.reset()`, which makes it a precise trigger.

Activation is by environment variable only; no file under `asem/`, `eval/` or
`scripts/` is modified, so the running pipeline is untouched.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_STATE: Dict[str, Any] = {
    "dir": None,
    "index_path": None,
    "snapshots": [],
    "installed": False,
    "bank_path": None,
    "pending": None,      # metadata for the question whose graph is in the bank
    "clear_count": 0,
}


def _read_bank(path: str) -> Dict[str, Any]:
    uri = "file:" + os.path.abspath(path).replace("?", "%3f").replace("#", "%23")
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    try:
        rows = list(conn.execute(
            "SELECT id, entities, L, timestamp_iso, session_id, t, c FROM notes"
        ))
    finally:
        conn.close()
    n = len(rows)
    if not n:
        return {"n_notes": 0}

    rel: Counter = Counter()
    undirected: set = set()
    ent_df: Counter = Counter()
    note_ents: List[set] = []
    sup_targets: Dict[str, List[str]] = {}
    n_ts_null = 0
    n_today = 0
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sids: set = set()
    meta: Dict[str, Dict[str, Any]] = {}

    for nid, ents_raw, links_raw, ts_iso, sid, t_raw, content in rows:
        try:
            ents = json.loads(ents_raw) if ents_raw else []
        except (json.JSONDecodeError, TypeError):
            ents = []
        entset = {str(e).strip().lower() for e in (ents or []) if str(e).strip()}
        note_ents.append(entset)
        for e in entset:
            ent_df[e] += 1

        if not ts_iso:
            n_ts_null += 1
        elif str(ts_iso).startswith(today):
            n_today += 1
        if sid:
            sids.add(str(sid))

        meta[nid] = {
            "session_id": str(sid) if sid else None,
            "t": t_raw,
            "c": (content or "")[:400],
            "entities": sorted(entset),
        }

        try:
            links = json.loads(links_raw) if links_raw else []
        except (json.JSONDecodeError, TypeError):
            links = []
        for link in links or []:
            if isinstance(link, dict):
                r = str(link.get("relation") or "linked")
                target = str(link.get("target_id") or "")
            else:
                r, target = "linked", str(link)
            rel[r] += 1
            if r == "superseded_by" and target:
                sup_targets.setdefault(nid, []).append(target)
            if target:
                undirected.add(tuple(sorted((nid, target))))

    # Entity fan-out.
    ent_to_notes: Dict[str, set] = {}
    for i, es in enumerate(note_ents):
        for e in es:
            ent_to_notes.setdefault(e, set()).add(i)
    fan: List[int] = []
    for i, es in enumerate(note_ents):
        peers: set = set()
        for e in es:
            peers |= ent_to_notes.get(e, set())
        peers.discard(i)
        fan.append(len(peers))
    fan_sorted = sorted(fan)

    def pct(p: float) -> float:
        if not fan_sorted:
            return 0.0
        k = min(len(fan_sorted) - 1, max(0, int(round((p / 100.0) * (len(fan_sorted) - 1)))))
        return float(fan_sorted[k])

    # Supersession split: same-session restatement vs cross-session replacement.
    seen_pairs: set = set()
    pairs: List[Dict[str, Any]] = []
    for a, targets in sup_targets.items():
        for b in targets:
            key = tuple(sorted((a, b)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            ma, mb = meta.get(a, {}), meta.get(b, {})
            same = bool(ma.get("session_id")) and ma.get("session_id") == mb.get("session_id")
            pairs.append({
                "a": a, "b": b,
                "same_session": same,
                "a_session": ma.get("session_id"), "b_session": mb.get("session_id"),
                "a_date": ma.get("t"), "b_date": mb.get("t"),
                "a_sentence": ma.get("c"), "b_sentence": mb.get("c"),
                "a_entities": ma.get("entities"), "b_entities": mb.get("entities"),
            })
    n_same = sum(1 for p in pairs if p["same_session"])
    n_cross = len(pairs) - n_same

    return {
        "n_notes": n,
        "n_edges": sum(rel.values()),
        "n_unique_undirected_pairs": len(undirected),
        "edge_histogram": dict(sorted(rel.items(), key=lambda kv: -kv[1])),
        "entity_fanout_mean": round(sum(fan) / n, 4),
        "entity_fanout_median": float(fan_sorted[n // 2]),
        "entity_fanout_p90": pct(90),
        "entity_fanout_max": max(fan),
        "n_distinct_entities": len(ent_df),
        "n_notes_with_empty_entities": sum(1 for es in note_ents if not es),
        "n_superseded_by_edges": rel.get("superseded_by", 0),
        "n_notes_with_supersession": sum(1 for k in sup_targets if sup_targets[k]),
        "n_supersession_pairs": len(pairs),
        "n_supersession_pairs_same_session": n_same,
        "n_supersession_pairs_cross_session": n_cross,
        "supersession_pairs": pairs,
        "n_notes_timestamp_iso_null": n_ts_null,
        "n_notes_timestamp_today": n_today,
        "n_notes_date_unknown": n_ts_null,
        "session_ids": sorted(sids),
        "top10_entities_by_df": [
            {"entity": e, "df": c, "df_over_n": round(c / n, 4)}
            for e, c in ent_df.most_common(10)
        ],
    }


def _capture() -> Optional[Dict[str, Any]]:
    bank = _STATE.get("bank_path")
    if not bank or not os.path.exists(bank):
        return None
    row: Dict[str, Any] = {"captured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    row.update(_STATE.get("pending") or {})
    try:
        row.update(_read_bank(bank))
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def install(out_dir: str) -> bool:
    if _STATE.get("installed"):
        return True
    os.makedirs(out_dir, exist_ok=True)
    _STATE["dir"] = out_dir
    _STATE["index_path"] = os.path.join(out_dir, "_index.json")

    import asem.memory_bank as mb

    _STATE["bank_path"] = os.path.join(
        os.environ.get("ASEM_SNAPSHOT_DB_DIR", "data/benchmarks/longmemeval_banks"),
        "asem_thg.sqlite",
    )
    target = os.path.abspath(_STATE["bank_path"])
    original_clear = mb.MemoryBank.clear

    def clear_with_snapshot(self):  # noqa: ANN001
        try:
            if os.path.abspath(self._db_path) == target:
                _STATE["clear_count"] = int(_STATE.get("clear_count", 0)) + 1
                row = _capture()
                if row is not None and row.get("n_notes", 0) > 0:
                    _STATE["snapshots"].append(row)
                    _write_index()
                    print(
                        f"[snapshot] q{row.get('idx')} {row.get('question_id')} -> "
                        f"{row.get('n_notes')} notes, {row.get('n_edges')} edges, "
                        f"sup_pairs={row.get('n_supersession_pairs')}",
                        flush=True,
                    )
                else:
                    print(f"[snapshot] clear #{_STATE['clear_count']}: bank empty (first question)",
                          flush=True)
                _STATE["pending"] = None
        except Exception as exc:  # noqa: BLE001
            print(f"[snapshot] failed: {type(exc).__name__}: {exc}", flush=True)
        return original_clear(self)

    mb.MemoryBank.clear = clear_with_snapshot

    # Feed question identity: wrap the runner's ingest dispatch so we know which
    # question's graph the bank currently holds. The runner calls
    # `_ingest_question(system, name, sessions)` per question.
    try:
        import scripts.run_longmemeval_eval as runner

        original_ingest = runner._ingest_question
        counter = {"i": 0}

        def ingest_wrapped(system, name, sessions):  # noqa: ANN001
            if name in ("ASEM-THG", "ASEM_THG"):
                idx = counter["i"]
                counter["i"] += 1
                _STATE["pending"] = {
                    "idx": idx,
                    "question_id": None,   # filled by the reader wrapper below
                    "n_haystack_sessions": len(list(sessions)),
                    "n_haystack_turns": sum(
                        len(getattr(s, "turns", []) or []) for s in sessions
                    ),
                }
            return original_ingest(system, name, sessions)

        runner._ingest_question = ingest_wrapped

        # Question identity: the ONLY thing the snapshot needs at capture time is
        # "which question produced the graph currently in the bank". The runner
        # answers questions in `groups` order and appends one prediction row per
        # question, so the question is identified at capture time by the count of
        # questions already started. We record that ordinal here and join it to
        # `question_id` afterwards from the runner's own predictions file
        # (see `analysis/build_per_question_index.py --preds`).
        import scripts.run_longmemeval_eval as runner

        orig_sfg = runner.sessions_for_group
        counter = {"i": 0}

        def sfg_wrapped(group):  # noqa: ANN001
            try:
                if group and _STATE.get("pending") is not None:
                    # New question starting: stamp the ordinal of the PREVIOUS
                    # one we are about to finish is already set; here we set the
                    # ordinal for the question about to be ingested.
                    item = group[0].get("_item")
                    _STATE["pending"] = dict(_STATE.get("pending") or {})
                    _STATE["pending"]["idx"] = counter["i"]
                    _STATE["pending"]["question_id"] = getattr(item, "question_id", None)
                    _STATE["pending"]["question_type"] = getattr(item, "question_type", None)
                    counter["i"] += 1
            except Exception:  # noqa: BLE001
                pass
            return orig_sfg(group)

        runner.sessions_for_group = sfg_wrapped
        _STATE["_orig_sfg"] = orig_sfg
    except Exception as exc:  # noqa: BLE001
        print(f"[snapshot] runner patch skipped: {type(exc).__name__}: {exc}", flush=True)

    _STATE["installed"] = True
    print(f"[snapshot] installed -> {out_dir} (bank={_STATE['bank_path']})", flush=True)
    return True


def _write_index() -> None:
    path = _STATE.get("index_path")
    if not path:
        return
    payload = {
        "tool": "analysis/snapshot_hook.py",
        "note": "One row per LongMemEval question, captured in-process at MemoryBank.clear().",
        "n_questions": len(_STATE["snapshots"]),
        "per_question": _STATE["snapshots"],
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)
