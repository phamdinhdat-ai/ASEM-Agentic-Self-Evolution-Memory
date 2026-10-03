"""Graph metrics for ASEM-THG memory banks (READ-ONLY analysis tool).

Owned by WS2 (`bank-runner`). Reads a persisted ASEM bank (SQLite) and reports
the graph-shape numbers the thesis needs:

  * notes / total links / link-relation histogram
  * entity fan-out per note (notes sharing >= 1 entity), mean + median + p90 + max
  * share of notes with >= 1 `superseded_by` link, plus verbatim example pairs
  * how many notes render `date unknown`

Nothing here mutates a bank. It only opens SQLite in read-only mode.

Usage
-----
    python analysis/graph_metrics.py --bank <path/to/asem_thg.sqlite> [--examples 5]
    python analysis/graph_metrics.py --dir <bank-dir> --label gb_full --json out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Bank loading
# ---------------------------------------------------------------------------

def _table_names(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return [r[0] for r in rows]


# Columns stored as JSON blobs inside the `notes` table.
_JSON_COLS = ("K", "G", "e", "L", "z", "entities")


def load_notes(bank_path: str) -> List[Dict[str, Any]]:
    """Load every note row from a bank as a dict.

    The persisted bank (``asem/memory_bank.py``) is a **columnar** SQLite table
    ``notes(id, c, t, K, G, X, e, L, z, q, session_id, session_date,
    timestamp_iso, entities, speaker)`` where ``K/G/e/L/z/entities`` are JSON
    text. A dict-payload layout is also tolerated for forward compatibility.
    """
    uri = "file:" + os.path.abspath(bank_path).replace("?", "%3f").replace("#", "%23")
    conn = sqlite3.connect(uri, uri=True)
    try:
        tables = _table_names(conn)
        cand = [t for t in tables if "note" in t.lower()] or tables
        notes: List[Dict[str, Any]] = []
        for table in cand:
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
            if not cols:
                continue

            # (a) columnar layout: the table itself IS the note schema.
            if "id" in cols and "c" in cols:
                select = ", ".join(f'"{c}"' for c in cols)
                for row in conn.execute(f'SELECT {select} FROM "{table}"'):
                    rec = dict(zip(cols, row))
                    for col in _JSON_COLS:
                        if col in rec:
                            rec[col] = _maybe_json(rec[col])
                    notes.append(rec)
                if notes:
                    break

            # (b) payload layout: one JSON blob per row.
            payload_col = next(
                (n for n in ("payload", "data", "json", "note") if n in cols), None
            )
            if payload_col is None:
                continue
            for (raw,) in conn.execute(f'SELECT "{payload_col}" FROM "{table}"'):
                rec = _maybe_json(raw)
                if isinstance(rec, dict) and "id" in rec and "c" in rec:
                    notes.append(rec)
            if notes:
                break
        return notes
    finally:
        conn.close()


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{\"" and text not in ("null", "true", "false"):
        return value
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return value


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _norm_entity(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _link_iter(note: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for item in note.get("L") or []:
        if isinstance(item, dict):
            yield item
        elif isinstance(item, str):
            yield {"target_id": item, "relation": "linked"}


def _parse_dt(value: Any):
    from datetime import datetime

    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def compute_metrics(
    notes: List[Dict[str, Any]], n_examples: int = 5
) -> Dict[str, Any]:
    by_id = {n["id"]: n for n in notes}

    # --- link histogram -------------------------------------------------
    rel_counts: Counter = Counter()
    undirected: Counter = Counter()          # canonical unordered pair count
    per_note_rel: Dict[str, Counter] = {n["id"]: Counter() for n in notes}
    superseded_in: Counter = Counter()       # note -> #superseded_by links
    for note in notes:
        for link in _link_iter(note):
            rel = str(link.get("relation") or "linked")
            target = str(link.get("target_id") or "")
            rel_counts[rel] += 1
            per_note_rel[note["id"]][rel] += 1
            if rel == "superseded_by":
                superseded_in[note["id"]] += 1
            if target:
                key = tuple(sorted((note["id"], target)))
                undirected[key] += 1

    n_links = sum(rel_counts.values())

    # --- entity fan-out -------------------------------------------------
    entity_to_notes: Dict[str, set] = defaultdict(set)
    note_entities: Dict[str, set] = {}
    for note in notes:
        ents = {_norm_entity(e) for e in (note.get("entities") or []) if _norm_entity(e)}
        note_entities[note["id"]] = ents
        for ent in ents:
            entity_to_notes[ent].add(note["id"])

    fanout: Dict[str, int] = {}
    for note in notes:
        peers: set = set()
        for ent in note_entities[note["id"]]:
            peers |= entity_to_notes[ent]
        peers.discard(note["id"])
        fanout[note["id"]] = len(peers)

    fan_values = list(fanout.values())

    def _pct(values: List[int], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
        return float(ordered[idx])

    share_metric = {
        "mean": statistics.fmean(fan_values) if fan_values else 0.0,
        "median": statistics.median(fan_values) if fan_values else 0.0,
        "p90": _pct(fan_values, 90),
        "max": max(fan_values) if fan_values else 0,
        "zero_share": (sum(1 for v in fan_values if v == 0) / len(fan_values))
        if fan_values
        else 0.0,
    }

    # Same-entity edges implied by the entity channel (clique check).
    implied_pairs = 0
    for _ent, ids in entity_to_notes.items():
        k = len(ids)
        if k > 1:
            implied_pairs += k * (k - 1) // 2

    # --- supersession -------------------------------------------------
    n_sup_notes = sum(1 for nid in superseded_in if superseded_in[nid] > 0)
    sup_pairs: List[Tuple[str, str, Optional[str], Optional[str]]] = []
    for note in notes:
        for link in _link_iter(note):
            if str(link.get("relation")) == "superseded_by":
                tgt = str(link.get("target_id") or "")
                sup_pairs.append(
                    (
                        note["id"],
                        tgt,
                        link.get("valid_from"),
                        link.get("valid_to"),
                    )
                )

    # Deduplicate symmetric pairs, keep a stable order.
    seen_pairs: set = set()
    uniq_pairs: List[Tuple[str, str, Optional[str], Optional[str]]] = []
    for a, b, vf, vt in sup_pairs:
        key = tuple(sorted((a, b)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        uniq_pairs.append((a, b, vf, vt))

    examples = []
    for a, b, vf, vt in uniq_pairs[: max(0, n_examples)]:
        na, nb = by_id.get(a, {}), by_id.get(b, {})
        a_dt, b_dt = _parse_dt(na.get("t")), _parse_dt(nb.get("t"))
        if a_dt and b_dt:
            older, newer = (na, nb) if a_dt <= b_dt else (nb, na)
        else:
            older, newer = na, nb
        examples.append(
            {
                "note_a": a,
                "note_b": b,
                "note_a_sentence": na.get("c", ""),
                "note_b_sentence": nb.get("c", ""),
                "note_a_date": na.get("session_date") or na.get("t"),
                "note_b_date": nb.get("session_date") or nb.get("t"),
                "link_valid_from": vf,
                "link_valid_to": vt,
                "older_note": older.get("id"),
                "newer_note": newer.get("id"),
                "older_entities": older.get("entities") or [],
                "newer_entities": newer.get("entities") or [],
                "older_keywords": older.get("K") or [],
                "newer_keywords": newer.get("K") or [],
            }
        )

    # --- dates --------------------------------------------------------
    n_ts_iso = sum(1 for n in notes if n.get("timestamp_iso"))
    n_ts_none = len(notes) - n_ts_iso
    n_date_unknown = sum(
        1
        for n in notes
        if not n.get("timestamp_iso") and not _parse_dt(n.get("t"))
    )

    # --- session ids --------------------------------------------------
    sid_present = [n for n in notes if n.get("session_id")]
    kinds = Counter()
    for n in sid_present:
        sid = str(n["session_id"])
        if sid.startswith("answer_"):
            kinds["haystack_answer_id"] += 1
        elif sid.lower().startswith("session_") or (
            len(sid) <= 4 and sid[:1].lower() == "s" and sid[1:].isdigit()
        ):
            kinds["ordinal"] += 1
        else:
            kinds["other"] += 1

    return {
        "n_notes": len(notes),
        "n_link_records": n_links,
        "n_unique_undirected_pairs": len(undirected),
        "relation_histogram": dict(sorted(rel_counts.items(), key=lambda kv: -kv[1])),
        "entity_fanout": share_metric,
        "n_distinct_entities": len(entity_to_notes),
        "implied_same_entity_pairs": implied_pairs,
        "n_notes_with_superseded_by": n_sup_notes,
        "share_notes_with_superseded_by": (n_sup_notes / len(notes)) if notes else 0.0,
        "n_supersession_pairs": len(uniq_pairs),
        "supersession_examples": examples,
        "n_notes_timestamp_iso_present": n_ts_iso,
        "n_notes_timestamp_iso_null": n_ts_none,
        "n_notes_date_unknown": n_date_unknown,
        "session_id_present": len(sid_present),
        "session_id_kinds": dict(kinds),
        "session_id_samples": [n["session_id"] for n in sid_present[:5]],
        "n_notes_entities_empty": sum(
            1 for n in notes if not (n.get("entities") or [])
        ),
    }


def render_markdown(label: str, metrics: Dict[str, Any]) -> str:
    lines = [f"### {label}", ""]
    lines.append(f"- notes: **{metrics['n_notes']}**")
    lines.append(f"- link records: **{metrics['n_link_records']}**")
    lines.append(f"- unique undirected pairs: {metrics['n_unique_undirected_pairs']}")
    hist = ", ".join(f"`{k}`={v}" for k, v in metrics["relation_histogram"].items())
    lines.append(f"- relation histogram: {hist}")
    fo = metrics["entity_fanout"]
    lines.append(
        f"- entity fan-out per note: mean **{fo['mean']:.2f}**, median "
        f"**{fo['median']:.1f}**, p90 {fo['p90']:.0f}, max {fo['max']}, "
        f"zero-fanout share {fo['zero_share']:.3f}"
    )
    lines.append(f"- distinct entities: {metrics['n_distinct_entities']}")
    lines.append(f"- implied same-entity pairs from entities: {metrics['implied_same_entity_pairs']}")
    lines.append(
        f"- notes with >=1 superseded_by: **{metrics['n_notes_with_superseded_by']}** "
        f"({metrics['share_notes_with_superseded_by'] * 100:.2f}%)"
    )
    lines.append(f"- supersession pairs (undirected, deduped): {metrics['n_supersession_pairs']}")
    lines.append(
        f"- date unknown (no timestamp_iso AND unparseable t): "
        f"**{metrics['n_notes_date_unknown']}** (timestamp_iso null on "
        f"{metrics['n_notes_timestamp_iso_null']})"
    )
    lines.append(
        f"- notes with non-null session_id: {metrics['session_id_present']} "
        f"({metrics['session_id_kinds']}) samples={metrics['session_id_samples']}"
    )
    lines.append(f"- notes with empty entities list: {metrics['n_notes_entities_empty']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_banks(target: str) -> List[str]:
    if os.path.isdir(target):
        found = glob.glob(os.path.join(target, "**", "*.sqlite"), recursive=True)
        return sorted(found)
    return [target]


def main() -> int:
    parser = argparse.ArgumentParser(description="ASEM-THG bank graph metrics (read-only)")
    parser.add_argument("--bank", help="Path to a single .sqlite bank")
    parser.add_argument("--dir", help="Directory to scan recursively for .sqlite banks")
    parser.add_argument("--label", default="bank", help="Label for the report")
    parser.add_argument("--examples", type=int, default=5, help="Supersession examples to print")
    parser.add_argument("--json", help="Write full metrics JSON here")
    parser.add_argument("--out", help="Write markdown report here")
    args = parser.parse_args()

    target = args.bank or args.dir
    if not target:
        parser.error("one of --bank or --dir is required")

    banks = _find_banks(target)
    if not banks:
        print(f"no .sqlite banks found under {target}", file=sys.stderr)
        return 1

    report: Dict[str, Any] = {}
    md: List[str] = []
    for path in banks:
        try:
            notes = load_notes(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {path}: {exc}", file=sys.stderr)
            continue
        if not notes:
            continue
        m = compute_metrics(notes, n_examples=args.examples)
        m["bank_path"] = path
        label = f"{args.label} :: {os.path.relpath(path)}"
        report[label] = m
        md.append(render_markdown(label, m))
        md.append("")
        print(f"[{path}] notes={m['n_notes']} links={m['n_link_records']} "
              f"hist={m['relation_histogram']}")

    text = "\n".join(md)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"markdown -> {args.out}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"json -> {args.json}")
    if not args.out and not args.json:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
