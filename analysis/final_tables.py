"""Final per-conversation table for a bank tag: notes/edges/metrics/elapsed.

Owned by WS2. Joins the manifest (notes, link_edges, elapsed_sec) with the
computed graph metrics + hub-only fraction + supersession split, for every
conversation in a tag. Read-only.

The manifest also carries the LLM cost of each build (``llm_calls`` /
``llm_tokens``), which is the other half of the efficiency story: ASEM-THG's
linking is LLM-free, so its call count should track sessions, not edges. Those
counts are reported per conversation and per system alongside the graph metrics,
so the "cheap graph" claim is a measured number rather than an architectural
assertion.

Usage:
    python analysis/final_tables.py --tag-root static/memory_banks/locomo10/thg_v2 \
        --out analysis/_thg_v2_final_tables.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.graph_metrics import compute_metrics, load_notes  # noqa: E402
from analysis.hub_only_corpus import measure as hub_measure  # noqa: E402


def supersession_split(bank: str) -> Dict[str, Any]:
    notes = load_notes(bank)
    by_id = {n["id"]: n for n in notes}
    seen = set()
    pairs = []
    for nid, rec in by_id.items():
        for link in rec.get("L") or []:
            if isinstance(link, dict):
                rel = str(link.get("relation") or "")
                tgt = str(link.get("target_id") or "")
            else:
                rel, tgt = "", ""
            if rel != "superseded_by" or not tgt or tgt not in by_id:
                continue
            key = tuple(sorted((nid, tgt)))
            if key in seen:
                continue
            seen.add(key)
            a, b = by_id[key[0]], by_id[key[1]]
            same = bool(a.get("session_id")) and a.get("session_id") == b.get("session_id")
            pairs.append({"a": key[0], "b": key[1], "same_session": same,
                          "a_session": a.get("session_id"), "b_session": b.get("session_id"),
                          "a_date": a.get("session_date"), "b_date": b.get("session_date"),
                          "a_sentence": a.get("c"), "b_sentence": b.get("c")})
    n_same = sum(1 for p in pairs if p["same_session"])
    return {"n_pairs": len(pairs), "n_same_session": n_same,
            "n_cross_session": len(pairs) - n_same, "pairs": pairs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hub-df", type=float, default=0.5)
    ap.add_argument("--topn", type=int, default=2)
    ap.add_argument("--examples", type=int, default=0)
    args = ap.parse_args()

    manifest_path = os.path.join(args.tag_root, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    # Graph-build cost per system, summed over the tag. Read from the manifest
    # totals when present (a build run computes them); otherwise summed here so
    # older manifests still yield a number instead of nothing.
    build_cost: Dict[str, Dict[str, Any]] = {}
    for conv in manifest["conversations"]:
        for name, state in (conv.get("systems") or {}).items():
            slot = build_cost.setdefault(name, {"llm_calls": 0, "llm_tokens": 0, "elapsed_sec": 0.0})
            slot["llm_calls"] += int(state.get("llm_calls") or 0)
            slot["llm_tokens"] += int(state.get("llm_tokens") or 0)
            slot["elapsed_sec"] += float(state.get("elapsed_sec") or 0.0)

    rows: List[Dict[str, Any]] = []
    for conv in manifest["conversations"]:
        cid = conv["conversation_id"]
        sysinfo = conv["systems"].get("ASEM-THG", {})
        bank = sysinfo.get("bank")
        if not bank or not os.path.exists(bank):
            print(f"  !! {cid}: bank missing")
            continue
        notes = load_notes(bank)
        m = compute_metrics(notes, n_examples=args.examples)
        hub = hub_measure(bank, args.hub_df, args.topn)
        sup = supersession_split(bank)
        rows.append({
            "conversation": cid,
            "sessions": conv.get("sessions"),
            "turns": conv.get("turns"),
            "notes": m["n_notes"],
            "link_records": m["n_link_records"],
            "unique_undirected_pairs": m["n_unique_undirected_pairs"],
            "records_per_note": round(m["n_link_records"] / m["n_notes"], 4) if m["n_notes"] else None,
            "pairs_per_note": round(m["n_unique_undirected_pairs"] / m["n_notes"], 4) if m["n_notes"] else None,
            "edge_histogram": m["relation_histogram"],
            "fanout_mean": m["entity_fanout"]["mean"],
            "fanout_median": m["entity_fanout"]["median"],
            "fanout_p90": m["entity_fanout"]["p90"],
            "fanout_max": m["entity_fanout"]["max"],
            "fanout_zero_share": m["entity_fanout"]["zero_share"],
            "n_distinct_entities": m["n_distinct_entities"],
            "n_notes_empty_entities": m["n_notes_entities_empty"],
            "implied_pairs_per_entity_sum": m["implied_same_entity_pairs"],
            "hub": hub,
            "supersession": {k: v for k, v in sup.items() if k != "pairs"},
            "supersession_pairs": sup["pairs"],
            "n_notes_date_unknown": m["n_notes_date_unknown"],
            "session_ids": m["session_id_samples"],
            "elapsed_sec": sysinfo.get("elapsed_sec"),
            # LLM cost of building THIS conversation's ASEM-THG bank. None on a
            # manifest written before the counter existed.
            "llm_calls": sysinfo.get("llm_calls"),
            "llm_tokens": sysinfo.get("llm_tokens"),
            "bank": bank,
            "bank_bytes": sysinfo.get("bytes"),
        })

    tot_notes = sum(r["notes"] for r in rows)
    tot_records = sum(r["link_records"] for r in rows)
    tot_pairs = sum(r["unique_undirected_pairs"] for r in rows)
    tot_hub_edges = sum(r["hub"]["n_same_entity_edges"] for r in rows)
    tot_hub_only = sum(r["hub"]["n_hub_only"] for r in rows)
    tot_disc = sum(r["hub"]["n_discriminative"] for r in rows)
    tot_sup = sum(r["supersession"]["n_pairs"] for r in rows)
    tot_sup_same = sum(r["supersession"]["n_same_session"] for r in rows)

    print(f"{'conversation':<14}{'notes':>6}{'records':>9}{'pairs':>7}{'rec/n':>7}"
          f"{'prs/n':>7}{'fan_med':>9}{'fan_p90':>9}{'disc':>6}{'sup':>5}{'elapsed':>9}")
    for r in rows:
        print(f"{r['conversation']:<14}{r['notes']:>6}{r['link_records']:>9}"
              f"{r['unique_undirected_pairs']:>7}{r['records_per_note']:>7}"
              f"{r['pairs_per_note']:>7}{r['fanout_median']:>9.0f}{r['fanout_p90']:>9.0f}"
              f"{r['hub']['n_discriminative']:>6}{r['supersession']['n_pairs']:>5}"
              f"{r['elapsed_sec']:>9.1f}")
    print(f"\nTOTALS: conversations={len(rows)} notes={tot_notes} records={tot_records} "
          f"unique_pairs={tot_pairs}")
    print(f"  same-entity edges={tot_hub_edges} hub_only={tot_hub_only} "
          f"discriminative={tot_disc} frac={tot_hub_only/tot_hub_edges if tot_hub_edges else 0:.4f}")
    print(f"  supersession pairs={tot_sup} (same-session={tot_sup_same}, "
          f"cross-session={tot_sup - tot_sup_same})")
    print(f"  total elapsed={sum(r['elapsed_sec'] or 0 for r in rows):.1f}s "
          f"(manifest elapsed_sec={manifest.get('elapsed_sec')})")

    # Graph-build cost: the "cheap graph" claim, per system. A system whose
    # linking is LLM-free (ASEM-THG) shows calls ~= sessions; one that spends an
    # LLM call per link shows calls scaling with the edge count.
    if any(v["llm_calls"] for v in build_cost.values()):
        print(f"\n  {'system':<18}{'llm_calls':>11}{'llm_tokens':>12}{'calls/conv':>12}"
              f"{'elapsed_s':>11}")
        n_conv = max(1, len(rows))
        for name, cost in sorted(build_cost.items()):
            print(f"  {name:<18}{cost['llm_calls']:>11}{cost['llm_tokens']:>12}"
                  f"{cost['llm_calls'] / n_conv:>12.1f}{cost['elapsed_sec']:>11.1f}")

    payload = {
        "tag_root": args.tag_root,
        "manifest_elapsed_sec": manifest.get("elapsed_sec"),
        "manifest_run": manifest.get("run"),
        "manifest_totals": manifest.get("totals"),
        "input_sha256": manifest.get("input_sha256"),
        "config_sha256": manifest.get("config_sha256"),
        "model": manifest.get("model"),
        "embedder_name": manifest.get("embedder_name"),
        "created_at": manifest.get("created_at"),
        "updated_at": manifest.get("updated_at"),
        "totals": {
            "conversations": len(rows), "notes": tot_notes,
            "link_records": tot_records, "unique_undirected_pairs": tot_pairs,
            "same_entity_edges": tot_hub_edges, "hub_only": tot_hub_only,
            "discriminative": tot_disc,
            "hub_only_fraction": (tot_hub_only / tot_hub_edges) if tot_hub_edges else None,
            "supersession_pairs": tot_sup,
            "supersession_same_session": tot_sup_same,
            "supersession_cross_session": tot_sup - tot_sup_same,
            "elapsed_sec_sum": sum(r["elapsed_sec"] or 0 for r in rows),
        },
        "build_cost": build_cost,
        "per_conversation": rows,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"json -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
