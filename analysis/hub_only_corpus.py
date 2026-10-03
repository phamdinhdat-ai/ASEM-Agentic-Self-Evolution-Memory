"""Corpus-level hub-only edge fraction across every finished bank in a tag.

Owned by WS2. Runs `hub_only_edges.py` logic over all conversations and reports
per-conversation + aggregate hub_only_edge_fraction and the count of
discriminative same-entity edges (the amount of real entity signal).

Usage:
    python analysis/hub_only_corpus.py --root static/memory_banks/locomo10/thg_v2/ASEM-THG
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.hub_only_edges import _load, _norm  # noqa: E402


def measure(bank: str, hub_df: float, topn: int) -> Dict[str, Any]:
    from collections import Counter

    notes = _load(bank)
    n = len(notes)
    df: Counter = Counter()
    for rec in notes.values():
        for e in _norm(rec["entities"]):
            df[e] += 1
    hubs = {e for e, c in df.items() if c / n >= hub_df}
    for e, _c in df.most_common(topn):
        hubs.add(e)

    seen = set()
    n_same = n_hub_only = n_disc = n_nointer = 0
    for nid, rec in notes.items():
        for link in rec["L"]:
            if isinstance(link, dict):
                rel = str(link.get("relation") or "linked")
                tgt = str(link.get("target_id") or "")
            else:
                rel, tgt = "linked", str(link)
            if rel != "same-entity" or not tgt or tgt not in notes:
                continue
            key = tuple(sorted((nid, tgt)))
            if key in seen:
                continue
            seen.add(key)
            n_same += 1
            inter = _norm(notes[nid]["entities"]) & _norm(notes[tgt]["entities"])
            if not inter:
                n_nointer += 1
            elif inter <= hubs:
                n_hub_only += 1
            else:
                n_disc += 1

    return {
        "n_notes": n,
        "hubs": sorted(hubs),
        "n_same_entity_edges": n_same,
        "n_hub_only": n_hub_only,
        "n_discriminative": n_disc,
        "n_no_intersection": n_nointer,
        "hub_only_edge_fraction": (n_hub_only / n_same) if n_same else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--hub-df", type=float, default=0.5)
    ap.add_argument("--topn", type=int, default=2)
    ap.add_argument("--out", default="analysis/_hub_only_corpus.json")
    args = ap.parse_args()

    banks = sorted(glob.glob(os.path.join(args.root, "*", "asem_thg.sqlite")))
    print(f"banks: {len(banks)}")
    rows: List[Dict[str, Any]] = []
    for bank in banks:
        conv = os.path.basename(os.path.dirname(bank))
        m = measure(bank, args.hub_df, args.topn)
        m["conversation"] = conv
        m["bank"] = bank
        rows.append(m)
        frac = m["hub_only_edge_fraction"]
        print(f"  {conv}: notes={m['n_notes']:4d} same_entity_edges={m['n_same_entity_edges']:5d} "
              f"hub_only={m['n_hub_only']:5d} discriminative={m['n_discriminative']:3d} "
              f"frac={frac if frac is None else round(frac,4)}")

    tot_edges = sum(r["n_same_entity_edges"] for r in rows)
    tot_hub = sum(r["n_hub_only"] for r in rows)
    tot_disc = sum(r["n_discriminative"] for r in rows)
    agg = (tot_hub / tot_edges) if tot_edges else None
    print(f"\n=== AGGREGATE ({len(rows)} conversations) ===")
    print(f"  same-entity edges : {tot_edges}")
    print(f"  hub-only          : {tot_hub}")
    print(f"  discriminative    : {tot_disc}")
    print(f"  hub_only fraction : {agg if agg is None else round(agg,4)}")
    print(f"  discriminative per conversation: "
          f"{round(tot_disc/len(rows),2) if rows else 0}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "root": args.root,
            "n_conversations": len(rows),
            "total_same_entity_edges": tot_edges,
            "total_hub_only": tot_hub,
            "total_discriminative": tot_disc,
            "aggregate_hub_only_edge_fraction": agg,
            "per_conversation": rows,
        }, fh, indent=2)
    print(f"json -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
