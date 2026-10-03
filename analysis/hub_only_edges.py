"""Hub-only edge fraction: how much of the entity channel is uninformative?

Owned by WS2. For a bank, counts the STORED `same-entity` edges whose two notes
share at least one entity but whose entity intersection contains nothing besides
a set of "hub" entities (the highest-df entities in the bank).

An edge like that carries zero discriminative information: it says only "both
notes are about the conversation's speakers", which is true of most of the bank.

Usage:
    python analysis/hub_only_edges.py --bank <path> [--hub-df 0.5] [--topn 2]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from typing import Any, Dict, List, Set, Tuple


def _load(path: str) -> Dict[str, Dict[str, Any]]:
    uri = "file:" + os.path.abspath(path).replace("?", "%3f").replace("#", "%23")
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = list(conn.execute("SELECT id, entities, L FROM notes"))
    finally:
        conn.close()
    out: Dict[str, Dict[str, Any]] = {}
    for nid, ents_raw, links_raw in rows:
        for src, dst in ((ents_raw, "entities"), (links_raw, "L")):
            pass
        def _j(v):
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    return []
            return v or []
        out[str(nid)] = {"entities": _j(ents_raw), "L": _j(links_raw)}
    return out


def _norm(seq) -> Set[str]:
    return {str(e).strip().lower() for e in (seq or []) if str(e).strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--hub-df", type=float, default=0.5,
                    help="An entity is a HUB if its document frequency >= this fraction of notes.")
    ap.add_argument("--topn", type=int, default=2,
                    help="Also treat the top-N most frequent entities as hubs.")
    ap.add_argument("--json")
    args = ap.parse_args()

    notes = _load(args.bank)
    n = len(notes)
    df: Counter = Counter()
    for rec in notes.values():
        for e in _norm(rec["entities"]):
            df[e] += 1

    hubs = {e for e, c in df.items() if c / n >= args.hub_df}
    for e, _c in df.most_common(args.topn):
        hubs.add(e)

    print(f"bank   : {args.bank}")
    print(f"notes  : {n}")
    print(f"hubs   : {sorted(hubs)}  (df>= {args.hub_df:.0%} or top-{args.topn})")
    print("top-10 df: " + ", ".join(f"{e}={c}({c/n:.1%})" for e, c in df.most_common(10)))

    # Count stored same-entity edges (unique undirected) and classify each.
    seen: Set[Tuple[str, str]] = set()
    n_same = 0
    n_hub_only = 0
    n_discriminative = 0
    n_no_intersection = 0
    hub_only_examples: List[Dict[str, Any]] = []
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
                n_no_intersection += 1
            elif inter <= hubs:
                n_hub_only += 1
                if len(hub_only_examples) < 5:
                    hub_only_examples.append({
                        "a": key[0], "b": key[1],
                        "shared": sorted(inter),
                        "a_entities": sorted(_norm(notes[key[0]]["entities"])),
                        "b_entities": sorted(_norm(notes[key[1]]["entities"])),
                    })
            else:
                n_discriminative += 1

    frac = (n_hub_only / n_same) if n_same else 0.0
    print(f"\nstored same-entity edges (unique undirected): {n_same}")
    print(f"  hub-only (intersection subset of hubs)     : {n_hub_only}  ({frac:.1%})")
    print(f"  discriminative (shares a non-hub entity)   : {n_discriminative}  "
          f"({(n_discriminative/n_same if n_same else 0):.1%})")
    print(f"  no entity intersection at all              : {n_no_intersection}")
    print(f"\nhub_only_edge_fraction = {frac:.4f}")
    for ex in hub_only_examples:
        print(f"  e.g. shared={ex['shared']}  A={ex['a_entities']}  B={ex['b_entities']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({
                "bank": args.bank, "n_notes": n, "hubs": sorted(hubs),
                "hub_df_threshold": args.hub_df, "topn": args.topn,
                "n_same_entity_edges": n_same,
                "n_hub_only": n_hub_only,
                "n_discriminative": n_discriminative,
                "n_no_intersection": n_no_intersection,
                "hub_only_edge_fraction": frac,
                "top10_df": [{"entity": e, "df": c, "df_over_n": c / n}
                             for e, c in df.most_common(10)],
                "examples": hub_only_examples,
            }, fh, indent=2)
        print(f"json -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
