"""Reproduce the entity-set-pair denominator and reconcile two counting rules.

Two defensible denominators exist for "implied same-entity pairs" and they are
NOT the same number:

  (A) unordered pairs that actually share >= 1 entity  -> C(k,2) summed with
      overlap removed by set union (the count we report as "implied pairs").
  (B) sum over entities of C(df_e, 2)  -> counts a pair once PER SHARED ENTITY,
      so a pair sharing 2 entities contributes 2.

The lead measured 92,114 on locomo_0000; our union count is 46,084. This script
prints both, plus the naive sum, so the difference is attributable.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.hub_only_edges import _load, _norm  # noqa: E402


def main() -> int:
    p = sys.argv[1] if len(sys.argv) > 1 else (
        "static/memory_banks/locomo10/thg_v2/ASEM-THG/locomo_0000/asem_thg.sqlite"
    )
    notes = _load(p)
    n = len(notes)
    df: Counter = Counter()
    for r in notes.values():
        for e in _norm(r["entities"]):
            df[e] += 1

    hubs = {e for e, c in df.items() if c / n >= 0.5}
    for e, _c in df.most_common(2):
        hubs.add(e)

    ids = list(notes)
    union_pairs = set()
    per_entity_sum = 0
    for e, c in df.items():
        if c > 1:
            per_entity_sum += c * (c - 1) // 2
    for i, a in enumerate(ids):
        ea = _norm(notes[a]["entities"])
        if not ea:
            continue
        for b in ids[i + 1:]:
            if ea & _norm(notes[b]["entities"]):
                union_pairs.add((a, b))

    hub_only_u = disc_u = 0
    for a, b in union_pairs:
        inter = _norm(notes[a]["entities"]) & _norm(notes[b]["entities"])
        if inter <= hubs:
            hub_only_u += 1
        else:
            disc_u += 1

    # (B) per-entity counting, classified by whether the shared entity is a hub.
    hub_pairs_per_entity = 0
    disc_pairs_per_entity = 0
    for e, c in df.items():
        contrib = c * (c - 1) // 2
        if e in hubs:
            hub_pairs_per_entity += contrib
        else:
            disc_pairs_per_entity += contrib

    print(f"bank: {p}")
    print(f"notes: {n}   distinct entities: {len(df)}")
    print(f"hubs: {sorted(hubs)}")
    print()
    print("(A) UNION denominator — pairs sharing >=1 entity, deduped:")
    print(f"    implied distinct pairs : {len(union_pairs)}")
    print(f"    hub-only               : {hub_only_u}")
    print(f"    discriminative         : {disc_u}")
    print(f"    hub fraction           : {hub_only_u/len(union_pairs):.6f}")
    print()
    print("(B) PER-ENTITY-SUM denominator — sum_e C(df_e,2), pairs counted per shared entity:")
    print(f"    total                  : {per_entity_sum}")
    print(f"    hub-entity contribution: {hub_pairs_per_entity}")
    print(f"    non-hub contribution   : {disc_pairs_per_entity}")
    print(f"    hub fraction           : {hub_pairs_per_entity/per_entity_sum:.6f}")
    print()
    print("NOTE: (B) > (A) because a pair sharing caroline AND melanie counts twice")
    print("      in (B) but once in (A). Both are correct for their own question.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
