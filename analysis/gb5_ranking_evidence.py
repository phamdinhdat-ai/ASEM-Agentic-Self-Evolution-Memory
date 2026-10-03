"""ITEM 1 - GB5 frozen-fixture RANKING evidence (before vs after).

WHY RANKING AND NOT AGGREGATE: GB5 shipped with choice (b) - fan-out is
unchanged and only the ORDERING changes (hub-only pairs remain weak peers;
rare-entity peers outrank them). So the aggregate edge count CANNOT show the fix,
and an aggregate-only report would look like a no-op. The valid evidence is
per-note peer order with printed scores.

BEFORE = peer score is the bare cosine (the pre-GB5 ranking).
AFTER  = peer score is (1 + max(idf of shared keys)) * cosine, ranked by strength
         with an id tie-break.

Acceptance is BOTH directions:
  * discriminative peers must INCREASE and keep existing (13 -> 27 stored edges);
  * hub-only peers must be DEMOTED (not merely deleted - deleting them would be a
    regression, because the cap would then supply no signal at all).

Run: PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE python analysis/gb5_ranking_evidence.py
The frozen bank is copied to a temp path first; it is never written in place.
"""

from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from pathlib import Path

from asem.hyper_graph import TemporalHyperGraph
from asem.memory_bank import MemoryBank

REPO = Path(__file__).resolve().parent.parent
FROZEN = REPO / "static/memory_banks/locomo10/thg_v2/ASEM-THG/locomo_0000/asem_thg.sqlite"
HUBS = {"caroline", "melanie"}


def _cosine_only_ranking(graph, note_id, peers, keys):
    """BEFORE: rank by bare cosine."""
    ei = graph._entity_index
    out = []
    for pid, _ in peers:
        sim = graph._cosine(
            graph._fact_nodes[note_id].e, graph._fact_nodes[pid].e
        )
        out.append((pid, sim, keys & ei.get(pid, set())))
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


def _idf_ranking(graph, note_id, keys):
    """AFTER: the shipped `_entity_peers` ranking."""
    ei = graph._entity_index
    return [
        (pid, score, keys & ei.get(pid, set()))
        for pid, score in graph._entity_peers(graph._fact_nodes[note_id], keys)
    ]


def _label(shared):
    if not shared:
        return "none"
    if shared <= HUBS:
        return "hub-only"
    if shared & HUBS:
        return "mixed"
    return "RARE"


def main() -> None:
    tmp = Path(tempfile.mkdtemp()) / "bank.sqlite"
    shutil.copy2(FROZEN, tmp)
    bank = MemoryBank(str(tmp))
    notes = bank.list_notes()
    bank.close()

    graph = TemporalHyperGraph(
        semantic_tau=2.0, max_entity_links=5, max_semantic_links=0,
        enable_supersession=False,
    )
    for n in notes:
        graph.add_note(n)

    df = Counter()
    for n in notes:
        for key in graph._entity_index[n.id]:
            df[key] += 1
    N = len(notes)

    print("=" * 78)
    print("GB5 IDF SEPARATION (frozen locomo_0000, N=%d)" % N)
    print("=" * 78)
    for key in ("caroline", "melanie", "oliver", "sweden", "grand canyon", "sara bareilles"):
        w = graph._entity_weight(key)
        print(f"  {key:16s} df={df.get(key, 0):4d}  idf={w:.3f}")
    weights = [graph._entity_weight(k) for k in df]
    print(f"  -> max/min separation = {max(weights) / max(min(weights), 1e-9):.1f}x")

    print()
    print("=" * 78)
    print("RANKED PEERS, BEFORE (cosine) vs AFTER (idf-weighted)")
    print("=" * 78)
    chosen = []
    for key in ("oliver", "sweden", "grand canyon"):
        for nid in sorted(graph._entity_to_notes.get(key, ())):
            if nid not in chosen:
                chosen.append(nid)
                break

    for nid in chosen:
        keys = graph._entity_index[nid]
        rare = [k for k in keys if df.get(k, 0) <= 3]
        print()
        print(f"note {nid[:8]} carries rare {rare} (df="
              f"{[df[k] for k in rare]}); keys={sorted(keys)}")
        before = _cosine_only_ranking(
            graph, nid, graph._entity_peers(graph._fact_nodes[nid], keys), keys
        )
        after = _idf_ranking(graph, nid, keys)
        for title, rows in (("BEFORE (cosine only)", before), ("AFTER (idf strength)", after)):
            print(f"  {title}:")
            for i, (pid, score, shared) in enumerate(rows[:5], 1):
                print(f"    {i}. {pid[:8]} score={score:.4f} shared={sorted(shared)} "
                      f"[{_label(shared)}]")

    print()
    print("=" * 78)
    print("AGGREGATE (both directions)")
    print("=" * 78)
    hub_only = disc = top5_hub = top5_disc = 0
    notes_with_rare_top5 = 0
    for nid, note in graph._fact_nodes.items():
        keys = graph._entity_index[nid]
        after = _idf_ranking(graph, nid, keys)[:5]
        before = _cosine_only_ranking(
            graph, nid, graph._entity_peers(note, keys), keys
        )[:5]
        for _, _, shared in after:
            if shared and shared <= HUBS:
                hub_only += 1
            else:
                disc += 1
        for _, _, shared in before:
            if shared and shared <= HUBS:
                top5_hub += 1
            else:
                top5_disc += 1
        if any(s and not (s <= HUBS) for _, _, s in after):
            notes_with_rare_top5 += 1

    print(f"  top-5 slots BEFORE: hub-only={top5_hub} discriminative={top5_disc}")
    print(f"  top-5 slots AFTER : hub-only={hub_only} discriminative={disc}")
    print(f"  notes whose top-5 contains a rare peer: {notes_with_rare_top5}/{N}")


if __name__ == "__main__":
    main()
