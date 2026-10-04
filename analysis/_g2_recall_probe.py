"""G2 falsification: does the TYPED, entity-anchored walk change what the model
actually reads, and does that change improve evidence recall?

The G1 probe settled one thing: at the answer window (top-8) the hydrated graph
changed NOTHING (0.9289 session_recall@8 in both arms, identical per category).
Its candidate-level gain was pure pool size. So the bar for G2 is not "does the
pool grow" but "does the top-8 change, and for the better".

Rungs over the SAME frozen bank, same embedder, same queries:

  dark       use_graph_rank=False    -- no walk
  note-only  entity_seed_weight=0.0  -- G1 behaviour: seeds = base winners,
                                        typed weights but note-only reset
  typed+ent  entity_seed_weight=0.5  -- G2: typed transitions + query-named
                                        entity nodes in the reset distribution
  ent-only   entity_seed_weight=1.0  -- reset entirely on the question's entities

`--set key=value ...` applies retriever overrides to every arm, so the same
probe can separate "the walk carries no signal" from "the walk is under-weighted"
without editing the config.

Reported per rung: session recall@8, plus two attribution diagnostics that say
whether the walk is even CAPABLE of mattering at this k:

  top-8 changed  fraction of queries whose top-8 ordering differs from `dark`
  slots in/out   mean top-8 slots the walk gained and lost against `dark`

If the slot churn is ~0, no recall movement is possible no matter how good the
walk is, and the honest conclusion is that Phase D's weight on the graph term is
the binding constraint — not the walk's quality.

No API cost: retrieval is embedding-driven and generation is never called.
"""
import argparse
import json
import os
import shutil
import tempfile
from collections import defaultdict

import yaml

from asem.backends.langchain_backend import _build_embedder
from asem.backends.openai_backend import OpenAIBackend
from analysis.evidence_recall import gold_sessions, load_note_sessions
from eval.systems import build_asem_thg_system

CFG = "configs/models/thg_v2_postfix_off.yaml"
BANK = "static/memory_banks/locomo10/gpt56_luna/ASEM-THG/locomo_0000/asem_thg.sqlite"
DATASET = "datasets/locomo/locomo10.json"
K = 8

# arm -> retriever overrides. `dark` is the control: no walk at all, so every
# difference is attributable to the walk. `note-only` isolates the TYPED
# transitions (seeds stay base winners). `typed+ent` adds the entity-anchored
# reset. `ent-only` flips entity_seed_weight to 1.0 — the walk starts entirely
# from the question's entities, which is the sharpest test of whether the
# entity graph carries query-relevant signal at all.
ARMS = {
    "dark": {"use_graph_rank": False},
    "note-only": {"use_graph_rank": True, "entity_seed_weight": 0.0},
    "typed+ent": {"use_graph_rank": True, "entity_seed_weight": 0.5},
    "ent-only": {"use_graph_rank": True, "entity_seed_weight": 1.0},
}

_ap = argparse.ArgumentParser()
_ap.add_argument("--limit", type=int, default=0, help="score only the first N questions")
_ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
_ap.add_argument("--config", default=CFG)
_ap.add_argument("--bank", default=BANK)
# Free-form retriever overrides applied to EVERY arm, e.g.
#   --set ppr_weight=0.6 base_weight=0.30
# so the same probe can answer "is the walk inert, or merely under-weighted?".
_ap.add_argument("--set", nargs="*", default=[],
                 help="key=value retriever overrides applied to every arm")
_args = _ap.parse_args()
CFG, BANK = _args.config, _args.bank
ARMS = {k: ARMS[k] for k in _args.arms}
GLOBAL_OVERRIDES = {}
for kv in _args.set:
    k, _, v = kv.partition("=")
    GLOBAL_OVERRIDES[k] = yaml.safe_load(v)


class StubClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kw):
                raise RuntimeError("no generation in this probe")


embedder = _build_embedder(
    {"embedder_provider": "huggingface",
     "embedder_name": "sentence-transformers/all-MiniLM-L6-v2"}
)


def build(arm: str, tmp: str):
    cfg = yaml.safe_load(open(CFG))
    cfg.setdefault("hyper_graph", {})["hydrate_on_load"] = True
    ret = cfg.setdefault("retriever", {})
    ret.update(ARMS[arm])
    ret.update(GLOBAL_OVERRIDES)
    p = os.path.join(tmp, f"cfg_{arm}.yaml")
    yaml.safe_dump(cfg, open(p, "w"))
    d = os.path.join(tmp, f"db_{arm}")
    os.makedirs(d, exist_ok=True)
    shutil.copy2(BANK, os.path.join(d, "asem_thg.sqlite"))
    backend = OpenAIBackend(client=StubClient(), model="stub", embedder=embedder)
    return build_asem_thg_system(p, d, backend=backend)


ds = json.load(open(DATASET))
qa = ds[0]["qa"]
if _args.limit:
    qa = qa[:_args.limit]
tmp = tempfile.mkdtemp()
arms = {}
for name in ARMS:
    arms[name] = build(name, tmp)
    print(f"built arm {name}", flush=True)
note_sessions = load_note_sessions(BANK)

# category -> [n, hits_any, hits_all, used]
agg = {arm: defaultdict(lambda: [0, 0, 0, 0]) for arm in arms}
top8 = {arm: {} for arm in arms}          # arm -> {qidx: tuple(top-8 ids)}
pool = {arm: {} for arm in arms}          # arm -> {qidx: pool size}
# Recall@8 is COARSE here: one session owns many notes, so swapping one note for
# another inside the same session is invisible to it. Session MRR is the sharper
# instrument — it asks whether the walk moved the FIRST gold session earlier in
# the list, which recall@8 cannot see at all.
mrr = {arm: [] for arm in arms}           # arm -> [1/rank of first gold session]

for qi, item in enumerate(qa):
    q = item.get("question")
    if not q:
        continue
    gold = gold_sessions(item.get("evidence") or [])
    if not gold:
        continue
    cat = item.get("category") or item.get("category_name") or "unknown"
    for arm, system in arms.items():
        retr = system.pipeline.retriever
        notes = retr.retrieve(q, system.pipeline.memory_bank)
        used = [n.id for n in notes[:K]]
        top8[arm][qi] = tuple(used)
        pool[arm][qi] = len(notes)
        # Session of each note, in rank order, deduped — the list recall@8 and
        # MRR are both computed over.
        sess_rank = list(dict.fromkeys(
            note_sessions[nid] for nid in used if nid in note_sessions
        ))
        first = next((i for i, s in enumerate(sess_rank, 1) if s in gold), 0)
        mrr[arm].append(1.0 / first if first else 0.0)
        got = set(sess_rank)
        b = agg[arm][cat]
        b[0] += 1
        b[3] += len(used)
        if got & gold:
            b[1] += 1
        if gold <= got:
            b[2] += 1

print(f"bank={BANK}")
print(f"notes in bank={len(note_sessions)}  questions scored="
      f"{sum(v[0] for v in agg['dark'].values())}  K={K}")

base = top8["dark"]
base_pool = pool["dark"]
for arm in ARMS:
    per_cat = agg[arm]
    n = sum(v[0] for v in per_cat.values())
    recall = sum(v[1] for v in per_cat.values()) / n if n else 0.0
    allsess = sum(v[2] for v in per_cat.values()) / n if n else 0.0
    print(f"\n=== {arm} ===")
    print(f"    overall session_recall@{K}={recall:.4f}  all_sessions={allsess:.4f}  "
          f"session_MRR={sum(mrr[arm])/max(len(mrr[arm]),1):.4f}  n={n}")
    for cat, v in sorted(per_cat.items()):
        print(f"    {cat:<16} n={v[0]:>4}  recall={v[1]/v[0]:.4f}  "
              f"all={v[2]/v[0]:.4f}  used={v[3]/v[0]:.1f}")
    if arm != "dark":
        shared = sorted(set(base) & set(top8[arm]))
        # The diagnostic that matters: how many top-8 SLOTS the walk changed.
        # If this is ~0 the walk cannot move recall at this k no matter how
        # good it is, and the binding constraint is Phase D's weight on it.
        new_slots = sum(len(set(top8[arm][k]) - set(base[k])) for k in shared)
        lost_slots = sum(len(set(base[k]) - set(top8[arm][k])) for k in shared)
        reordered = sum(1 for k in shared if base[k] != top8[arm][k])
        mp = sum(pool[arm][k] for k in shared) / max(len(shared), 1)
        mb = sum(base_pool[k] for k in shared) / max(len(shared), 1)
        # Session-level churn: the metric is scored at SESSION granularity, so a
        # note-level slot swap inside one session is invisible to it. These two
        # counts say whether the walk changed which SESSIONS are in the top-8 —
        # the only churn recall@8 can actually see.
        gb, ga = [], []
        for k in shared:
            gb.append({note_sessions[n] for n in base[k] if n in note_sessions})
            ga.append({note_sessions[n] for n in top8[arm][k] if n in note_sessions})
        sess_in = sum(len(a - b) for a, b in zip(ga, gb))
        sess_out = sum(len(b - a) for a, b in zip(ga, gb))
        print(f"    --- attribution ---")
        print(f"    pool size      dark={mb:.1f}  {arm}={mp:.1f}")
        print(f"    top-8 changed  {reordered}/{len(shared)} queries "
              f"({reordered/len(shared):.3f});  "
              f"slots in={new_slots/len(shared):.2f}  out={lost_slots/len(shared):.2f} per query")
        print(f"    SESSIONS in={sess_in/len(shared):.2f}  out={sess_out/len(shared):.2f} per query")
