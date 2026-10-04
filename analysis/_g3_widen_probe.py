"""G3 alternative: widen the pool, then spread the window across sessions.

The reachability probe changed the diagnosis. The missing session is topically
NEAR the question (mean cosine 0.673 vs 0.758 for what was found — a gap of
0.085, and all 16 above 0.30). So the missing fact is about the SAME topic, in a
different session. It is not unguessable; it is being crowded out — by notes
that are marginally more similar, from sessions that are already represented.

If that is right, the fix is not an LLM planner at all. It is:

  1. widen the candidate pool (the missing session is at 0.673 but the pool
     cutoff sits above it), then
  2. spread the answer window across sessions instead of taking the top-8 by
     similarity alone.

Both are free — no LLM call, no graph walk. This measures whether they recover
the 16 incomplete cat-1 questions, against the `off` baseline and the
oracle_fused ceiling from `_g3_planner_probe.py` (0.8148).

One system, retrieval only, no generation, no API cost.
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

_ap = argparse.ArgumentParser()
_ap.add_argument("--cat", type=int, default=1)
_ap.add_argument("--config", default=CFG)
_ap.add_argument("--bank", default=BANK)
_ap.add_argument("--dataset", default=DATASET)
_ap.add_argument("--k1", type=int, default=200)
_ap.add_argument("--k2", type=int, default=40,
                 help="pool RETURNED by the retriever; must exceed the 8-note "
                      "answer window or there is nothing to spread")
_ap.add_argument("--delta", type=float, default=0.0)
_args = _ap.parse_args()
CFG, BANK, DATASET = _args.config, _args.bank, _args.dataset


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

tmp = tempfile.mkdtemp()
cfg = yaml.safe_load(open(CFG))
cfg.setdefault("hyper_graph", {})["hydrate_on_load"] = True
cfg.setdefault("retriever", {})["use_graph_rank"] = False
p = os.path.join(tmp, "cfg.yaml")
yaml.safe_dump(cfg, open(p, "w"))
d = os.path.join(tmp, "db")
os.makedirs(d, exist_ok=True)
shutil.copy2(BANK, os.path.join(d, "asem_thg.sqlite"))
backend = OpenAIBackend(client=StubClient(), model="stub", embedder=embedder)
system = build_asem_thg_system(p, d, backend=backend)
retr = system.pipeline.retriever
M = system.pipeline.memory_bank
retr.use_graph_rank = False

ds = json.load(open(DATASET))
conv0 = ds[0]
note_sessions = load_note_sessions(BANK)


def sess(notes):
    return {note_sessions[n.id] for n in notes[:K] if n.id in note_sessions}


def diversify(notes, k):
    """Best-ranked note per session, then fill by rank."""
    seen, out = set(), []
    for n in notes:
        s = note_sessions.get(n.id)
        if s is not None and s not in seen:
            seen.add(s)
            out.append(n)
        if len(out) == k:
            return out
    for n in notes:
        if n not in out:
            out.append(n)
        if len(out) == k:
            break
    return out


items = []
for item in conv0["qa"]:
    if item.get("category") != _args.cat:
        continue
    gold = gold_sessions(item.get("evidence") or [])
    if len(gold) <= 1:
        continue
    items.append((item, gold))

arms = ["off", "wide", "wide+div"]
res = {a: [0, 0] for a in arms}   # [any, all]
pool_has = 0

for item, gold in items:
    q = item["question"]

    # off: default pool.
    retr.k1, retr.k2, retr.delta = 20, 10, 0.30
    retr.ppr_scores = {}
    base = retr.retrieve(q, M)
    for a in ("off",):
        got = sess(base)
        res[a][0] += 1 if (got & gold) else 0
        res[a][1] += 1 if (gold <= got) else 0

    # wide: a much larger pool, top-8 by the retriever's own order.
    retr.k1, retr.k2, retr.delta = _args.k1, _args.k2, _args.delta
    retr.ppr_scores = {}
    wide = retr.retrieve(q, M)
    got = sess(wide)
    res["wide"][0] += 1 if (got & gold) else 0
    res["wide"][1] += 1 if (gold <= got) else 0

    pool_sess = {note_sessions[n.id] for n in wide if n.id in note_sessions}
    if (gold - sess(base)) & pool_sess:
        pool_has += 1

    # wide+div: same pool, window spread across sessions.
    got = sess(diversify(wide, K))
    res["wide+div"][0] += 1 if (got & gold) else 0
    res["wide+div"][1] += 1 if (gold <= got) else 0

n = len(items)
print(f"category={_args.cat}  K={K}  multi-gold questions={n}")
print(f"  wide pool: k1={_args.k1}  delta={_args.delta}")
print()
print(f"  {'arm':<12} {'sess_recall':>12} {'all_sessions':>13}")
for a in arms:
    print(f"  {a:<12} {res[a][0]/n:>12.4f} {res[a][1]/n:>13.4f}")
print()
print(f"  questions whose missing session IS in the wide pool: {pool_has} of {n}")
print()
print("Compare against _g3_planner_probe.py: off all_sessions=0.4074,")
print("oracle_fused=0.8148. If wide+div approaches the oracle at ZERO LLM cost,")
print("the planner (G3) is unnecessary — the fix was pool width + session spread.")

try:
    M.close()
except Exception:
    pass
