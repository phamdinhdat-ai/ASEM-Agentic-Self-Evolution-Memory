"""Is there headroom for a RESERVED-SLOT fusion, or is the walk simply empty?

Run 1 (`_g2_recall_probe.py`) settled that the additive Phase D term cannot move
the answer window: a walk-added note enters the pool at the bottom, is scored
-1.67 z-units on rank and negative on similarity (it is a note dense retrieval
MISSED — that is what "added" means), and a 0.052-weight PPR term cannot close
that. The measured effect was +0.0101 overall = exactly one adversarial question,
with session MRR going DOWN.

The plan's principled fix is HippoRAG 2's "recall then precision" split: let the
walk SELECT part of the context rather than compete as a small additive weight.
But that is only worth implementing if the walk has anything to select. This
probe measures the CEILING before any production code is written:

  For each query, reserve R context slots for the walk's top-R additions and ask
  how many gold SESSIONS that newly covers, against the no-walk base.

If reserving R=1..3 slots buys no gold sessions, the walk carries no recoverable
evidence at this k and the honest move is to drop G2's ranking claim rather than
re-tune it. If it does buy sessions, the reserved-slot fusion is justified.

The gain must be attributable to the walk's RANKING, not merely to admitting
different notes. So every reserve arm is run against a RANDOM control: the same
candidate set (the walk's additions), reserved in random order. If random
reservation scores as well as PPR-ordered reservation, the walk's ordering is
not doing the work and the result is not evidence for the graph.

Cheap and light: ONE system, two retrieve() calls per query (walk off, walk on),
no generation, no API cost, no second bank copy.
"""
import argparse
import json
import os
import random
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
RESERVES = [0, 1, 2, 3]

_ap = argparse.ArgumentParser()
_ap.add_argument("--limit", type=int, default=0)
_ap.add_argument("--config", default=CFG)
_ap.add_argument("--bank", default=BANK)
_ap.add_argument("--entity-seed-weight", type=float, default=0.5)
_args = _ap.parse_args()
CFG, BANK = _args.config, _args.bank


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
ret = cfg.setdefault("retriever", {})
ret["use_graph_rank"] = True
ret["entity_seed_weight"] = _args.entity_seed_weight
p = os.path.join(tmp, "cfg.yaml")
yaml.safe_dump(cfg, open(p, "w"))
d = os.path.join(tmp, "db")
os.makedirs(d, exist_ok=True)
shutil.copy2(BANK, os.path.join(d, "asem_thg.sqlite"))
backend = OpenAIBackend(client=StubClient(), model="stub", embedder=embedder)
system = build_asem_thg_system(p, d, backend=backend)
retr = system.pipeline.retriever
M = system.pipeline.memory_bank

ds = json.load(open(DATASET))
qa = ds[0]["qa"]
if _args.limit:
    qa = qa[:_args.limit]
note_sessions = load_note_sessions(BANK)

# category -> n; reserve R -> [hits_any, hits_all]
agg = defaultdict(int)
res = {r: defaultdict(lambda: [0, 0]) for r in RESERVES}
# Controls. `ord` reserves the SAME candidate set the walk picked, in random
# order (isolates the PPR ordering). `set` reserves random notes from the whole
# below-K pool (isolates the candidate set as well). If either matches the walk,
# the walk's ranking is not what moved recall.
ctl_ord = {r: defaultdict(lambda: [0, 0]) for r in RESERVES}
ctl_set = {r: defaultdict(lambda: [0, 0]) for r in RESERVES}
# Diagnostics that say WHY, if the ceiling is low:
new_sess_total = 0        # walk additions whose session is NOT in the base top-K
add_total = 0             # walk additions seen (below-K pool entries the walk ranked)
add_in_top8 = 0           # of those, how many the additive Phase D already promoted


def sess_of(ids):
    return [note_sessions[i] for i in ids if i in note_sessions]


for item in qa:
    q = item.get("question")
    if not q:
        continue
    gold = gold_sessions(item.get("evidence") or [])
    if not gold:
        continue
    cat = item.get("category") or item.get("category_name") or "unknown"

    # Dark pass: walk off, and CLEAR the stale ppr_scores or Phase D reads the
    # previous query's ranking (retrieve() only refreshes them when the walk runs).
    retr.use_graph_rank = False
    retr.ppr_scores = {}
    base = retr.retrieve(q, M)
    base_ids = [n.id for n in base[:K]]
    base_pool = {n.id for n in base}

    # Walk pass.
    retr.use_graph_rank = True
    full = retr.retrieve(q, M)
    full_ids = [n.id for n in full[:K]]
    ppr = dict(retr.ppr_scores)

    # Walk additions: pool entries the base pass never returned, ranked by PPR.
    additions = [n.id for n in full if n.id not in base_pool]
    additions.sort(key=lambda nid: ppr.get(nid, 0.0), reverse=True)
    add_total += len(additions)
    add_in_top8 += sum(1 for nid in additions if nid in set(full_ids))

    # The below-K pool in BASE order — the candidates a random reservation draws
    # from, so `ctl_set` perturbs membership without the walk's ranking at all.
    below_k = [n.id for n in base[K:]]

    base_sess = set(sess_of(base_ids))
    for nid in additions:
        s = note_sessions.get(nid)
        if s and s not in base_sess:
            new_sess_total += 1

    agg[cat] += 1
    rng = random.Random(len(agg) * 1000003 + sum(ord(c) for c in q))
    for r in RESERVES:
        # Reserve r slots: the walk's top-r additions displace the base's last r.
        keep = base_ids[: max(K - r, 0)]

        take = [nid for nid in additions[:r] if nid not in set(keep)]
        got = set(sess_of(keep + take))
        res[r][cat][0] += 1 if (got & gold) else 0
        res[r][cat][1] += 1 if (gold <= got) else 0

        # Control 1: same candidate set, random order.
        shuffled = list(additions)
        rng.shuffle(shuffled)
        take_o = [nid for nid in shuffled[:r] if nid not in set(keep)]
        got_o = set(sess_of(keep + take_o))
        ctl_ord[r][cat][0] += 1 if (got_o & gold) else 0
        ctl_ord[r][cat][1] += 1 if (gold <= got_o) else 0

        # Control 2: random members of the below-K pool (not the walk's set).
        pool_shuf = list(below_k)
        rng.shuffle(pool_shuf)
        take_s = [nid for nid in pool_shuf[:r] if nid not in set(keep)]
        got_s = set(sess_of(keep + take_s))
        ctl_set[r][cat][0] += 1 if (got_s & gold) else 0
        ctl_set[r][cat][1] += 1 if (gold <= got_s) else 0

n = sum(agg.values())
print(f"bank={BANK}")
print(f"questions={n}  K={K}  entity_seed_weight={_args.entity_seed_weight}")
print(f"walk additions/query (below-K pool entries the walk ranked): "
      f"{add_total/max(n,1):.2f}")
print(f"  ...of which additive Phase D ALREADY put in the top-8: "
      f"{add_in_top8/max(n,1):.2f}/query")
print(f"  ...whose session is NOT already in the base top-8: "
      f"{new_sess_total/max(n,1):.2f}/query   <-- the recoverable headroom")
print()
print("session_recall@8 by reserved-slot count (walk-ordered vs two controls):")
print(f"  {'reserve':<8}  {'walk':<8}  {'rand-order':<11}  {'rand-pool':<10}")
for r in RESERVES:
    w = sum(v[0] for v in res[r].values()) / n if n else 0.0
    o = sum(v[0] for v in ctl_ord[r].values()) / n if n else 0.0
    s = sum(v[0] for v in ctl_set[r].values()) / n if n else 0.0
    print(f"  {r:<8}  {w:.4f}    {o:.4f}       {s:.4f}")
print()
print("all_sessions@8:")
print(f"  {'reserve':<8}  {'walk':<8}  {'rand-order':<11}  {'rand-pool':<10}")
for r in RESERVES:
    w = sum(v[1] for v in res[r].values()) / n if n else 0.0
    o = sum(v[1] for v in ctl_ord[r].values()) / n if n else 0.0
    s = sum(v[1] for v in ctl_set[r].values()) / n if n else 0.0
    print(f"  {r:<8}  {w:.4f}    {o:.4f}       {s:.4f}")
print()
print("per-category session_recall@K (walk arms):")
cats = sorted(agg)
print(f"  {'category':<18}" + "".join(f"  R={r:<7}" for r in RESERVES) + "     n")
for c in cats:
    row = f"  {c:<18}"
    for r in RESERVES:
        row += f"  {res[r][c][0]/agg[c]:.4f}   "
    row += f"  {agg[c]:>4}"
    print(row)

try:
    M.close()
except Exception:
    pass
