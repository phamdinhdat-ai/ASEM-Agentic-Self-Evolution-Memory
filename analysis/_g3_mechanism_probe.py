"""G3 mechanism: is the missing session CROWDED OUT or NOT RETRIEVED?

The oracle (`_g3_oracle_probe.py`) proved the missing evidence is reachable by
text: 16/16. But it reached it with the gold turn's OWN text, which a planner
will never produce. So the oracle bounds the ceiling without saying which
mechanism gets there.

This splits the 16 missing sessions by where they are lost:

  crowded_out   the missing session HAS a note in the candidate pool, but the
                top-8 window filled with other sessions first. A session-aware
                selection (diversification) recovers it for free — no new
                retrieval, no LLM, no graph.
  not_retrieved the missing session has NO note anywhere in the pool. Only a
                different QUERY can find it — this is what a planner (G3) is for.

The two need different fixes, so building the wrong one is wasted effort. This
measures the split before either is written. One base pass per question, no
generation, no API cost.
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
_ap.add_argument("--config", default=CFG)
_ap.add_argument("--bank", default=BANK)
_ap.add_argument("--dataset", default=DATASET)
_ap.add_argument("--cat", type=int, default=1)
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


def sess(ids):
    return {note_sessions[i] for i in ids if i in note_sessions}


# Session-diversified selection: greedily keep the best-ranked note from each
# session not yet represented, then fill remaining slots by rank. This is the
# cheapest possible "spread the window across sessions" rule — no MMR scoring,
# no graph — so if it recovers the crowded-out sessions, the mechanism is
# session coverage, not ranking quality.
def diversify(notes, k):
    seen, out = set(), []
    for n in notes:                      # best note per session, in rank order
        s = note_sessions.get(n.id)
        if s is not None and s not in seen:
            seen.add(s)
            out.append(n)
        if len(out) == k:
            return out
    for n in notes:                      # then fill by rank
        if n not in out:
            out.append(n)
        if len(out) == k:
            break
    return out


buckets = defaultdict(int)   # crowded_out / not_retrieved / already_ok
detail = []
for item in conv0["qa"]:
    if item.get("category") != _args.cat:
        continue
    gold = gold_sessions(item.get("evidence") or [])
    if len(gold) <= 1:
        continue
    q = item["question"]
    retr.ppr_scores = {}
    pool = retr.retrieve(q, M)
    top8 = pool[:K]

    top_sess = sess([n.id for n in top8])
    if gold <= top_sess:
        buckets["already_ok"] += 1
        continue
    missing = gold - top_sess
    pool_sess = sess([n.id for n in pool])
    crowded = missing & pool_sess
    absent = missing - pool_sess
    if crowded:
        buckets["crowded_out"] += 1
    if absent:
        buckets["not_retrieved"] += 1
    buckets["missing_total"] += 1

    div = diversify(pool, K)
    div_sess = sess([n.id for n in div])
    if gold <= div_sess:
        buckets["diversify_fixes"] += 1
    detail.append((q, sorted(missing), sorted(crowded), sorted(absent),
                   gold <= div_sess, len(pool)))

n_multi = buckets["already_ok"] + buckets["missing_total"]
print(f"category={_args.cat}  K={K}  walk=OFF  multi-gold questions={n_multi}")
print(f"  already complete in base top-8 : {buckets['already_ok']}")
print(f"  missing >=1 gold session       : {buckets['missing_total']}")
print(f"    ...CROWDED OUT (in pool, not top-8) : {buckets['crowded_out']}")
print(f"    ...NOT RETRIEVED (absent from pool) : {buckets['not_retrieved']}")
print()
print(f"  session-diversified top-8 fixes the question: "
      f"{buckets['diversify_fixes']} of {buckets['missing_total']}")
print()
print("Reading: 'crowded out' questions are recovered by spreading the window")
print("across sessions — no LLM, no graph. 'not retrieved' questions need a")
print("different QUERY, which is what a planner (G3) provides.")
print()
print(f"  {'question':<58} {'miss':<14} {'crowd':<10} {'absent':<10} div")
for q, miss, crowd, absent, fixed, pl in detail:
    print(f"  {q[:56]:<58} {','.join(miss):<14} {','.join(crowd):<10} "
          f"{','.join(absent):<10} {'Y' if fixed else 'n'}")

try:
    M.close()
except Exception:
    pass
