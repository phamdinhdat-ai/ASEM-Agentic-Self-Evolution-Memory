"""G3 headroom: is there room for multi-hop decomposition to help?

The G2 null (`_g2_recall_probe.py`, `_g2_reserve_probe.py`) says the graph walk
cannot move the answer window. But both were scored on `session_recall`, and the
bank has only 19 sessions of ~54 notes each — the base already sits at 0.9239, so
the metric is near its ceiling and *cannot* show much movement either way.

Before building G3 (multi-hop query decomposition) this measures the metric that
is NOT saturated: `all_sessions`, which requires EVERY gold session to be
represented. Multi-hop questions name two sessions; a decomposition that
retrieves each sub-question directly would admit the second premise's session,
which is exactly what `all_sessions` counts and `session_recall` does not.

Base retrieval only (walk off), one pass, no generation. Fast.
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
_ap.add_argument("--limit", type=int, default=0)
_ap.add_argument("--config", default=CFG)
_ap.add_argument("--bank", default=BANK)
_ap.add_argument("--k", type=int, default=K)
_args = _ap.parse_args()
CFG, BANK, K = _args.config, _args.bank, _args.k


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
qa = ds[0]["qa"]
if _args.limit:
    qa = qa[:_args.limit]
note_sessions = load_note_sessions(BANK)

# category -> [n, hits_any, hits_all, n_missing_any, sum_missing]
agg = defaultdict(lambda: [0, 0, 0, 0, 0])
multi = [0, 0, 0]  # multi-gold-session questions: n, all_present, any_missing

for item in qa:
    q = item.get("question")
    if not q:
        continue
    gold = gold_sessions(item.get("evidence") or [])
    if not gold:
        continue
    cat = item.get("category") or item.get("category_name") or "unknown"

    retr.ppr_scores = {}
    notes = retr.retrieve(q, M)
    used = [n.id for n in notes[:K]]
    got = {note_sessions[nid] for nid in used if nid in note_sessions}

    b = agg[cat]
    b[0] += 1
    if got & gold:
        b[1] += 1
    if gold <= got:
        b[2] += 1
    missing = gold - got
    if missing:
        b[3] += 1
        b[4] += len(missing)

    if len(gold) > 1:
        multi[0] += 1
        if gold <= got:
            multi[1] += 1
        else:
            multi[2] += 1

n = sum(v[0] for v in agg.values())
print(f"bank={BANK}")
print(f"questions={n}  K={K}  walk=OFF (base retrieval only)")
print()
print(f"  {'category':<16} {'n':>4} {'sess_recall':>11} {'all_sess':>9} "
      f"{'miss>=1':>8} {'miss/qs':>8}")
for cat, v in sorted(agg.items()):
    print(f"  {cat:<16} {v[0]:>4} {v[1]/v[0]:>11.4f} {v[2]/v[0]:>9.4f} "
          f"{v[3]/v[0]:>8.4f} {v[4]/v[0]:>8.2f}")
tot_any = sum(v[1] for v in agg.values()) / n
tot_all = sum(v[2] for v in agg.values()) / n
tot_miss = sum(v[3] for v in agg.values()) / n
print(f"  {'OVERALL':<16} {n:>4} {tot_any:>11.4f} {tot_all:>9.4f} {tot_miss:>8.4f}")
print()
print(f"multi-gold-session questions: n={multi[0]}  all_present={multi[1]} "
      f"({multi[1]/max(multi[0],1):.4f})  missing>=1={multi[2]}")
print("  ^ this last number is G3's target: questions where a gold session is")
print("    entirely absent from the answer window. session_recall@8 is saturated")
print(f"    ({tot_any:.4f}); all_sessions ({tot_all:.4f}) is the metric with room.")

try:
    M.close()
except Exception:
    pass
