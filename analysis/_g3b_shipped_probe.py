"""G3b: verify the SHIPPED mechanism, not a probe-local copy.

`_g3_widen_probe.py` and `_g3_div_probe.py` reimplemented the session-spread
rule inside the probe (`diversify()`). That proves the IDEA; it does not prove
the code that will actually run. This builds the system from config ONCE and
flips `retriever.session_diversity` / `k1` / `k2` / `delta` on the live
retriever, so the arm that scores is the REAL `EnhancedHybridRetriever.retrieve`
with the shipped `_diversify_sessions`.

Arms:
  off        session_diversity=false, default pool (k1=20, k2=10)
  div_only   session_diversity=true,  default pool          <- the missing arm:
             does diversity help WITHOUT widening? If yes, the wide pool (and
             its extra work) is unnecessary.
  shipped    session_diversity=true,  wide pool (k1/k2 from --k1/--k2)

One system, mutated per arm — 1/3 the memory of three builds on the 7.9GB box.
Every category, so the single-hop regression check comes from the same pass.

Retrieval only, no generation, no API cost.
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
_ap.add_argument("--k1", type=int, default=200)
_ap.add_argument("--k2", type=int, default=40)
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
# Start with the shipped default; the arm flips it on the live instance below.
cfg.setdefault("retriever", {})["session_diversity"] = False
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

# (session_diversity, k1, k2, delta)
ARMS = {
    "off": (False, 20, 10, 0.30),
    "div_only": (True, 20, 10, 0.30),
    "shipped": (True, _args.k1, _args.k2, _args.delta),
}
ARM_ORDER = ["off", "div_only", "shipped"]

ds = json.load(open(DATASET))
conv0 = ds[0]
note_sessions = load_note_sessions(BANK)


def sess(notes):
    return {note_sessions[n.id] for n in notes[:K] if n.id in note_sessions}


by_cat = defaultdict(lambda: {a: [0, 0, 0] for a in ARM_ORDER})

for item in conv0["qa"]:
    cat = item.get("category")
    if cat is None:
        continue
    gold = gold_sessions(item.get("evidence") or [])
    if not gold:
        continue
    q = item["question"]

    for a in ARM_ORDER:
        div, k1, k2, delta = ARMS[a]
        retr.session_diversity = div
        retr.k1, retr.k2, retr.delta = k1, k2, delta
        retr.ppr_scores = {}
        got = sess(retr.retrieve(q, M))
        by_cat[cat][a][0] += 1 if (got & gold) else 0
        by_cat[cat][a][1] += 1 if (gold <= got) else 0
        by_cat[cat][a][2] += 1

print(f"SHIPPED path check  K={K}  shipped pool k1={_args.k1} k2={_args.k2} delta={_args.delta}")
print()
hdr = "  ".join(f"{a:>17}" for a in ARM_ORDER)
print(f"  {'cat':>4} {'n':>5}  {hdr}   {'d_shipped':>9}")
tot = {a: [0, 0, 0] for a in ARM_ORDER}
for cat in sorted(by_cat):
    cells = []
    for a in ARM_ORDER:
        v = by_cat[cat][a]
        cells.append(f"{v[0]/v[2]:>7.3f}/{v[1]/v[2]:<9.3f}")
        for i in range(3):
            tot[a][i] += v[i]
    d = (by_cat[cat]["shipped"][1] - by_cat[cat]["off"][1]) / by_cat[cat]["off"][2]
    n = by_cat[cat]["off"][2]
    print(f"  {cat:>4} {n:>5}  " + "  ".join(cells) + f"   {d:>+9.3f}")
print()
cells = []
for a in ARM_ORDER:
    v = tot[a]
    cells.append(f"{v[0]/v[2]:>7.3f}/{v[1]/v[2]:<9.3f}")
d = (tot["shipped"][1] - tot["off"][1]) / tot["off"][2]
print(f"  {'ALL':>4} {tot['off'][2]:>5}  " + "  ".join(cells) + f"   {d:>+9.3f}")
print()
print("  each cell = sess_recall / all_sessions (any gold session / ALL gold)")
print("  d_shipped = all_sessions change, shipped vs off. Positive is better.")
print("  Watch cat 4 (single-hop, 1 gold session): the spread must not cost it.")
print("  If div_only ~ shipped, widening is unnecessary and can be dropped.")

try:
    M.close()
except Exception:
    pass

