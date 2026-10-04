"""The structural reason the G2 walk cannot reach the answer window.

The gap probe measured the Phase D arithmetic; this measures the two SIGNALS the
walk-added notes actually carry, to show the problem is not a tuning mistake but
a design tension:

  * rank_z — where the note enters Phase D. Walk-added notes are appended after
    the base results, so they enter at the BOTTOM of the pool.
  * local_z — how similar the note is to the query. A note the walk reached but
    dense retrieval did not is, by construction, one whose embedding the query
    did not match well.

A walk's unique value is reaching a bridging premise that shares no query terms
with the question — and that is exactly the note Phase D penalises on both
terms at once. This probe prints the distributions so the claim is measured, not
asserted.

No API cost: retrieval only, generation stubbed.
"""
import json
import os
import shutil
import tempfile

import numpy as np
import yaml

from asem.backends.langchain_backend import _build_embedder
from asem.backends.openai_backend import OpenAIBackend
from eval.systems import build_asem_thg_system

CFG = "configs/models/thg_v2_postfix_off.yaml"
BANK = "static/memory_banks/locomo10/gpt56_luna/ASEM-THG/locomo_0000/asem_thg.sqlite"
DATASET = "datasets/locomo/locomo10.json"
K = 8
NQ = 40


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
qa = [it for it in ds[0]["qa"] if it.get("question")][:NQ]

rendered_local, added_local, added_rank_frac = [], [], []
for item in qa:
    q = item["question"]
    notes = retr.retrieve(q, M)
    n = len(notes)
    if n <= K:
        continue
    e_q = retr.backend.embed(q)
    sims = np.array([retr._cosine(e_q, x.e) for x in notes])
    rendered_local.extend(sims[:K].tolist())
    # Notes outside the top-K that the walk ranked (ppr_scores non-empty).
    for i in range(K, n):
        if retr.ppr_scores.get(notes[i].id, 0.0) > 0.0:
            added_local.append(float(sims[i]))
            added_rank_frac.append(i / max(n - 1, 1))

def stats(name, xs):
    a = np.array(xs)
    if a.size == 0:
        print(f"  {name}: none")
        return
    print(f"  {name}: n={a.size:5d}  mean={a.mean():.3f}  "
          f"p25={np.percentile(a,25):.3f}  median={np.median(a):.3f}  "
          f"p75={np.percentile(a,75):.3f}")

print(f"queries={len(qa)}  K={K}")
print("Query-to-note cosine similarity:")
stats("rendered (top-K)   ", rendered_local)
stats("walk-added (below K)", added_local)
print("Position the walk-added notes enter Phase D (0=top, 1=bottom):")
stats("rank fraction      ", added_rank_frac)

try:
    M.close()
except Exception:
    pass
