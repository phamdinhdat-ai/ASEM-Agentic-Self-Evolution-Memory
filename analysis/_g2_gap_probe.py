"""Why is the G2 walk inert at the answer window? Score-gap decomposition.

The recall probe establishes WHAT happens (top-8 barely moves). This says WHY,
by instrumenting Phase D on real queries: for every note the walk ADDS, report

  * its rank in the base order (where it entered the pool),
  * the hybrid-score gap it must close to reach slot K,
  * the size of its PPR term, in the same units.

If `ppr contribution` is smaller than the gap for every added note, the walk is
arithmetically incapable of reaching the answer window at the current weights,
and the fix is a weight change (or a wider `ppr_topk`), not a better walk.

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

# Retriever overrides for this run: the default Phase D weights give the walk
# ~1 rank slot of authority; the sweep asks what a larger share buys.
RETRIEVER_OVERRIDES = json.loads(os.environ.get("G2_RET_OVERRIDES", "{}"))


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
cfg.setdefault("retriever", {}).update(RETRIEVER_OVERRIDES)
p = os.path.join(tmp, "cfg.yaml")
yaml.safe_dump(cfg, open(p, "w"))
d = os.path.join(tmp, "db")
os.makedirs(d, exist_ok=True)
shutil.copy2(BANK, os.path.join(d, "asem_thg.sqlite"))
backend = OpenAIBackend(client=StubClient(), model="stub", embedder=embedder)
system = build_asem_thg_system(p, d, backend=backend)
retr = system.pipeline.retriever
M = system.pipeline.memory_bank

# Reproduce Phase D's arithmetic on the retriever's own state after a retrieve.
def decompose(q):
    notes = retr.retrieve(q, M)
    n = len(notes)
    if n < 2:
        return None
    e_q = retr.backend.embed(q)
    base_rank = np.array([1.0 - i / max(n - 1, 1) for i in range(n)])
    local = np.array([retr._cosine(e_q, x.e) for x in notes])
    glob = np.array([retr._compute_global_score(x, notes[0]) for x in notes])
    qv = np.array([x.q for x in notes])
    z = retr._zscore
    rank_z, local_z, glob_z = z(base_rank), z(local), z(glob)
    q_z = z(qv) if qv.std() > 1e-6 else np.zeros(n)
    raw_ppr = np.array([retr.ppr_scores.get(x.id, 0.0) for x in notes])
    hi = raw_ppr.max() if raw_ppr.size else 0.0
    ppr_z = raw_ppr / hi if hi > 0 else np.zeros(n)
    w_ppr = retr.ppr_weight if ppr_z.any() else 0.0
    w_graph = retr.alpha + retr.beta + (retr.gamma if qv.std() > 1e-6 else 0.0) + w_ppr
    scale = (1.0 - retr.base_weight) / (w_graph or 1.0)
    # Per-note contribution of the PPR term, in hybrid units.
    contrib = scale * w_ppr * ppr_z
    return dict(notes=notes, ppr_z=ppr_z, contrib=contrib, scale=scale, w_ppr=w_ppr)


ds = json.load(open(DATASET))
qa = [it for it in ds[0]["qa"] if it.get("question")][:NQ]

added_ranks, gaps, contribs = [], [], []
for item in qa:
    d0 = decompose(item["question"])
    if not d0:
        continue
    notes, contrib = d0["notes"], d0["contrib"]
    in_pool = len(notes)
    # Notes the walk ranked but that sit OUTSIDE the top-K: these are the ones
    # the walk would have to promote for recall to move.
    for i in range(K, in_pool):
        if contrib[i] <= 0:
            continue
        added_ranks.append(i)
        # Gap in hybrid units between slot K-1 (last rendered) and this note is
        # approximated by their base-rank z difference scaled by `a` — the walk
        # term is what must cover it.
        gaps.append(d0["scale"] * d0["w_ppr"])   # max the term can contribute
        contribs.append(contrib[i])

if added_ranks:
    print(f"queries={len(qa)}  notes outside top-{K} carrying PPR mass: {len(added_ranks)}")
    print(f"  their pool ranks: min={min(added_ranks)}  median={int(np.median(added_ranks))}  max={max(added_ranks)}")
    print(f"  mean PPR contribution (hybrid units) = {np.mean(contribs):.5f}")
    print(f"  max  PPR contribution possible       = {np.mean(gaps):.5f}")
    print(f"  a base-rank step at n={in_pool} is a*step = "
          f"{retr.base_weight * (1.0/max(in_pool-1,1)) * np.sqrt(12):.5f} (approx z units)")
else:
    print("no notes outside the answer window carry PPR mass — the walk only "
          "re-ranks what the base already rendered")
