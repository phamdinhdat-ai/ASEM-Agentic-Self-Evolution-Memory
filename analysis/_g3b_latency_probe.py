"""G3b efficiency: what does the wider pool + reorder cost in retrieval time?

The plan's efficiency axis asks for retrieval latency alongside answer tokens.
`session_diversity` reorders a list (free), but the pool it spreads is only wide
enough to help when k2 is raised, and k2=40 vs 10 means the retriever ranks and
returns 4x the candidates. That is a real cost and the write-up must state it
rather than imply the mechanism is free.

Times the retrieval leg only (embedding + channels + walk + Phase D), with the
stub generator so no answer call is made, over all questions of one
conversation. Reports median and p90 per arm.

Retrieval only, no generation, no API cost.
"""
import argparse
import json
import os
import shutil
import statistics
import tempfile
import time
from collections import defaultdict

import yaml

from asem.backends.langchain_backend import _build_embedder
from asem.backends.openai_backend import OpenAIBackend
from analysis.evidence_recall import gold_sessions
from eval.systems import build_asem_thg_system

CFG = "configs/models/thg_v2_postfix_off.yaml"
BANK = "static/memory_banks/locomo10/gpt56_luna/ASEM-THG/locomo_0000/asem_thg.sqlite"
DATASET = "datasets/locomo/locomo10.json"

_ap = argparse.ArgumentParser()
_ap.add_argument("--config", default=CFG)
_ap.add_argument("--bank", default=BANK)
_ap.add_argument("--dataset", default=DATASET)
_ap.add_argument("--k1", type=int, default=200)
_ap.add_argument("--k2", type=int, default=40)
_ap.add_argument("--delta", type=float, default=0.0)
_ap.add_argument("--repeat", type=int, default=3, help="timed repeats per query")
_ap.add_argument("--limit", type=int, default=60,
                 help="questions to time (a subset is enough for a median; the "
                      "wide arm is slow and the machine is memory-throttled)")
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

ARMS = {
    "off": (False, 20, 10, 0.30),
    "div_only": (True, 20, 10, 0.30),
    "shipped": (True, _args.k1, _args.k2, _args.delta),
}
ARM_ORDER = ["off", "div_only", "shipped"]

ds = json.load(open(DATASET))
qs = [it["question"] for it in ds[0]["qa"]
      if it.get("question") and gold_sessions(it.get("evidence") or [])]
qs = qs[: _args.limit]

# Warm up the graph + embedder cache once per arm so the first query's cold cost
# does not dominate the median.
timings = defaultdict(list)
for a in ARM_ORDER:
    div, k1, k2, delta = ARMS[a]
    retr.session_diversity = div
    retr.k1, retr.k2, retr.delta = k1, k2, delta
    retr.retrieve(qs[0], M)  # warm

for q in qs:
    for a in ARM_ORDER:
        div, k1, k2, delta = ARMS[a]
        retr.session_diversity = div
        retr.k1, retr.k2, retr.delta = k1, k2, delta
        best = None
        for _ in range(_args.repeat):
            retr.ppr_scores = {}
            t0 = time.perf_counter()
            retr.retrieve(q, M)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)  # min = least noise
        timings[a].append(best)

print(f"retrieval latency over {len(qs)} questions, best of {_args.repeat}")
print(f"  shipped pool k1={_args.k1} k2={_args.k2} delta={_args.delta}")
print()
print(f"  {'arm':<10} {'median_ms':>10} {'p90_ms':>9} {'mean_ms':>9}")
base_med = None
for a in ARM_ORDER:
    ts = sorted(t * 1000 for t in timings[a])
    med = statistics.median(ts)
    if a == "off":
        base_med = med
    p90 = ts[int(0.9 * (len(ts) - 1))]
    print(f"  {a:<10} {med:>10.1f} {p90:>9.1f} {statistics.mean(ts):>9.1f}")
print()
for a in ARM_ORDER:
    ts = sorted(t * 1000 for t in timings[a])
    med = statistics.median(ts)
    print(f"  {a:<10} median {med/base_med:.2f}x off")
print()
print("  Answer tokens are unchanged by this mechanism (the answer agent caps the")
print("  window at max_context_notes before rendering); the cost is retrieval time.")

try:
    M.close()
except Exception:
    pass
