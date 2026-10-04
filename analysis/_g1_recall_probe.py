"""G1 re-baseline, retrieval level: does hydration move evidence recall?

The G1 probe already showed the RANKING changes on every question. The question
that matters for the write-up is whether it changes the EVIDENCE RECALL — the
G7 metric that attributes an answer to retrieval rather than to the model. This
runs both arms over the frozen conv0 bank, uses the REAL local embedder and a
stub generator, and scores session-level recall with the same join the
instrument uses (gold "D1:3" -> bank session_id "s1").

No API cost: retrieval is embedding-driven and generation is never called.
"""
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


def build(hydrate: bool, tmp):
    cfg = yaml.safe_load(open(CFG))
    cfg.setdefault("hyper_graph", {})["hydrate_on_load"] = hydrate
    p = os.path.join(tmp, f"cfg_{hydrate}.yaml")
    yaml.safe_dump(cfg, open(p, "w"))
    d = os.path.join(tmp, f"db_{hydrate}")
    os.makedirs(d, exist_ok=True)
    shutil.copy2(BANK, os.path.join(d, "asem_thg.sqlite"))
    backend = OpenAIBackend(client=StubClient(), model="stub", embedder=embedder)
    return build_asem_thg_system(p, d, backend=backend)


ds = json.load(open(DATASET))
qa = ds[0]["qa"]
tmp = tempfile.mkdtemp()
arms = {"live": build(True, tmp), "dark": build(False, tmp)}
note_sessions = load_note_sessions(BANK)

# The deployed answer reads at most `answer.max_context_notes` notes, and with
# context_window=32768 / max_tokens=512 the budget never bites, so the rendered
# set is exactly the top-8 of the candidate list. Scoring recall over the TOP-8
# (not the whole candidate list) is what removes the pool-size confound: the
# live arm returns ~26 candidates and the dark arm ~15, so a candidate-level
# comparison would credit the graph for merely returning MORE notes. At k=8 both
# arms see the same number of notes, and the metric answers the question that
# matters: did the graph put better evidence in front of the model?
K = 8

# category -> [n, hits_any, hits_all, used]
agg = {arm: defaultdict(lambda: [0, 0, 0, 0]) for arm in arms}
for i, item in enumerate(qa):
    q = item.get("question")
    if not q:
        continue
    gold = gold_sessions(item.get("evidence") or [])
    if not gold:
        continue
    cat = item.get("category") or item.get("category_name") or "unknown"
    for arm, system in arms.items():
        notes = system.pipeline.retriever.retrieve(q, system.pipeline.memory_bank)
        used = [n.id for n in notes[:K]]
        got = {note_sessions[nid] for nid in used if nid in note_sessions}
        b = agg[arm][cat]
        b[0] += 1
        b[3] += len(used)
        if got & gold:
            b[1] += 1
        if gold <= got:
            b[2] += 1

print(f"bank={BANK}")
print(f"notes in bank={len(note_sessions)}  questions scored="
      f"{sum(v[0] for v in agg['live'].values())}  K={K}")
for arm in ("live", "dark"):
    per_cat = agg[arm]
    n = sum(v[0] for v in per_cat.values())
    recall = sum(v[1] for v in per_cat.values()) / n if n else 0.0
    allsess = sum(v[2] for v in per_cat.values()) / n if n else 0.0
    print(f"\n=== {arm} ===")
    print(f"    overall session_recall@{K}={recall:.4f}  all_sessions={allsess:.4f}  n={n}")
    for cat, v in sorted(per_cat.items()):
        print(f"    {cat:<16} n={v[0]:>4}  recall={v[1]/v[0]:.4f}  "
              f"all={v[2]/v[0]:.4f}  used={v[3]/v[0]:.1f}")

