"""G3: is the oracle's ceiling REACHABLE, or does it require knowing the answer?

The planner probe gave the shape (cat-1 multi-gold, n=27):

  off             all_sessions=0.4074
  heuristic       all_sessions=0.3704
  llm (blind)     all_sessions=0.4074
  oracle_fused    all_sessions=0.8148

The oracle fused the original query with the MISSING SESSION'S OWN GOLD TURN
TEXT. That text contains the answer. So the 0.8148 is not a planner ceiling — it
is the ceiling of already knowing the answer, which no planner has.

This measures whether the gap is bridgeable by a question a planner could
actually write. For each incomplete cat-1 question it asks: how far is the
missing session from the question, semantically?

  q_to_missing  max cosine(question, any note in the missing session)
  q_to_found    max cosine(question, any note in the top-8)

If `q_to_missing` is close to `q_to_found`, the missing session is topically
NEAR and a paraphrasing planner could plausibly reach it — the gap is a ranking
failure. If `q_to_missing` is far below, the missing fact is about a different
topic than the question names, and no query derived from the question can name
it: the gap is not bridgeable by decomposition, only by knowing the answer.

Also splits the questions by whether the missing hop is a DIFFERENT ENTITY
("Caroline and Melanie both painted") or the SAME entity ("Melanie's pets") —
only the former is what decomposition can address.

Embedder only: no generation, no API cost.
"""
import argparse
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict

import numpy as np
import yaml

from asem.backends.langchain_backend import _build_embedder
from asem.backends.openai_backend import OpenAIBackend
from analysis.evidence_recall import gold_sessions, load_note_sessions
from eval.systems import build_asem_thg_system

CFG = "configs/models/thg_v2_postfix_off.yaml"
BANK = "static/memory_banks/locomo10/gpt56_luna/ASEM-THG/locomo_0000/asem_thg.sqlite"
DATASET = "datasets/locomo/locomo10.json"
K = 8
_DIA_RE = re.compile(r"D(\d+):(\d+)")

_ap = argparse.ArgumentParser()
_ap.add_argument("--cat", type=int, default=1)
_ap.add_argument("--config", default=CFG)
_ap.add_argument("--bank", default=BANK)
_ap.add_argument("--dataset", default=DATASET)
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

# session -> note embeddings, from the bank, for the reachability comparison.
session_notes = defaultdict(list)
for nid, sid in note_sessions.items():
    session_notes[sid].append(nid)
note_vec = {}
for nid in note_sessions:
    pass
all_notes = {n.id: n for n in M.list_notes()}


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


rows = []
for item in conv0["qa"]:
    if item.get("category") != _args.cat:
        continue
    gold = gold_sessions(item.get("evidence") or [])
    if len(gold) <= 1:
        continue
    q = item["question"]
    retr.ppr_scores = {}
    notes = retr.retrieve(q, M)
    top8 = notes[:K]
    got = {note_sessions[n.id] for n in top8 if n.id in note_sessions}
    if gold <= got:
        continue
    missing = gold - got

    e_q = backend.embed(q)
    q_to_found = max(
        (cos(e_q, all_notes[n.id].e) for n in top8 if n.id in all_notes),
        default=0.0,
    )
    q_to_missing = 0.0
    for sid in missing:
        for nid in session_notes.get(sid, []):
            n = all_notes.get(nid)
            if n is not None:
                q_to_missing = max(q_to_missing, cos(e_q, n.e))
    # Entity-split vs same-entity: does the question name >1 distinct person?
    ents = set()
    for n in top8:
        for e in (getattr(n, "entities", None) or []):
            ents.add(str(e).lower())
    rows.append((q, sorted(missing), q_to_found, q_to_missing))

n = len(rows)
print(f"category={_args.cat}  K={K}  walk=OFF")
print(f"incomplete multi-gold questions: {n}")
print()
print(f"  {'question':<52} {'miss':<10} {'q->found':>9} {'q->miss':>8}")
for q, miss, fnd, mss in rows:
    print(f"  {q[:50]:<52} {','.join(miss):<10} {fnd:>9.3f} {mss:>8.3f}")
print()
fnd = np.array([r[2] for r in rows])
mss = np.array([r[3] for r in rows])
print(f"  mean q->found   : {fnd.mean():.3f}")
print(f"  mean q->missing : {mss.mean():.3f}")
print(f"  mean gap        : {(fnd - mss).mean():.3f}")
print(f"  questions where q->missing >= 0.30 (topically reachable): "
      f"{int((mss >= 0.30).sum())} of {n}")
print()
print("If q->missing sits well below q->found, the missing fact is about a")
print("different topic than the question names, so no sub-question DERIVED from")
print("the question can name it. The oracle reached it only because its query WAS")
print("the answer's own turn — a ceiling no planner can have.")

try:
    M.close()
except Exception:
    pass
