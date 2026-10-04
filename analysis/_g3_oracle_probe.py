"""G3 oracle: can a PERFECT sub-question retrieve the session the base missed?

The headroom probe (`_g3_headroom_probe.py`) located G3's target precisely:
category-1 multi-hop questions have `all_sessions` = 0.4688, and 18 of 30
multi-gold questions are missing a gold session ENTIRELY from the answer window.
`session_recall@8` was saturated at 0.92; `all_sessions` is the metric with room.

Before writing a query planner (G3) this asks the gating question the plan's
falsification criterion implies: **is the missing session reachable by retrieval
at all?** The oracle replaces the planner with ground truth — instead of an LLM
guessing a sub-question, it retrieves with the missing session's OWN gold turn
text. That is the strongest possible sub-question.

  * If even the oracle cannot surface the missing session, no decomposition can,
    and G3 should be dropped rather than built.
  * If the oracle surfaces it, G3's ceiling is real and the remaining question is
    only whether an LLM planner approximates the oracle well enough.

One base pass + one oracle pass per cat-1 question. No generation, no API cost.
"""
import argparse
import json
import os
import re
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
_DIA_RE = re.compile(r"D(\d+):(\d+)")

_ap = argparse.ArgumentParser()
_ap.add_argument("--config", default=CFG)
_ap.add_argument("--bank", default=BANK)
_ap.add_argument("--dataset", default=DATASET)
_ap.add_argument("--cat", type=int, default=1, help="category to probe")
_args = _ap.parse_args()
CFG, BANK, DATASET = _args.config, _args.bank, _args.dataset


class StubClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kw):
                raise RuntimeError("no generation in this probe")


def turn_texts(conv: dict) -> dict:
    """dia_id -> text, across every session in the conversation."""
    out = {}
    for key, val in conv.items():
        m = re.fullmatch(r"session_(\d+)", str(key))
        if not m or not isinstance(val, list):
            continue
        for turn in val:
            if isinstance(turn, dict) and turn.get("dia_id"):
                out[str(turn["dia_id"])] = str(turn.get("text", ""))
    return out


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
turns = turn_texts(conv0["conversation"])
note_sessions = load_note_sessions(BANK)

# Only questions whose gold spans >1 session — the ones all_sessions penalises.
multi = []
for item in conv0["qa"]:
    if item.get("category") != _args.cat:
        continue
    gold = gold_sessions(item.get("evidence") or [])
    if len(gold) > 1:
        multi.append((item, gold))

stats = {
    "n": 0,
    "base_all": 0,          # all gold sessions already present (base)
    "missing_qs": 0,        # questions with >=1 missing session
    "oracle_covered": 0,    # of those, oracle retrieval covers the missing one
    "oracle_adds_new": 0,   # oracle surfaces a session the base top-8 lacked
}

for item, gold in multi:
    q = item["question"]
    stats["n"] += 1

    retr.ppr_scores = {}
    base = retr.retrieve(q, M)
    base_used = [n.id for n in base[:K]]
    base_sess = {note_sessions[i] for i in base_used if i in note_sessions}
    if gold <= base_sess:
        stats["base_all"] += 1
        continue

    missing = gold - base_sess
    stats["missing_qs"] += 1

    # Oracle: retrieve with the missing session's own gold turn texts.
    ev = item.get("evidence") or []
    texts = []
    for e in ev:
        m = _DIA_RE.search(str(e))
        if not m:
            continue
        if f"s{int(m.group(1))}" in missing:
            t = turns.get(str(e))
            if t:
                texts.append(t)
    if not texts:
        continue
    oracle_q = " ".join(texts)
    retr.ppr_scores = {}
    orc = retr.retrieve(oracle_q, M)
    orc_sess = {note_sessions[i] for i in [n.id for n in orc[:K]] if i in note_sessions}

    if missing & orc_sess:
        stats["oracle_covered"] += 1
    if orc_sess - base_sess:
        stats["oracle_adds_new"] += 1

print(f"category={_args.cat}  K={K}  walk=OFF")
print(f"multi-gold questions: {stats['n']}")
print(f"  all gold sessions already in base top-8: {stats['base_all']} "
      f"({stats['base_all']/max(stats['n'],1):.3f})")
print(f"  missing >=1 gold session (G3's target): {stats['missing_qs']}")
print(f"  oracle retrieval COVERS a missing session: {stats['oracle_covered']} "
      f"of {stats['missing_qs']} "
      f"({stats['oracle_covered']/max(stats['missing_qs'],1):.3f})")
print(f"  oracle surfaces ANY session the base lacked: {stats['oracle_adds_new']}")
print()
print("Reading: 'oracle covers' is the ceiling a PERFECT sub-question achieves.")
print("If it is ~0, the missing evidence is not retrievable by text and G3 cannot")
print("help. If it is high, G3's ceiling is real and only planner quality remains.")

try:
    M.close()
except Exception:
    pass
