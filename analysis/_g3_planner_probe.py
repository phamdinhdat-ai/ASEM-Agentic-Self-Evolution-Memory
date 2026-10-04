"""G3: does sub-question fusion recover the missing hop?

The probes established: (a) 16 of 27 cat-1 multi-gold questions miss a gold
session; (b) 13 of those misses are absent from the POOL, not merely outranked;
(c) an oracle sub-question (the gold turn's own text) recovers 16/16. This
measures how much of that ceiling the ACTUAL planner reaches.

Arms (all over the frozen bank, graph walk OFF so only the planner varies):

  off          query_planner=off                    -- baseline
  heuristic    query_planner=heuristic              -- free conjunction split
  oracle_fused original + the missing hop's gold text, FUSED -- the true ceiling

The oracle arm FUSES: it retrieves the original query AND a gold-text query for
each missing session, then fuses. Replacing the query with gold text (the first
version of this probe) is NOT the ceiling — it loses the questions the original
already answered. Fusing is what a working planner would do, so oracle_fused is
the number a planner has to be compared against.

Scored on the metric with room: all_sessions (every gold session present in the
answer window) and session_recall, category 1 only.

One system, one pass per arm, no generation, no API cost.
"""
import argparse
import json
import os
import re
import shutil
import tempfile

import yaml

from asem.backends.langchain_backend import _build_embedder
from asem.backends.openai_backend import OpenAIBackend
from asem.query_planner import QueryPlanner
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


def turn_texts(conv):
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


def build(planner_mode, tmp):
    cfg = yaml.safe_load(open(CFG))
    cfg.setdefault("hyper_graph", {})["hydrate_on_load"] = True
    ret = cfg.setdefault("retriever", {})
    ret["use_graph_rank"] = False
    ret["query_planner"] = planner_mode
    p = os.path.join(tmp, f"cfg_{planner_mode}.yaml")
    yaml.safe_dump(cfg, open(p, "w"))
    d = os.path.join(tmp, f"db_{planner_mode}")
    os.makedirs(d, exist_ok=True)
    shutil.copy2(BANK, os.path.join(d, "asem_thg.sqlite"))
    backend = OpenAIBackend(client=StubClient(), model="stub", embedder=embedder)
    return build_asem_thg_system(p, d, backend=backend)


ds = json.load(open(DATASET))
conv0 = ds[0]
turns = turn_texts(conv0["conversation"])
note_sessions = load_note_sessions(BANK)

# (item, gold, per-gold-session gold texts) for cat-1 multi-gold questions.
items = []
for item in conv0["qa"]:
    if item.get("category") != _args.cat:
        continue
    gold = gold_sessions(item.get("evidence") or [])
    if len(gold) <= 1:
        continue
    by_sess = {}
    for e in item.get("evidence") or []:
        m = _DIA_RE.search(str(e))
        t = turns.get(str(e))
        if m and t:
            by_sess.setdefault(f"s{int(m.group(1))}", []).append(t)
    items.append((item, gold, by_sess))

tmp = tempfile.mkdtemp()


def _sessions(notes, K):
    used = [x.id for x in notes[:K]]
    return {note_sessions[i] for i in used if i in note_sessions}


def score(system, oracle=False):
    retr = system.pipeline.retriever
    M = system.pipeline.memory_bank
    n = allp = anyp = 0
    for item, gold, by_sess in items:
        q = item["question"]
        if oracle:
            # Fuse the original query with one query per gold session's own
            # turn texts — the strongest sub-questions a planner could emit.
            pools = [retr._retrieve_core(q, M)]
            for sess in sorted(gold):
                texts = by_sess.get(sess)
                if texts:
                    pools.append(retr._retrieve_core(" ".join(texts), M))
            notes = retr._fuse_subquestions(pools, [q] * len(pools))
        else:
            retr.ppr_scores = {}
            notes = retr.retrieve(q, M)
        got = _sessions(notes, K)
        n += 1
        if got & gold:
            anyp += 1
        if gold <= got:
            allp += 1
    return n, anyp, allp


results = {}
for arm, oracle in (("off", False), ("heuristic", False), ("oracle_fused", True)):
    system = build("heuristic" if arm == "heuristic" else "off", tmp)
    print(f"built {arm}", flush=True)
    results[arm] = score(system, oracle=oracle)
    try:
        system.pipeline.memory_bank.close()
    except Exception:
        pass
    del system
    import gc
    gc.collect()
    print(f"scored {arm}", flush=True)

print()
print(f"category={_args.cat}  K={K}  multi-gold questions={results['off'][0]}")
print(f"  {'arm':<14} {'sess_recall':>12} {'all_sessions':>13}")
for arm in ("off", "heuristic", "oracle_fused"):
    n, anyp, allp = results[arm]
    print(f"  {arm:<14} {anyp/n:>12.4f} {allp/n:>13.4f}")
print()
print("off -> heuristic   = the gain from the FREE split (no LLM).")
print("off -> oracle_fused = the ceiling a planner reaches if it names every")
print("                      gold session's hop; any real planner sits below it.")
