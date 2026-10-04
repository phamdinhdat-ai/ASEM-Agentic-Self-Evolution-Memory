"""G3: does the REAL LLM planner approach the oracle ceiling?

The mechanism probe (`_g3_planner_probe.py`) established the shape:

  off             all_sessions=0.4074
  heuristic       all_sessions=0.3704   (free split: net-NEGATIVE — crude)
  oracle_fused    all_sessions=0.8148   (fuse original + per-gold-session text)

So the fusion mechanism is worth +0.41, but only if the sub-questions are good.
The free heuristic's are not. This measures the real `llm` planner — one
backbone call per multi-hop question — against both, and prints the plans so
planner quality is visible, not just its score.

API cost: one planning call per cat-1 multi-gold question (~27). Retrieval is
embedding-driven and makes no generation calls.
"""
import argparse
import json
import os
import re
import shutil
import tempfile

import yaml

from asem.backends.langchain_backend import _build_embedder
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
_ap.add_argument("--show", type=int, default=4, help="how many plans to print")
_args = _ap.parse_args()
CFG, BANK, DATASET = _args.config, _args.bank, _args.dataset


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


def build(mode, tmp, tag):
    cfg = yaml.safe_load(open(CFG))
    cfg.setdefault("hyper_graph", {})["hydrate_on_load"] = True
    ret = cfg.setdefault("retriever", {})
    ret["use_graph_rank"] = False
    ret["query_planner"] = mode
    p = os.path.join(tmp, f"cfg_{tag}.yaml")
    yaml.safe_dump(cfg, open(p, "w"))
    d = os.path.join(tmp, f"db_{tag}")
    os.makedirs(d, exist_ok=True)
    shutil.copy2(BANK, os.path.join(d, "asem_thg.sqlite"))
    # backend=None -> build_backend(cfg["inference"]) -> the real OpenAI-compatible
    # client from .env. Only the planner calls generate(); retrieval does not.
    return build_asem_thg_system(p, d, backend=None)


ds = json.load(open(DATASET))
conv0 = ds[0]
turns = turn_texts(conv0["conversation"])
note_sessions = load_note_sessions(BANK)

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


def sess(notes):
    return {note_sessions[i] for i in [n.id for n in notes[:K]] if i in note_sessions}


def score(system, oracle=False, show=0):
    retr = system.pipeline.retriever
    M = system.pipeline.memory_bank
    n = allp = anyp = 0
    shown = 0
    for item, gold, by_sess in items:
        q = item["question"]
        if oracle:
            pools = [retr._retrieve_core(q, M)]
            for s in sorted(gold):
                if by_sess.get(s):
                    pools.append(retr._retrieve_core(" ".join(by_sess[s]), M))
            notes = retr._fuse_subquestions(pools, [q] * len(pools))
        else:
            retr.ppr_scores = {}
            notes = retr.retrieve(q, M)
            if show and shown < show and retr.stats.get("planner_fired"):
                print(f"    Q: {q}")
                for sq in retr.stats.get("planner_subquestion_list", []):
                    print(f"       - {sq}")
                shown += 1
        got = sess(notes)
        n += 1
        if got & gold:
            anyp += 1
        if gold <= got:
            allp += 1
    return n, anyp, allp


results = {}
for arm, mode, oracle in (("off", "off", False),
                          ("heuristic", "heuristic", False),
                          ("llm", "llm", False),
                          ("oracle_fused", "off", True)):
    system = build(mode, tmp, arm)
    print(f"built {arm}", flush=True)
    results[arm] = score(system, oracle=oracle,
                         show=_args.show if arm == "llm" else 0)
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
for arm in ("off", "heuristic", "llm", "oracle_fused"):
    n, anyp, allp = results[arm]
    print(f"  {arm:<14} {anyp/n:>12.4f} {allp/n:>13.4f}")
print()
print("If `llm` lands near `oracle_fused`, the planner is the lever. If it lands")
print("near `off`/`heuristic`, the LLM's sub-questions are as noisy as the free")
print("split and G3 does not pay for its extra call.")
