"""G3 candidate mechanism: wide pool + session-diverse window. Two questions.

`_g3_widen_probe.py` showed, on cat-1 multi-gold only:

  off        all_sessions=0.4074
  wide       all_sessions=0.3704   <- width alone is WORSE than off
  wide+div   all_sessions=0.7037   <- diversification supplies +0.333

Two things that probe did not establish:

  1. MECHANISM. Is the win from spreading across sessions at all, or from the
     *ranked* spread (best note per session)? Add a control that spreads across
     sessions at RANDOM (one random note per random session). If rand-spread ~
     ranked-spread, the rule is "cover sessions"; if ranked >> rand, the ranking
     inside the spread carries signal and must be kept.

  2. REGRESSION. Multi-gold questions are cat 1. Single-hop (cat 4) has exactly
     one gold session, so spending 7 of 8 window slots on other sessions can only
     hurt. A mechanism that fixes multi-hop by breaking single-hop is not a fix.
     Measure every category.

Retrieval only, no generation, no API cost. Deterministic (seeded RNG).
"""
import argparse
import json
import os
import random
import shutil
import tempfile
import zlib
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
SEED = 20261004

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


def sess(notes):
    return {note_sessions[n.id] for n in notes[:K] if n.id in note_sessions}


def div_ranked(notes, k):
    """Best-ranked note per session, then fill by rank."""
    seen, out = set(), []
    for n in notes:
        s = note_sessions.get(n.id)
        if s is not None and s not in seen:
            seen.add(s)
            out.append(n)
        if len(out) == k:
            return out
    for n in notes:
        if n not in out:
            out.append(n)
        if len(out) == k:
            break
    return out


def div_random(notes, k, rng):
    """One RANDOM note per session, sessions in random order, fill by rank."""
    by_sess = defaultdict(list)
    for n in notes:
        s = note_sessions.get(n.id)
        if s is not None:
            by_sess[s].append(n)
    order = list(by_sess)
    rng.shuffle(order)
    out, seen = [], set()
    for s in order:
        out.append(rng.choice(by_sess[s]))
        seen.add(s)
        if len(out) == k:
            return out
    for n in notes:
        if n not in out:
            out.append(n)
        if len(out) == k:
            break
    return out


ARMS = ["off", "wide", "wide+div", "wide+randdiv"]
by_cat = defaultdict(lambda: {a: [0, 0, 0] for a in ARMS})  # any, all, n

for item in conv0["qa"]:
    cat = item.get("category")
    if cat is None:
        continue
    gold = gold_sessions(item.get("evidence") or [])
    if not gold:
        continue
    q = item["question"]

    retr.k1, retr.k2, retr.delta = 20, 10, 0.30
    retr.ppr_scores = {}
    base = retr.retrieve(q, M)

    retr.k1, retr.k2, retr.delta = _args.k1, _args.k2, _args.delta
    retr.ppr_scores = {}
    wide = retr.retrieve(q, M)

    # zlib.crc32, not hash(): str hashing is salted per process, which would
    # make the "seeded" control non-reproducible across runs.
    rng = random.Random(SEED + zlib.crc32(q.encode("utf-8")))

    pools = {
        "off": base,
        "wide": wide,
        "wide+div": div_ranked(wide, K),
        "wide+randdiv": div_random(wide, K, rng),
    }
    for a in ARMS:
        got = sess(pools[a])
        by_cat[cat][a][0] += 1 if (got & gold) else 0
        by_cat[cat][a][1] += 1 if (gold <= got) else 0
        by_cat[cat][a][2] += 1

print(f"K={K}  wide pool: k1={_args.k1} k2={_args.k2} delta={_args.delta}")
print()
print(f"  {'cat':>4} {'n':>5}  " + "  ".join(f"{a:>13}" for a in ARMS))
for cat in sorted(by_cat):
    cells = []
    for a in ARMS:
        any_, all_, n = by_cat[cat][a]
        cells.append(f"{any_/n:>5.3f}/{all_/n:<7.3f}")
    n = by_cat[cat]["off"][2]
    print(f"  {cat:>4} {n:>5}  " + "  ".join(cells))
print()
print("  each cell = sess_recall / all_sessions   (any gold session / ALL gold)")
print()
print("  Mechanism: if wide+randdiv ~ wide+div, the win is session COVERAGE;")
print("             if wide+div >> wide+randdiv, ranked selection matters.")
print("  Regression: compare 'off' vs 'wide+div' on cat 4 (single-hop, 1 gold")
print("             session) — a fix that breaks single-hop is not a fix.")

try:
    M.close()
except Exception:
    pass
