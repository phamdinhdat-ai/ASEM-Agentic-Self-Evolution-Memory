"""G3b on the FULL LoCoMo set: all 10 conversations, n=1982.

The conv0 result is real but thin — 32 multi-hop questions, where a 2-question
swing is 6 points. The plan's own risk note says so: "Multi-hop is only n=32 in
locomo_0000; a 2-question swing is 6 points. Use n=1986 for the multi-hop claim."
This is that run (1982 = the questions that carry gold evidence; the rest have
none and cannot be scored for recall).

For each of the 10 conversations it builds the ASEM-THG system from that
conversation's frozen bank, then runs three arms over its questions through the
REAL `EnhancedHybridRetriever.retrieve`:

  off        session_diversity=false, default pool (k1=20, k2=10)
  div_only   session_diversity=true,  default pool
  shipped    session_diversity=true,  wide pool (k1/k2 from --k1/--k2)

One system per conversation, closed before the next — the 7.9GB box cannot hold
ten banks plus ten graphs at once. Session recall is scored per category and
pooled across all conversations, so the multi-hop number has n=282, not 32.

Retrieval only, no generation, no API cost. Expect ~2-3h on the 7.9GB machine.
"""
import argparse
import gc
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
BANK_ROOT = "static/memory_banks/locomo10/gpt56_luna/ASEM-THG"
DATASET = "datasets/locomo/locomo10.json"
K = 8

_ap = argparse.ArgumentParser()
_ap.add_argument("--config", default=CFG)
_ap.add_argument("--bank-root", default=BANK_ROOT)
_ap.add_argument("--dataset", default=DATASET)
_ap.add_argument("--k1", type=int, default=200)
_ap.add_argument("--k2", type=int, default=40)
_ap.add_argument("--delta", type=float, default=0.0)
_ap.add_argument("--convs", type=int, default=10, help="how many conversations")
_ap.add_argument("--json-out", default=None,
                 help="write per-conversation + pooled results here after each "
                      "conversation, so a crash at conv N does not lose 0..N-1")
_args = _ap.parse_args()
CFG, BANK_ROOT, DATASET = _args.config, _args.bank_root, _args.dataset


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

# (session_diversity, k1, k2, delta)
ARMS = {
    "off": (False, 20, 10, 0.30),
    "div_only": (True, 20, 10, 0.30),
    "shipped": (True, _args.k1, _args.k2, _args.delta),
}
ARM_ORDER = ["off", "div_only", "shipped"]

ds = json.load(open(DATASET))

# category -> arm -> [any, all, n]
by_cat = defaultdict(lambda: {a: [0, 0, 0] for a in ARM_ORDER})

tmp = tempfile.mkdtemp()
for ci in range(min(_args.convs, len(ds))):
    conv_id = f"locomo_{ci:04d}"
    bank_dir = os.path.join(BANK_ROOT, conv_id)
    if not os.path.isdir(bank_dir):
        print(f"[skip] no bank for {conv_id}")
        continue
    bank_src = None
    for name in sorted(os.listdir(bank_dir)):
        if name.endswith(".sqlite"):
            bank_src = os.path.join(bank_dir, name)
            break
    if bank_src is None:
        print(f"[skip] no sqlite in {bank_dir}")
        continue

    cfg = yaml.safe_load(open(CFG))
    cfg.setdefault("hyper_graph", {})["hydrate_on_load"] = True
    cfg.setdefault("retriever", {})["use_graph_rank"] = False
    cfg.setdefault("retriever", {})["session_diversity"] = False
    p = os.path.join(tmp, f"cfg_{ci}.yaml")
    yaml.safe_dump(cfg, open(p, "w"))
    d = os.path.join(tmp, f"db_{ci}")
    os.makedirs(d, exist_ok=True)
    shutil.copy2(bank_src, os.path.join(d, "asem_thg.sqlite"))

    backend = OpenAIBackend(client=StubClient(), model="stub", embedder=embedder)
    system = build_asem_thg_system(p, d, backend=backend)
    retr = system.pipeline.retriever
    M = system.pipeline.memory_bank
    retr.use_graph_rank = False
    note_sessions = load_note_sessions(bank_src)

    def sess(notes):
        return {note_sessions[n.id] for n in notes[:K] if n.id in note_sessions}

    n_q = 0
    for item in ds[ci]["qa"]:
        cat = item.get("category")
        if cat is None:
            continue
        gold = gold_sessions(item.get("evidence") or [])
        if not gold:
            continue
        q = item["question"]

        # `off` and `div_only` share ONE pool (k1=20,k2=10); only the reorder
        # differs. Retrieve it once and apply the SHIPPED `_diversify_sessions`
        # to the same list for the div arm — exact, since the method is a pure
        # reorder, and it saves a third of the run's retrievals.
        retr.session_diversity = False
        retr.k1, retr.k2, retr.delta = 20, 10, 0.30
        retr.ppr_scores = {}
        pool_default = retr.retrieve(q, M)

        retr.session_diversity = True
        retr.k1, retr.k2, retr.delta = _args.k1, _args.k2, _args.delta
        retr.ppr_scores = {}
        pool_wide = retr.retrieve(q, M)

        arm_notes = {
            "off": pool_default,
            "div_only": retr._diversify_sessions(list(pool_default)),
            "shipped": pool_wide,
        }
        for a in ARM_ORDER:
            got = sess(arm_notes[a])
            by_cat[cat][a][0] += 1 if (got & gold) else 0
            by_cat[cat][a][1] += 1 if (gold <= got) else 0
            by_cat[cat][a][2] += 1
        n_q += 1

    try:
        M.close()
    except Exception:
        pass
    # Ten sequential banks on a 7.9GB box: drop the per-conversation objects and
    # collect, so a native FAISS index that `close()` does not release cannot
    # accumulate across iterations.
    del system, retr, M, backend, note_sessions
    gc.collect()
    print(f"[done] {conv_id}: {n_q} scored questions", flush=True)
    if _args.json_out:
        with open(_args.json_out, "w", encoding="utf-8") as fh:
            json.dump({str(c): {a: by_cat[c][a] for a in ARM_ORDER}
                       for c in sorted(by_cat)}, fh, indent=2)

print()
print(f"FULL LoCoMo ({_args.convs} conversations)  K={K}  "
      f"shipped pool k1={_args.k1} k2={_args.k2} delta={_args.delta}")
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
print("  The multi-hop row (cat 1) now has n=282, not the conv0 n=32.")
