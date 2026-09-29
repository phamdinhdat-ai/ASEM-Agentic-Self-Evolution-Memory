"""Retrieval-layer diagnosis: for each adversarial question, is the correct
evidence note in the top-k retrieved? And is the trap distractor's note also there?
"""
from __future__ import annotations
import json, os, re, sys
sys.path.insert(0, os.getcwd())

DATA = "datasets/locomo/locomo10.json"
PRED = "data/benchmarks/results/static/locomo10/preds/ds_thg__deepseek_v4_flash__ASEM-THG.jsonl"
BANK = "static/memory_banks/locomo10/ds_thg/ASEM-THG/locomo_0000/asem_thg.sqlite"
CONFIG = "configs/locomo_openai.yaml"

_REFUSAL_RE = re.compile(
    r"not mentioned|no information|i don'?t know|i do not know|"
    r"cannot (?:be )?(?:answer|determine|find|tell)|does ?n[o']t (?:mention|say|state)|"
    r"no memory|nothing (?:in the|is )", re.I)
_STOP = set("""the a an and of to in for is are was were it that this with on at as be
have has had you your we they she he i me my her his so but or if not no yes just really
very much more also even still now then here there when where what who why how do does did
about after before over under""".split())


def norm(s): return " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))
def toks(s): return {t for t in norm(s).split() if t not in _STOP}


def main():
    data = json.load(open(DATA, encoding="utf-8"))
    conv = data[0]["conversation"]
    rows = [json.loads(l) for l in open(PRED, encoding="utf-8") if l.strip()]
    cat5 = [r for r in rows if r.get("category") == 5]

    from asem.memory_bank import MemoryBank
    from asem.retriever import HybridRetriever
    from asem.enhanced_retriever import EnhancedHybridRetriever
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import yaml

    class LocalEmbedBackend:
        def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
            self.model = SentenceTransformer(model_name)
        def embed(self, text):
            return self.model.encode(text, normalize_embeddings=True)
        def generate(self, prompt, **kw):
            return ""

    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    backend = LocalEmbedBackend()
    hp = cfg["hyperparameters"]
    retriever = EnhancedHybridRetriever(
        backend=backend, k1=hp["k1"], k2=hp["k2"], delta=hp["delta"],
        lambda_weight=hp["lambda"], max_hops=2, hop_decay=0.7, multi_hop_topn=5,
        alpha=0.35, beta=0.25, gamma=0.40,
        enable_global_semantics=True, enable_intent_q=True,
    )
    mb = MemoryBank(db_path=BANK)
    notes = mb.list_notes() if hasattr(mb, "list_notes") else mb.get_all_notes()
    print(f"notes in bank: {len(notes)}")

    # For each cat-5 question, find the best evidence note and check retrieval
    ev_in_topk = 0
    ev_in_bank = 0
    trapped_but_ev_retrieved = 0
    trapped_ev_not_retrieved = 0
    refused_ev_retrieved = 0
    refused_ev_not_retrieved = 0

    for r in cat5:
        idx = r["idx"]; q = data[0]["qa"][idx]
        pred = r.get("pred") or ""
        ref = bool(_REFUSAL_RE.search(pred))
        query = r.get("query") or q["question"]

        # Evidence dialogue text
        ev_texts = []
        for ev in q.get("evidence", []):
            sess, dia = ev.split(":")
            sess_list = conv.get(f"session_{sess[1:]}", [])
            d = sess_list[int(dia)] if int(dia) < len(sess_list) else {}
            ev_texts.append(d.get("text", ""))
        ev_tok = toks(" ".join(ev_texts))

        # Find best evidence note in bank
        best, best_ov = None, 0
        for n in notes:
            nt = toks(f"{n.c} {n.X}")
            if not ev_tok: continue
            ov = len(ev_tok & nt)
            if ov > best_ov:
                best, best_ov = n, ov
        thr = max(3, int(0.35 * len(ev_tok)))
        has_ev = best is not None and best_ov >= thr
        if has_ev:
            ev_in_bank += 1

        # Retrieve top-k for this query
        retrieved = retriever.retrieve(query, mb)
        retrieved_ids = {n.id for n in retrieved}
        ev_retrieved = has_ev and best.id in retrieved_ids

        if ev_retrieved:
            ev_in_topk += 1
        if ref and ev_retrieved:
            refused_ev_retrieved += 1
        elif ref and not ev_retrieved:
            refused_ev_not_retrieved += 1
        elif not ref and ev_retrieved:
            trapped_but_ev_retrieved += 1
        elif not ref and not ev_retrieved:
            trapped_ev_not_retrieved += 1

        status = "REFUSED" if ref else "TRAPPED"
        ev_flag = "EV_IN_TOPK" if ev_retrieved else ("EV_IN_BANK" if has_ev else "EV_MISSING")
        print(f"[{idx:3d}] {status:7s} {ev_flag:10s}  Q: {q['question'][:55]}")

    print(f"\n=== RETRIEVAL SUMMARY ===")
    print(f"Evidence note in bank: {ev_in_bank}/{len(cat5)}")
    print(f"Evidence note in top-k retrieved: {ev_in_topk}/{len(cat5)}")
    print(f"\nREFUSED + evidence retrieved: {refused_ev_retrieved}")
    print(f"REFUSED + evidence NOT retrieved: {refused_ev_not_retrieved}")
    print(f"TRAPPED + evidence retrieved: {trapped_but_ev_retrieved}")
    print(f"TRAPPED + evidence NOT retrieved: {trapped_ev_not_retrieved}")


if __name__ == "__main__":
    main()
