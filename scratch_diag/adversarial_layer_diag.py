"""Layer-by-layer diagnosis of the adversarial (cat-5) failures.

For each of the 47 cat-5 questions, answer three separate questions:

  L1 INGESTION  : does the bank contain a note whose text carries the FACT
                  (the distractor string), and does that note attribute it to
                  the RIGHT speaker?
  L2 RETRIEVAL  : does that note appear in the top-k retrieved for this query?
  L3 ANSWERING  : given the retrieved context, does the model refuse (correct)
                  or fall into the trap?

This separates "the graph never stored it" from "the retriever never found it"
from "the model saw it and still got it wrong".
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())

DATA = "datasets/locomo/locomo10.json"
PRED = "data/benchmarks/results/static/locomo10/preds/ds_thg__deepseek_v4_flash__ASEM-THG.jsonl"

_REFUSAL_RE = re.compile(
    r"not mentioned|no information|i don'?t know|i do not know|"
    r"cannot (?:be )?(?:answer|determine|find|tell)|does ?n[o']t (?:mention|say|state)|"
    r"no memory|nothing (?:in the|is )",
    re.I,
)


def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))


def main() -> None:
    data = json.load(open(DATA, encoding="utf-8"))
    conv = data[0]["conversation"]
    rows = [json.loads(l) for l in open(PRED, encoding="utf-8") if l.strip()]
    cat5 = [r for r in rows if r.get("category") == 5]

    # ---- Load the frozen bank ------------------------------------------
    from eval.phase_runner import bank_file

    bank_path = bank_file("static/memory_banks/locomo10", "ds_thg", "ASEM-THG", "locomo_0000")
    print(f"bank: {bank_path}  exists={os.path.exists(bank_path)}")

    from asem.memory_bank import MemoryBank

    mb = MemoryBank(db_path=bank_path)
    notes = mb.list_notes() if hasattr(mb, "list_notes") else mb.get_all_notes()
    print(f"notes in bank: {len(notes)}")

    # ---- Build a text index over the notes -----------------------------
    note_texts = []
    for n in notes:
        desc = getattr(n, "description", "") or ""
        content = getattr(n, "content", "") or ""
        speaker = getattr(n, "speaker", "") or ""
        note_texts.append({
            "id": getattr(n, "id", ""),
            "text": norm(desc + " " + content),
            "speaker": speaker,
            "raw": (desc or content)[:300],
        })

    # ---- For each cat-5 item, locate the evidence dialogue --------------
    print("\n" + "=" * 100)
    print("LAYER DIAGNOSIS — 47 adversarial questions")
    print("=" * 100)

    stats = {"fact_in_bank": 0, "fact_missing": 0, "refused": 0, "trapped": 0}
    rows_out = []

    for r in cat5:
        idx = r["idx"]
        q = data[0]["qa"][idx]
        distractor = q.get("adversarial_answer", "")
        pred = r.get("pred") or ""
        refused = bool(_REFUSAL_RE.search(pred))

        # L1: is the distractor's content present anywhere in the bank?
        d_tokens = set(norm(distractor).split()) - {"the", "a", "an", "and", "of", "to", "in", "for"}
        hits = []
        for nt in note_texts:
            if not d_tokens:
                continue
            overlap = len(d_tokens & set(nt["text"].split()))
            if overlap >= max(2, int(0.6 * len(d_tokens))):
                hits.append(nt)
        in_bank = bool(hits)

        # L2: was the fact-bearing note retrieved? (approximate: check if any
        # hit's id appears in the retrieved context — we don't have it here, so
        # we mark it as "unknown" and rely on L1 + L3.)
        if in_bank:
            stats["fact_in_bank"] += 1
        else:
            stats["fact_missing"] += 1
        if refused:
            stats["refused"] += 1
        else:
            stats["trapped"] += 1

        rows_out.append({
            "idx": idx,
            "q": q["question"],
            "distractor": distractor,
            "in_bank": in_bank,
            "n_hits": len(hits),
            "hit_speakers": sorted({h["speaker"] for h in hits if h["speaker"]})[:3],
            "refused": refused,
            "pred": pred[:220],
        })

    print(f"\nL1 INGESTION : fact text present in bank : {stats['fact_in_bank']}/{len(cat5)}"
          f"   MISSING: {stats['fact_missing']}")
    print(f"L3 ANSWERING  : refused (correct)         : {stats['refused']}/{len(cat5)}"
          f"   TRAPPED: {stats['trapped']}")

    print("\n--- Cases where the fact is MISSING from the bank (ingestion loss) ---")
    for row in rows_out:
        if not row["in_bank"]:
            print(f"  [idx {row['idx']}] {row['q'][:70]}")
            print(f"      distractor: {row['distractor'][:80]}")

    print("\n--- Cases where the fact IS in the bank but the model still failed ---")
    for row in rows_out:
        if row["in_bank"] and not row["refused"]:
            print(f"  [idx {row['idx']}] {row['q'][:70]}")
            print(f"      distractor: {row['distractor'][:80]}")
            print(f"      speakers on matching notes: {row['hit_speakers']}")
            print(f"      predicted : {row['pred'][:150]}")

    with open("scratch_diag/adversarial_layer_diag.json", "w", encoding="utf-8") as fh:
        json.dump(rows_out, fh, indent=2, ensure_ascii=False)
    print("\n-> scratch_diag/adversarial_layer_diag.json")


if __name__ == "__main__":
    main()
