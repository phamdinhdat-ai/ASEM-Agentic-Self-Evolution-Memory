"""Correct layer diagnosis: for adversarial questions, the distractor is the WRONG
answer, so "missing from bank" is EXPECTED. What matters is:

  L1: does the bank contain the TRUE fact (the one that lets the model see the
      attribution is wrong)?
  L2: does retrieval surface it?
  L3: does the model refuse?

For the 45 attribution traps (no 'answer' key), the true fact is the evidence
dialogue itself. For the 2 denial traps, the true answer is "No".
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

    from eval.phase_runner import bank_file
    from asem.memory_bank import MemoryBank

    bank_path = bank_file("static/memory_banks/locomo10", "ds_thg", "ASEM-THG", "locomo_0000")
    mb = MemoryBank(db_path=bank_path)
    notes = mb.list_notes() if hasattr(mb, "list_notes") else mb.get_all_notes()

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

    print(f"notes in bank: {len(notes)}")
    print("\n" + "=" * 100)
    print("CORRECTED LAYER DIAGNOSIS — 47 adversarial questions")
    print("  For attribution traps, the distractor SHOULD be absent; the evidence dialogue is the truth.")
    print("=" * 100)

    # For each cat-5 item, check whether the EVIDENCE dialogue's text is in the bank.
    for r in cat5:
        idx = r["idx"]
        q = data[0]["qa"][idx]
        pred = r.get("pred") or ""
        refused = bool(_REFUSAL_RE.search(pred))
        ev_ids = q.get("evidence", [])
        # Collect the evidence dialogue texts
        ev_texts = []
        for ev in ev_ids:
            sess, dia = ev.split(":")
            key = f"session_{sess[1:]}"
            try:
                d = conv[key][int(dia)]
                ev_texts.append(d.get("text", ""))
            except Exception:
                pass
        ev_norm = norm(" ".join(ev_texts))
        ev_tokens = set(ev_norm.split()) - {"the", "a", "an", "and", "of", "to", "in", "for", "is", "are", "was", "were", "it", "that", "this", "with", "on", "at", "as", "be", "have", "has", "had", "you", "your", "we", "they", "she", "he", "i", "me", "my", "her", "his", "so", "but", "or", "if", "not", "no", "yes", "just", "really", "very", "much", "more", "also", "even", "still", "now", "then", "here", "there", "when", "where", "what", "who", "why", "how"}
        # Check if the evidence text's key tokens appear in any note
        hits = []
        for nt in note_texts:
            if not ev_tokens:
                continue
            overlap = len(ev_tokens & set(nt["text"].split()))
            if overlap >= max(2, int(0.4 * len(ev_tokens))):
                hits.append(nt)

        status = "REFUSED" if refused else "TRAPPED"
        ev_short = " ".join(ev_texts)[:80].replace("\n", " ")
        print(f"\n[idx {idx:3d}] {status:7s}  ev={ev_ids}  hits={len(hits)}  Q: {q['question'][:60]}")
        print(f"         evidence: {ev_short}")
        if hits:
            print(f"         bank hit : {hits[0]['raw'][:120]}  (speaker={hits[0]['speaker']})")
        else:
            print(f"         bank hit : NONE — evidence dialogue not captured as a note")
        print(f"         predicted: {pred[:150]}")


if __name__ == "__main__":
    main()
