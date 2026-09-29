"""Re-score category 5 (adversarial) under the OFFICIAL LoCoMo protocol.

Why this exists
---------------
Our harness (scripts/run_locomo10_experiments.py:convert_locomo10_to_eval) does:

    if category == 5:
        gold_answer = qa.get("adversarial_answer")

and then scores EM(pred, ref) as usual. But `adversarial_answer` is NOT the
correct answer to the question -- it is the *distractor*: the plausible answer
that a naive system produces by grabbing the nearby text, which attributes a
fact to the WRONG person.

Evidence (conv-0):
  Q  : What did Caroline realize after her charity race?   [D2:3]
  ev : Caroline: "...Great that you're prioritizing self-care."
  adv: "self-care is important"
  -> The realization was MELANIE's, not Caroline's. Correct answer = "not mentioned".

The official evaluator (snap-research/locomo, task_eval/evaluation.py) therefore
scores cat-5 as a REFUSAL test, never as string match:

    elif line['category'] in [5]:
        if 'no information available' in output.lower() or 'not mentioned' in output.lower():
            all_ems.append(1)
        else:
            all_ems.append(0)

and frames the question as a forced 2-way choice between
"Not mentioned in the conversation" and the gold answer.

Consequence: our current cat-5 number of 0.0 EM means "the system never emitted
the distractor verbatim", i.e. it partly REWARDED falling into the trap.
"""
import json
import re
from collections import Counter

PRED = "data/benchmarks/results/static/locomo10/preds/ds_thg__deepseek_v4_flash__ASEM-THG.jsonl"
DATA = "datasets/locomo/locomo10.json"

# Refusal markers: the official two, plus the phrasings our prompts produce.
REFUSAL_PATTERNS = [
    r"not mentioned",
    r"no information",
    r"i don'?t know",
    r"i do not know",
    r"cannot (?:be )?(?:answer|determine|find|tell)",
    r"can'?t (?:answer|determine|tell)",
    r"does ?n[o']t (?:mention|say|state|specify)",
    r"nothing (?:in the|is )",
    r"no memory",
    r"unclear",
    r"\bunspecified\b",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.I)


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text or ""))


def main() -> None:
    data = json.load(open(DATA, encoding="utf-8"))
    rows = [json.loads(l) for l in open(PRED, encoding="utf-8") if l.strip()]

    # Map flat prediction index -> the original cat-5 QA object.
    conv0_qa = data[0]["qa"]
    adv_items = [
        (i, q) for i, q in enumerate(conv0_qa) if q.get("category") == 5
    ]
    cat5_rows = [r for r in rows if r.get("category") == 5]

    print(f"cat-5 predictions: {len(cat5_rows)}   cat-5 QA in dataset: {len(adv_items)}")
    print()

    n_refused = n_binary = 0
    official_hits = 0
    old_em = 0.0
    breakdown = Counter()
    mistakes = []

    for r in cat5_rows:
        idx = r["idx"]
        q = conv0_qa[idx]
        pred = r.get("pred") or ""
        true_answer = q.get("answer")  # present on only 2/446 items

        if true_answer is not None:
            # Denial trap: the real answer is a short string ("No").
            from eval.metrics import exact_match, em_loose

            hit = exact_match(pred, true_answer) == 1.0 or em_loose(pred, true_answer) == 1.0
            kind = "binary"
            n_binary += 1
        else:
            # Attribution trap: the only correct behaviour is to refuse.
            hit = is_refusal(pred)
            kind = "refusal"
            if is_refusal(pred):
                n_refused += 1
            else:
                mistakes.append((idx, q["question"], q.get("adversarial_answer", ""), pred))

        if hit:
            official_hits += 1
        else:
            breakdown[kind] += 1
        old_em += r.get("em") or 0.0

    n = len(cat5_rows)
    print("=" * 78)
    print("CATEGORY 5 — ASEM-THG, two scorings side by side")
    print("=" * 78)
    print(f"  n                      : {n}")
    print(f"  refusals emitted       : {n_refused} ({n_refused / n:.0%})")
    print(f"  binary/denial items    : {n_binary}")
    print()
    print(f"  OLD scoring  EM(pred, adversarial_answer) = {old_em / n:.4f}   <-- rewards the trap")
    print(f"  OFFICIAL     refusal-based accuracy        = {official_hits / n:.4f}")
    print()
    print(f"  remaining misses: {sum(breakdown.values())}  {dict(breakdown)}")
    print()

    print("-" * 78)
    print("Remaining misses (refused nothing AND did not give the true answer):")
    print("-" * 78)
    for idx, question, adv, pred in mistakes[:15]:
        print(f"\n[idx {idx}] {question}")
        print(f"   distractor : {adv[:90]}")
        print(f"   predicted  : {pred[:200]}")


if __name__ == "__main__":
    main()
