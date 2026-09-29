"""Dump the adversarial (and a few open_domain) Q/A pairs to see why EM=0."""
import json
from collections import Counter

PRED = "data/benchmarks/results/static/locomo10/preds/ds_thg__deepseek_v4_flash__ASEM-THG.jsonl"

rows = [json.loads(l) for l in open(PRED, encoding="utf-8") if l.strip()]
print("total rows:", len(rows))
print("keys:", list(rows[0].keys()))


def get(r, *names):
    for n in names:
        if n in r and r[n] is not None:
            return r[n]
    return None


cat_key = "category" if "category" in rows[0] else ("cat" if "cat" in rows[0] else None)
print("category key:", cat_key)
if cat_key:
    print("category counts:", Counter(r.get(cat_key) for r in rows))


def is_idk(text):
    t = (text or "").strip().lower().rstrip(".")
    return t in {"i don't know", "i dont know", "i do not know", "unknown",
                 "not mentioned", "no information", "i don't know.",
                 "cannot answer", "no mentioned"}


for cat in ("adversarial", "open_domain"):
    sub = [r for r in rows if r.get(cat_key) == cat]
    print("\n" + "=" * 80)
    print(f"CATEGORY: {cat}  (n={len(sub)})")
    n_idk = sum(1 for r in sub if is_idk(get(r, "prediction", "pred")))
    print(f"refusals (I don't know): {n_idk}/{len(sub)} = {n_idk / max(1, len(sub)):.1%}")
    print("=" * 80)
    for i, r in enumerate(sub):
        q = get(r, "question", "query")
        g = get(r, "gold", "answer", "ground_truth", "gt")
        p = get(r, "prediction", "pred")
        em = get(r, "em", "exact_match")
        print(f"\n[{i}] EM={em}")
        print(f"  Q: {q}")
        print(f"  G: {g}")
        print(f"  P: {(p or '')[:400]}")
