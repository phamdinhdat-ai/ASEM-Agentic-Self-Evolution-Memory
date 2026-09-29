"""Dump the adversarial (and open_domain) Q/A pairs with the full answer context,
so we can see whether the failure is retrieval, prompt, or gold-answer format."""
import json
from collections import Counter, defaultdict

PRED = "data/benchmarks/results/static/locomo10/preds/ds_thg__deepseek_v4_flash__ASEM-THG.jsonl"

rows = [json.loads(l) for l in open(PRED, encoding="utf-8") if l.strip()]

# map numeric -> name from category_name
name_of = {}
for r in rows:
    if r.get("category_name"):
        name_of[r["category"]] = r["category_name"]
print("category map:", name_of)
print("counts:", Counter(name_of.get(r["category"], r["category"]) for r in rows))


def is_idk(text):
    t = (text or "").strip().lower().rstrip(".!")
    return any(t.startswith(x) for x in (
        "i don't know", "i dont know", "i do not know", "unknown",
        "not mentioned", "no information", "cannot answer",
    ))


for cat_id in (5, 3):  # adversarial, open_domain
    cat = name_of.get(cat_id, str(cat_id))
    sub = [r for r in rows if r["category"] == cat_id]
    n_idk = sum(1 for r in sub if is_idk(r.get("pred")))
    print("\n" + "=" * 80)
    print(f"CATEGORY {cat_id} = {cat}  (n={len(sub)})   refusals={n_idk} ({n_idk/max(1,len(sub)):.0%})")
    print("=" * 80)
    for r in sub:
        print(f"\n[idx {r['idx']}] em={r.get('em')} em_loose={r.get('em_loose')} rougeL={r.get('rougeL')}")
        print(f"  Q: {r['question']}")
        print(f"  G: {r['ref']}")
        print(f"  P: {(r.get('pred') or '')[:400]}")
