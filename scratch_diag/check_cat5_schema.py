"""Check: do ALL category-5 items lack 'answer'? What does the official LoCoMo
eval convention do with them?"""
import json

data = json.load(open("datasets/locomo/locomo10.json", encoding="utf-8"))

stats = {}
for ci, conv in enumerate(data):
    for q in conv["qa"]:
        cat = q.get("category")
        key = (cat, frozenset(q.keys()))
        stats[key] = stats.get(key, 0) + 1

print("(category, key-set) -> count")
for k, v in sorted(stats.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
    print(f"  cat={k[0]}  keys={sorted(k[1])}  n={v}")

print("\n--- 5 sample category-5 items ---")
n = 0
for conv in data:
    for q in conv["qa"]:
        if q.get("category") == 5:
            print(json.dumps(q, ensure_ascii=False))
            n += 1
            if n >= 5:
                break
    if n >= 5:
        break

# how many cat-5 have evidence?
tot = ev = 0
for conv in data:
    for q in conv["qa"]:
        if q.get("category") == 5:
            tot += 1
            if q.get("evidence"):
                ev += 1
print(f"\ncat-5 with evidence: {ev}/{tot}")
