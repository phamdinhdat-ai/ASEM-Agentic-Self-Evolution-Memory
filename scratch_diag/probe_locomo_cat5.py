"""Understand LoCoMo category 5 (adversarial) and gather the evidence for a few
of the adversarial question indices, so we can tell whether the answer is
extractable from the conversation."""
import json

D = "datasets/locomo/locomo10.json"
data = json.load(open(D, encoding="utf-8"))
print("type:", type(data), "len:", len(data))
sample = data[0] if isinstance(data, list) else list(data.values())[0]
print("sample keys:", list(sample.keys()))

conv = data[0]
qa = conv["qa"]
print("\nnum qa:", len(qa))
print("qa[0] keys:", list(qa[0].keys()))

# show a full category-5 example
for q in qa:
    if q.get("category") == 5 or q.get("category") == "5":
        print("\nFULL ADVERSARIAL QA OBJECT:")
        print(json.dumps(q, indent=2, ensure_ascii=False)[:1500])
        break

# count by category
from collections import Counter
print("\ncat counts:", Counter(str(q.get("category")) for q in qa))
