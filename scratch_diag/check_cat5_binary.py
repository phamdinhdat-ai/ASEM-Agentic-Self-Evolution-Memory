"""Check the two cat-5 items that carry a real 'answer' key (denial-type traps)."""
import json

data = json.load(open("datasets/locomo/locomo10.json", encoding="utf-8"))
conv = data[0]["conversation"]

for ci, qi, q in [(0, 167, None), (0, 178, None)]:
    q = data[ci]["qa"][qi]
    print(f"=== conv={ci} qa_idx={qi} cat={q['category']} ===")
    print("Q :", q["question"])
    print("answer            :", q.get("answer"))
    print("adversarial_answer:", q.get("adversarial_answer"))
    for ev in q.get("evidence", []):
        sess, dia = ev.split(":")
        key = f"session_{sess[1:]}"
        d = conv[key][int(dia)]
        print(f"  [{ev}] {d.get('speaker')}: {d.get('text','')[:200]}")
print()

# how many cat-5 questions in conv 0 are binary (Did/Is/Does...) vs wh-?
cats = data[0]["qa"]
trap_yesno = [q for q in cats if q["category"] == 5 and q["question"].lstrip().lower().startswith(("did ", "is ", "does ", "was ", "are ", "has "))]
print(f"cat-5 yes/no-style in conv0: {len(trap_yesno)}/47")
for q in trap_yesno:
    print("   ", q["question"][:80], "| adv:", q.get("adversarial_answer", "")[:40], "| ans:", q.get("answer", ""))