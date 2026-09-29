"""Resolve the semantics of LoCoMo category 5.

Official LoCoMo eval (snap-research/locomo, task_eval/evaluation.py) does:
    elif line['category'] in [5]:
        if 'no information available' in output.lower() or 'not mentioned' in output.lower():
            all_ems.append(1)
        else:
            all_ems.append(0)
and builds the question as a 2-way choice between
    'Not mentioned in the conversation'  vs  qa['answer'].
So category 5 = 'is this actually in the conversation?' -> refusal is CORRECT.

This script checks whether the evidence dialogue really supports the adversarial
answer or not.
"""
import json

data = json.load(open("datasets/locomo/locomo10.json", encoding="utf-8"))
conv = data[0]["conversation"]

for ev in ["D2:3", "D2:8", "D2:12", "D2:14", "D2:1"]:
    sess, dia = ev.split(":")
    key = f"session_{sess[1:]}"
    try:
        d = conv[key][int(dia)]
    except Exception as e:
        print(f"{ev}: ERR {e}")
        continue
    print(f"--- {ev} ({d.get('speaker')} -> {d.get('blip_caption','')}) ---")
    print("   ", d.get("text", "").replace("\n", " ")[:400])
