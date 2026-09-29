"""Verify category-5 semantics against official LoCoMo and our data.

Official snap-research/locomo eval (task_eval/evaluation.py, eval_question_answering):
    elif line['category'] in [5]:
        if 'no information available' in output.lower() or 'not mentioned' in output.lower():
            all_ems.append(1)
        else:
            all_ems.append(0)

And the question framer (gpt_utils.get_gpt_answers):
    qa['question'] + " Select the correct answer: (a) Not mentioned in the conversation (b) {qa['answer']}"

=> Category 5 (adversarial) is an *attribution trap*: correct behaviour is to REFUSE
   ("not mentioned"), because the premise attributes a fact to the wrong person.

Checks:
1. Do our cat-5 items carry an 'answer' key? (only 2/446 in the 10-conv set)
2. For a sample of cat-5 items, is the 'adversarial_answer' really absent from the
   evidence dialogue in the way the question implies?
"""
import json
from collections import Counter

data = json.load(open("datasets/locomo/locomo10.json", encoding="utf-8"))

# 1) which cat-5 items have 'answer'
have = []
for ci, conv in enumerate(data):
    for qi, q in enumerate(conv["qa"]):
        if q.get("category") == 5 and "answer" in q:
            have.append((ci, qi, q))
print("cat-5 items WITH 'answer':", len(have))
for ci, qi, q in have:
    print(f"  conv={ci} qa_idx={qi}")
    print("   ", json.dumps(q, ensure_ascii=False)[:400])