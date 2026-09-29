"""Verify adversarial scoring logic in eval/metrics.py."""
import sys
sys.path.insert(0, '.')

from eval.metrics import (
    is_refusal, adversarial_em, adversarial_em_loose,
    adversarial_token_f1, adversarial_rouge_l,
    compute_metrics, per_category_metrics,
)

# Test is_refusal
assert is_refusal('Not mentioned') == True, 'Not mentioned should be refusal'
assert is_refusal('No information available') == True
assert is_refusal('There is no information about that in the conversation.') == True
assert is_refusal('I dont know') == True
assert is_refusal('The answer is Paris') == False
assert is_refusal('No') == False, 'bare No must NOT be refusal (denial trap)'
assert is_refusal('Yes') == False
assert is_refusal('') == False
assert is_refusal(None) == False
print('is_refusal: OK')

# Test adversarial_em
assert adversarial_em('Not mentioned', 'Not mentioned') == 1.0
assert adversarial_em('No information available', 'Not mentioned') == 1.0
assert adversarial_em('Paris', 'Not mentioned') == 0.0
assert adversarial_em('No', 'No') == 1.0  # denial trap: exact match
assert adversarial_em('Not mentioned', 'No') == 1.0  # refusal still scores 1
print('adversarial_em: OK')

# Test compute_metrics with adversarial_flags
preds = ['Not mentioned', 'Paris', 'Not mentioned', 'London']
refs  = ['Not mentioned', 'Paris', 'Not mentioned', 'London']
adv   = [True, False, True, False]
# All correct: 2 refusals (cat5) + 2 exact matches (normal)
r = compute_metrics(preds, refs, ['em'], adversarial_flags=adv)
assert r['em'] == 1.0, f"expected 1.0 got {r['em']}"

# One wrong: cat5 non-refusal
preds2 = ['Paris', 'Paris', 'Not mentioned', 'London']
r2 = compute_metrics(preds2, refs, ['em'], adversarial_flags=adv)
assert r2['em'] == 0.75, f"expected 0.75 got {r2['em']}"
print('compute_metrics with adversarial_flags: OK')

# Test per_category_metrics
preds3 = ['Not mentioned', 'Paris', 'Not mentioned', 'London']
refs3  = ['Not mentioned', 'Paris', 'Not mentioned', 'London']
cats   = ['adversarial', 'single_hop', 'adversarial', 'single_hop']
adv3   = [True, False, True, False]
pc = per_category_metrics(preds3, refs3, cats, ['em'], adversarial_flags=adv3)
assert pc['adversarial']['em'] == 1.0
assert pc['single_hop']['em'] == 1.0
print('per_category_metrics with adversarial_flags: OK')

# Test without adversarial_flags (backward compat)
r3 = compute_metrics(['Paris', 'Paris'], ['Paris', 'London'], ['em'])
assert r3['em'] == 0.5
print('backward compat (no adversarial_flags): OK')

# Test adversarial_token_f1 and adversarial_rouge_l
assert adversarial_token_f1('Not mentioned', 'Not mentioned') == 1.0
assert adversarial_rouge_l('Not mentioned', 'Not mentioned') == 1.0
assert adversarial_token_f1('Paris', 'Not mentioned') == 0.0
assert adversarial_rouge_l('Paris', 'Not mentioned') == 0.0
print('adversarial_token_f1/rouge_l: OK')

print('\nAll checks passed!')
