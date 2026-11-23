# CARE Bridge v2 versus Qwen3-8B NoMemory and FullContext-8K

All methods answered the same 1,986 LoCoMo questions with pinned Qwen3-8B. The NoMemory prompt follows the prior best-guess baseline. FullContext-8K uses the first five dated turns plus the longest recent suffix that fits the 8,192-token model window; it is not an untruncated full-history result.

| Arm | Group | n | Baseline LoCoMo | CARE LoCoMo | Delta | Baseline BERTScore raw F1 | CARE BERTScore raw F1 | Delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| no_memory | all | 1986 | 0.0774 | 0.5256 | +0.4482 | 0.8776 | 0.9088 | +0.0312 |
| no_memory | answerable | 1540 | 0.0998 | 0.4648 | +0.3650 | 0.8782 | 0.9149 | +0.0367 |
| no_memory | adversarial | 446 | 0.0000 | 0.7354 | +0.7354 | 0.8755 | 0.8877 | +0.0123 |
| full_context_8k | all | 1986 | 0.1961 | 0.5256 | +0.3294 | 0.8912 | 0.9088 | +0.0175 |
| full_context_8k | answerable | 1540 | 0.2406 | 0.4648 | +0.2242 | 0.8949 | 0.9149 | +0.0200 |
| full_context_8k | adversarial | 446 | 0.0426 | 0.7354 | +0.6928 | 0.8786 | 0.8877 | +0.0091 |

## FullContext-8K coverage

1515/5882 conversation turns retained across 10 conversations (25.8%); all conversations were truncated. Per-conversation range: 21.9%–43.9%.

## Per-category LoCoMo score

| Category | n | CARE | NoMemory | FullContext-8K |
|---:|---:|---:|---:|---:|
| 1 | 282 | 0.2609 | 0.0966 | 0.2353 |
| 2 | 321 | 0.4415 | 0.0336 | 0.2367 |
| 3 | 96 | 0.1647 | 0.1256 | 0.2014 |
| 4 | 841 | 0.5763 | 0.1232 | 0.2484 |
| 5 | 446 | 0.7354 | 0.0000 | 0.0426 |

## Interpretation

- The LoCoMo overall score combines answerable token F1 and binary category-5 refusal accuracy; compare those groups separately.
- BERTScore is semantic similarity, not factual correctness or citation support.
- NoMemory's old-style best-guess prompt and CARE's refusal contract differ. A category-5 gap partly reflects that contract, not only memory ability.
- FullContext-8K omits most of each long conversation. This experiment does not establish how CARE compares with a truly untruncated long-context model.

Baseline merge SHA256: `1e6d172c06c28baad050ba63c21c1477e9debe4cbd5cf2d092d018b57654fb20`. Baseline BERTScore SHA256: `65a0a0ad22bd4d0a7dbb0524500266fbd4b355c2e40abfe7fbd88037d7d955f7`.
