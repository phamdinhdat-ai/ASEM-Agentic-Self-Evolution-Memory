# CARE full LoCoMo: final results and post-hoc BERTScore

All five Kaggle shards completed. The sealed merge covers 10 conversations, 1,986 questions and 3,972 question-arm predictions, with zero inference errors.
This is a text plus image-caption adapter, not an official multimodal run.

BERTScore uses roberta-large at pinned revision `722cf37b1afa9454edce342e7895e588b6ff1d59`, layer 17, no IDF, slow tokenizer and the official bert-score package. It is a post-hoc semantic similarity metric; it does not verify factual support or citations. Category 5 uses the standard `Not mentioned` reference and is reported separately. Category 3 uses gold text before the semicolon, as in the LoCoMo scorer.

| Group | n | Arm | LoCoMo score | BERTScore raw F1 | BERTScore rescaled F1 |
|---|---:|---|---:|---:|---:|
| all | 1986 | bridge_v2 | 0.5256 | 0.9088 | 0.4595 |
| all | 1986 | relation_temporal | 0.5251 | 0.9083 | 0.4565 |
| answerable | 1540 | bridge_v2 | 0.4648 | 0.9149 | 0.4956 |
| answerable | 1540 | relation_temporal | 0.4603 | 0.9141 | 0.4911 |
| adversarial | 446 | bridge_v2 | 0.7354 | 0.8877 | 0.3347 |
| adversarial | 446 | relation_temporal | 0.7489 | 0.8881 | 0.3371 |

## By category

| Category | n | Arm | LoCoMo score | BERTScore raw F1 | BERTScore rescaled F1 |
|---|---:|---|---:|---:|---:|
| 1 | 282 | bridge_v2 | 0.2609 | 0.8843 | 0.3147 |
| 1 | 282 | relation_temporal | 0.2609 | 0.8843 | 0.3147 |
| 2 | 321 | bridge_v2 | 0.4415 | 0.9152 | 0.4977 |
| 2 | 321 | relation_temporal | 0.4384 | 0.9148 | 0.4950 |
| 3 | 96 | bridge_v2 | 0.1647 | 0.8650 | 0.1998 |
| 3 | 96 | relation_temporal | 0.1647 | 0.8650 | 0.1998 |
| 4 | 841 | bridge_v2 | 0.5763 | 0.9307 | 0.5892 |
| 4 | 841 | relation_temporal | 0.5692 | 0.9295 | 0.5820 |
| 5 | 446 | bridge_v2 | 0.7354 | 0.8877 | 0.3347 |
| 5 | 446 | relation_temporal | 0.7489 | 0.8881 | 0.3371 |

## Paired BERTScore raw F1, Relation+Temporal minus Bridge v2

| Group | n | Mean delta | Conversation bootstrap 95% | Wins | Losses | Ties |
|---|---:|---:|---|---:|---:|---:|
| all | 1986 | -0.00050 | [-0.00105, -0.00006] | 5 | 10 | 1971 |
| answerable | 1540 | -0.00076 | [-0.00149, -0.00019] | 0 | 9 | 1531 |
| adversarial | 446 | +0.00040 | [+0.00014, +0.00065] | 5 | 1 | 440 |

Answers differ for 15 of 1,986 paired questions.
The LoCoMo aggregate mixes token F1 for categories 1-4 and binary refusal accuracy for category 5. Interpret answerable F1 and adversarial accuracy separately.

Metric environment: {'bert-score': '0.3.13', 'transformers': '5.8.0', 'torch': '2.11.0+cu130', 'huggingface-hub': '1.14.0'}.
Input payload SHA256: `4312add18aca7372ae3ff2732c00e729a254cb3e468b6b4b0eaa0bfd18917a23`.
Kaggle output SHA256: `62ee515b7ae4cd1a8bc83a8ba8eeb6f5e7dc3c68dfc50ccdd91fc30f5e1c1e24`.
