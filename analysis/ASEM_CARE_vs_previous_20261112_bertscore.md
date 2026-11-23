# CARE Bridge v2 versus previous methods: paired BERTScore

Both sides were rescored with the same pinned roberta-large revision, layer 17, no IDF, slow tokenizer and package versions. Raw predictions and original references were retained. Category 5 uses `Not mentioned` as the refusal reference and is reported separately.

| Previous method | Group | n | Previous raw F1 | CARE raw F1 | Delta | 95% cluster bootstrap | CARE W/L/T |
|---|---|---:|---:|---:|---:|---|---|
| hybrid_attributed | all | 746 | 0.8428 | 0.9083 | +0.0655 | [+0.0565, +0.0758] | 593/147/6 |
| hybrid_attributed | answerable | 590 | 0.8353 | 0.9139 | +0.0786 | [+0.0666, +0.0926] | 462/122/6 |
| hybrid_attributed | adversarial | 156 | 0.8712 | 0.8870 | +0.0158 | [+0.0137, +0.0180] | 131/25/0 |
| graph_memory | all | 746 | 0.8310 | 0.9083 | +0.0774 | [+0.0598, +0.0963] | 575/167/4 |
| graph_memory | answerable | 590 | 0.8224 | 0.9139 | +0.0915 | [+0.0696, +0.1154] | 445/141/4 |
| graph_memory | adversarial | 156 | 0.8631 | 0.8870 | +0.0239 | [+0.0166, +0.0340] | 130/26/0 |
| bm25 | all | 1128 | 0.7979 | 0.9093 | +0.1114 | [+0.0983, +0.1359] | 826/289/13 |
| bm25 | answerable | 954 | 0.7863 | 0.9131 | +0.1268 | [+0.1109, +0.1529] | 677/264/13 |
| bm25 | adversarial | 174 | 0.8611 | 0.8882 | +0.0271 | [+0.0172, +0.0336] | 149/25/0 |

## Baseline-rescaled F1

| Previous method | Group | Previous rescaled F1 | CARE rescaled F1 |
|---|---|---:|---:|
| hybrid_attributed | all | 0.0688 | 0.4568 |
| hybrid_attributed | answerable | 0.0243 | 0.4901 |
| hybrid_attributed | adversarial | 0.2371 | 0.3305 |
| graph_memory | all | -0.0016 | 0.4568 |
| graph_memory | answerable | -0.0520 | 0.4901 |
| graph_memory | adversarial | 0.1891 | 0.3305 |
| bm25 | all | -0.1977 | 0.4625 |
| bm25 | answerable | -0.2661 | 0.4853 |
| bm25 | adversarial | 0.1772 | 0.3376 |

BERTScore is semantic text similarity and does not validate facts, entity binding or citation support. Compare the answerable group and LoCoMo category-specific score alongside it. Old graph/BM25 cohorts are incomplete; the comparison does not isolate a single architectural component.
