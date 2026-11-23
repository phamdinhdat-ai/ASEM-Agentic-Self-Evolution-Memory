# Retrieval: BM25 vs Qwen3-Embedding-0.6B vs hybrid RRF (17/11/2026)

Embedding: Kaggle GPU kernel `asem-care-embed-qwen3-0-6b` (Qwen3-Embedding-0.6B, revision 97b0c614, fp16, last-token pooling, instruction-prefixed queries, docs ≤ 1,024 tokens; 0/5,882 LoCoMo and 1/10,960 LongMemEval turns truncated). Sanity: relevant 0.79 vs irrelevant 0.30. Embedded text matches the local turn builders (SHA-256 identical).
Evaluation: `python -m scripts.eval_embed_retrieval` (CPU, no LLM). Gold evidence used only for scoring. Full numbers: `CARE_embed_retrieval_qwen3_0p6b_20261117.json`.

| Recall of annotated evidence | BM25@8 | BM25-stem@8 | Dense@8 | Hybrid@8 | BM25@18 | Dense@18 | Hybrid@18 |
|---|---|---|---|---|---|---|---|
| LoCoMo (1,531 answerable with evidence) | 0.539 | 0.578 | **0.617** | 0.612 | 0.624 | **0.725** | 0.712 |
| LongMemEval-oracle turn-level (470) | 0.837 | 0.846 | **0.926** | 0.911 | 0.941 | **0.988** | 0.979 |

- Dense@8 beats BM25@8 in **10/10 LoCoMo conversations** (per-conversation gain +0.006 to +0.151).
- Largest gains: LoCoMo category 1 (list-style answers) 0.225 → 0.378; LongMemEval multi-session 0.670 → 0.858.
- Hybrid RRF does not beat dense alone overall; it is best only on LoCoMo category 4 (0.695 vs 0.671).

Implication for the CARE paper: its "BM25 beats dense" result used MiniLM-L6-v2 (paper: source recall 0.5353 vs 0.4576 at top-8). With Qwen3-Embedding-0.6B the ranking reverses at the retrieval level. This is recall only; the end-to-end QA comparison with the same reader has not been run. The 30-question LongMemEval probe showed higher recall alone did not raise multi-session F1, so recall gains may not carry to QA.
