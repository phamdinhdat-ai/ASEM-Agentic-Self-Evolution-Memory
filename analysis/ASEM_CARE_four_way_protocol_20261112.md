# CARE four-way LoCoMo experiment protocol

## Fixed inputs and scoring

- Preserve the completed CARE Bridge v2 results in `build/care/relation_full_merged.json` and `build/care/relation_full_bertscore.json`.
- Evaluate every arm on the same locked `datasets/locomo/locomo10.json` (SHA256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`), all 1,986 questions from 10 conversations. Five shards contain whole conversations.
- Answer model: `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`, served with vLLM on Kaggle T4 x2; 8,192-token window, maximum 224 answer tokens, temperature 0, thinking disabled.
- LoCoMo score: token F1 for categories 1–4; category 5 is binary accuracy for refusal wording. Calculate answerable and adversarial groups separately.
- BERTScore: `roberta-large@722cf37b1afa9454edce342e7895e588b6ff1d59`, layer 17, no IDF, raw and baseline-rescaled F1. The same posthoc scorer and references are used for all arms.
- Gold answers and evidence IDs are unavailable to all inference prompts and retrieval. They are used only after each prediction for scoring and diagnostics.

## Arms

| Arm | Context construction | Answer calls |
|---|---|---:|
| CARE Bridge v2 | Original turns, source-first candidate ranking, citation checks, optional evidence repair | 1 or more |
| NoMemory | Question only; prior best-guess baseline prompt | 1 |
| FullContext-8K | Original dated turns: first 5 plus longest recent suffix fitting 8K | 1 |
| SimRetrieval | MiniLM-L6-v2 cosine top-18 original dated turns, 10,000-character evidence cap | 1 |

The SimRetrieval embedder is `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Turn and query vectors use 256-token truncation, attention-mask mean pooling and L2 normalization. Retrieval uses cosine similarity with stable turn-order tie breaking. The answer model sees original turn text, speaker and date, plus available image captions; the embedder does not see gold.

## Interpretation limits

FullContext-8K is a context-limited baseline because LoCoMo conversations exceed 8K. The report will record how much history it retains. Prompts differ by arm, especially NoMemory's best-guess instruction versus CARE's refusal rule; the comparison measures complete methods and cannot isolate memory quality alone. BERTScore is semantic overlap rather than factual correctness.

## Execution

The existing FullContext-8K and NoMemory Kaggle scheduler finishes first, followed by its BERTScore job. The SimRetrieval scheduler then runs a 4-question pilot and five full shards with at most two active Kaggle jobs. Its finalizer runs BERTScore and writes `analysis/ASEM_CARE_four_way_20261112.md` after validating question IDs, dataset/model revisions, score recomputation and BERTScore identity.
