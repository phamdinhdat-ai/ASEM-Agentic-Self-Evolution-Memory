# CARE paper follow-up status (14 November 2026)

## Completed

- The 1,986-question matched-reader comparison is complete. CARE's sparse
  top-eight selector plus conditional repair scores 0.5256 versus 0.4994
  for dense top-eight with the same reader and repair policy. The paired
  difference is 0.0261, with a conversation-cluster 95% bootstrap interval
  of [0.0153, 0.0377].
- The saved first-pass audit shows a +0.0667 aggregate repair change for
  CARE but 46 correct refusals lost. Both results are incorporated in the
  15-page `paper_asem/asem_v2.pdf`.
- A blinded, balanced 100-answer source-support annotation queue is frozen
  in JSON and CSV. No entailment labels exist yet, so no support rate is
  claimed.
- The full-history 40K, LongMemEval cleaned-oracle, and Qwen3-4B protocols,
  runners, packaging, and local aggregation are prepared. The 4B adapter and
  server-name override passed local tests.

## GPU queue and automatic sequence

At the latest check, the 40K full-history pilot notebook
`phamdinhdatai/asem-care-full-history-40k-pilot` remains `QUEUED`. Kaggle
reports 108,000 seconds of weekly GPU quota allowed and zero seconds used or
reserved; the queue is not explained by a spent quota. The exact reason for
pending allocation is not available from the status response.

Three hidden local scheduler processes monitor the following sequential
chain, submitting one new GPU notebook at a time:

1. `build/care/full_history_40k_scheduler_status.json`: validate the 2-question
   40K pilot, run five ten-question shards, and summarize paired CARE results.
2. `build/care/longmemeval_scheduler_status.json`: after the first experiment
   terminates, run a 4-question pilot, then all 500 cleaned-oracle questions
   in five shards, and summarize first-versus-conditional results.
3. `build/care/second_backbone_4b_scheduler_status.json`: after LongMemEval
   terminates, run a 2-question Qwen3-4B pilot, then the frozen 50-question
   cohort, and pair it with saved Qwen3-8B results.

Each scheduler records terminal failures and does not submit the next shard
after a failed pilot. New quantitative claims will enter the manuscript only
after the relevant outputs are downloaded, checked, and interpreted at their
proper scope.
