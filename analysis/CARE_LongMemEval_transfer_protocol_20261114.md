# External transfer: LongMemEval cleaned oracle

This preregistered diagnostic uses the 500 records in the SHA-256 pinned
`longmemeval_oracle.json` file. Five disjoint SHA-256 question-ID shards contain
98, 110, 100, 97, and 95 questions. Thirty question IDs end in `_abs` and
test abstention. The original user and assistant turns, session order, and
session dates enter the CARE source adapter. The model never receives the
reference answer, `answer_session_ids`, or `has_answer` flags.

The model is the same pinned Qwen3-8B used in the LoCoMo study, with thinking
disabled, temperature zero, an 8,192-token server window, and at most 224
generated tokens per call. Each question yields a first-pass answer and the
CARE conditional result; the latter can use at most one additional read.
The first-pass result is recovered from the same physical model call rather
than generated independently.

The primary *diagnostic* is normalized token F1 on answerable questions.
Exact match and phrase-based abstention on the 30 `_abs` questions are
reported separately. Source-session recall checks whether annotated answer
sessions occur in the retrieved bundle. These metrics are **not** the
official LongMemEval LLM-judged accuracy. The cleaned oracle setting also
supplies only the annotated relevant sessions, not the full distractor
haystack. No leaderboard or state-of-the-art comparison follows from this
experiment. The purpose is to test whether CARE's refusal and repair behavior
transfers to a different conversational QA collection under a controlled
adapter.

After all five shards complete, merge only if the 500 question IDs are
distinct, the dataset checksums match, and every row has both arms. Report
paired first-versus-conditional differences with question-level bootstrap
intervals, plus answerable and abstention outcomes. Any claim about
generalization must be limited to this cleaned-oracle diagnostic.
