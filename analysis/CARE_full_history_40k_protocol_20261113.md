# Full-history reader sensitivity protocol

The complete FullContext-8K baseline in the CARE paper is window limited.
This follow-up checks the same Qwen3-8B reader contract used by CARE while
showing it **every original conversation turn** in a 40,960-token window.

The cohort is fixed before generation: five questions per each of the ten
LoCoMo conversations, chosen by SHA-256 of question ID within a round-robin
category ordering. It contains 50 questions. Selection uses neither answers
nor gold source labels. The reader makes one call, uses the CARE JSON prompt
and source-binding checks, and does not invoke repair. All turns retain their
source ID, speaker, and session date. This is a sensitivity study; it cannot
replace a complete 1,986-question full-history comparison.

Local preflight with the pinned Qwen3-8B tokenizer found 50 full prompts
between 21,409 and 40,557 tokens. Adding the 224-token answer reserve leaves
all selected prompts within the 40,960-token server window. The worker
rechecks this bound before every model call and verifies that every turn is
visible. A pilot on two long-history questions precedes the five shards.

Report the LoCoMo score, answerable score, accepted-refusal accuracy,
per-question prompt and completion tokens, and paired CARE results on the
same selected question IDs. Bootstrap whole conversations. Interpret any
result as conditional on this 50-question cohort and the enlarged window.
