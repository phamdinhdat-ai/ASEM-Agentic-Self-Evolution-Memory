# Second-backbone sensitivity: Qwen3-4B

Use the same 50-question, ten-conversation cohort frozen for the full-history
40K sensitivity study. The cohort contains five SHA-256 selected questions per
conversation with a round-robin category selection. Run unchanged CARE
retrieval, reader prompt, source-binding checks, and conditional repair with
the pinned Qwen3-4B model. Keep the 8,192-token window, temperature zero,
non-thinking mode, and 224-token answer cap. A two-question pilot precedes
the full cohort. Gold answers and annotated source IDs enter only after model
generation.

Score first-pass and conditional outputs with the same LoCoMo scorer used
for Qwen3-8B. Pair each Qwen3-4B result with the saved Qwen3-8B result on
the identical question ID. Report aggregate score, answerable score,
accepted-refusal count, repair gains and losses, calls, model tokens, and
conversation-cluster bootstrap intervals. This is a small backbone
sensitivity check; it cannot establish model-family generality or a
full-dataset superiority claim.

The pinned model is [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B).
