# `static/` — frozen (ingest-once) memory banks

This directory holds **memory banks built once and reused by every later
evaluation run**. Nothing in here is produced during answering: the evaluation
phase copies each bank out before opening it, so the artifacts are immutable
inputs to as many retrieval backbones, sweeps, and ablations as you like.

## Layout

```
static/memory_banks/<dataset>/<tag>/
├── manifest.json                              # what was built, note counts, errors, config hash
├── banks.json                                 # {System: {conversation: {path, notes, status, bytes}}}
├── <System>/<locomo_000N>/<bank>.sqlite       # one frozen bank per system per conversation
└── ...
```

With the defaults (`--static-root static/memory_banks`, dataset derived from the
input filename) a LoCoMo-10 run with `--tag gpt54` produces:

```
static/memory_banks/locomo10/gpt54/
├── manifest.json
├── banks.json
├── FastASEM/locomo_0000/fast_asem.sqlite
├── SimRetrieval/locomo_0000/simretrieval.sqlite
└── ...
```

`<System>` uses the same bank filenames as the phase benchmark
(`BANK_FILE_NAMES` in `eval/systems.py`): `asem`, `asem_v2`, `fast_asem`,
`simretrieval`, `atomiclinking`, `rlmanageronly`, `valueretrievalonly`.

## The one rule: one embedder

Every bank stores the **embedder's vectors**. Any config used for ingestion or
for answering must therefore share one `embedder_name`
(`sentence-transformers/all-MiniLM-L6-v2`). The LLM may differ freely between
phases — that is the whole point of freezing the banks.
`scripts/validate_model_configs.py` enforces this.

## Ingest once, evaluate many

```bash
# Phase A — build the banks (idempotent; safe to re-run)
python scripts/build_static_banks.py --tag gpt54 \
    --ingest-config configs/models/gpt54_api.yaml

# Phase B — answer full QA from those banks, vs FullContext and NoMemory
python scripts/run_static_eval.py --tag gpt54 \
    --config configs/models/gpt54_api.yaml \
    --judge-config configs/models/judge_api.yaml

# Retrieval-only: swap the backbone, reuse the same frozen banks
python scripts/run_static_eval.py --tag gpt54 \
    --config configs/models/qwen3_4b_api.yaml \
    --systems FastASEM SimRetrieval FullContext

# One conversation at a time: same command, narrower scope
python scripts/run_static_eval.py --tag gpt54 \
    --config configs/models/gpt54_api.yaml \
    --conversations locomo_0003
```

`--conversations` accepts dataset ids (`locomo_0003`) or bare indices (`3`) and is
applied before `--limit`. It is purely a scope filter: answers are keyed by QA
index, so evaluating `locomo_0003` today and running the full command tomorrow
reuses today's answers and only fills in the rest.

Every result file carries three views of the same run — `overall`,
`per_category` and `per_conversation` — and the generated Markdown repeats them
as tables (per-system conversation tables plus one conversation × system matrix
per metric). `--no-per-conversation` skips the conversation block for a smaller
file.

Or just run the ready-made command list:

```bash
bash run_static_experiments.sh      # Phase A, then Phase B
```

`run_static_experiments.sh` (and `run_static_experiments.ps1` on Windows) holds
the literal commands — edit the paths/systems there and re-run it.

## Resume semantics

Both phases continue rather than restart:

| Phase | Progress key | Re-running the same command |
|-------|--------------|-----------------------------|
| ingest | `status: "ok"` per bank in `manifest.json` | skips banks recorded as finished; `--force` rebuilds |
| eval | `results/static/<dataset>/preds/<tag>__<model>__<System>.jsonl` | answers only the missing QA pairs |

Resume is per QA pair, not per run, which is what makes `--conversations` useful:
the prediction file is shared across scoped and unscoped invocations, so a
conversation-at-a-time workflow and a later full sweep accumulate into one
result set instead of overwriting each other.

**Resume granularity is one `(system, conversation)`.** A bank counts as finished
only when the manifest says so — *not* merely because the file has notes. If a run
dies mid-conversation, the half-written bank is **deleted and re-ingested from
scratch** on the next run, and reported as `re-ingested` in the summary. That
matters because ingestion is slow (a LoCoMo conversation is ~26 min for FastASEM),
and accepting a partial bank would silently freeze an incomplete memory graph.

Interrupting with Ctrl-C:

* **First Ctrl-C** — stops after the in-flight conversation finishes, saves the
  manifest, and exits 130. Re-run the same command to continue with the rest.
* **Second Ctrl-C** — aborts immediately. The manifest is still written, and the
  in-flight conversation is re-ingested next time.

A partially-ingested tag still works: systems with complete banks are evaluated
and `--allow-missing-banks` downgrades the rest to a skip.

To genuinely re-run a backbone, delete its prediction file or pass `--fresh`.
To genuinely re-ingest, pass `--force`.

## Failure handling

* An exception while answering a QA pair is recorded (`error` on the row,
  `pred=""`) and the run continues.
* While ingesting, a failure on one `(system, conversation)` is recorded in the
  manifest with its traceback; the run continues with the next one. On retry
  that unit is re-ingested (a partial bank is never treated as complete).
* A judge or BERTScore failure degrades to `None` / `0.0` and is recorded — it
  can never lose an answer.
* After 8 consecutive answer failures (configurable with
  `--abort-after-consecutive-errors`) the current **system** is aborted, not the
  whole run, so a dead endpoint cannot burn the remaining budget.
* Each system runs in its own pass by default (`--no-isolate-systems` to change
  that), so a fatal error in one system cannot discard another's saved work.
* The aggregate JSON is rewritten atomically after every conversation.

## Housekeeping

The `*.sqlite` artifacts are git-ignored (they are large, machine-local, and
fully reproducible from the manifest's config hash). `manifest.json` and
`banks.json` are kept so a tag's provenance stays reviewable.

```bash
# drop saved results for a tag (keeps the banks)
rm -f data/benchmarks/results/static/locomo10/gpt54__*
rm -f data/benchmarks/results/static/locomo10/{preds,scores,logs}/gpt54__*
# drop the frozen banks themselves
rm -rf static/memory_banks/locomo10/gpt54
```

## Adding another dataset

Point `--input` at the new file and pass `--dataset-name`; everything else is
dataset-agnostic as long as the JSON follows the LoCoMo shape read by
`eval/phase_runner.extract_sessions` (`conversation.session_<n>` plus
`session_<n>_date_time`) and the QA items produced by
`scripts/run_locomo10_experiments.convert_locomo10_to_eval`.
