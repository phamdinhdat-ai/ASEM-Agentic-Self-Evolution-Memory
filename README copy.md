# ASEM: Agentic Self-Evolving Memory

ASEM is a five-stage memory framework for LLM agents that maintains a living knowledge
network across sessions. The backbone model stays frozen; adaptation happens via the
external memory bank and utility estimates.

## Highlights

- Multi-attribute atomic notes (keywords, tags, description + embeddings)
- RL-trained memory manager (GRPO) for write operations
- Two-phase hybrid retrieval with value-aware re-ranking
- Non-parametric utility updates with EMA
- Pluggable inference backend (HuggingFace or LangChain)

## Repository Structure

```
asem/
  ase m/               Core library
  training/            GRPO training loops
  eval/                Evaluation harness + baselines
  configs/             Default hyperparameters
  data/                Prompts and benchmark assets
  scripts/             Utilities (downloads, profiling)
  tests/               Unit and integration tests
```

## Quickstart

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Download the base model (optional if already cached):

```
python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct
```

3. Run tests:

```
pytest tests/
```

4. Profile pipeline latency:

```
python scripts/profile_pipeline.py --config configs/default.yaml --turns 5
```

## Training

Memory Manager and Answer Agent training use GRPO via HuggingFace models.

- Memory Manager: see [training/train_manager.py](training/train_manager.py)
- Answer Agent: see [training/train_answer.py](training/train_answer.py)

Enable W&B logging by setting `use_wandb=True` in the training config.

## Evaluation

1. Download datasets using [data/benchmarks/download_datasets.py](data/benchmarks/download_datasets.py).
2. Run baselines + ASEM with [eval/run_full_evaluation.py](eval/run_full_evaluation.py).
3. Generate a results table with [eval/results_table.py](eval/results_table.py).

## Human Evaluation

- Build blinded annotation sets with [eval/human_eval.py](eval/human_eval.py)
- Open the UI at [data/benchmarks/human_eval/index.html](data/benchmarks/human_eval/index.html)

## Notes

- The LangGraph pipeline wiring is optional and can be added after core pipeline validation.
- Training always uses the HuggingFace backend, even if inference uses LangChain.
