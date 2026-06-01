"""
Fine-tune the ASEM Answer Agent with GRPO on LoCoMo training data.

Usage
-----
    python scripts/run_training.py \
        --train  data/training/train.jsonl \
        --val    data/training/val.jsonl \
        --config configs/locomo_0.5b.yaml \
        --output-dir checkpoints/answer_agent_0.5b \
        --epochs 3

The script loads the generated training data (from scripts/generate_training_data.py),
wires it into training/train_answer.py:train_answer_agent(), and saves the
fine-tuned checkpoint to --output-dir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import yaml


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune ASEM Answer Agent with GRPO on LoCoMo data"
    )
    parser.add_argument(
        "--train",
        default="data/training/train.jsonl",
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--val",
        default="data/training/val.jsonl",
        help="Path to validation JSONL file (used for logging only)",
    )
    parser.add_argument(
        "--config",
        default="configs/locomo_0.5b.yaml",
        help="Path to YAML config (inference + hyperparameters)",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/answer_agent_0.5b",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="GRPO group size (number of samples per prompt)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="GRPO KL penalty coefficient",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for generation during training",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        default="asem-answer-agent-locomo",
        help="W&B project name",
    )
    parser.add_argument(
        "--category",
        type=int,
        default=None,
        help="Only train on a specific QA category (1-5). Omit for all.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    print(f"Loading config from {args.config} ...")
    cfg = load_yaml(args.config)
    hf_cfg = cfg.get("inference", {}).get("huggingface", {})
    model_name = hf_cfg.get("model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")

    # ------------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------------
    print(f"Loading training data from {args.train} ...")
    training_data = load_jsonl(args.train)

    if args.category is not None:
        training_data = [ex for ex in training_data if ex.get("category") == args.category]
        print(f"  Filtered to category {args.category}: {len(training_data)} examples")
    else:
        print(f"  Loaded {len(training_data)} examples")

    # Category breakdown
    from collections import Counter
    cat_counts = Counter(ex.get("category") for ex in training_data)
    for cat in sorted(cat_counts):
        print(f"    category {cat}: {cat_counts[cat]}")

    # Load val data for reference
    print(f"Loading val data from {args.val} ...")
    val_data = load_jsonl(args.val)
    print(f"  Loaded {len(val_data)} val examples")

    # ------------------------------------------------------------------
    # Build AnswerTrainingConfig
    # ------------------------------------------------------------------
    # Import here so missing deps give a clear error message
    try:
        from training.train_answer import AnswerTrainingConfig, train_answer_agent
    except ImportError as e:
        print(f"ERROR: Could not import training module: {e}")
        print("Make sure you are running from the project root and dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    config = AnswerTrainingConfig(
        model_name_or_path=model_name,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        beta=args.beta,
        group_size=args.group_size,
        wandb_project=args.wandb_project,
        use_wandb=args.use_wandb,
    )

    print(f"\nTraining config:")
    print(f"  model:        {config.model_name_or_path}")
    print(f"  output_dir:   {config.output_dir}")
    print(f"  epochs:       {args.epochs}")
    print(f"  group_size:   {config.group_size}")
    print(f"  beta:         {config.beta}")
    print(f"  max_new_tok:  {config.max_new_tokens}")
    print(f"  use_wandb:    {config.use_wandb}")

    # ------------------------------------------------------------------
    # Trainer kwargs (passed through to trl.GRPOTrainer)
    # ------------------------------------------------------------------
    trainer_kwargs: Dict[str, Any] = {
        "num_train_epochs": args.epochs,
    }

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\nStarting GRPO training on {len(training_data)} examples ...")
    os.makedirs(args.output_dir, exist_ok=True)

    result = train_answer_agent(
        training_data=training_data,
        config=config,
        trainer_kwargs=trainer_kwargs,
    )

    print(f"\nTraining complete.")
    print(f"  Checkpoint saved to: {result['output_dir']}")

    # Save a training manifest for reproducibility
    manifest = {
        "model": model_name,
        "train_file": args.train,
        "val_file": args.val,
        "config_file": args.config,
        "n_train": len(training_data),
        "n_val": len(val_data),
        "epochs": args.epochs,
        "group_size": args.group_size,
        "beta": args.beta,
        "category_filter": args.category,
        "output_dir": args.output_dir,
    }
    manifest_path = os.path.join(args.output_dir, "training_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
