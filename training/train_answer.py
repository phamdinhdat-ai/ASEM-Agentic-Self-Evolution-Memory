"""Training loop for the Answer Agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import json

from datasets import Dataset

from .grpo import GRPOConfig
from asem.logging_utils import init_experiment_tracker, setup_logging


@dataclass
class AnswerTrainingConfig:
    """Configuration for Answer Agent training."""

    model_name_or_path: str
    output_dir: str
    max_new_tokens: int = 128
    temperature: float = 0.0
    beta: float = 0.1
    group_size: int = 8
    wandb_project: str = "asem-answer-agent"
    use_wandb: bool = False


def build_answer_examples(
    triples: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prepare training examples from (query, candidates, gold_answer) triples."""

    return [dict(item) for item in triples]


def build_dataset(examples: List[Dict[str, Any]]) -> Dataset:
    return Dataset.from_list(examples)


def format_prompt(query: str, candidates: List[Dict[str, Any]]) -> str:
    payload = json.dumps(candidates)
    return (
        "You are an answer agent. Given a query and candidate notes, "
        "select the minimal subset needed and answer accurately.\n\n"
        f"Query: {query}\n"
        f"Candidates: {payload}\n\n"
        "Return ONLY JSON with keys: selected_ids (list of ids), answer (string)."
    )


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def exact_match_reward(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def _reward_fn(samples: List[Dict[str, Any]], responses: List[str], **kwargs: Any) -> List[float]:
    rewards = []
    for sample, response in zip(samples, responses):
        gold = sample.get("gold_answer", "")
        rewards.append(exact_match_reward(response, gold))
    return rewards


def train_answer_agent(
    training_data: List[Dict[str, Any]],
    config: AnswerTrainingConfig,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run GRPO training for the Answer Agent."""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOTrainer
    except ImportError as exc:
        raise ImportError("transformers and trl are required for training") from exc

    setup_logging()
    trainer_kwargs = trainer_kwargs or {}

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    dataset = build_dataset(training_data)

    def _format(sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt = format_prompt(sample["query"], sample["candidates"])
        return {
            "prompt": prompt,
            "gold_answer": sample.get("gold_answer", ""),
        }

    dataset = dataset.map(_format)

    grpo_config = GRPOConfig(beta=config.beta, group_size=config.group_size)

    run = None
    if config.use_wandb:
        run = init_experiment_tracker(config.wandb_project, config.__dict__)

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_fn=_reward_fn,
        **trainer_kwargs,
    )

    train_result = trainer.train()
    trainer.save_model(config.output_dir)

    if run is not None:
        run.finish()

    return {
        "result": train_result,
        "output_dir": config.output_dir,
    }
