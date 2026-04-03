"""Training loop for the Memory Manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .grpo import GRPOConfig, GRPOTrainerWrapper
from asem.logging_utils import init_experiment_tracker, setup_logging


@dataclass
class ManagerTrainingConfig:
    """Configuration for Memory Manager training."""

    group_size: int = 8
    beta: float = 0.1
    wandb_project: str = "asem-memory-manager"
    use_wandb: bool = False


def build_training_examples(
    fact_pairs: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prepare training examples from sequential fact pairs."""

    return [pair for pair in fact_pairs]


def train_memory_manager(
    training_data: List[Dict[str, Any]],
    trainer_kwargs: Dict[str, Any],
    config: ManagerTrainingConfig,
) -> Dict[str, Any]:
    """Run GRPO training for the Memory Manager."""

    setup_logging()
    grpo_config = GRPOConfig(beta=config.beta, group_size=config.group_size)
    trainer = GRPOTrainerWrapper(config=grpo_config, **trainer_kwargs)
    run = None
    if config.use_wandb:
        run = init_experiment_tracker(config.wandb_project, config.__dict__)

    result = trainer.train()
    if run is not None:
        run.finish()
    return result
