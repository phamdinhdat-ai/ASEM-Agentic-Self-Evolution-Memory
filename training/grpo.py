"""GRPO training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    beta: float = 0.1
    group_size: int = 8


class GRPOTrainerWrapper:
    """Thin wrapper around trl.GRPOTrainer when available."""

    def __init__(self, config: GRPOConfig, **kwargs: Any) -> None:
        self.config = config
        self.kwargs = kwargs
        self._trainer = self._init_trainer()

    def _init_trainer(self) -> Any:
        try:
            from trl import GRPOTrainer
        except ImportError as exc:
            raise ImportError("trl is required for GRPO training") from exc

        return GRPOTrainer(**self.kwargs)

    def train(self) -> Dict[str, Any]:
        return self._trainer.train()
