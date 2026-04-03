"""Logging and experiment tracking helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a simple, consistent format."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def init_experiment_tracker(
    project: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Initialize Weights & Biases if available, otherwise no-op."""

    try:
        import wandb
    except ImportError:
        return None

    return wandb.init(project=project, config=config or {})
