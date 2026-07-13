"""Logging and experiment tracking helpers with loguru."""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional

from loguru import logger


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure loguru with a consistent format for console and optional file output.

    Args:
        level: One of TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL.
        log_file: Optional path to a file for persistent logs.
    """
    # Remove default handler
    logger.remove()

    # Console sink — colorized, compact
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[stage]: <12}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File sink — machine-readable, full timestamps, no color
    if log_file:
        logger.add(
            log_file,
            level="DEBUG",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                "{extra[stage]: <12} | {message}"
            ),
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            enqueue=True,  # non-blocking file writes
        )

    # Bind a default stage so format strings never fail on missing extra
    logger.configure(extra={"stage": "init"})


def get_logger(stage: str = "asem"):
    """Return a logger instance bound to a specific pipeline stage name.

    Usage:
        log = get_logger("S2.memory_manager")
        log.info("Selected op ADD for note {}", note_id)
    """
    return logger.bind(stage=stage)


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

