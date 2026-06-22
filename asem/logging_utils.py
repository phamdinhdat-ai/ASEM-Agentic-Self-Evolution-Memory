"""Logging and experiment tracking with loguru.

Provides a centralized logging setup that can be configured via:
  - Programmatic call: setup_logging(level="DEBUG")
  - Environment variable: LOG_LEVEL=DEBUG
  - YAML config: logging.level: "DEBUG"

All ASEM components should use get_logger(__name__) to obtain a
loguru logger instance with consistent formatting and context.

Usage:
    from asem.logging_utils import setup_logging, get_logger

    setup_logging(level="INFO", log_file="logs/asem.log")
    logger = get_logger(__name__)
    logger.info("Pipeline started with config: {}", config)
    logger.debug("Input content: {!r}", content)
"""

from __future__ import annotations

import os
import sys
from functools import wraps
from typing import Any, Callable, Dict, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Default format strings
# ---------------------------------------------------------------------------

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[component]:<24}</cyan> | "
    "<level>{message}</level>"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{extra[component]:<24} | "
    "{name}:{function}:{line} | "
    "{message}"
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    serialize_json: bool = False,
) -> None:
    """Configure loguru with console and optional file sinks.

    Parameters
    ----------
    level : str
        Minimum log level (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL).
        Also read from LOG_LEVEL env var (takes precedence over this argument).
    log_file : str | None
        If provided, also write logs to this file with rotation.
    rotation : str
        When to rotate the log file (size or time, e.g. "10 MB", "1 day").
    retention : str
        How long to keep rotated logs (e.g. "7 days", "1 month").
    serialize_json : bool
        If True, write structured JSON logs to file (useful for log aggregation).
    """
    # Let env var override the level argument
    env_level = os.environ.get("LOG_LEVEL", "").upper()
    if env_level:
        level = env_level

    # Remove any previously configured handlers to avoid duplicates
    logger.remove()

    # ---- Console sink (colored) ----
    logger.add(
        sys.stderr,
        format=_CONSOLE_FORMAT,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ---- File sink (optional) ----
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_format = (
            # JSON serialization for structured logging
            "{message}" if serialize_json else _FILE_FORMAT
        )
        logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="gz",
            serialize=serialize_json,
            backtrace=True,
            diagnose=True,
        )
        logger.bind(component="logging").info(
            "Log file: {} (rotation={}, retention={})", log_file, rotation, retention
        )

    logger.bind(component="logging").debug("Logging initialized at level {}", level)


def get_logger(name: str = "asem") -> Any:
    """Return a loguru logger with the component name bound as extra context.

    Parameters
    ----------
    name : str
        Component name (use __name__ in each module).

    Returns
    -------
    A loguru logger with ``component`` bound in ``extra``.
    """
    # Shorten module paths for readability in console output
    short_name = name.split(".")[-1] if "." in name else name
    return logger.bind(component=short_name)


def log_call(
    level: str = "DEBUG",
    log_args: bool = True,
    log_result: bool = False,
    max_result_len: int = 200,
) -> Callable:
    """Decorator that logs function calls with arguments and results.

    Parameters
    ----------
    level : str
        Log level for entry/exit messages.
    log_args : bool
        If True, log positional and keyword arguments.
    log_result : bool
        If True, log the return value (truncated to max_result_len).
    max_result_len : int
        Max characters of result to log.

    Usage::

        @log_call(level="DEBUG", log_result=True)
        def my_func(x, y):
            return x + y
    """

    def decorator(func: Callable) -> Callable:
        func_logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if log_args:
                args_repr = ", ".join(repr(a) for a in args)
                kwargs_repr = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
                sig = ", ".join(filter(None, [args_repr, kwargs_repr]))
                func_logger.log(level, "→ {}({})", func.__name__, sig)
            else:
                func_logger.log(level, "→ {}()", func.__name__)

            try:
                result = func(*args, **kwargs)
            except Exception:
                func_logger.exception("✗ {} failed", func.__name__)
                raise

            if log_result:
                result_str = repr(result)
                if len(result_str) > max_result_len:
                    result_str = result_str[:max_result_len] + "..."
                func_logger.log(level, "← {} → {}", func.__name__, result_str)
            else:
                func_logger.log(level, "← {} done", func.__name__)

            return result

        return wrapper

    return decorator


def log_config(config: Dict[str, Any], prefix: str = "Config") -> None:
    """Log all settings from a configuration dictionary at INFO level.

    Parameters
    ----------
    config : dict
        Configuration dictionary (e.g. loaded from YAML).
    prefix : str
        Label to use in the log header.
    """
    cfg_logger = logger.bind(component="config")

    cfg_logger.info("═" * 50)
    cfg_logger.info("{} settings:", prefix)

    def _flatten(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
        items: Dict[str, Any] = {}
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(_flatten(v, new_key))
            else:
                items[new_key] = v
        return items

    for key, value in _flatten(config).items():
        # Mask potential secrets
        if any(secret in key.lower() for secret in ("key", "secret", "token", "password")):
            value_str = "***"
        elif isinstance(value, str) and len(value) > 80:
            value_str = value[:77] + "..."
        else:
            value_str = repr(value)
        cfg_logger.info("  {} = {}", key, value_str)

    cfg_logger.info("═" * 50)


def setup_logging_from_config(config: Dict[str, Any]) -> None:
    """Configure logging from an optional 'logging' section in a YAML config.

    If no 'logging' key is present, uses defaults (INFO to stderr only).

    Parameters
    ----------
    config : dict
        Full config dict that may contain a 'logging' key.
    """
    log_cfg = config.get("logging", {})
    level = log_cfg.get("level", os.environ.get("LOG_LEVEL", "INFO"))
    log_file = log_cfg.get("log_file")
    rotation = log_cfg.get("rotation", "10 MB")
    retention = log_cfg.get("retention", "7 days")
    serialize_json = log_cfg.get("serialize_json", False)

    setup_logging(
        level=level,
        log_file=log_file,
        rotation=rotation,
        retention=retention,
        serialize_json=serialize_json,
    )


def log_error(
    logger_instance: Any,
    message: str,
    exc: Optional[Exception] = None,
    **context: Any,
) -> None:
    """Log an error with optional exception traceback and extra context.

    Parameters
    ----------
    logger_instance : loguru.Logger
        The logger to use.
    message : str
        Human-readable error description.
    exc : Exception | None
        If provided, log the full traceback.
    **context : Any
        Additional key-value pairs logged alongside the error.
    """
    ctx = " | ".join(f"{k}={v!r}" for k, v in context.items()) if context else ""
    full_msg = f"{message}" + (f" [{ctx}]" if ctx else "")

    if exc is not None:
        logger_instance.opt(exception=exc).error(full_msg)
    else:
        logger_instance.error(full_msg)


# ---------------------------------------------------------------------------
# Legacy compatibility: keep old function signature for existing callers
# ---------------------------------------------------------------------------


def init_experiment_tracker(
    project: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Initialize Weights & Biases if available, otherwise no-op."""
    try:
        import wandb
    except ImportError:
        logger.debug("wandb not installed — experiment tracking disabled")
        return None

    return wandb.init(project=project, config=config or {})


# ---------------------------------------------------------------------------
# Auto-configure on import from environment
# ---------------------------------------------------------------------------

# Only auto-configure if no sinks are registered yet (e.g. first import)
if not logger._core.handlers:
    _default_level = os.environ.get("LOG_LEVEL", "INFO")
    setup_logging(level=_default_level)

