"""Backend interface definition."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class InferenceBackend(ABC):
    """Interface for inference backends used by ASEM stages."""

    def __init__(self) -> None:
        self._token_count: int = 0

    @property
    def token_count(self) -> int:
        """Total tokens consumed by all generate() calls so far."""
        return self._token_count

    def reset_token_count(self) -> None:
        """Reset the token counter to zero."""
        self._token_count = 0

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Single-turn text generation."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a dense vector for the given text."""
