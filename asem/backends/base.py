"""Backend interface definition."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class InferenceBackend(ABC):
    """Interface for inference backends used by ASEM stages."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Single-turn text generation."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a dense vector for the given text."""
