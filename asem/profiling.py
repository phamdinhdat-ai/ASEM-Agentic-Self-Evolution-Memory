"""Lightweight profiling utilities for pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List


@dataclass
class StageProfiler:
    """Collect stage-level durations in milliseconds."""

    durations_ms: Dict[str, List[float]] = field(default_factory=dict)

    def record(self, stage: str, duration_ms: float) -> None:
        self.durations_ms.setdefault(stage, []).append(duration_ms)

    def summary(self) -> Dict[str, float]:
        return {
            stage: sum(values) / len(values)
            for stage, values in self.durations_ms.items()
            if values
        }


class stage_timer:
    """Context manager for timing a named stage."""

    def __init__(self, profiler: StageProfiler, stage: str) -> None:
        self._profiler = profiler
        self._stage = stage
        self._start = 0.0

    def __enter__(self) -> None:
        self._start = perf_counter()

    def __exit__(self, exc_type, exc, tb) -> None:
        end = perf_counter()
        self._profiler.record(self._stage, (end - self._start) * 1000.0)
