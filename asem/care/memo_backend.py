"""Prompt-keyed memoizing backend with persisted raw replies.

Several ablation arms send byte-identical prompts (the same first read, and the same repair read whenever
a flag does not change the outcome). ``MemoBackend`` answers the repeat from a cache, so the physical
model is called once, while ``calls`` still records one receipt per *logical* call. Per-arm call and
token counts therefore equal what the arm would cost alone; ``physical`` counts real model calls.
Raw replies are persisted, which makes the run resumable and lets any arm be re-scored offline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

Complete = Callable[[str, float, int], tuple[str, dict[str, Any]]]


def prompt_key(prompt: str, temperature: float, max_tokens: int) -> str:
    return hashlib.sha256(f"{temperature}|{max_tokens}|{prompt}".encode()).hexdigest()


class MemoBackend:
    def __init__(self, complete: Complete, path: Path | None = None, max_new_calls: int | None = None):
        self.complete, self.path, self.max_new_calls = complete, path, max_new_calls
        self.store: dict[str, tuple[str, dict[str, Any]]] = {}
        self.calls: list[dict[str, Any]] = []     # one receipt per logical call
        self.physical = 0                          # model calls made in this process
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    row = json.loads(line)
                    self.store[row["key"]] = (row["text"], row["receipt"])

    def generate(self, prompt: str, **kwargs: Any) -> str:
        temperature = float(kwargs.get("temperature", 0.0))
        max_tokens = int(kwargs.get("max_tokens", 224))
        key = prompt_key(prompt, temperature, max_tokens)
        cached = key in self.store
        if not cached:
            if self.max_new_calls is not None and self.physical >= self.max_new_calls:
                raise RuntimeError("ablation call cap reached")
            text, receipt = self.complete(prompt, temperature, max_tokens)
            self.store[key] = (text, receipt)
            self.physical += 1
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"key": key, "text": text, "receipt": receipt},
                                            ensure_ascii=False, separators=(",", ":")) + "\n")
        text, receipt = self.store[key]
        self.calls.append({**receipt, "key": key, "cached": cached})
        return text
