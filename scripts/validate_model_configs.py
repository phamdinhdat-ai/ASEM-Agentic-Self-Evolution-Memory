#!/usr/bin/env python
"""Validate the small-model registry and configs used by the phase benchmark.

Checks that every registry entry points at an existing, parseable config, that
each config has the blocks the builders require, and that all configs share one
embedder (required because one ingested bank is reused across backbones).

Usage::

    python scripts/validate_model_configs.py
    python scripts/validate_model_configs.py --registry configs/models/registry.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

REQUIRED_BLOCKS = ["inference", "hyperparameters", "write_gate", "retriever", "answer", "ingestion"]


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _model_id_and_embedder(inference: Dict[str, Any]):
    backend = inference.get("backend")
    block = inference.get(backend, {}) or {}
    if backend == "huggingface":
        return block.get("model_name_or_path", "?"), block.get("embedder_name")
    return block.get("model", "?"), block.get("embedder_name")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate model registry + configs")
    parser.add_argument("--registry", default="configs/models/registry.yaml")
    args = parser.parse_args()

    from asem.config import ASEMConfig

    registry_path = os.path.join(_PROJECT_ROOT, args.registry)
    registry = _load_yaml(registry_path)
    expected_embedder = registry.get("embedder")

    problems: List[str] = []
    embedders: set = set()

    for tag, meta in (registry.get("models") or {}).items():
        cfg_path = os.path.join(_PROJECT_ROOT, meta["config"])
        if not os.path.exists(cfg_path):
            problems.append(f"{tag}: config not found: {meta['config']}")
            continue

        raw = _load_yaml(cfg_path)
        missing = [k for k in REQUIRED_BLOCKS if k not in raw]
        if missing:
            problems.append(f"{tag}: missing blocks {missing}")

        try:
            cfg = ASEMConfig.load(cfg_path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{tag}: ASEMConfig.load failed: {exc}")
            continue

        inference = raw.get("inference", {})
        model_id, embedder = _model_id_and_embedder(inference)
        embedders.add(embedder)
        if embedder != expected_embedder:
            problems.append(
                f"{tag}: embedder {embedder!r} != registry {expected_embedder!r}"
            )

        status = "OK" if not missing else "BAD"
        print(f"[{status}] {tag:<18} size={str(meta.get('size_b')):<4} "
              f"backend={str(inference.get('backend')):<12} model={model_id:<34} "
              f"preset={cfg.preset}")

    print(f"\nembedders seen: {embedders}")
    if len(embedders) > 1:
        problems.append("embedders are not uniform across configs")

    if problems:
        print("\nPROBLEMS:")
        for item in problems:
            print("  -", item)
        return 1

    print("All configs valid and embedder-consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
