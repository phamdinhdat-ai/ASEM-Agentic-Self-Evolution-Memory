"""Sanity-check a backbone config without making any API call.

Verifies:
  * the YAML parses and `asem.config.ASEMConfig.load` maps every block;
  * the ANSWER budget/knobs actually reach the parsed config
    (max_tokens / context_window / recovery_*);
  * the inference backend constructs (client construction does NOT hit the API).

Usage:
  python scratch_diag/check_config.py configs/models/qwen3_4b_openai.yaml
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_env(path: str = ".env") -> None:
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return
    with open(full, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_env()

import yaml  # noqa: E402
from asem.config import ASEMConfig  # noqa: E402
from asem.backends import build_backend  # noqa: E402
from asem.token_budget import SAFETY_MARGIN_TOKENS  # noqa: E402


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/models/qwen3_4b_openai.yaml"
    full = os.path.join(ROOT, path)
    with open(full, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    inf = raw.get("inference", {})
    backend_name = inf.get("backend")
    block_key = backend_name if backend_name in {"openai", "langchain"} else "langchain"
    block = inf.get(block_key) or {}

    print(f"config        : {path}")
    print(f"backend       : {backend_name}   (block: inference.{block_key})")
    print(f"model         : {block.get('model')}")
    print(f"max_tokens    : {block.get('max_tokens')}   <- client-wide default")
    print(f"temperature   : {block.get('temperature')}")
    print(f"timeout       : {block.get('timeout')}")
    print(f"embedder      : {block.get('embedder_name')}")
    print(f"extra_body    : {block.get('extra_body')}")

    cfg = ASEMConfig.load(path)
    a = cfg.answer
    print("\nparsed ASEMConfig.answer:")
    for attr in ("direct_mode", "include_dates", "max_context_notes",
                 "max_tokens", "context_window", "temperature",
                 "recovery_enabled", "recovery_k2", "recovery_delta"):
        print(f"  {attr:<20} = {getattr(a, attr, '<MISSING>')}")
    # Assert the parsed budget EQUALS what the YAML declares (never hardcode a
    # window: each config declares its own, and `None` = trimming disabled).
    declared_cap = (raw.get("answer", {}) or {}).get("max_tokens")
    declared_window = (raw.get("answer", {}) or {}).get("context_window")
    assert a.max_tokens == declared_cap, "answer.max_tokens did not parse"
    assert a.context_window == declared_window, "answer.context_window did not parse"

    prompt_budget = (
        None if not a.context_window
        else a.context_window - (a.max_tokens or 0) - SAFETY_MARGIN_TOKENS
    )
    print(f"  {'prompt budget':<20} = {prompt_budget}   "
          f"(context_window - max_tokens - {SAFETY_MARGIN_TOKENS} safety)")

    hp = cfg.hyperparameters
    print(f"\nhyperparameters: k1={hp.k1} k2={hp.k2} delta={hp.delta} lambda={hp.lambda_weight}")

    # Building the backend loads sentence-transformers/transformers, which is
    # slow here and is NOT needed to validate the config mapping. Opt in.
    if "--build" in sys.argv:
        print("\nbuilding backend (no API call) ...")
        backend = build_backend(cfg.inference)
        print(f"  OK -> {type(backend).__name__}")
    else:
        print("\n(skipped backend build; pass --build to include it)")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
