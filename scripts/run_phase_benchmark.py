#!/usr/bin/env python
"""Phase-separated LoCoMo benchmark: ingest once, retrieve with many backbones.

Two independent phases that talk only through a shared bank directory:

  Phase A — ingest      build + persist one memory bank per system/conversation
  Phase B — retrieve    load those banks and answer QA (swap the backbone freely)

This lets you answer a model-size sweep (1B / 1.5B / 4B ...) against ONE fixed
ingestion result — the backbone is the only variable in the retrieval phase.

Examples
--------
::

    # 1) Ingest once with the 1.5B config into the "shared" bank tag
    python scripts/run_phase_benchmark.py --phase ingest \
        --ingest-config configs/models/qwen2.5_1.5b_hf.yaml --tag shared

    # 2) Sweep retrieval across three backbones reusing that bank
    python scripts/run_phase_benchmark.py --phase retrieve --tag shared \
        --models-group hf_small --systems FastASEM ASEMv2 \
        --metrics em rougeL

    # 3) Single-model retrieval run
    python scripts/run_phase_benchmark.py --phase retrieve --tag shared \
        --config configs/models/qwen3_4b_api.yaml

    # 4) Do both in one go (ingest with the 1.5B, retrieve with all HF models)
    python scripts/run_phase_benchmark.py --phase combined \
        --ingest-config configs/models/qwen2.5_1.5b_hf.yaml \
        --models-group hf_small
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Load .env (OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL) if present.
_dotenv_path = os.path.join(_PROJECT_ROOT, ".env")
if os.path.exists(_dotenv_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(_dotenv_path, override=False)
    except ImportError:
        with open(_dotenv_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


DEFAULT_REGISTRY = "configs/models/registry.yaml"
DEFAULT_INGEST_CONFIG = "configs/models/qwen2.5_1.5b_hf.yaml"


# ---------------------------------------------------------------------------
# Small config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_bank_tag(config_path: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    cfg = _load_yaml(config_path)
    phase = cfg.get("phase") or {}
    return str(phase.get("bank_tag") or "shared")


def resolve_ingest_config(config_path: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    cfg = _load_yaml(config_path)
    phase = cfg.get("phase") or {}
    ic = phase.get("ingest_config")
    if ic:
        return str(ic)
    # No bank-builder named by the config: use the shared default when present.
    if os.path.exists(DEFAULT_INGEST_CONFIG):
        return DEFAULT_INGEST_CONFIG
    return config_path


def registry_models(registry_path: str) -> Dict[str, Dict[str, Any]]:
    reg = _load_yaml(registry_path)
    return reg.get("models") or {}


def registry_group(registry_path: str, group: str) -> List[str]:
    reg = _load_yaml(registry_path)
    groups = reg.get("groups") or {}
    if group not in groups:
        raise SystemExit(
            f"Unknown --models-group {group!r}. Available: {list(groups)}"
        )
    return list(groups[group])


def resolve_retrieval_configs(
    args: argparse.Namespace,
) -> List[Dict[str, str]]:
    """Return a list of ``{"tag": ..., "config": ...}`` retrieval backbones."""
    registry = registry_models(args.registry)

    tags: List[str] = []
    if args.models_group:
        tags = registry_group(args.registry, args.models_group)
    if args.models:
        tags = list(args.models)

    if tags:
        resolved: List[Dict[str, str]] = []
        for tag in tags:
            if tag not in registry:
                raise SystemExit(
                    f"Unknown model tag {tag!r}. Available: {list(registry)}"
                )
            resolved.append({"tag": tag, "config": str(registry[tag]["config"])})
        return resolved

    if args.config:
        return [{"tag": "", "config": args.config}]

    return [{"tag": "", "config": args.ingest_config or DEFAULT_INGEST_CONFIG}]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-separated LoCoMo benchmark (ingest once, retrieve many)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default="datasets/locomo/locomo10.json",
                        help="Path to locomo10.json")

    # Phase
    parser.add_argument("--phase", choices=["ingest", "retrieve", "combined"],
                        default=None,
                        help="Default: taken from the config's `phase.mode` (else combined)")
    parser.add_argument("--tag", default=None,
                        help="Shared bank tag/directory name (default: config phase.bank_tag)")

    # Configs
    parser.add_argument("--config", default=None,
                        help="Retrieval/answer backbone config (single-model run)")
    parser.add_argument("--ingest-config", default=None,
                        help="Backbone config used to build the banks (phase=ingest). "
                             "Default: config's phase.ingest_config, else "
                             f"{DEFAULT_INGEST_CONFIG}")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY,
                        help="Model registry YAML for --models/--models-group")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Registry tags to sweep in the retrieval phase")
    parser.add_argument("--models-group", default=None,
                        help="Registry group to sweep (e.g. hf_small, api_small)")

    # Scope
    parser.add_argument("--systems", nargs="+", default=None,
                        help="Systems to run (default: all)")
    parser.add_argument("--metrics", nargs="+", default=["em", "rougeL"])
    parser.add_argument("--per-category", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N QA pairs (smoke tests)")
    parser.add_argument("--bank-root", default="data/benchmarks/ingested_banks")
    parser.add_argument("--results-dir", default="data/benchmarks/results/phased")
    parser.add_argument("--work-root", default="data/benchmarks/retrieval_work",
                        help="Per-run bank working copies (keeps ingested banks pristine)")
    parser.add_argument("--no-work-copy", action="store_true",
                        help="Answer against the ingested banks in place (mutating systems "
                             "will modify them; not recommended for sweeps)")
    parser.add_argument("--allow-missing-banks", action="store_true",
                        help="Skip (instead of failing) systems whose bank is absent")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    from asem.logging_utils import setup_logging
    from eval.phase_runner import (
        ALL_SYSTEMS,
        bank_exists,
        build_backend_from_config,
        model_tag_from_config,
        render_sweep_table,
        run_ingest_phase,
        run_retrieve_phase,
        write_results,
    )

    setup_logging(level=args.log_level)

    # ---- Data ---------------------------------------------------------
    if not os.path.exists(args.input):
        raise SystemExit(f"Dataset not found: {args.input}")

    from scripts.run_locomo10_experiments import (
        convert_locomo10_to_eval,
        group_by_conversation,
    )
    from eval.phase_runner import load_raw_dataset

    print(f"Loading {args.input} ...")
    raw_dataset = load_raw_dataset(args.input)
    eval_items = convert_locomo10_to_eval(args.input, limit=args.limit)
    groups = group_by_conversation(eval_items)
    total_qa = sum(len(g) for g in groups)
    print(f"  {len(groups)} conversations | {total_qa} QA pairs")

    systems = args.systems or list(ALL_SYSTEMS)
    unknown = [s for s in systems if s not in ALL_SYSTEMS]
    if unknown:
        raise SystemExit(f"Unknown systems {unknown}. Known: {ALL_SYSTEMS}")

    # ---- Determine phase & configs -----------------------------------
    retrieval_cfgs = resolve_retrieval_configs(args)
    primary_config = retrieval_cfgs[0]["config"]

    cfg_phase = (_load_yaml(args.config or primary_config).get("phase") or {})
    phase = args.phase or cfg_phase.get("mode") or "combined"

    ingest_config = resolve_ingest_config(primary_config, args.ingest_config)
    tag = resolve_bank_tag(ingest_config, args.tag)

    print(f"Phase       : {phase}")
    print(f"Bank tag    : {tag}  (root: {args.bank_root})")
    if phase in ("ingest", "combined"):
        print(f"Ingest cfg  : {ingest_config}")
    if phase in ("retrieve", "combined"):
        print("Retrieve    : " + ", ".join(
            f"{c['tag'] or model_tag_from_config(c['config'])} -> {c['config']}"
            for c in retrieval_cfgs
        ))

    os.makedirs(args.results_dir, exist_ok=True)

    # ---- Phase A: ingest ---------------------------------------------
    if phase in ("ingest", "combined"):
        print("\n=== Phase A — ingest ===")
        ingest_backend = build_backend_from_config(ingest_config)
        manifest = run_ingest_phase(
            raw_dataset=raw_dataset,
            groups=groups,
            systems=systems,
            ingest_config=ingest_config,
            bank_root=args.bank_root,
            tag=tag,
            backend=ingest_backend,
        )
        print(f"Ingested {manifest['n_conversations']} conversations "
              f"into {args.bank_root}/{tag}")

    if phase == "ingest":
        print("\nDone (ingest only).")
        return

    # ---- Phase B: retrieve -------------------------------------------
    print("\n=== Phase B — retrieve ===")
    sweep: Dict[str, Dict[str, Any]] = {}

    for entry in retrieval_cfgs:
        config_path = entry["config"]
        tag_hint = entry["tag"]
        model_tag = model_tag_from_config(config_path)
        display_tag = tag_hint or model_tag

        missing = [
            s for s in systems
            if s not in ("NoMemory", "FullContext")
            and not all(
                bank_exists(args.bank_root, tag, s, str(g[0].get("session_id", "")))
                for g in groups
            )
        ]
        if missing and not args.allow_missing_banks:
            raise SystemExit(
                f"Missing banks for systems {missing} under "
                f"{args.bank_root}/{tag}. Run --phase ingest first "
                f"(or pass --allow-missing-banks)."
            )

        print(f"\n--- Retrieve model: {display_tag} ({config_path}) ---")
        backend = build_backend_from_config(config_path)
        preds_dir = os.path.join(args.results_dir, "preds")
        results = run_retrieve_phase(
            groups=groups,
            systems=systems,
            config_path=config_path,
            bank_root=args.bank_root,
            tag=tag,
            backend=backend,
            metric_names=args.metrics,
            preds_dir=preds_dir,
            model_tag=display_tag,
            per_category=args.per_category,
            require_banks=not args.allow_missing_banks,
            work_root=None if args.no_work_copy else args.work_root,
        )

        out_path = os.path.join(args.results_dir, f"{tag}__{display_tag}.json")
        write_results(out_path, results)
        print(f"Results -> {out_path}")
        for name, res in results["systems"].items():
            print(f"  {name:<20} n={res['n']:<5} {res['overall']}")

        sweep[display_tag] = results

    if len(sweep) > 1:
        table = render_sweep_table(sweep, args.metrics)
        table_path = os.path.join(args.results_dir, f"{tag}__sweep_table.md")
        with open(table_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Phase benchmark sweep — bank tag `{tag}`\n\n")
            fh.write(table + "\n")
        sweep_path = os.path.join(args.results_dir, f"{tag}__sweep.json")
        write_results(sweep_path, sweep)
        print(f"\nSweep table -> {table_path}")
        print(f"Sweep JSON  -> {sweep_path}")
        print("\n" + table)


if __name__ == "__main__":
    main()
