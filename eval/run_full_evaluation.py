"""Run full evaluation across datasets and baselines."""

from __future__ import annotations

import argparse
import importlib

from eval.evaluate import EvalConfig, DatasetPaths, load_datasets, run_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full ASEM evaluation")
    parser.add_argument("--systems-module", required=True, help="Module with get_systems()")
    parser.add_argument("--longmemeval", required=True)
    parser.add_argument("--locomo", required=True)
    parser.add_argument("--personalmembench", required=True)
    parser.add_argument("--results", required=True)

    args = parser.parse_args()

    module = importlib.import_module(args.systems_module)
    systems = module.get_systems()

    config = EvalConfig(
        datasets=DatasetPaths(
            longmemeval=args.longmemeval,
            locomo=args.locomo,
            personalmembench=args.personalmembench,
        ),
        results_path=args.results,
    )

    datasets = load_datasets(config)
    run_all(systems, datasets, config)


if __name__ == "__main__":
    main()
