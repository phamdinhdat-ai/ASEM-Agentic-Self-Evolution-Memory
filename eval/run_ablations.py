"""Run ablation sweeps for ASEM components."""

from __future__ import annotations

import argparse
import importlib

from eval.evaluate import EvalConfig, DatasetPaths, load_datasets, run_all


def _disable_link_evolver(system) -> None:
    if hasattr(system, "link_evolver"):
        system.link_evolver = _NoopLinkEvolver()


class _NoopLinkEvolver:
    def link_and_evolve(self, m_new, M):
        return None


def _set_retriever_zscore(system, enabled: bool) -> None:
    if hasattr(system, "retriever"):
        setattr(system.retriever, "use_zscore", enabled)


def _set_lambda(system, value: float) -> None:
    if hasattr(system, "retriever"):
        setattr(system.retriever, "lambda_weight", value)


def _set_alpha(system, value: float) -> None:
    if hasattr(system, "utility_updater"):
        setattr(system.utility_updater, "alpha", value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation sweeps")
    parser.add_argument("--systems-module", required=True, help="Module with get_systems()")
    parser.add_argument("--longmemeval", required=True)
    parser.add_argument("--locomo", required=True)
    parser.add_argument("--personalmembench", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--disable-link", action="store_true")
    parser.add_argument("--disable-zscore", action="store_true")
    parser.add_argument("--lambda", dest="lambda_weight", type=float, default=None)
    parser.add_argument("--alpha", dest="alpha", type=float, default=None)

    args = parser.parse_args()

    module = importlib.import_module(args.systems_module)
    systems = module.get_systems()

    for system in systems.values():
        if args.disable_link:
            _disable_link_evolver(system)
        if args.disable_zscore:
            _set_retriever_zscore(system, False)
        if args.lambda_weight is not None:
            _set_lambda(system, args.lambda_weight)
        if args.alpha is not None:
            _set_alpha(system, args.alpha)

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
