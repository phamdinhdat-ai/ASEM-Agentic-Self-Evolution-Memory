#!/usr/bin/env python
"""Visualize an ASEM memory bank as an interactive graph.

Usage::

    python scripts/visualize_memory.py --db data/benchmarks/eval_banks_openai/asem.sqlite
    python scripts/visualize_memory.py --db data/benchmarks/demo_bank.sqlite --stats
    python scripts/visualize_memory.py --db bank.sqlite --format png --output graph.png
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure the project root is on sys.path so `asem` imports work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asem.visualizer import visualize_bank


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ASEM memory bank as an interactive graph."
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to the SQLite memory bank file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: memory_graph.html or memory_graph.png).",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=["html", "png"],
        default="html",
        help="Output format: interactive HTML or static PNG (default: html).",
    )
    parser.add_argument(
        "--title",
        default="ASEM Memory Graph",
        help="Title for the graph (default: 'ASEM Memory Graph').",
    )
    parser.add_argument(
        "--no-physics",
        action="store_true",
        help="Disable force-directed physics (nodes stay where placed).",
    )
    parser.add_argument(
        "--no-infer-types",
        action="store_true",
        help="Skip heuristic edge-type inference (all edges shown as 'linked').",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print graph metrics (centrality, clustering, etc.) to stdout.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    # Default output path
    if args.output is None:
        ext = "html" if args.fmt == "html" else "png"
        args.output = f"memory_graph.{ext}"

    metrics = visualize_bank(
        db_path=args.db,
        output=args.output,
        fmt=args.fmt,
        title=args.title,
        physics=not args.no_physics,
        infer_types=not args.no_infer_types,
        stats=args.stats,
    )

    if metrics["nodes"] == 0:
        print("Memory bank is empty — no graph generated.", file=sys.stderr)
        sys.exit(0)

    print(f"Graph saved -> {args.output}")
    if args.fmt == "html":
        print(f"Open in browser to explore: file:///{os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
