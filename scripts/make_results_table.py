"""
Generate a Markdown results table from the benchmark results JSON.

Thin wrapper around eval/results_table.py that can be called directly
from the scripts/ directory.

Usage
-----
    python scripts/make_results_table.py \
        --results data/benchmarks/results/locomo_baseline.json \
        --output  data/benchmarks/results/locomo_baseline_table.md
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.results_table import main

if __name__ == "__main__":
    main()
