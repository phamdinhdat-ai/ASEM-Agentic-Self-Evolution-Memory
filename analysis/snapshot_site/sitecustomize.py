"""PYTHONPATH shim: auto-install the in-process bank snapshot hook.

Activation is purely by environment variable, so no repo file is edited:

    $env:ASEM_SNAPSHOT_DIR="data/benchmarks/longmemeval_banks/gb_full_per_question"
    $env:PYTHONPATH="analysis/snapshot_site;."

Python imports `sitecustomize` automatically at startup, before the target
script's module-level code runs, so the `MemoryBank.clear` wrapper is in place
before any question is ingested.
"""

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

_out = os.environ.get("ASEM_SNAPSHOT_DIR")
if _out:
    try:
        from analysis import snapshot_hook

        snapshot_hook.install(_out)
    except Exception as _exc:  # noqa: BLE001
        print(f"[snapshot_site] hook not installed: {type(_exc).__name__}: {_exc}",
              file=sys.stderr)
