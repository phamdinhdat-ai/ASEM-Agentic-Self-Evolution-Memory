"""Preflight probe: is the judge config's model actually ENTITLED for this key?

WHY THIS EXISTS
---------------
On 2026-10-03 a judge run printed

    judge: 0.0, judge_coverage: 0.0

which reads like "the judge disagreed with every answer". It was not a score at
all: every judge call had raised ``403 FORBIDDEN: Please subscribe to model in
the API Key: deepseek-v4-flash``, the per-row handler swallowed the exception
into ``judge_error`` with ``judge_correct=None``, and the aggregate rendered
``None`` as ``0.0``. A TOTAL OUTAGE WAS INDISTINGUISHABLE FROM A MEASURED ZERO.

Entitlement is a property of the API key and can change — the config that worked
historically stopped working. So a judge run must be *probed*, not assumed.

WHAT IT CHECKS
--------------
For each judge config given (default: every ``configs/models/judge*.yaml``):
  1. the advertised model list from ``GET /v1/models``;
  2. a real 1-token chat completion against the configured model id.

It exits NON-ZERO if any configured judge model is not callable, so it can be
used as a gate before a long evaluation run:

    & $py scripts/check_judge_entitlement.py            # probe all judge configs
    & $py scripts/check_judge_entitlement.py --config configs/models/judge_api.yaml

A WARNING ABOUT AGGREGATES (the second half of the same lesson): any reported
rate must carry its denominator AND its error count, and must refuse to print a
rate when errors > 0. This script exists to make the entitlement failure loud
BEFORE it becomes a number in a table.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _model_of(cfg: dict) -> str:
    """Configured model id for either backend shape (langchain/openai)."""
    inference = cfg.get("inference") or {}
    for backend in ("langchain", "openai"):
        block = inference.get(backend) or {}
        if block.get("model"):
            return str(block["model"])
    return ""


def probe(config_path: Path, client, advertised: list[str] | None) -> bool:
    """Probe one judge config. Returns True when the model is callable."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model = _model_of(cfg)
    print(f"\n=== {config_path.relative_to(REPO_ROOT)}")
    print(f"    backend : {(cfg.get('inference') or {}).get('backend')}")
    print(f"    model   : {model!r}")
    if not model:
        print("    RESULT  : FAIL — no model id configured")
        return False
    if advertised is not None and model not in advertised:
        print(f"    NOTE    : not in the advertised list {advertised} (probing anyway)")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            max_tokens=8,
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        print(f"    RESULT  : OK — 200 {text[:40]!r}")
        return True
    except Exception as exc:  # noqa: BLE001 - we want the raw provider error
        print(f"    RESULT  : FAIL — {str(exc)[:200]}")
        print("              This config would produce judge=0.0, judge_coverage=0.0")
        print("              and the zero would be a SILENT OUTAGE, not a score.")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Judge config YAML to probe (repeatable). Default: every judge*.yaml.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        print("FAIL: OPENAI_API_KEY is not set (expected in .env)")
        return 1
    print(f"base_url: {base_url}")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    advertised: list[str] | None
    try:
        advertised = sorted(m.id for m in client.models.list().data)
        print(f"advertised models: {advertised}")
    except Exception as exc:  # noqa: BLE001
        advertised = None
        print(f"advertised models: <could not list — {str(exc)[:120]}>")

    if args.config:
        paths = [Path(c) for c in args.config]
    else:
        paths = sorted((REPO_ROOT / "configs" / "models").glob("judge*.yaml"))
    if not paths:
        print("FAIL: no judge configs found")
        return 1

    results = [(p, probe(p, client, advertised)) for p in paths]

    ok = [p for p, good in results if good]
    bad = [p for p, good in results if not good]
    print("\n" + "=" * 70)
    print(f"PROBED {len(results)} config(s): {len(ok)} OK, {len(bad)} FAILING")
    for p in ok:
        print(f"  OK      {p.relative_to(REPO_ROOT)}")
    for p in bad:
        print(f"  FAILING {p.relative_to(REPO_ROOT)}")
    print("=" * 70)
    if bad:
        print("Do NOT run a judged evaluation with a failing config — the judge")
        print("column would be an outage rendered as 0.0.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
