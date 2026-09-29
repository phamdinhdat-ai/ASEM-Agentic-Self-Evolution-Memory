"""Debug: reproduce the ASEM-THG single-pass extraction call and inspect it.

Answers the only question that matters when the log prints

    SinglePassSessionIngestor | JSON extraction fallback triggered
    (raw_len=3392 parsed=NoneType)

is the model writing bad JSON, or was the completion CUT OFF before the JSON
was closed? The two need opposite fixes (prompt/structured-output vs a bigger
output/context budget), and the raw text plus ``finish_reason`` tells them apart.

Usage (always in the project's conda env):

    python scratch_diag/debug_thg_extraction.py <config> [conv_idx] [session_idx]

    # the local vLLM backbone that was failing
    python scratch_diag/debug_thg_extraction.py configs/locomo_vllm_qwen3_4b-2507.yaml
    # the DeepSeek no-thinking backbone that parses cleanly
    python scratch_diag/debug_thg_extraction.py configs/models/deepseek_openai.yaml 0 3

Writes the full raw response to scratch_diag/_thg_raw.txt (editor-friendly) and
prints: prompt/output token estimates, finish_reason, the raw head/tail, and
the outcome of the same parser the ingestor uses.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # keys already in the environment
    pass

import yaml

from asem.backends import build_backend
from asem.llm_validator import validate_fact_array
from asem.note import _try_extract_json
from asem.single_pass_ingest import _SINGLE_PASS_PROMPT_TEMPLATE
from asem.token_budget import estimate_tokens

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_thg_raw.txt")


def main() -> int:
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/locomo_vllm_qwen3_4b-2507.yaml"
    conv_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    session_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    cfg = yaml.safe_load(open(os.path.join(ROOT, config), encoding="utf-8"))
    infer_cfg = cfg["inference"]
    block = infer_cfg.get(infer_cfg.get("backend")) or {}
    backend = build_backend(infer_cfg)
    print(f"config     : {config}")
    print(f"backend    : {infer_cfg.get('backend')} "
          f"model={block.get('model') or block.get('model_name_or_path')} "
          f"base_url={block.get('base_url') or os.environ.get('OPENAI_BASE_URL', '(unset)')}")
    print(f"max_tokens : {block.get('max_tokens', '(unset -> server decides)')}")

    # Same session parsing + header reconstruction as the phase runner.
    from scripts.run_asem_v2 import _parse_sessions

    with open(os.path.join(ROOT, "datasets/locomo/locomo10.json"), encoding="utf-8") as fh:
        dataset = json.load(fh)
    sessions = _parse_sessions(dataset[conv_idx]["conversation"])
    sess_num, date_str, turns = sessions[session_idx]

    header = f"[Session {sess_num}" + (f" — {date_str}" if date_str else "") + "]"
    dialogue = [header] + turns
    prompt = _SINGLE_PASS_PROMPT_TEMPLATE.format(
        session_date=date_str or "Unknown", dialogue="\n".join(dialogue)
    )
    print(f"conv/session: locomo_{conv_idx:04d} / session {session_idx} "
          f"({len(turns)} turns, {date_str})")
    print(f"prompt      : {len(prompt)} chars ~ {estimate_tokens(prompt)} tokens")

    print("\n>>> calling model ...")
    raw = backend.generate(prompt)
    finish_reason = getattr(backend, "last_finish_reason", None)

    with open(RAW_DUMP, "w", encoding="utf-8") as fh:
        fh.write(raw or "")
    print(f"raw         : {len(raw or '')} chars ~ {estimate_tokens(raw or '')} tokens "
          f"-> {RAW_DUMP}")
    print(f"finish_reason: {finish_reason!r}")
    if finish_reason == "length":
        print("  !! TRUNCATED — the model was cut off mid-answer. Raise the served")
        print("     --max-model-len and/or this call's max_tokens. No prompt fix.")
    print("\n--- raw head ---")
    print((raw or "")[:600])
    print("--- raw tail ---")
    print((raw or "")[-400:])

    parsed = _try_extract_json(raw, expect_array=True)
    print(f"\nparse       : {type(parsed).__name__}"
          + (f" ({len(parsed)} facts)" if isinstance(parsed, list) else ""))
    if isinstance(parsed, list):
        result = validate_fact_array(parsed)
        print(f"validate    : valid={result.valid} "
              f"errors={result.errors[:4]}")
        for i, item in enumerate(parsed[:5]):
            if isinstance(item, dict):
                print(f"  [{i}] fact={str(item.get('fact'))[:70]!r} "
                      f"entities={item.get('entities')}")
            else:
                print(f"  [{i}] NOT A DICT: {type(item).__name__}")
    else:
        print("  -> the ingestor would fall back to one note per dialogue line,")
        print("     losing every triplet / entity edge in this session.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
