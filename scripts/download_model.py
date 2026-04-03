"""Download or prepare the base model weights."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Download base model weights")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory",
    )

    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required to download models") from exc

    AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir)

    print(f"Downloaded model: {args.model}")


if __name__ == "__main__":
    main()
