"""Quick manual test for LangChainBackend with Ollama.

Usage examples:
  python quick_test.py
  python quick_test.py --model llama3.2:3b --prompt "Introduce yourself in one sentence."
  python quick_test.py --test-embed
"""

from __future__ import annotations

import argparse
import sys

from asem.backends.langchain_backend import LangChainBackend


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run a quick Ollama inference via LangChainBackend")
	parser.add_argument("--model", default="gemma3:1b", help="Ollama chat model name")
	parser.add_argument(
		"--embedder",
		default="granite-embedding:latest",
		help="Ollama embedding model used by LangChainBackend",
	)
	parser.add_argument(
		"--prompt",
		default="You are testing ASEM backend wiring. Reply in one short sentence.",
		help="Prompt text for one-shot inference",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.0,
		help="Sampling temperature passed to ChatOllama",
	)
	parser.add_argument(
		"--test-embed",
		action="store_true",
		help="Also call backend.embed and print embedding shape",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()

	cfg = {
		"provider": "ollama",
		"model": args.model,
		"temperature": args.temperature,
		"embedder_provider": "ollama",
		"embedder_name": args.embedder,
	}

	try:
		backend = LangChainBackend.from_config(cfg)
		answer = backend.generate(args.prompt)
	except Exception as exc:
		print("Failed to run Ollama inference via LangChainBackend.", file=sys.stderr)
		print(f"Error: {exc}", file=sys.stderr)
		print("Hints:", file=sys.stderr)
		print("  1. Start Ollama server: ollama serve", file=sys.stderr)
		print(f"  2. Pull chat model: ollama pull {args.model}", file=sys.stderr)
		print(f"  3. Pull embed model: ollama pull {args.embedder}", file=sys.stderr)
		return 1

	print("=== LangChainBackend + Ollama ===")
	print(f"Model: {args.model}")
	print(f"Prompt: {args.prompt}")
	print("Response:")
	print(answer)

	if args.test_embed:
		try:
			vec = backend.embed(args.prompt)
			print(f"Embedding shape: {tuple(vec.shape)}")
		except Exception as exc:
			print(f"Embedding call failed: {exc}", file=sys.stderr)
			return 1

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
