"""Temporary environment probe for ASEM small-model config design."""
import importlib
import sys

print("python:", sys.version.split()[0])

mods = [
    "torch",
    "transformers",
    "sentence_transformers",
    "faiss",
    "langchain",
    "langchain_openai",
    "openai",
    "yaml",
    "dotenv",
    "evaluate",
    "accelerate",
    "bitsandbytes",
]
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: OK {getattr(mod, '__version__', '')}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name}: MISSING ({type(exc).__name__})")
