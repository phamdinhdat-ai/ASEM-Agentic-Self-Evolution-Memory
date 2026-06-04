OPENAI_API_KEY="sk-a7003232a711a6d441aa360c64eb54d57a71e42db3a0883e8631c38326b912c4" OPENAI_BASE_URL="https://ckey.vn/v1" PYTHONPATH="." python scripts/run_locomo_benchmark.py  --val     data/training/val.jsonl
  --config  configs/locomo_openai.yaml \
  --results data/benchmarks/results/locomo_openai.json \
  --db-dir  data/benchmarks/eval_banks_openai \
  --systems NoMemory FullContext SimRetrieval \
  --metrics em rougeL 2>&1 | tail -30