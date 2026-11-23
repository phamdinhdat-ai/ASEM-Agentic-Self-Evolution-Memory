# CARE versus FullContext-8K, NoMemory, SimRetrieval

All four arms use the same 1,986 LoCoMo questions, pinned Qwen3-8B, 8,192-token window, 224-token answer cap, temperature 0, no thinking, and identical LoCoMo and BERTScore scorers. CARE results were preserved.

| Group | Arm | n | LoCoMo | BERTScore raw F1 | BERTScore rescaled F1 | CARE minus arm LoCoMo |
|---|---|---:|---:|---:|---:|---:|
| all | bridge_v2 | 1986 | 0.5256 | 0.9088 | 0.4595 | -- |
| all | no_memory | 1986 | 0.0774 | 0.8776 | 0.2745 | +0.4482 |
| all | full_context_8k | 1986 | 0.1961 | 0.8912 | 0.3556 | +0.3294 |
| all | sim_retrieval | 1986 | 0.4069 | 0.8896 | 0.3460 | +0.1187 |
| answerable | bridge_v2 | 1540 | 0.4648 | 0.9149 | 0.4956 | -- |
| answerable | no_memory | 1540 | 0.0998 | 0.8782 | 0.2781 | +0.3650 |
| answerable | full_context_8k | 1540 | 0.2406 | 0.8949 | 0.3773 | +0.2242 |
| answerable | sim_retrieval | 1540 | 0.3124 | 0.8910 | 0.3544 | +0.1524 |
| adversarial | bridge_v2 | 446 | 0.7354 | 0.8877 | 0.3347 | -- |
| adversarial | no_memory | 446 | 0.0000 | 0.8755 | 0.2621 | +0.7354 |
| adversarial | full_context_8k | 446 | 0.0426 | 0.8786 | 0.2807 | +0.6928 |
| adversarial | sim_retrieval | 446 | 0.7332 | 0.8848 | 0.3171 | +0.0022 |

## Protocol limits

FullContext-8K keeps the first five turns and a recent suffix; its middle turns are omitted to fit the context window. SimRetrieval uses top-18 cosine matches from original dated turns with MiniLM-L6-v2, a 10,000-character evidence cap and one answer call. CARE uses its own retrieval, source checks and optional repair. NoMemory follows the prior best-guess prompt. These systems have different answer prompts, so score gaps measure whole pipelines, not the isolated memory mechanism. Category-5 refusal policy especially differs for NoMemory.

BERTScore measures semantic overlap, not factual correctness. The LoCoMo overall score combines token F1 for categories 1-4 with binary refusal accuracy for category 5.

## Files and integrity

- `build\care\relation_full_merged.json` SHA256 `ca88a9b46ccfb986cbc0a376c916d02fd797d2f3f9a244fb32ef273ba7babb7d`
- `build\care\context_baselines_merged.json` SHA256 `1e6d172c06c28baad050ba63c21c1477e9debe4cbd5cf2d092d018b57654fb20`
- `build\care\sim_retrieval_merged.json` SHA256 `d067349048e4acbbeb7a4591351e5a29da21953c4b6081e655b54d781fe2a641`
- `build\care\relation_full_bertscore.json` SHA256 `62ee515b7ae4cd1a8bc83a8ba8eeb6f5e7dc3c68dfc50ccdd91fc30f5e1c1e24`
- `build\care\context_baselines_bertscore.json` SHA256 `65a0a0ad22bd4d0a7dbb0524500266fbd4b355c2e40abfe7fbd88037d7d955f7`
- `build\care\sim_retrieval_bertscore.json` SHA256 `2563868e8431b4b36915d428f8d40ae1722486f472a5ca7a65e1404fde4ede03`
