"""Package the matched-reader ablation with a Qwen3-Embedding-0.6B dense first bundle.

Same reader, prompt, source-binding checks and conditional repair as the MiniLM
matched-reader run; only the frozen dense rankings change
(``build/care/qwen3_0p6b_dense_rankings.json``, from ``scripts.make_qwen3_dense_rankings``).
"""

from __future__ import annotations

import json

from scripts.prepare_care_full_kaggle import ROOT
from scripts.prepare_care_matched_reader_kaggle import build


def main() -> None:
    rankings = (ROOT / "build/care/qwen3_0p6b_dense_rankings.json").read_bytes()
    root = ROOT / "build/kaggle/care-mr-qwen3emb"
    print(json.dumps([build(root / f"shard-{i}", i, rankings,
                            slug_prefix="asem-care-mr-qwen3emb",
                            title_prefix="ASEM CARE MR Qwen3Emb",
                            output_prefix="care_mr_qwen3emb",
                            encoder_note="dense top-8 first bundle from frozen Qwen3-Embedding-0.6B rankings")
                      for i in range(5)], indent=2))


if __name__ == "__main__":
    main()
