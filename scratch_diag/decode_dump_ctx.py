# -*- coding: utf-8 -*-
"""Scratch: decode logs/dump_ctx.log (likely UTF-16) to a readable copy."""
ROOT = r"C:\Users\Dat Pham\Documents\datpd\master-phenikaa\thesis_master\ASEM-Masters\ASEM-Agentic-Self-Evolution-Memory"
src = ROOT + r"\logs\dump_ctx.log"
raw = open(src, "rb").read()
if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
    txt = raw.decode("utf-16", errors="replace")
else:
    txt = raw.decode("utf-8", errors="replace")
open(ROOT + r"\scratch_diag\dump_ctx_decoded.txt", "w", encoding="utf-8").write(txt)
print("chars:", len(txt))
