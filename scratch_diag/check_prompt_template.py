"""Verify the single-pass prompt template formats cleanly and check brace balance."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.getcwd())

from asem.single_pass_ingest import _SINGLE_PASS_PROMPT_TEMPLATE as T

out = T.format(session_date="8 May 2023", dialogue="Caroline: Hi\nMelanie: Hello")
print("format OK, length:", len(out))

# Brace balance in the RAW template (before format) must be even for .format
print("raw open braces :", T.count("{"))
print("raw close braces:", T.count("}"))

# The schema block must survive intact
assert '"fact"' in out, "fact key missing"
assert "speaker" in out, "speaker key missing"
print("\n--- rendered rule 5 ---")
seg = out.split("5. REACTIVE")[1].split("DIALOGUE:")[0]
print("5. REACTIVE" + seg[:700])

print("\n--- rendered schema block ---")
blk = out.split("Return a JSON array")[1]
print(blk[:600])

# Check no literal braces remain in the rendered prompt (outside the schema block)
header = out.split("Return a JSON array")[0]
if "{" in header:
    print("\nWARNING: literal { in header (may be OK if it's in the schema example)")
else:
    print("\nNo literal braces in header — OK")
