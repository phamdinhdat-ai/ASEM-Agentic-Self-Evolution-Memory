# -*- coding: utf-8 -*-
"""Scratch diagnostic (delete after use): analyze FastASEM ingestion gate.
Reads logs/fastasem_locomo10_full.log, writes a report to scratch_diag/gate_report.txt
"""
from collections import Counter, defaultdict
import io

ROOT = r"C:\Users\Dat Pham\Documents\datpd\master-phenikaa\thesis_master\ASEM-Masters\ASEM-Agentic-Self-Evolution-Memory"
log = ROOT + r"\logs\fastasem_locomo10_full.log"
out = io.StringIO()

def read_text(path):
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")

def p(*a):
    print(*a, file=out)

lines = read_text(log).splitlines()

cnt = Counter()
noops, adds = [], []
merge_targets = defaultdict(list)

for ln in lines:
    kind = None
    for k in ("NOOP", "UPDATE", "ADD"):
        if f"Gated {k}" in ln:
            kind = k
            break
    if kind is None:
        continue
    cnt[kind] += 1
    fact = ln.split("fact='", 1)[1].split("'", 1)[0] if "fact='" in ln else ln
    if kind == "NOOP":
        noops.append(fact)
    elif kind == "UPDATE":
        m = ln.split("id=", 1)[1].split(" ", 1)[0] if "id=" in ln else None
        merge_targets[m].append(fact)
    else:
        adds.append(fact)

p("=== GATE DECISION COUNTS ===")
p(dict(cnt))
p("")

p("=== ALL NOOP (fact DROPPED as near-duplicate, tau_redund>=0.90) ===")
for fact in noops:
    p(f"  DROPPED | {fact[:200]}")
p("")

p("=== UPDATE merge sizes (facts folded into one note) ===")
sizes = sorted(merge_targets.items(), key=lambda kv: -len(kv[1]))
for tid, facts in sizes[:15]:
    p(f"\n  note {tid}  <-- {len(facts)} facts merged:")
    for f in facts:
        p(f"      - {f[:160]}")
p("")

p("=== SEARCH for facts relevant to failing questions ===")
targets = {
    "married 5": "idx90 married-5yrs",
    "sunflower": "idx37/114 sunflower",
    "warmth and happiness": "idx114 sunflowers",
    "workshop": "idx97 workshop",
    "adoption": "idx3/85 adoption research",
    "bowl": "idx94/101 bowl",
    "dog face": "idx110 dog-face cup",
    "dinosaurs": "idx19 dinosaurs",
    "sunset": "idx37/55 sunset",
    "sweden": "idx11 Sweden",
    "home country": "idx11 home country",
    "ally": "idx46 ally",
}
seen = set()
for ln in lines:
    low = ln.lower()
    for key, label in targets.items():
        if key.lower() in low:
            tag = f"[{label}]"
            if tag in seen:
                continue
            seen.add(tag)
            p(f"\n{tag}\n  {ln[:300]}")
            break

with open(ROOT + r"\scratch_diag\gate_report.txt", "w", encoding="utf-8") as f:
    f.write(out.getvalue())
print("report written to scratch_diag/gate_report.txt")
