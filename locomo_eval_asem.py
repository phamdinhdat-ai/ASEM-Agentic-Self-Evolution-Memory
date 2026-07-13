"""
LoCoMo QA Evaluation with ASEM Pipeline + Baselines — Knowledge-Base-Aware

Quy trình cho mỗi conversation (theo conv-id):
  1. Load toàn bộ session hội thoại
  2. Với mỗi system (NoMemory, FullContext, SimRetrieval, ..., ASEM):
     a. Reset knowledge base
     b. Ingest TẤT CẢ turn vào KB theo cách riêng của system
     c. Với mỗi QA: truy vấn KB → sinh câu trả lời
  3. So sánh EM, ROUGE-L, LLM-as-Judge giữa các system

Usage:
    PYTHONPATH="." python locomo_eval_asem.py \
        --data-file datasets/locomo/locomo10.json \
        --config configs/locomo_openai.yaml \
        --out-file outputs/locomo10_asem_eval.json \
        --systems NoMemory FullContext ASEM \
        --limit 2

    PYTHONPATH="." python locomo_eval_asem.py \
        --data-file datasets/locomo/locomo10.json \
        --config configs/locomo_openai.yaml \
        --systems ASEM --judge
"""

import os, sys, json, time, re, argparse, traceback
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import OrderedDict

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_dotenv = os.path.join(_PROJECT_ROOT, ".env")
if os.path.exists(_dotenv):
    try:
        from dotenv import load_dotenv
        load_dotenv(_dotenv, override=False)
    except ImportError:
        with open(_dotenv, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import yaml
from eval.systems import (
    build_asem_system, build_baselines, _load_config,
    _NO_MEMORY_PROMPT, _FULL_CONTEXT_PROMPT,
)
from asem.backends import build_backend

# Lazy import — evaluate can be slow to load
_evaluate_available = False
try:
    import evaluate as _hf_evaluate
    _evaluate_available = True
except ImportError:
    pass


def _compute_em(preds: List[str], refs: List[str]) -> float:
    """Exact match — lightweight, no external deps."""
    if not preds:
        return 0.0
    matches = sum(1 for p, r in zip(preds, refs)
                  if " ".join(p.strip().lower().split()) ==
                     " ".join(r.strip().lower().split()))
    return matches / len(preds)


def _compute_rougeL(preds: List[str], refs: List[str]) -> float:
    """ROUGE-L via evaluate (if available) or fallback."""
    if _evaluate_available:
        try:
            rouge = _hf_evaluate.load("rouge")
            scores = rouge.compute(predictions=preds, references=refs)
            return float(scores.get("rougeL", 0.0))
        except Exception:
            pass
    return 0.0


def compute_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """Compute EM and ROUGE-L without heavy deps."""
    return {
        "em": _compute_em(preds, refs),
        "rougeL": _compute_rougeL(preds, refs),
    }

CATEGORY_NAMES = {1: "single_hop", 2: "temporal", 3: "commonsense",
                  4: "conversational", 5: "adversarial"}
SYSTEM_NAMES = ["NoMemory", "FullContext", "SimRetrieval", "AtomicLinking",
                "RLManagerOnly", "ValueRetrievalOnly", "ASEM"]

# ── Data loaders ────────────────────────────────────────────

def _parse_dia_id(dia_id: str) -> Tuple[int, int]:
    m = re.match(r"D(\d+):(\d+)", dia_id)
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def _format_turn(turn: Dict) -> str:
    speaker = turn.get("speaker", "Unknown")
    text = turn.get("text", "")
    blip = turn.get("blip_caption", "")
    result = f"[{speaker}] {text}"
    if blip:
        result += f" (shared photo: {blip})"
    return result


def get_all_turns(conv: Dict) -> List[Dict]:
    """Extract ALL turns from a conversation, sorted by session + turn."""
    turns = []
    sess_keys = sorted(
        [k for k in conv if k.startswith("session_") and "_date_time" not in k],
        key=lambda k: int(re.findall(r"\d+", k.split("session_")[-1])[0]))

    for sk in sess_keys:
        sn = int(re.findall(r"\d+", sk.split("session_")[-1])[0])
        date_str = conv.get(f"session_{sn}_date_time", "")
        for turn in conv[sk]:
            did = turn.get("dia_id", "")
            _, tn = _parse_dia_id(did)
            turns.append({
                "dia_id": did,
                "speaker": turn.get("speaker", "Unknown"),
                "text": turn.get("text", ""),
                "blip_caption": turn.get("blip_caption", ""),
                "session_num": sn,
                "turn_num": tn,
                "date": date_str,
                "content": _format_turn(turn),
            })
    return turns


def get_qa_items(sample: Dict) -> List[Dict]:
    """Extract QA items from a sample."""
    return [{
        "question": qa.get("question", ""),
        "answer": qa.get("answer", ""),
        "category": qa.get("category", 0),
        "category_name": CATEGORY_NAMES.get(qa.get("category", 0), "unknown"),
        "evidence": qa.get("evidence", []),
    } for qa in sample.get("qa", [])]


# ── System Runners ──────────────────────────────────────────

class SystemRunner:
    """Base class for running a system on a full conversation."""
    def __init__(self, name: str):
        self.name = name

    def reset(self) -> None:
        pass

    def ingest_turn(self, turn: Dict) -> None:
        pass

    def answer(self, question: str) -> str:
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        return {}


class NoMemoryRunner(SystemRunner):
    """Backbone-only — ignores all history."""
    def __init__(self, backend, prompt: str):
        super().__init__("NoMemory")
        self._b = backend
        self._p = prompt

    def answer(self, q: str) -> str:
        return self._b.generate(self._p.format(query=q))


class FullContextRunner(SystemRunner):
    """All conversation concatenated into context window."""
    def __init__(self, backend, prompt: str, max_turns: int = 300):
        super().__init__("FullContext")
        self._b = backend
        self._p = prompt
        self._max = max_turns
        self._ctx: List[str] = []
        self._n = 0

    def reset(self) -> None:
        self._ctx = []
        self._n = 0

    def ingest_turn(self, t: Dict) -> None:
        if self._n < self._max:
            self._ctx.append(t["content"])
            self._n += 1

    def answer(self, q: str) -> str:
        ctx = "\n".join(self._ctx[-self._max:]) if self._ctx else "(no conversation)"
        return self._b.generate(self._p.format(query=q, context=ctx))

    def stats(self) -> Dict:
        return {"turns_ingested": self._n}


class ASEMRunner(SystemRunner):
    """Full ASEM pipeline — memory bank + linking + retrieval."""
    def __init__(self, asem_system):
        super().__init__("ASEM")
        self._s = asem_system
        self._ni = 0

    def reset(self) -> None:
        self._s.reset()
        self._ni = 0

    def ingest_turn(self, t: Dict) -> None:
        try:
            self._s.ingest(t["content"])
            self._ni += 1
        except Exception:
            pass

    def answer(self, q: str) -> str:
        try:
            _, a = self._s.pipeline.read_path(q)
            return a
        except Exception:
            return ""

    def stats(self) -> Dict:
        g = self._s.pipeline.memory_bank.get_link_graph()
        return {
            "turns_ingested": self._ni,
            "bank_size": self._s.bank_size,
            "graph_nodes": len(g["nodes"]),
            "graph_edges": len(g["edges"]),
        }


class BaselineRunner(SystemRunner):
    """Runner for SimRetrieval, AtomicLinking, RLManagerOnly, ValueRetrievalOnly."""
    def __init__(self, name: str, system):
        super().__init__(name)
        self._s = system
        self._turns: List[str] = []
        self._ni = 0

    def reset(self) -> None:
        if hasattr(self._s, "reset"):
            self._s.reset()
        self._turns = []
        self._ni = 0

    def ingest_turn(self, t: Dict) -> None:
        self._turns.append(t["content"])
        self._ni += 1

    def answer(self, q: str) -> str:
        try:
            return self._s.answer(q, self._turns)
        except Exception:
            return ""

    def stats(self) -> Dict:
        return {"turns_ingested": self._ni}


# ── Builder ─────────────────────────────────────────────────

def build_runner(name: str, config_path: str, db_dir: str,
                 max_turns: int = 300) -> SystemRunner:
    cfg = _load_config(config_path)
    if name == "NoMemory":
        return NoMemoryRunner(build_backend(cfg["inference"]), _NO_MEMORY_PROMPT)
    if name == "FullContext":
        return FullContextRunner(build_backend(cfg["inference"]),
                                _FULL_CONTEXT_PROMPT, max_turns)
    if name == "ASEM":
        return ASEMRunner(build_asem_system(config_path, db_dir))
    bl = build_baselines(config_path, db_dir, max_history_turns=max_turns)
    if name in bl:
        return BaselineRunner(name, bl[name])
    raise ValueError(f"Unknown system: {name}")


# ── Main eval loop ──────────────────────────────────────────

def run_eval(samples: List[Dict], systems: List[str], config_path: str,
             out_path: str, db_dir: str, overwrite: bool = False,
             judge: bool = False) -> Dict:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)

    # Resume support
    results: Dict = {}
    if os.path.exists(out_path) and not overwrite:
        with open(out_path, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                pass

    completed: Set[Tuple] = set()
    if "___completed___" in results:
        completed = set(tuple(x) for x in results["___completed___"])

    # Build runners
    runners: Dict[str, SystemRunner] = {}
    for sn in systems:
        if sn not in SYSTEM_NAMES:
            continue
        print(f"Building {sn} runner ...")
        runners[sn] = build_runner(sn, config_path, db_dir)

    total_qa = sum(len(s.get("qa", [])) for s in samples)
    print(f"\n{'='*60}")
    print(f"EVAL: {len(samples)} conversations, ~{total_qa} QA items")
    print(f"Systems: {list(runners.keys())}")
    print(f"{'='*60}\n")

    all_preds = {sn: {"preds": [], "refs": []} for sn in runners}

    for si, sample in enumerate(samples):
        sid = sample.get("sample_id", f"conv_{si}")
        conv = sample.get("conversation", sample)
        all_turns = get_all_turns(conv)
        qa_items = get_qa_items(sample)

        print(f"[Conv {si+1}/{len(samples)}] id={sid} | "
              f"{len(all_turns)} turns | {len(qa_items)} QA")

        for sn, runner in runners.items():
            ck = (str(sid), sn)
            if ck in completed:
                print(f"  [{sn}] already done — skip")
                continue

            print(f"  [{sn}] reset + ingest {len(all_turns)} turns ...",
                  end=" ", flush=True)
            t0 = time.time()
            try:
                runner.reset()
                for t in all_turns:
                    runner.ingest_turn(t)
                st = runner.stats()
                print(f"done ({time.time()-t0:.1f}s) | {st}")

                cp, cr = [], []
                for qi, qa in enumerate(qa_items):
                    a = runner.answer(qa["question"])
                    cp.append(a)
                    cr.append(qa["answer"])
                    if (qi + 1) % 10 == 0 or (qi + 1) == len(qa_items):
                        print(f"    [{sn}] QA {qi+1}/{len(qa_items)} | "
                              f"latest: {a[:60]!r}")

                ckey = f"{sid}/{sn}"
                results.setdefault(sn, {})[ckey] = {
                    "sample_id": sid,
                    "qa": [{"question": q["question"], "gold": q["answer"],
                            "pred": p, "category": q["category"]}
                           for q, p in zip(qa_items, cp)],
                    "stats": st,
                    "n_turns": len(all_turns),
                    "n_qa": len(qa_items),
                }

                all_preds[sn]["preds"].extend(cp)
                all_preds[sn]["refs"].extend(cr)

                if cp:
                    cm = compute_metrics(cp, cr)
                    results[sn][ckey]["metrics"] = cm
                    print(f"    [{sn}] EM={cm.get('em',0):.3f} "
                          f"ROUGE-L={cm.get('rougeL',0):.3f}")

                completed.add(ck)
                results["___completed___"] = [list(x) for x in completed]

            except Exception as exc:
                print(f"\n  [{sn}] ERROR: {exc}")
                traceback.print_exc()
                continue

        _flush(out_path, results)

    # ── Final summary ──
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    final = {}
    for sn, data in all_preds.items():
        if data["preds"]:
            m = compute_metrics(data["preds"], data["refs"])
            final[sn] = m
            results[f"__summary__{sn}"] = m
            print(f"  {sn:25s}  EM={m.get('em',0):.4f}  "
                  f"ROUGE-L={m.get('rougeL',0):.4f}  (n={len(data['preds'])})")

    results["__final__"] = final
    _flush(out_path, results)
    print(f"\n✅ Saved to: {out_path}")
    return results


def _flush(path: str, data: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ── CLI ─────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="LoCoMo QA Eval — ASEM Pipeline + Baselines")
    p.add_argument("--data-file", default="datasets/locomo/locomo10.json")
    p.add_argument("--config", default="configs/locomo_openai.yaml")
    p.add_argument("--out-file", default="outputs/locomo10_asem_eval.json")
    p.add_argument("--db-dir", default="data/benchmarks/eval_banks_locomo")
    p.add_argument("--systems", nargs="+",
                   default=["NoMemory", "FullContext", "ASEM"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--max-turns", type=int, default=300)
    args = p.parse_args()

    unknown = set(args.systems) - set(SYSTEM_NAMES)
    if unknown:
        print(f"WARNING: Unknown systems: {unknown}")
    args.systems = [s for s in args.systems if s in SYSTEM_NAMES]

    with open(args.data_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if args.limit:
        samples = samples[:args.limit]
    print(f"Loaded {len(samples)} conversations from {args.data_file}")

    run_eval(samples, args.systems, args.config, args.out_file,
             args.db_dir, args.overwrite, args.judge)


if __name__ == "__main__":
    main()
