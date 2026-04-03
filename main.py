"""Runnable ASEM demo entrypoint.

This script provides a deterministic local demo backend so the full ASEM
pipeline can be demonstrated without external model APIs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from asem.answer_agent import AnswerAgent
from asem.backends import build_backend
from asem.backends.base import InferenceBackend
from asem.link_evolver import LinkEvolver
from asem.memory_bank import MemoryBank
from asem.memory_manager import MemoryManager, Op
from asem.note import Note, NoteConstructor
from asem.pipeline import ASEMPipeline
from asem.retriever import HybridRetriever
from asem.utility_updater import UtilityUpdater


NOTE_PROMPT = "ASEM_STAGE=NOTE\nCONTENT:{content}"
WRITE_PROMPT = "ASEM_STAGE=WRITE_OP\nCONTENT:{content}\nMEMORY:{memory}"
LINK_PROMPT = "ASEM_STAGE=LINK\nNEW_NOTE:{new_note}\nNEIGHBORS:{neighbors}"
EVOLVE_PROMPT = "ASEM_STAGE=EVOLVE\nEXISTING_NOTE:{existing_note}\nNEW_NOTE:{new_note}"
ANSWER_PROMPT = "ASEM_STAGE=ANSWER\nQUERY:{query}\nCANDIDATES:{candidates}"
BASELINE_PROMPT = "ASEM_STAGE=BASELINE\nQUERY:{query}\nCONTEXT:{context}"
SUMMARY_PROMPT = "ASEM_STAGE=SUMMARY\nQUERY:{query}\nANSWER:{answer}\nREWARD:{reward}"

DEFAULT_FACTS = [
	"My name is Alex.",
	"I adopted a dog named Buddy.",
	"I also adopted another dog named Scout.",
	"I work as a data scientist at a tech company.",
]

DEFAULT_QUERIES = [
	"What dogs do I have?",
	"What is my job?",
	"What is my name?",
]


class DemoBackend(InferenceBackend):
	"""Deterministic local backend for first-run ASEM demonstrations."""

	def __init__(self, embed_dim: int = 64) -> None:
		self._embed_dim = embed_dim

	def generate(self, prompt: str, **kwargs) -> str:
		if "ASEM_STAGE=NOTE" in prompt:
			content = self._extract_block(prompt, "CONTENT:")
			return json.dumps(self._extract_note_fields(content))
		if "ASEM_STAGE=WRITE_OP" in prompt:
			content = self._extract_block(prompt, "CONTENT:", "MEMORY:")
			memory_raw = self._extract_block(prompt, "MEMORY:")
			memory = self._safe_json(memory_raw, default=[])
			return json.dumps(self._select_operation(content, memory))
		if "ASEM_STAGE=LINK" in prompt:
			new_note_raw = self._extract_block(prompt, "NEW_NOTE:", "NEIGHBORS:")
			neighbors_raw = self._extract_block(prompt, "NEIGHBORS:")
			new_note = self._safe_json(new_note_raw, default={})
			neighbors = self._safe_json(neighbors_raw, default=[])
			return json.dumps(self._link_relations(new_note, neighbors))
		if "ASEM_STAGE=EVOLVE" in prompt:
			existing_raw = self._extract_block(prompt, "EXISTING_NOTE:", "NEW_NOTE:")
			new_raw = self._extract_block(prompt, "NEW_NOTE:")
			existing = self._safe_json(existing_raw, default={})
			new_note = self._safe_json(new_raw, default={})
			return json.dumps(self._evolve_fields(existing, new_note))
		if "ASEM_STAGE=ANSWER" in prompt:
			query = self._extract_block(prompt, "QUERY:", "CANDIDATES:")
			candidates_raw = self._extract_block(prompt, "CANDIDATES:")
			candidates = self._safe_json(candidates_raw, default=[])
			return json.dumps(self._distil_and_answer(query, candidates))
		if "ASEM_STAGE=BASELINE" in prompt:
			return "I do not have enough memory context to answer confidently yet."
		if "ASEM_STAGE=SUMMARY" in prompt:
			query = self._extract_block(prompt, "QUERY:", "ANSWER:")
			answer = self._extract_block(prompt, "ANSWER:", "REWARD:")
			reward = self._extract_block(prompt, "REWARD:")
			return f"Query: {query.strip()} | Answer: {answer.strip()} | Reward: {reward.strip()}"

		return "{}"

	def embed(self, text: str):
		tokens = re.findall(r"[a-z0-9]+", text.lower())
		vec = [0.0] * self._embed_dim
		if not tokens:
			tokens = ["empty"]

		for token in tokens:
			digest = hashlib.md5(token.encode("utf-8")).hexdigest()
			idx = int(digest, 16) % self._embed_dim
			vec[idx] += 1.0

		norm = sum(v * v for v in vec) ** 0.5
		if norm == 0:
			return _to_vector(vec)
		return _to_vector([v / norm for v in vec])

	@staticmethod
	def _extract_block(prompt: str, start: str, end: Optional[str] = None) -> str:
		start_idx = prompt.find(start)
		if start_idx < 0:
			return ""
		start_idx += len(start)
		if end is None:
			return prompt[start_idx:].strip()
		end_idx = prompt.find(end, start_idx)
		if end_idx < 0:
			end_idx = len(prompt)
		return prompt[start_idx:end_idx].strip()

	@staticmethod
	def _safe_json(payload: str, default: Any) -> Any:
		try:
			return json.loads(payload)
		except json.JSONDecodeError:
			pass

		try:
			parsed = ast.literal_eval(payload)
		except (ValueError, SyntaxError):
			return default

		return parsed

	def _extract_note_fields(self, content: str) -> Dict[str, Any]:
		text = content.strip()
		lower = text.lower()

		keywords: List[str] = []
		tags: List[str] = []
		description = text

		if "name is" in lower:
			name = self._extract_name(lower)
			if name:
				keywords.extend([name, "name", "identity"])
				tags.extend(["profile", "personal"])
				description = f"User's name is {name.title()}."
		if "dog" in lower or "buddy" in lower or "scout" in lower:
			pet_names = self._extract_pet_names(text)
			keywords.extend(["dog", "pet"] + pet_names)
			tags.append("pets")
			if pet_names:
				pretty = ", ".join(name.title() for name in pet_names)
				description = f"User has dog(s): {pretty}."
		if "work" in lower or "job" in lower or "scientist" in lower:
			keywords.extend(["job", "career", "data scientist"])
			tags.extend(["career", "professional"])
			description = "User works as a data scientist at a tech company."

		if not keywords:
			keywords = list(_token_set(lower))[:4] or ["general"]
			tags = ["general"]
			description = text

		return {
			"keywords": list(dict.fromkeys(k for k in keywords if k)),
			"tags": list(dict.fromkeys(t for t in tags if t)),
			"description": description,
		}

	def _select_operation(self, content: str, memory: List[Dict[str, Any]]) -> Dict[str, Any]:
		if _is_query(content):
			return {"op": "NOOP", "target_id": None}

		fields = self._extract_note_fields(content)
		new_keywords = {k.lower() for k in fields["keywords"]}

		for note in memory:
			note_keywords = {str(k).lower() for k in note.get("keywords", [])}
			if new_keywords & note_keywords:
				return {"op": "UPDATE", "target_id": note.get("id")}

		return {"op": "ADD", "target_id": None}

	@staticmethod
	def _link_relations(new_note: Dict[str, Any], neighbors: List[Dict[str, Any]]) -> List[Dict[str, str]]:
		new_id = str(new_note.get("id", ""))
		new_tags = {str(t).lower() for t in new_note.get("tags", [])}
		relations: List[Dict[str, str]] = []

		for neighbor in neighbors:
			target = str(neighbor.get("id", ""))
			if not target or target == new_id:
				continue

			tags = {str(t).lower() for t in neighbor.get("tags", [])}
			rel = "semantic-related"
			if new_tags & tags:
				rel = "same-domain"
			relations.append({"source": new_id, "target": target, "relation": rel})

		return relations[:2]

	@staticmethod
	def _evolve_fields(existing: Dict[str, Any], new_note: Dict[str, Any]) -> Dict[str, Any]:
		keywords = list(dict.fromkeys(list(existing.get("keywords", [])) + list(new_note.get("keywords", []))))
		tags = list(dict.fromkeys(list(existing.get("tags", [])) + list(new_note.get("tags", []))))
		existing_desc = str(existing.get("description", "")).strip()
		new_desc = str(new_note.get("description", "")).strip()
		description = existing_desc if not new_desc else f"{existing_desc} | Related update: {new_desc}".strip(" |")
		return {"keywords": keywords, "tags": tags, "description": description}

	def _distil_and_answer(self, query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
		q = query.lower()
		scored: List[Tuple[float, Dict[str, Any]]] = []
		q_tokens = set(_token_set(q))

		for note in candidates:
			content = str(note.get("content", "")).lower()
			keywords = {str(k).lower() for k in note.get("keywords", [])}
			overlap = len(q_tokens & set(_token_set(content))) + len(q_tokens & keywords)
			score = float(overlap) + float(note.get("utility", 0.0))
			scored.append((score, note))

		scored.sort(key=lambda item: item[0], reverse=True)
		selected = [item[1] for item in scored[:2]] if scored else []
		selected_ids = [str(note.get("id", "")) for note in selected if note.get("id")]

		if "dog" in q or "pet" in q:
			dogs = self._extract_all_pet_names(selected)
			if dogs:
				answer = "You have two dogs: " + ", ".join(name.title() for name in dogs) + "."
			else:
				answer = "I could not find your dog names in memory yet."
		elif "job" in q or "work" in q or "career" in q:
			answer = "You work as a data scientist at a tech company."
		elif "name" in q or "who am i" in q:
			name = self._extract_name_from_candidates(selected)
			answer = f"Your name is {name.title()}." if name else "I could not find your name in memory yet."
		else:
			answer = str(selected[0].get("content", "I need more context.")) if selected else "I need more context."

		return {"selected_ids": selected_ids, "answer": answer}

	@staticmethod
	def _extract_name(text: str) -> str:
		m = re.search(r"name is\s+([a-z]+)", text)
		return m.group(1) if m else ""

	@staticmethod
	def _extract_pet_names(text: str) -> List[str]:
		found = re.findall(r"(?:named\s+|dog\s+named\s+|named\s+)([A-Za-z]+)", text)
		if not found:
			# Capture simple explicit names from known demo entities.
			for name in ("buddy", "scout"):
				if name in text.lower():
					found.append(name)
		return list(dict.fromkeys(name.lower() for name in found))

	def _extract_all_pet_names(self, notes: Iterable[Dict[str, Any]]) -> List[str]:
		names: List[str] = []
		for note in notes:
			names.extend(self._extract_pet_names(str(note.get("content", ""))))
			for kw in note.get("keywords", []):
				if str(kw).lower() in {"buddy", "scout"}:
					names.append(str(kw).lower())
		return list(dict.fromkeys(names))

	def _extract_name_from_candidates(self, notes: List[Dict[str, Any]]) -> str:
		for note in notes:
			content = str(note.get("content", "")).lower()
			name = self._extract_name(content)
			if name:
				return name
		return ""


def _to_vector(values: List[float]):
	import numpy as np

	return np.asarray(values, dtype=float)


def _token_set(text: str) -> List[str]:
	return re.findall(r"[a-z0-9]+", text.lower())


def _is_query(text: str) -> bool:
	return "?" in text or bool(re.match(r"^(what|who|where|when|how|which|do|does|is|are)\b", text.strip(), re.I))


def _load_text(path: str) -> str:
	with open(path, "r", encoding="utf-8") as handle:
		return handle.read()


def build_demo_pipeline(db_path: str) -> ASEMPipeline:
	backend = DemoBackend()

	note_constructor = NoteConstructor(backend=backend, prompt_template=NOTE_PROMPT, q0=0.5)
	memory_manager = MemoryManager(backend=backend, prompt_template=WRITE_PROMPT)
	link_evolver = LinkEvolver(
		backend=backend,
		link_prompt_template=LINK_PROMPT,
		evolve_prompt_template=EVOLVE_PROMPT,
		k=5,
	)
	retriever = HybridRetriever(
		backend=backend,
		k1=20,
		k2=5,
		delta=0.1,
		lambda_weight=0.4,
	)
	answer_agent = AnswerAgent(
		backend=backend,
		prompt_template=ANSWER_PROMPT,
		baseline_prompt_template=BASELINE_PROMPT,
	)
	updater = UtilityUpdater(
		backend=backend,
		alpha=0.1,
		q0=0.5,
		summary_prompt_template=SUMMARY_PROMPT,
		note_constructor=note_constructor,
	)

	return ASEMPipeline(
		memory_bank=MemoryBank(db_path),
		note_constructor=note_constructor,
		memory_manager=memory_manager,
		link_evolver=link_evolver,
		retriever=retriever,
		answer_agent=answer_agent,
		utility_updater=updater,
	)


def build_pipeline_from_config(config_path: str, db_path: str) -> ASEMPipeline:
	import yaml

	with open(config_path, "r", encoding="utf-8") as handle:
		cfg = yaml.safe_load(handle)

	backend = build_backend(cfg["inference"])
	hp = cfg["hyperparameters"]

	note_prompt = _load_text("data/prompts/P1_note_construction.txt")
	link_prompt = _load_text("data/prompts/P2_link_generation.txt")
	evolve_prompt = _load_text("data/prompts/P3_memory_evolution.txt")

	note_constructor = NoteConstructor(backend=backend, prompt_template=note_prompt, q0=hp["q0"])
	memory_manager = MemoryManager(backend=backend, prompt_template=WRITE_PROMPT)
	link_evolver = LinkEvolver(
		backend=backend,
		link_prompt_template=link_prompt,
		evolve_prompt_template=evolve_prompt,
		k=hp["k"],
	)
	retriever = HybridRetriever(
		backend=backend,
		k1=hp["k1"],
		k2=hp["k2"],
		delta=hp["delta"],
		lambda_weight=hp["lambda"],
	)
	answer_agent = AnswerAgent(
		backend=backend,
		prompt_template=ANSWER_PROMPT,
		baseline_prompt_template=BASELINE_PROMPT,
	)
	updater = UtilityUpdater(
		backend=backend,
		alpha=hp["alpha"],
		q0=hp["q0"],
		summary_prompt_template=SUMMARY_PROMPT,
		note_constructor=note_constructor,
	)

	return ASEMPipeline(
		memory_bank=MemoryBank(db_path),
		note_constructor=note_constructor,
		memory_manager=memory_manager,
		link_evolver=link_evolver,
		retriever=retriever,
		answer_agent=answer_agent,
		utility_updater=updater,
	)


def _write_with_trace(pipeline: ASEMPipeline, content: str, timestamp: datetime) -> None:
	note = pipeline.note_constructor.build(content, timestamp)
	existing = pipeline.memory_bank.list_notes()
	op, target = pipeline.memory_manager.select_op(content, existing)
	target_id = target.id if target else "-"
	print(f"[S2] OP={op.value} target={target_id} | content={content}")

	if op == Op.ADD:
		pipeline.memory_bank.add(note)
		pipeline.link_evolver.link_and_evolve(note, pipeline.memory_bank)
		print(f"[S1->S3] added note {note.id}")
		return

	if op == Op.UPDATE:
		updated = pipeline._merge_update(target, note)
		pipeline.memory_bank.add(updated)
		pipeline.link_evolver.link_and_evolve(updated, pipeline.memory_bank)
		print(f"[S1->S3] updated note {updated.id}")
		return

	if op == Op.DELETE:
		if target is not None:
			pipeline.memory_bank.delete(target.id)
			print(f"[S2] deleted note {target.id}")
		return

	print("[S2] noop (no write)")


def _read_update_with_trace(pipeline: ASEMPipeline, query: str, reward: float) -> None:
	candidates = pipeline.retriever.retrieve(query, pipeline.memory_bank)
	print(f"[S4] query='{query}' candidates={[n.id for n in candidates]}")

	used_notes, answer = pipeline.answer_agent.distil_and_answer(query, candidates)
	used_ids = [n.id for n in used_notes]
	print(f"[S4] distilled={used_ids}")
	print(f"[ANSWER] {answer}")

	before = {n.id: n.q for n in pipeline.memory_bank.list_notes()}
	pipeline.update_path(reward, used_notes, query, answer)
	after = {n.id: n.q for n in pipeline.memory_bank.list_notes()}

	print(f"[S5] reward={reward:.2f}")
	for note_id in used_ids:
		old = before.get(note_id)
		new = after.get(note_id)
		if old is not None and new is not None:
			print(f"  q({note_id}): {old:.3f} -> {new:.3f}")


def _print_bank_snapshot(pipeline: ASEMPipeline) -> None:
	notes = pipeline.memory_bank.list_notes()
	print("\n=== MEMORY BANK SNAPSHOT ===")
	if not notes:
		print("(empty)")
		return

	for note in notes:
		print(
			f"- id={note.id} q={note.q:.3f} links={len(note.L)} "
			f"K={note.K[:4]} G={note.G[:3]} X={note.X[:80]}"
		)


def run_seed_demo(pipeline: ASEMPipeline, reward: float) -> None:
	print("\n=== SEEDED DEMO: WRITE PATH ===")
	now = datetime.now(UTC)
	for fact in DEFAULT_FACTS:
		_write_with_trace(pipeline, fact, now)

	print("\n=== SEEDED DEMO: READ + UPDATE PATH ===")
	for query in DEFAULT_QUERIES:
		_read_update_with_trace(pipeline, query, reward)

	_print_bank_snapshot(pipeline)


def main() -> None:
	parser = argparse.ArgumentParser(description="Run a complete ASEM first demo pipeline")
	parser.add_argument(
		"--mode",
		choices=["demo", "config"],
		default="demo",
		help="demo: deterministic local backend; config: use configs/default.yaml backend",
	)
	parser.add_argument("--config", default="configs/default.yaml")
	parser.add_argument("--db", default="data/benchmarks/demo_bank.sqlite")
	parser.add_argument("--reward", type=float, default=0.8)
	parser.add_argument("--no-seed", action="store_true", help="Skip the seeded scenario")
	parser.add_argument("--interactive", action="store_true", help="Enter interactive chat after seeded run")
	parser.add_argument("--reset-db", action="store_true", help="Delete existing sqlite db before starting")

	args = parser.parse_args()

	os.makedirs(os.path.dirname(args.db), exist_ok=True)
	if args.reset_db and os.path.exists(args.db):
		os.remove(args.db)

	pipeline = build_demo_pipeline(args.db) if args.mode == "demo" else build_pipeline_from_config(args.config, args.db)

	if not args.no_seed:
		run_seed_demo(pipeline, args.reward)

	if not args.interactive:
		return

	print("\n=== INTERACTIVE MODE ===")
	print("Type '/exit' to quit. Prefix '/r 0.9' to set reward.")
	reward = args.reward

	while True:
		user_input = input("you> ").strip()
		if not user_input:
			continue
		if user_input == "/exit":
			break
		if user_input.startswith("/r "):
			try:
				reward = float(user_input.split(maxsplit=1)[1])
				print(f"reward set to {reward:.2f}")
			except ValueError:
				print("invalid reward value")
			continue

		if _is_query(user_input):
			_read_update_with_trace(pipeline, user_input, reward)
		else:
			_write_with_trace(pipeline, user_input, datetime.now(UTC))

	_print_bank_snapshot(pipeline)


if __name__ == "__main__":
	main()
