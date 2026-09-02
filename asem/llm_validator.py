"""Output validation and retry handling for LLM-generated structured data.

Small language models often produce malformed or schema-violating JSON.
This module provides:

* **Validators** — check parsed LLM output against the expected schema
  (note fields, link arrays, memory ops, batch results) and return a
  structured ``ValidationResult`` with human-readable error messages.
* **LLMRetryHandler** — wraps ``backend.generate()`` so that on parse or
  validation failure the prompt is re-issued with an explicit format
  correction (including the previous bad output), up to ``max_retries``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from .logging_utils import get_logger

_log = get_logger("llm_validator")

# Transient network errors that are safe to retry (DNS blips, connection
# resets, timeouts). Matched by class name so we don't hard-depend on the
# specific SDK (openai/httpx) that raised them.
_TRANSIENT_ERROR_NAMES = {
    "APIConnectionError", "APIStatusError", "ConnectError",
    "ConnectTimeout", "ReadTimeout", "Timeout", "ConnectionError",
    "RemoteProtocolError", "NetworkError", "InternalServerError",
    "RateLimitError",
}


def _is_transient_network_error(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like a transient network/transport error."""
    if type(exc).__name__ in _TRANSIENT_ERROR_NAMES:
        return True
    # Walk the cause chain (openai wraps httpx errors in APIConnectionError).
    cause = getattr(exc, "__cause__", None)
    while cause is not None:
        if type(cause).__name__ in _TRANSIENT_ERROR_NAMES:
            return True
        cause = getattr(cause, "__cause__", None)
    return False

# The exact relation labels the S3 link prompts are allowed to emit.
# Any other value is a format violation that triggers a retry.
RELATION_TYPES: Set[str] = {
    "contradicts", "extends", "causal",
    "same-topic", "temporal", "semantic",
}

# Memory-manager operations.
MEMORY_OPS: Set[str] = {"ADD", "UPDATE", "DELETE", "NOOP"}


@dataclass
class ValidationResult:
    """Outcome of validating parsed LLM output."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    parsed: Any = None

    @classmethod
    def ok(cls, parsed: Any) -> "ValidationResult":
        return cls(valid=True, parsed=parsed)

    @classmethod
    def fail(cls, errors: List[str], parsed: Any = None) -> "ValidationResult":
        return cls(valid=False, errors=list(errors), parsed=parsed)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_note_fields(data: Any) -> ValidationResult:
    """Validate a single note dict: keywords (list), tags (list), description (str)."""
    if not isinstance(data, dict):
        return ValidationResult.fail(["expected a JSON object, got "
                                      f"{type(data).__name__}"], data)

    errors: List[str] = []

    keywords = data.get("keywords")
    if not isinstance(keywords, list):
        errors.append("'keywords' must be a JSON array of strings")
    elif any(not isinstance(k, str) for k in keywords):
        errors.append("every entry in 'keywords' must be a string")

    tags = data.get("tags")
    if not isinstance(tags, list):
        errors.append("'tags' must be a JSON array of strings")
    elif any(not isinstance(t, str) for t in tags):
        errors.append("every entry in 'tags' must be a string")

    desc = data.get("description")
    if not isinstance(desc, str):
        errors.append("'description' must be a string")

    if errors:
        return ValidationResult.fail(errors, data)
    return ValidationResult.ok(data)


def validate_link_array(
    data: Any,
    valid_source_id: Optional[str] = None,
    valid_target_ids: Optional[Set[str]] = None,
    allow_unknown_relations: bool = False,
) -> ValidationResult:
    """Validate a link array: each entry has source/target IDs and a known relation.

    Args:
        data: Parsed LLM output (expected list of dicts).
        valid_source_id: If given, every ``source`` must equal this ID.
        valid_target_ids: If given, every ``target`` must be in this set.
        allow_unknown_relations: If True, unknown relation labels are kept
            (defaults to "semantic" downstream) instead of failing.
    """
    if not isinstance(data, list):
        return ValidationResult.fail(["expected a JSON array of link objects, got "
                                      f"{type(data).__name__}"], data)

    errors: List[str] = []
    cleaned: List[Dict[str, str]] = []

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"entry {i} is not an object")
            continue
        source = item.get("source")
        target = item.get("target")
        relation = str(item.get("relation", "")).lower()

        if not isinstance(source, str) or not source:
            errors.append(f"entry {i} is missing a non-empty 'source'")
        if not isinstance(target, str) or not target:
            errors.append(f"entry {i} is missing a non-empty 'target'")
        if not relation:
            errors.append(f"entry {i} is missing 'relation'")

        if valid_source_id is not None and source != valid_source_id:
            errors.append(f"entry {i} 'source' must be the new note ID "
                          f"{valid_source_id!r}, got {source!r}")
        if valid_target_ids is not None and target not in valid_target_ids:
            errors.append(f"entry {i} 'target' {target!r} is not a provided neighbor")

        if relation and relation not in RELATION_TYPES:
            if allow_unknown_relations:
                relation = "semantic"
            else:
                errors.append(f"entry {i} has invalid relation {relation!r} — "
                              f"must be one of {sorted(RELATION_TYPES)}")

        cleaned.append({
            "source": str(source or ""),
            "target": str(target or ""),
            "relation": relation or "semantic",
        })

    if errors:
        return ValidationResult.fail(errors, cleaned or data)
    return ValidationResult.ok(cleaned)


def validate_memory_ops(data: Any, num_notes: int = -1) -> ValidationResult:
    """Validate memory-op decisions: index (int), op in {ADD,UPDATE,DELETE,NOOP}."""
    if not isinstance(data, list):
        return ValidationResult.fail(["expected a JSON array of decision objects, got "
                                      f"{type(data).__name__}"], data)

    errors: List[str] = []
    cleaned: List[Dict[str, Any]] = []

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"entry {i} is not an object")
            continue
        idx = item.get("index")
        op = str(item.get("op", "")).upper()
        target_id = item.get("target_id")

        if not isinstance(idx, int):
            errors.append(f"entry {i} 'index' must be an integer, got {idx!r}")
        if op not in MEMORY_OPS:
            errors.append(f"entry {i} has invalid op {op!r} — "
                          f"must be one of {sorted(MEMORY_OPS)}")
        if op in {"UPDATE", "DELETE"} and not isinstance(target_id, str):
            errors.append(f"entry {i} '{op}' requires a string 'target_id'")

        cleaned.append({
            "index": idx if isinstance(idx, int) else i,
            "op": op if op in MEMORY_OPS else "ADD",
            "target_id": target_id,
        })

    if errors:
        return ValidationResult.fail(errors, cleaned or data)
    return ValidationResult.ok(cleaned)


def validate_batch_notes(data: Any, expected_count: int = -1,
                         require_content: bool = False) -> ValidationResult:
    """Validate an array of note dicts (batch extraction / batch evolution).

    ``require_content`` should be set by extraction callers (P4) whose notes
    must carry a non-empty ``content`` — a response whose entries have the
    wrong shape (e.g. a content-part array) would otherwise validate fine
    and then be silently dropped during embedding.
    """
    if not isinstance(data, list):
        return ValidationResult.fail(["expected a JSON array of note objects, got "
                                      f"{type(data).__name__}"], data)
    if expected_count > 0 and len(data) != expected_count:
        return ValidationResult.fail(
            [f"expected exactly {expected_count} notes, got {len(data)}"], data)

    errors: List[str] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"entry {i} is not an object")
            continue
        for key in ("keywords", "tags", "description"):
            if key not in item:
                errors.append(f"entry {i} is missing '{key}'")
            elif key != "description" and not isinstance(item[key], list):
                errors.append(f"entry {i} '{key}' must be an array")
            elif key == "description" and not isinstance(item[key], str):
                errors.append(f"entry {i} 'description' must be a string")
        if require_content:
            content = item.get("content", "")
            if not isinstance(content, str) or not content.strip():
                errors.append(f"entry {i} is missing non-empty 'content'")

    if errors:
        return ValidationResult.fail(errors, data)
    return ValidationResult.ok(data)


def validate_distil_response(data: Any) -> ValidationResult:
    """Validate the answer-agent output: selected_ids (list) + answer (str)."""
    if not isinstance(data, dict):
        return ValidationResult.fail(["expected a JSON object, got "
                                      f"{type(data).__name__}"], data)

    errors: List[str] = []
    ids = data.get("selected_ids")
    if not isinstance(ids, list):
        errors.append("'selected_ids' must be a JSON array")
    if not isinstance(data.get("answer"), str):
        errors.append("'answer' must be a string")

    if errors:
        return ValidationResult.fail(errors, data)
    return ValidationResult.ok(data)


def validate_summary(data: Any) -> ValidationResult:
    """Validate the utility-updater summary output (a short factual string)."""
    if isinstance(data, str) and data.strip():
        return ValidationResult.ok(data.strip())
    return ValidationResult.fail(["expected a non-empty string summary"], data)


# ---------------------------------------------------------------------------
# Retry handler
# ---------------------------------------------------------------------------

_FALLBACK_VALIDATOR: Callable[[Any], ValidationResult] = lambda data: (
    ValidationResult.ok(data)
)


@dataclass
class LLMRetryHandler:
    """Call an LLM with retry on parse or validation failure.

    The retry prompt appends a FORMAT CORRECTION block that quotes the
    previous (bad) output and names the exact errors — this gives small
    models a concrete target to fix rather than re-rolling the dice.
    """

    generate_fn: Callable[[str], str]
    max_retries: int = 2
    # Optional: an initial prompt wrapper injected before every call
    # (e.g. a system-style preamble). Kept for future use.

    def invoke(
        self,
        prompt: str,
        parse_fn: Callable[[str], Any],
        validate_fn: Callable[[Any], ValidationResult] = _FALLBACK_VALIDATOR,
    ) -> tuple[Any, int]:
        """Generate + parse + validate, retrying on failure.

        Returns ``(parsed, attempt)`` where ``attempt`` is the 0-based
        attempt that succeeded, or ``max_retries`` if all attempts failed
        (the last parsed-but-invalid result is still returned so callers
        can salvage it best-effort).
        """
        last_parsed: Any = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.generate_fn(prompt)
            except Exception as exc:  # noqa: BLE001
                if _is_transient_network_error(exc) and attempt < self.max_retries:
                    backoff = min(2 ** attempt, 15)
                    _log.warning(
                        "Transient network error (attempt {}/{}): {} — retrying in {}s",
                        attempt + 1, self.max_retries + 1,
                        type(exc).__name__, backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise
            last_parsed = parse_fn(raw)

            if last_parsed is None:
                if attempt < self.max_retries:
                    prompt = self._build_retry_prompt(
                        prompt, raw,
                        "Your previous response was NOT valid JSON — it could "
                        "not be parsed at all. Return ONLY the JSON object/array "
                        "with no markdown fences, no commentary, and no trailing text.",
                        attempt,
                    )
                    continue
                break

            result = validate_fn(last_parsed)
            if result.valid:
                # Return the cleaned data (validators may normalize labels),
                # falling back to the raw parse if the validator passed it
                # through unchanged.
                return result.parsed, attempt

            if attempt < self.max_retries:
                correction = "; ".join(result.errors[:6])
                prompt = self._build_retry_prompt(
                    prompt, raw,
                    f"Your previous response violated the output format: "
                    f"{correction}. Fix EVERY error and return ONLY the "
                    f"correctly formatted JSON.",
                    attempt,
                )
            else:
                _log.warning(
                    "Validation failed on final attempt | errors={}", result.errors
                )

        return last_parsed, self.max_retries

    # ------------------------------------------------------------------

    def _build_retry_prompt(
        self, original_prompt: str, previous_output: str,
        correction: str, attempt: int,
    ) -> str:
        snippet = previous_output.strip()[:500] if previous_output else ""
        return (
            f"{original_prompt}\n\n"
            f"=== FORMAT CORRECTION (attempt {attempt + 1}/"
            f"{self.max_retries + 1}) ===\n"
            f"{correction}\n"
            f"Your previous output was:\n```\n{snippet}\n```\n"
            f"Now produce the corrected output. Output ONLY the JSON."
        )
