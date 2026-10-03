"""Prompt-budget arithmetic shared by every system that answers a question.

Every system in the benchmark answers with **one** model call, and on an
OpenAI-compatible endpoint a too-long prompt is a hard HTTP 400 — the row is
lost, not degraded. So whatever assembles the context (``FullContext``
concatenating the whole history, a retrieval baseline joining its top-k notes,
or ``AnswerAgent`` rendering the JSON payload) has to respect the budget of the
answer call::

    estimate_tokens(prompt) <= context_window - max_tokens - SAFETY_MARGIN_TOKENS

``context_window`` is the model's total window and ``max_tokens`` the completion
reserved inside it. When a caller passes no cap, the backend's own client-wide
cap is used instead (``InferenceBackend.default_max_tokens``), so the
reservation always matches what the request will actually ask for.

Both the *estimate* and the *trimming arithmetic* live here, so the "max
history" of ``FullContext``, the top-k context of the retrieval baselines and
the ASEM note payload can never drift apart.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar

from .logging_utils import get_logger

_log = get_logger("token_budget")

# ~4 chars/token. Deliberately conservative (JSON punctuation, non-ASCII): it
# only decides how much low-ranked context to drop, and dropping one extra note
# is harmless while keeping one too many is a hard 400 on a small-context model.
CHARS_PER_TOKEN = 4

# Reserved on top of the completion: the character estimate can undercount and a
# 1-token overflow is still a 400.
SAFETY_MARGIN_TOKENS = 64

# A block this short cannot be halved usefully — drop it instead of clipping.
_MIN_BLOCK_CHARS = 24

T = TypeVar("T")


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for a rendered prompt."""
    return max(1, len(text or "") // CHARS_PER_TOKEN)


def default_output_cap(backend: Any) -> Optional[int]:
    """The backend's client-wide completion cap, when it advertises one."""
    cap = getattr(backend, "default_max_tokens", None)
    try:
        cap = int(cap)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return cap or None


def prompt_budget(
    context_window: Optional[int],
    max_tokens: Optional[int],
    *,
    safety: int = SAFETY_MARGIN_TOKENS,
) -> Optional[int]:
    """Tokens available to the *prompt*: window minus the reserved completion.

    Returns ``None`` when no window is declared, which disables trimming (the
    historical behaviour of ``answer.context_window`` being unset).
    """
    if not context_window:
        return None
    return int(context_window) - int(max_tokens or 0) - int(safety)


def resolve_budget(
    backend: Any,
    context_window: Optional[int],
    max_tokens: Optional[int],
    *,
    safety: int = SAFETY_MARGIN_TOKENS,
) -> Optional[int]:
    """``prompt_budget`` using the backend's own cap when none was passed."""
    cap = int(max_tokens) if max_tokens else default_output_cap(backend)
    return prompt_budget(context_window, cap, safety=safety)


def fits(prompt: str, budget: Optional[int]) -> bool:
    """True when a rendered prompt fits the budget (or there is no budget)."""
    return budget is None or estimate_tokens(prompt) <= budget


def clip_block(
    text: str,
    *,
    keep: str = "head",
    min_chars: int = _MIN_BLOCK_CHARS,
) -> Optional[str]:
    """Halve one context block; ``None`` when it cannot shrink usefully further.

    Used as the last-resort ``shrink`` callback: a single oversized turn/note can
    still blow a small window even after every other block was dropped.
    ``keep="tail"`` preserves the most recent half (useful for ``[date] text``
    turns, where the date prefix is at the front), ``keep="head"`` the opening.

    Both branches are **non-growing**: ``len(result) < len(text)`` always, so
    ``fit_items``' shrink loop is guaranteed to terminate. This property is what
    makes the final rendering a hard contract rather than a best effort — see
    ``fit_render_exact``.
    """
    if not text or len(text) <= min_chars:
        return None
    keep_len = max(1, len(text) // 2)
    if keep == "tail":
        return "… " + text[-keep_len:]
    return text[:keep_len].rstrip() + " …"


def fit_render_exact(
    items: Sequence[T],
    render: Callable[[Sequence[T]], str],
    *,
    budget: Optional[int],
    fallback: str,
) -> Tuple[str, List[T]]:
    """Last-resort guarantee that the returned prompt fits ``budget`` EXACTLY.

    ``fit_items`` trims by *dropping whole items* and then tries a ``shrink``
    callback. Neither can help when the budget is smaller than the fixed
    scaffolding the renderer cannot remove — the prompt template alone, or a
    ``min_keep=1`` item whose block is larger than the whole budget. In that case
    ``fit_items`` logs a warning and returns an over-budget prompt anyway, and on
    an OpenAI-compatible endpoint that is a hard HTTP 400.

    This wrapper closes that hole by construction: if the rendered prompt still
    exceeds ``budget`` it *truncates the rendered string itself* — dropping the
    highest-ranked items first so the retained text is always a prefix of the
    full render, and finally falling back to ``fallback`` (the caller's
    no-context prompt) when even that does not fit. Truncation is a character
    operation, so it cannot be defeated by how the renderer chose to lay text
    out.

    The postcondition is therefore unconditional::

        budget is None  or  estimate_tokens(prompt) <= budget

    Returns ``(prompt, kept_items)``; ``kept_items`` is a prefix of ``items``
    (empty when the caller's own scaffolding had to be truncated).
    """
    if budget is None:
        return render(list(items)), list(items)

    # `budget <= 0` is a real (if degenerate) budget: `estimate_tokens` floors at
    # 1, so no non-empty prompt can satisfy it. Handle it explicitly rather than
    # letting the arithmetic below fall through, and never return an over-budget
    # string — the caller's fallback is the only correct answer.
    if int(budget) <= 0:
        return fallback, []

    kept = list(items)
    prompt = render(kept)
    if fits(prompt, budget):
        return prompt, kept

    # Step 1: binary search over how MANY items the renderer can afford.
    lo, hi, best = 0, len(kept), None
    while lo <= hi:
        mid = (lo + hi) // 2
        # `mid` items means the FIRST `mid`: relevance order puts the best notes
        # at the front, so the retained block set is always a prefix.
        if fits(render(kept[:mid]), budget):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    kept = kept[: best if best is not None else 0]
    prompt = render(kept)
    if fits(prompt, budget):
        return prompt, kept

    # Step 2: the scaffolding itself does not fit. Truncate the rendered string.
    limit_chars = max(1, int(budget) * CHARS_PER_TOKEN)
    if len(prompt) > limit_chars:
        # `rsplit` on whitespace when there is one, so the tail is not a partial
        # token; a hard cut otherwise (a render with no spaces at all).
        cut = prompt[:limit_chars].rsplit(" ", 1)[0]
        prompt = cut or prompt[:limit_chars]
    # A truncation that still cannot fit (e.g. a non-ASCII render where chars and
    # tokens disagree) falls back to the caller's minimal prompt.
    if not fits(prompt, budget) and fits(fallback, budget):
        _log.warning(
            "{}: even a truncated render exceeds the {}-token budget; "
            "falling back to the bare prompt ({} chars).",
            "fit_render_exact", budget, len(fallback),
        )
        return fallback, []
    return prompt, kept


def _select(items: Sequence[T], n: int, *, drop: str, head_keep: int) -> List[T]:
    """Keep ``n`` items out of ``items`` under the ``drop`` policy."""
    if n >= len(items):
        return list(items)
    if drop == "tail":
        # Relevance order: the lowest-ranked items live at the end.
        return list(items[:n])
    if drop == "oldest":
        # Chronological order: keep the opening (setup/context) plus the most
        # RECENT items, so the history loses its middle, not its ends.
        head = list(items[: min(head_keep, n)])
        tail_n = n - len(head)
        return head + (list(items[len(items) - tail_n:]) if tail_n > 0 else [])
    raise ValueError(f"unknown drop policy: {drop!r} (expected 'tail' or 'oldest')")


def fit_items(
    items: Sequence[T],
    render: Callable[[Sequence[T]], str],
    *,
    budget: Optional[int] = None,
    min_keep: int = 1,
    drop: str = "tail",
    head_keep: int = 0,
    shrink: Optional[Callable[[T], Optional[T]]] = None,
    label: str = "prompt",
) -> Tuple[str, List[T]]:
    """Shrink ``items`` until ``render(kept)`` fits ``budget`` tokens.

    Returns ``(prompt, kept_items)``, where ``kept_items`` is always a subset of
    ``items`` in the original order:

    * ``drop="tail"``   — drops the lowest-ranked items first (retrieved notes,
      which arrive in relevance order).
    * ``drop="oldest"`` — keeps the first ``head_keep`` items plus as many of the
      most recent ones as fit (chronological history).

    The rendered size is monotone in the number of kept items, so the largest
    fitting slice is found by **binary search** (O(log n) renders) instead of
    dropping one item at a time — the difference between a fast and a quadratic
    FullContext answer. When even ``min_keep`` items do not fit, ``shrink`` is
    applied to the longest block until it fits or cannot shrink further; if the
    prompt *still* overflows (i.e. the instruction + query alone exceed the
    window) a warning is logged, because no amount of context trimming helps.
    """
    kept = list(items)
    prompt = render(kept)
    if budget is None:
        return prompt, kept

    if not fits(prompt, budget) and len(kept) > min_keep:
        lo, hi, best = min_keep, len(kept), None
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = _select(kept, mid, drop=drop, head_keep=head_keep)
            if fits(render(candidate), budget):
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        kept = _select(kept, best if best is not None else min_keep,
                       drop=drop, head_keep=head_keep)
        prompt = render(kept)

    while shrink is not None and kept and not fits(prompt, budget):
        # Shrink the longest block first: it frees the most tokens per char lost.
        shrank = False
        for index in sorted(range(len(kept)), key=lambda i: -len(str(kept[i]))):
            smaller = shrink(kept[index])
            if smaller is None:
                continue
            kept[index] = smaller
            prompt = render(kept)
            shrank = True
            if fits(prompt, budget):
                break
        if not shrank:
            break  # nothing left that can shrink

    est = estimate_tokens(prompt)
    if not fits(prompt, budget):
        _log.warning(
            "{}: ~{} tokens even after trimming to {}/{} blocks (budget {}) — "
            "lower the completion cap or raise the model context window.",
            label, est, len(kept), len(items), budget,
        )
    elif len(kept) < len(items) or kept != list(items):
        _log.debug(
            "{}: kept {}/{} blocks (~{} tokens, budget {})",
            label, len(kept), len(items), est, budget,
        )
    return prompt, kept


def generate_with_cap(
    backend: Any,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """One generation call, passing the completion cap where the backend takes it.

    The cap is the other half of the budget: reserving it in the prompt while
    letting the request use a larger client-wide default would still overflow.
    Backends/wrappers without per-call kwargs fall back to their own default.

    ``temperature`` is forwarded the same way when supplied. Without it the
    backend's config-level value applies, which is how `answer.temperature` in the
    YAML came to be silently ignored.
    """
    kwargs: Dict[str, Any] = {}
    if max_tokens:
        kwargs["max_tokens"] = int(max_tokens)
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if not kwargs:
        return backend.generate(prompt)
    try:
        return backend.generate(prompt, **kwargs)
    except TypeError as exc:
        if "unexpected keyword" in str(exc) or "positional argument" in str(exc):
            _log.debug("Backend ignores per-call kwargs: {}", exc)
            # Retry with only the cap, then bare — backends vary in what they take.
            if "max_tokens" in kwargs:
                try:
                    return backend.generate(prompt, max_tokens=kwargs["max_tokens"])
                except TypeError:
                    pass
            return backend.generate(prompt)
        raise
