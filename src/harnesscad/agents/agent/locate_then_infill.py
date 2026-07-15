"""Locate-then-infill edit-plan construction, mined from CAD-Editor
(ICML 2025, arXiv:2502.03997).

CAD-Editor frames text-based CAD editing as *locate-then-infill*: rather than
regenerate the whole model, it (1) locates the span of the source CAD sequence
the edit touches and (2) masks that span, leaving the generator to infill only
the masked region against the surrounding context. This localises the change,
so the untouched majority of the model is preserved verbatim -- the same reason
the harness prefers op-DAG deltas to whole re-synthesis.

The generation model is out of scope; the plan *construction* is deterministic
and is what this module extracts:

* :func:`locate_span` -- find the contiguous token span to edit, either by an
  explicit index range or by matching an anchor subsequence,
* :func:`build_infill_plan` -- replace that span with a single mask sentinel,
  producing the prefix / suffix context an infilling generator consumes,
* :func:`apply_infill` -- splice a generated infill back into the sequence,
* :func:`InfillPlan` -- the masked template plus the located span, so an edit is
  auditable and reversible.

Tokens are opaque strings (or CAD op tags); nothing here interprets them, so it
works for any linear CAD sequence representation. Stdlib-only, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "MASK_TOKEN",
    "locate_span",
    "InfillPlan",
    "build_infill_plan",
    "apply_infill",
]

#: Sentinel marking the masked (to-be-infilled) region.
MASK_TOKEN = "<MASK>"


def locate_span(
    tokens: Sequence[str],
    *,
    span: Optional[Tuple[int, int]] = None,
    anchor: Optional[Sequence[str]] = None,
) -> Tuple[int, int]:
    """Locate the ``[start, end)`` token span an edit targets.

    Exactly one locator must be given: an explicit ``span`` (validated against
    bounds) or an ``anchor`` subsequence to find (first occurrence). Raises
    ``ValueError`` if neither/both are given, the span is out of bounds, or the
    anchor is absent.
    """
    if (span is None) == (anchor is None):
        raise ValueError("provide exactly one of span or anchor")
    n = len(tokens)
    if span is not None:
        start, end = span
        if not (0 <= start <= end <= n):
            raise ValueError(f"span {span} out of bounds for length {n}")
        return start, end
    a = list(anchor or ())
    if not a:
        raise ValueError("anchor must be non-empty")
    for i in range(0, n - len(a) + 1):
        if list(tokens[i : i + len(a)]) == a:
            return i, i + len(a)
    raise ValueError(f"anchor {a!r} not found in token sequence")


@dataclass(frozen=True)
class InfillPlan:
    """A masked edit template: prefix + mask + suffix, plus the located span."""

    prefix: Tuple[str, ...]
    suffix: Tuple[str, ...]
    span: Tuple[int, int]
    masked: Tuple[str, ...]  # prefix + (MASK_TOKEN,) + suffix

    @property
    def removed(self) -> Tuple[int, int]:
        return self.span


def build_infill_plan(
    tokens: Sequence[str],
    *,
    span: Optional[Tuple[int, int]] = None,
    anchor: Optional[Sequence[str]] = None,
) -> InfillPlan:
    """Mask the located span, yielding the prefix/suffix context to infill.

    The masked template is ``prefix + [MASK] + suffix``; an infilling generator
    conditions on it and the surrounding context to produce replacement tokens
    for the single mask, which :func:`apply_infill` splices back in.
    """
    start, end = locate_span(tokens, span=span, anchor=anchor)
    prefix = tuple(tokens[:start])
    suffix = tuple(tokens[end:])
    return InfillPlan(
        prefix=prefix,
        suffix=suffix,
        span=(start, end),
        masked=prefix + (MASK_TOKEN,) + suffix,
    )


def apply_infill(plan: InfillPlan, infill: Sequence[str]) -> Tuple[str, ...]:
    """Splice a generated *infill* into the masked region, returning the edit.

    The result is ``prefix + infill + suffix`` -- everything outside the located
    span is preserved verbatim, which is the whole point of locate-then-infill.
    """
    return tuple(plan.prefix) + tuple(infill) + tuple(plan.suffix)
