"""Progressive assembly traces from part hierarchies.

Progressive object assembly via a visual chain of thought. The multimodal model
is trained, but its **supervision** is built by a deterministic pipeline that
turns a part-based CAD hierarchy into a step-aligned assembly trace: parts are
ordered into incremental construction steps, each yielding a cumulative state
``s_n``. A compositional benchmark then grades **component numeracy** (are the
right number of parts present?) and **trace faithfulness** (does each step add
parts and grow monotonically?).

This module provides:

*   :func:`assembly_order` -- a deterministic build order over a part hierarchy
    (parents before children, stable by name);
*   :func:`build_trace` -- the sequence of cumulative states ``s_0 .. s_N``;
*   :func:`component_numeracy` -- per-category count accuracy vs a target; and
*   :func:`trace_faithfulness` -- monotone, one-or-more-parts-per-step check.

Deterministic, stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Part",
    "assembly_order",
    "build_trace",
    "component_numeracy",
    "trace_faithfulness",
]


@dataclass(frozen=True)
class Part:
    """A part in the hierarchy: a name, a category, and an optional parent name."""

    name: str
    category: str
    parent: Optional[str] = None


def assembly_order(parts: Sequence[Part]) -> List[Part]:
    """Return a deterministic build order: parents before children, ties by name.

    Implemented as a stable topological sort (roots first). Raises on a cycle or a
    dangling parent reference.
    """
    by_name = {p.name: p for p in parts}
    for p in parts:
        if p.parent is not None and p.parent not in by_name:
            raise ValueError(f"part {p.name!r} references unknown parent {p.parent!r}")

    ordered: List[Part] = []
    placed: set = set()
    remaining = sorted(parts, key=lambda p: p.name)
    # iterate to depth; each pass places parts whose parent is already placed
    while remaining:
        progress = False
        next_remaining: List[Part] = []
        for p in remaining:
            if p.parent is None or p.parent in placed:
                ordered.append(p)
                placed.add(p.name)
                progress = True
            else:
                next_remaining.append(p)
        if not progress:
            raise ValueError("cycle detected in part hierarchy")
        remaining = next_remaining
    return ordered


def build_trace(parts: Sequence[Part]) -> List[Tuple[Part, ...]]:
    """Cumulative assembly states ``s_0 .. s_N`` (s_0 empty), one part added per step."""
    order = assembly_order(parts)
    trace: List[Tuple[Part, ...]] = [tuple()]
    acc: List[Part] = []
    for p in order:
        acc.append(p)
        trace.append(tuple(acc))
    return trace


def component_numeracy(
    parts: Sequence[Part], target_counts: Mapping[str, int]
) -> float:
    """Per-category count accuracy: fraction of target categories with the exact count.

    A category present in ``target_counts`` scores 1 if the assembled parts contain
    exactly that many of it, else 0. Extra categories not in the target are ignored.
    """
    if not target_counts:
        raise ValueError("target_counts must be non-empty")
    counts: Dict[str, int] = {}
    for p in parts:
        counts[p.category] = counts.get(p.category, 0) + 1
    hits = sum(1 for cat, n in target_counts.items() if counts.get(cat, 0) == n)
    return hits / len(target_counts)


def trace_faithfulness(trace: Sequence[Tuple[Part, ...]]) -> float:
    """Fraction of steps that strictly grow the state (monotone, non-empty additions).

    A faithful trace adds at least one part at every step and never removes parts;
    returns the fraction of transitions that satisfy this (1.0 == fully faithful).
    """
    if len(trace) < 2:
        raise ValueError("trace needs at least two states")
    good = 0
    transitions = len(trace) - 1
    for prev, nxt in zip(trace, trace[1:]):
        prev_set = set(p.name for p in prev)
        nxt_set = set(p.name for p in nxt)
        if prev_set < nxt_set and len(nxt) > len(prev):
            good += 1
    return good / transitions
