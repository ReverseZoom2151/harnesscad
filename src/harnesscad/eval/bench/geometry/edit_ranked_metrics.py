"""Metrics for ranked, history-free B-rep edits."""

from __future__ import annotations

from math import sqrt
from typing import Iterable, Mapping, Sequence


def pass_at_k(outcomes: Iterable[bool], k: int) -> bool:
    if k < 0:
        raise ValueError("k must be non-negative")
    return any(tuple(outcomes)[:k])


def retention(before: Iterable[object], after: Iterable[object]) -> float:
    a, b = set(before), set(after)
    return len(a & b) / len(a) if a else 1.0


def symmetric_chamfer(
    before: Iterable[Sequence[float]], after: Iterable[Sequence[float]]
) -> float | None:
    a = tuple(tuple(map(float, p)) for p in before)
    b = tuple(tuple(map(float, p)) for p in after)
    if not a or not b:
        return None

    def directed(x, y):
        return sum(min(sqrt(sum((u - v) ** 2 for u, v in zip(p, q))) for q in y)
                   for p in x) / len(x)
    return (directed(a, b) + directed(b, a)) / 2.0


def relation_preservation(
    before: Mapping[str, str], after: Mapping[str, str]
) -> dict:
    """Guard declared parallel/perpendicular/symmetry relationships."""
    supported = {"parallel", "perpendicular", "symmetry"}
    declared = {k: v for k, v in before.items() if v in supported}
    broken = tuple(sorted(k for k, v in declared.items() if after.get(k) != v))
    return {
        "preserved": len(declared) - len(broken),
        "declared": len(declared),
        "fraction": ((len(declared) - len(broken)) / len(declared)
                     if declared else 1.0),
        "broken": broken,
    }
