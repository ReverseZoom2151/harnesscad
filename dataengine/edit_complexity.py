"""Complexity bins and deterministic balanced sampling for edit datasets."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


def complexity_score(*, affected_faces: int, context_faces: int,
                     relation_count: int, operation_count: int) -> int:
    values = (affected_faces, context_faces, relation_count, operation_count)
    if min(values) < 0:
        raise ValueError("complexity counts cannot be negative")
    return 3 * affected_faces + context_faces + 2 * relation_count + operation_count


def complexity_bin(score: int, cuts: tuple[int, int] = (6, 15)) -> str:
    if score < 0 or cuts[0] >= cuts[1]:
        raise ValueError("invalid score or cuts")
    return "easy" if score < cuts[0] else ("medium" if score < cuts[1] else "hard")


def balanced_sample(
    items: Iterable[T], n_per_bin: int, *, bin_of: Callable[[T], str],
    key: Callable[[T], object] = repr,
) -> tuple[T, ...]:
    if n_per_bin < 0:
        raise ValueError("n_per_bin must be non-negative")
    groups = defaultdict(list)
    for item in items:
        groups[bin_of(item)].append(item)
    out = []
    for name in ("easy", "medium", "hard"):
        out.extend(sorted(groups[name], key=key)[:n_per_bin])
    return tuple(out)
