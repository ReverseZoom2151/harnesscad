"""Two-reviewer annotation, third-party adjudication, and stable QC sampling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True)
class AnnotationItem:
    id: str
    payload: object


@dataclass(frozen=True)
class AnnotationDecision:
    item_id: str
    first: object
    second: object
    final: object
    adjudicated: bool


def adjudicate(
    items: Iterable[AnnotationItem],
    *,
    first_review: Callable[[AnnotationItem], object],
    second_review: Callable[[AnnotationItem], object],
    third_review: Callable[[AnnotationItem, object, object], object],
) -> tuple[AnnotationDecision, ...]:
    """Resolve disagreements only; stable item-id ordering makes runs replayable."""
    decisions = []
    for item in sorted(items, key=lambda value: value.id):
        first = first_review(item)
        second = second_review(item)
        disputed = first != second
        final = third_review(item, first, second) if disputed else first
        decisions.append(AnnotationDecision(item.id, first, second, final, disputed))
    return tuple(decisions)


def qc_sample(
    items: Iterable[AnnotationItem],
    *,
    fraction: float = 0.30,
    salt: str = "",
) -> tuple[AnnotationItem, ...]:
    """Choose exactly round(N*fraction) items by salted content checksum."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between zero and one")
    values = tuple(items)
    n = round(len(values) * fraction)

    def key(item: AnnotationItem):
        digest = hashlib.sha256(f"{salt}\0{item.id}\0{item.payload!r}".encode()).hexdigest()
        return digest, item.id

    return tuple(sorted(values, key=key)[:n])


def decision_distribution(decisions: Iterable[AnnotationDecision]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        key = str(decision.final)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
