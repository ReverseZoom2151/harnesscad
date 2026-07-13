"""Cost-aware coarse-to-fine filtering with stage quality accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class FilterDecision:
    accepted: bool
    stage: str
    reason: str = ""


@dataclass(frozen=True)
class CascadeItem(Generic[T]):
    item: T
    coarse: FilterDecision
    fine: FilterDecision | None

    @property
    def accepted(self) -> bool:
        return bool(self.coarse.accepted and self.fine and self.fine.accepted)


@dataclass(frozen=True)
class CascadeReport(Generic[T]):
    items: tuple[CascadeItem[T], ...]
    coarse_calls: int
    fine_calls: int
    accepted_count: int


def cascade_filter(
    items: Iterable[T],
    coarse: Callable[[T], FilterDecision],
    fine: Callable[[T], FilterDecision],
) -> CascadeReport[T]:
    results: list[CascadeItem[T]] = []
    fine_calls = 0
    for item in items:
        first = coarse(item)
        if first.stage != "coarse":
            raise ValueError("coarse predicate returned the wrong stage")
        second = None
        if first.accepted:
            second = fine(item)
            fine_calls += 1
            if second.stage != "fine":
                raise ValueError("fine predicate returned the wrong stage")
        results.append(CascadeItem(item, first, second))
    return CascadeReport(
        tuple(results),
        len(results),
        fine_calls,
        sum(result.accepted for result in results),
    )
