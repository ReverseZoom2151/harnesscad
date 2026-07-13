"""Injected generate-filter-train-evaluate self-improvement controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar

T = TypeVar("T")
M = TypeVar("M")


@dataclass(frozen=True)
class ImprovementRound(Generic[M]):
    index: int
    input_model: M
    output_model: M
    generated: int
    accepted: int
    validation_score: float
    retained: bool


@dataclass(frozen=True)
class ImprovementRun(Generic[M]):
    best_model: M
    best_score: float
    rounds: tuple[ImprovementRound[M], ...]
    stop_reason: str


def self_improve(
    initial_model: M,
    generate: Callable[[M, int], Iterable[T]],
    accept: Callable[[T], bool],
    train: Callable[[M, tuple[T, ...]], M],
    evaluate: Callable[[M], float],
    *,
    maximum_rounds: int = 5,
    sample_cap: int = 2000,
    minimum_accepted: int = 1,
) -> ImprovementRun[M]:
    if maximum_rounds < 1 or sample_cap < 1 or minimum_accepted < 1:
        raise ValueError("round and sample limits must be positive")
    best_model = initial_model
    best_score = evaluate(initial_model)
    rounds: list[ImprovementRound[M]] = []
    stop_reason = "maximum_rounds"
    current = initial_model
    for index in range(1, maximum_rounds + 1):
        generated = tuple(generate(current, index))
        selected = tuple(item for item in generated if accept(item))[:sample_cap]
        if len(selected) < minimum_accepted:
            stop_reason = "insufficient_accepted_samples"
            break
        candidate = train(current, selected)
        score = evaluate(candidate)
        retained = score > best_score
        rounds.append(
            ImprovementRound(
                index, current, candidate, len(generated), len(selected), score, retained
            )
        )
        if not retained:
            stop_reason = "validation_degraded"
            break
        best_model, best_score, current = candidate, score, candidate
    return ImprovementRun(best_model, best_score, tuple(rounds), stop_reason)
