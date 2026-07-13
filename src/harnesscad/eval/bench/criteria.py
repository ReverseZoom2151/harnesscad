"""Typed, per-sample CAD criteria and BlenderLLM-style score aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Callable, Iterable, Mapping


class Dimension(str, Enum):
    ATTRIBUTE = "attribute"
    SPATIAL = "spatial"
    INSTRUCTION = "instruction"


class Subdimension(str, Enum):
    SHAPE = "shape"
    COLOR = "color"
    SIZE = "size"
    PROPORTION = "proportion"
    TEXTURE = "texture"
    SPACE = "space"
    CONTACT = "contact"
    EXECUTE = "execute"


class Modality(str, Enum):
    IMAGE = "image"
    SCRIPT = "script"


_PARENT = {
    Subdimension.SHAPE: Dimension.ATTRIBUTE,
    Subdimension.COLOR: Dimension.ATTRIBUTE,
    Subdimension.SIZE: Dimension.ATTRIBUTE,
    Subdimension.PROPORTION: Dimension.ATTRIBUTE,
    Subdimension.TEXTURE: Dimension.ATTRIBUTE,
    Subdimension.SPACE: Dimension.SPATIAL,
    Subdimension.CONTACT: Dimension.SPATIAL,
    Subdimension.EXECUTE: Dimension.INSTRUCTION,
}


@dataclass(frozen=True)
class Criterion:
    id: str
    dimension: Dimension
    subdimension: Subdimension
    modality: Modality
    requirement: str

    def __post_init__(self) -> None:
        if not self.id or not self.requirement.strip():
            raise ValueError("criterion id and requirement are required")
        if _PARENT[self.subdimension] is not self.dimension:
            raise ValueError("subdimension does not belong to dimension")


@dataclass(frozen=True)
class CriterionResult:
    criterion: Criterion
    passed: bool


@dataclass(frozen=True)
class CriteriaScore:
    criterion_scores: Mapping[str, int]
    subdimension_scores: Mapping[str, float]
    dimension_scores: Mapping[str, float]
    overall: float
    standard_deviation: float


def route_and_evaluate(
    criteria: Iterable[Criterion],
    *,
    image_evaluator: Callable[[Criterion], bool],
    script_evaluator: Callable[[Criterion], bool],
) -> tuple[CriterionResult, ...]:
    """Route each binary criterion to exactly one evaluator."""
    results = []
    for criterion in criteria:
        evaluator = (
            image_evaluator if criterion.modality is Modality.IMAGE
            else script_evaluator
        )
        results.append(CriterionResult(criterion, bool(evaluator(criterion))))
    return tuple(results)


def aggregate(results: Iterable[CriterionResult]) -> CriteriaScore:
    items = tuple(results)
    by_sub: dict[Subdimension, list[int]] = {}
    by_dim: dict[Dimension, dict[Subdimension, list[int]]] = {}
    raw = {}
    for result in items:
        score = int(result.passed)
        raw[result.criterion.id] = score
        sub = result.criterion.subdimension
        dim = result.criterion.dimension
        by_sub.setdefault(sub, []).append(score)
        by_dim.setdefault(dim, {}).setdefault(sub, []).append(score)
    sub_scores = {
        sub.value: sum(values) / len(values) for sub, values in by_sub.items()
    }
    # Paper equation: dimensions average their represented subdimensions rather
    # than weighting subdimensions with more criteria more heavily.
    dim_scores = {}
    for dim, groups in by_dim.items():
        scores = [sum(values) / len(values) for values in groups.values()]
        dim_scores[dim.value] = sum(scores) / len(scores)
    values = list(dim_scores.values())
    overall = sum(values) / len(values) if values else 0.0
    sd = sqrt(sum((value - overall) ** 2 for value in values) / len(values)) if values else 0.0
    return CriteriaScore(raw, sub_scores, dim_scores, overall, sd)


def failure_rate(outcomes: Iterable[bool]) -> float | None:
    values = tuple(bool(value) for value in outcomes)
    return (sum(not value for value in values) / len(values)) if values else None


def syntax_failure_rate(executed: Iterable[bool]) -> float | None:
    return failure_rate(executed)


def render_failure_rate(rendered: Iterable[bool]) -> float | None:
    return failure_rate(rendered)
