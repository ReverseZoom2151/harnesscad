"""Deterministic procedural design selection, trials, placement, and coverage.

The module contains no CAD-kernel or model dependency.  Generation and evaluation
are injected, making the same orchestration useful for sketches, parts, assemblies,
and synthetic-data jobs while remaining exactly replayable in tests.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


ENGINEERING_REQUIREMENTS = (
    "precision",
    "repeatability",
    "cost",
    "manufacturability",
)


@dataclass(frozen=True)
class ProceduralTechnique:
    """A procedural technique and its normalized engineering capability scores."""

    name: str
    scores: Mapping[str, float]
    tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        unknown = set(self.scores) - set(ENGINEERING_REQUIREMENTS)
        if unknown:
            raise ValueError(f"unknown requirements: {sorted(unknown)}")
        if any(not 0.0 <= value <= 1.0 for value in self.scores.values()):
            raise ValueError("technique scores must be in [0, 1]")


class TechniqueRegistry:
    """Registry with deterministic, requirement-weighted technique selection."""

    def __init__(self, techniques: Iterable[ProceduralTechnique] = ()) -> None:
        self._techniques: dict[str, ProceduralTechnique] = {}
        for technique in techniques:
            self.register(technique)

    def register(self, technique: ProceduralTechnique) -> None:
        if technique.name in self._techniques:
            raise ValueError(f"duplicate technique: {technique.name}")
        self._techniques[technique.name] = technique

    def select(
        self,
        weights: Mapping[str, float],
        *,
        minimums: Mapping[str, float] | None = None,
        required_tags: Iterable[str] = (),
    ) -> list[tuple[ProceduralTechnique, float]]:
        """Rank eligible techniques; higher weighted engineering fit wins."""

        unknown = set(weights) - set(ENGINEERING_REQUIREMENTS)
        if unknown:
            raise ValueError(f"unknown requirements: {sorted(unknown)}")
        if any(value < 0 for value in weights.values()):
            raise ValueError("weights must be non-negative")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("at least one positive weight is required")
        minimums = minimums or {}
        tags = frozenset(required_tags)
        ranked: list[tuple[ProceduralTechnique, float]] = []
        for technique in self._techniques.values():
            if not tags <= technique.tags:
                continue
            if any(technique.scores.get(key, 0.0) < value for key, value in minimums.items()):
                continue
            score = sum(
                weight * technique.scores.get(requirement, 0.0)
                for requirement, weight in weights.items()
            ) / total
            ranked.append((technique, score))
        return sorted(ranked, key=lambda item: (-item[1], item[0].name))


@dataclass(frozen=True)
class TrialAttempt:
    index: int
    seed: int
    status: str
    score: float | None = None
    result: Any = None
    diagnostic: str | None = None
    elapsed: float | None = None


@dataclass(frozen=True)
class TrialRun:
    master_seed: int
    attempts: tuple[TrialAttempt, ...]
    winning_seed: int | None
    winning_result: Any = None
    winning_score: float | None = None

    @property
    def failures(self) -> tuple[TrialAttempt, ...]:
        return tuple(attempt for attempt in self.attempts if attempt.status != "ok")


Generator = Callable[[int], Any]
Evaluator = Callable[[Any], float]
Clock = Callable[[], float]


def derive_child_seeds(master_seed: int, count: int) -> tuple[int, ...]:
    """Derive stable, distinct 63-bit child seeds from one master seed."""

    if count < 0:
        raise ValueError("count must be non-negative")
    rng = random.Random(master_seed)
    seeds: list[int] = []
    seen: set[int] = set()
    while len(seeds) < count:
        seed = rng.getrandbits(63)
        if seed not in seen:
            seen.add(seed)
            seeds.append(seed)
    return tuple(seeds)


def run_trials(
    generator: Generator,
    evaluator: Evaluator,
    *,
    master_seed: int,
    attempts: int,
    timeout: float | None = None,
    clock: Clock | None = None,
) -> TrialRun:
    """Run a bounded multi-start search and retain an exactly replayable winner."""

    if attempts < 0:
        raise ValueError("attempts must be non-negative")
    if timeout is not None and (timeout < 0 or clock is None):
        raise ValueError("a non-negative timeout requires an injected clock")
    records: list[TrialAttempt] = []
    for index, seed in enumerate(derive_child_seeds(master_seed, attempts)):
        started = clock() if clock else None
        try:
            result = generator(seed)
            score = float(evaluator(result))
            if not math.isfinite(score):
                raise ValueError("evaluator returned a non-finite score")
            elapsed = (clock() - started) if clock and started is not None else None
            if timeout is not None and elapsed is not None and elapsed > timeout:
                records.append(TrialAttempt(
                    index, seed, "timeout", result=result,
                    diagnostic=f"elapsed {elapsed:.6g}s exceeded {timeout:.6g}s",
                    elapsed=elapsed,
                ))
            else:
                records.append(TrialAttempt(index, seed, "ok", score, result, elapsed=elapsed))
        except Exception as exc:  # trial failure is data; the bounded run continues
            elapsed = (clock() - started) if clock and started is not None else None
            records.append(TrialAttempt(
                index, seed, "failed", diagnostic=f"{type(exc).__name__}: {exc}",
                elapsed=elapsed,
            ))
    valid = [record for record in records if record.status == "ok"]
    winner = max(valid, key=lambda record: (record.score, -record.index)) if valid else None
    return TrialRun(
        master_seed,
        tuple(records),
        winner.seed if winner else None,
        winner.result if winner else None,
        winner.score if winner else None,
    )


def replay(generator: Generator, evaluator: Evaluator, seed: int) -> TrialAttempt:
    """Replay one recorded seed, preserving the same failure-as-data semantics."""

    try:
        result = generator(seed)
        score = float(evaluator(result))
        if not math.isfinite(score):
            raise ValueError("evaluator returned a non-finite score")
        return TrialAttempt(0, seed, "ok", score, result)
    except Exception as exc:
        return TrialAttempt(0, seed, "failed", diagnostic=f"{type(exc).__name__}: {exc}")


@dataclass(frozen=True)
class Placement:
    item: str
    point: tuple[float, ...]
    cluster: str | None = None


@dataclass(frozen=True)
class PlacementRules:
    """Declarative modular-placement constraints in two or more dimensions."""

    adjacency: tuple[tuple[str, str, float], ...] = ()
    cluster_radius: float | None = None
    obstacles: tuple[tuple[tuple[float, ...], float], ...] = ()
    clearance: float = 0.0


def validate_placements(
    placements: Sequence[Placement],
    rules: PlacementRules,
) -> tuple[str, ...]:
    """Return stable diagnostics for adjacency, clustering, and obstacle violations."""

    by_name = {placement.item: placement for placement in placements}
    if len(by_name) != len(placements):
        return ("duplicate placement item",)
    dimensions = {len(placement.point) for placement in placements}
    if len(dimensions) > 1:
        return ("placement dimensions do not match",)
    diagnostics: list[str] = []

    def distance(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    for left, right, maximum in rules.adjacency:
        if left not in by_name or right not in by_name:
            diagnostics.append(f"adjacency references missing item: {left}, {right}")
        elif distance(by_name[left].point, by_name[right].point) > maximum:
            diagnostics.append(f"{left} and {right} exceed adjacency distance {maximum:g}")
    if rules.cluster_radius is not None:
        groups: dict[str, list[Placement]] = {}
        for placement in placements:
            if placement.cluster is not None:
                groups.setdefault(placement.cluster, []).append(placement)
        for name, members in sorted(groups.items()):
            for i, left in enumerate(members):
                for right in members[i + 1:]:
                    if distance(left.point, right.point) > rules.cluster_radius:
                        diagnostics.append(
                            f"cluster {name} exceeds radius between {left.item} and {right.item}"
                        )
    for placement in placements:
        for center, radius in rules.obstacles:
            if len(center) != len(placement.point):
                diagnostics.append(f"obstacle dimension mismatch for {placement.item}")
            elif distance(placement.point, center) < radius + rules.clearance:
                diagnostics.append(f"{placement.item} intersects obstacle clearance")
    return tuple(diagnostics)


@dataclass(frozen=True)
class CoverageReport:
    dimension_coverage: Mapping[str, float]
    configuration_coverage: float
    diversity: float
    unique_configurations: int


def solution_space_coverage(
    dimensions: Mapping[str, Sequence[Any]],
    configurations: Iterable[Mapping[str, Any]],
) -> CoverageReport:
    """Measure declared-dimension coverage and normalized pairwise diversity."""

    declared = {name: tuple(values) for name, values in dimensions.items()}
    if any(not values for values in declared.values()):
        raise ValueError("declared dimensions must be non-empty")
    configs = [dict(config) for config in configurations]
    names = tuple(sorted(declared))
    canonical: set[tuple[Any, ...]] = set()
    observed = {name: set() for name in names}
    for config in configs:
        missing = set(names) - set(config)
        if missing:
            raise ValueError(f"configuration missing dimensions: {sorted(missing)}")
        for name in names:
            value = config[name]
            if value not in declared[name]:
                raise ValueError(f"undeclared value for {name}: {value!r}")
            observed[name].add(value)
        canonical.add(tuple(config[name] for name in names))
    per_dimension = {
        name: len(observed[name]) / len(declared[name])
        for name in names
    }
    total = math.prod(len(declared[name]) for name in names)
    unique = sorted(canonical, key=repr)
    distances: list[float] = []
    for index, left in enumerate(unique):
        for right in unique[index + 1:]:
            distances.append(sum(a != b for a, b in zip(left, right)) / len(names))
    diversity = sum(distances) / len(distances) if distances else 0.0
    return CoverageReport(per_dimension, len(unique) / total, diversity, len(unique))
