"""Deterministic metrics for sketch-constraint intent and solver alignment.

The module owns no solver or geometry kernel.  Evaluation functions are
injected so the metrics can be used with SolveSpace, a CAD backend, or small
offline fixtures without changing their semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import comb, dist, floor
from typing import Callable, Hashable, Iterable, Mapping, Sequence

Constraint = Hashable
Parameters = Mapping[str, float]
Point = tuple[float, ...]
Geometry = Mapping[str, Point]


class SolveCondition(str, Enum):
    FULLY_CONSTRAINED = "fully_constrained"
    UNDER_CONSTRAINED = "under_constrained"
    OVER_CONSTRAINED = "over_constrained"
    UNSOLVABLE = "unsolvable"


@dataclass(frozen=True)
class SolveSnapshot:
    """Normalized result returned by an injected sketch solver."""

    condition: SolveCondition
    geometry: Geometry
    residual_dof: int = 0
    solved: bool = True

    def __post_init__(self) -> None:
        if self.residual_dof < 0:
            raise ValueError("residual_dof cannot be negative")
        if self.condition is SolveCondition.UNSOLVABLE and self.solved:
            raise ValueError("an unsolvable snapshot cannot be marked solved")


@dataclass(frozen=True)
class StabilityCase:
    name: str
    parameters: Parameters


@dataclass(frozen=True)
class StabilityResult:
    case: str
    stable: bool
    same_entities: bool
    same_spatial_bins: bool
    max_displacement: float | None
    condition: SolveCondition


@dataclass(frozen=True)
class StabilityReport:
    results: tuple[StabilityResult, ...]

    @property
    def stable(self) -> bool:
        return bool(self.results) and all(result.stable for result in self.results)

    @property
    def stable_fraction(self) -> float:
        return (
            sum(result.stable for result in self.results) / len(self.results)
            if self.results
            else 0.0
        )


def parameter_stability(
    baseline: SolveSnapshot,
    cases: Iterable[StabilityCase],
    solve: Callable[[Parameters], SolveSnapshot],
    *,
    spatial_bin: float = 1.0,
    tolerance: float = 1e-6,
) -> StabilityReport:
    """Check whether parameter edits preserve solve and geometric intent.

    Entity identity and coarse spatial bins detect branch/topology jumps, while
    ``tolerance`` absorbs harmless solver noise near bin boundaries.
    """

    if spatial_bin <= 0 or tolerance < 0:
        raise ValueError("spatial_bin must be positive and tolerance non-negative")
    results: list[StabilityResult] = []
    baseline_keys = set(baseline.geometry)
    for case in cases:
        current = solve(case.parameters)
        same_entities = set(current.geometry) == baseline_keys
        displacements = (
            [dist(baseline.geometry[key], current.geometry[key]) for key in baseline_keys]
            if same_entities and _same_dimensions(baseline.geometry, current.geometry)
            else []
        )
        same_bins = same_entities and _bins_match(
            baseline.geometry, current.geometry, spatial_bin, tolerance
        )
        max_displacement = max(displacements, default=None)
        stable = (
            baseline.solved
            and current.solved
            and current.condition not in {SolveCondition.OVER_CONSTRAINED,
                                          SolveCondition.UNSOLVABLE}
            and same_entities
            and same_bins
        )
        results.append(
            StabilityResult(
                case.name,
                stable,
                same_entities,
                same_bins,
                max_displacement,
                current.condition,
            )
        )
    return StabilityReport(tuple(results))


@dataclass(frozen=True)
class IntentScorecard:
    fully_constrained: bool
    under_constrained: bool
    over_constrained: bool
    unsolvable: bool
    stable: bool

    @property
    def condition(self) -> SolveCondition:
        flags = {
            SolveCondition.FULLY_CONSTRAINED: self.fully_constrained,
            SolveCondition.UNDER_CONSTRAINED: self.under_constrained,
            SolveCondition.OVER_CONSTRAINED: self.over_constrained,
            SolveCondition.UNSOLVABLE: self.unsolvable,
        }
        selected = [condition for condition, enabled in flags.items() if enabled]
        if len(selected) != 1:
            raise ValueError("scorecard must contain exactly one solve condition")
        return selected[0]


def score_intent(snapshot: SolveSnapshot, stability: StabilityReport) -> IntentScorecard:
    return IntentScorecard(
        fully_constrained=snapshot.condition is SolveCondition.FULLY_CONSTRAINED,
        under_constrained=snapshot.condition is SolveCondition.UNDER_CONSTRAINED,
        over_constrained=snapshot.condition is SolveCondition.OVER_CONSTRAINED,
        unsolvable=snapshot.condition is SolveCondition.UNSOLVABLE,
        stable=stability.stable,
    )


@dataclass(frozen=True)
class ConstraintEconomy:
    dimensional: int
    geometric: int
    duplicate: int = 0
    ineffective: int = 0
    reference_only_dimensions: int = 0

    def __post_init__(self) -> None:
        if min(
            self.dimensional,
            self.geometric,
            self.duplicate,
            self.ineffective,
            self.reference_only_dimensions,
        ) < 0:
            raise ValueError("constraint counts cannot be negative")

    @property
    def total(self) -> int:
        return self.dimensional + self.geometric

    @property
    def dimension_to_geometric_ratio(self) -> float:
        if self.geometric == 0:
            return float("inf") if self.dimensional else 0.0
        return self.dimensional / self.geometric

    @property
    def useful_fraction(self) -> float:
        if not self.total:
            return 1.0
        waste = min(self.total, self.duplicate + self.ineffective)
        return (self.total - waste) / self.total

    def reward_hacking_diagnostics(
        self, *, max_dimension_ratio: float = 2.0
    ) -> tuple[str, ...]:
        if max_dimension_ratio < 0:
            raise ValueError("max_dimension_ratio cannot be negative")
        diagnostics: list[str] = []
        if self.dimension_to_geometric_ratio > max_dimension_ratio:
            diagnostics.append("dimension_stuffing")
        if self.duplicate:
            diagnostics.append("duplicate_constraints")
        if self.ineffective:
            diagnostics.append("ineffective_constraints")
        if self.reference_only_dimensions:
            diagnostics.append("reference_dimension_inflation")
        return tuple(diagnostics)


@dataclass(frozen=True)
class BlameStep:
    index: int
    constraint: Constraint
    prefix_condition: SolveCondition
    blamed_by_transition: bool
    drop_condition: SolveCondition
    blamed_by_drop: bool


def constraint_blame_trace(
    constraints: Sequence[Constraint],
    evaluate: Callable[[Sequence[Constraint]], SolveSnapshot],
) -> tuple[BlameStep, ...]:
    """Trace prefix failures and leave-one-out recovery for each constraint."""

    full = evaluate(constraints)
    steps: list[BlameStep] = []
    previous = evaluate(())
    bad = {SolveCondition.OVER_CONSTRAINED, SolveCondition.UNSOLVABLE}
    for index, constraint in enumerate(constraints):
        prefix = evaluate(constraints[: index + 1])
        dropped = evaluate(constraints[:index] + constraints[index + 1 :])
        steps.append(
            BlameStep(
                index=index,
                constraint=constraint,
                prefix_condition=prefix.condition,
                blamed_by_transition=previous.condition not in bad
                and prefix.condition in bad,
                drop_condition=dropped.condition,
                blamed_by_drop=full.condition in bad and dropped.condition not in bad,
            )
        )
        previous = prefix
    return tuple(steps)


@dataclass(frozen=True)
class VerifiedAttempt:
    generated: bool
    solver_verified: bool


def solver_verified_pass_at_k(
    attempts: Sequence[VerifiedAttempt], k: int
) -> float:
    """Unbiased pass@k estimate, counting only solver-verified generations."""

    if k <= 0:
        raise ValueError("k must be positive")
    generated = [attempt for attempt in attempts if attempt.generated]
    n = len(generated)
    if n < k:
        raise ValueError("k cannot exceed the number of generated attempts")
    c = sum(attempt.solver_verified for attempt in generated)
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _same_dimensions(left: Geometry, right: Geometry) -> bool:
    return all(len(left[key]) == len(right[key]) for key in left)


def _bins_match(
    left: Geometry, right: Geometry, width: float, tolerance: float
) -> bool:
    if not _same_dimensions(left, right):
        return False
    for key in left:
        for a, b in zip(left[key], right[key]):
            # Either the quantized region agrees, or the values straddle a
            # boundary by no more than numerical tolerance.
            if floor(a / width) != floor(b / width) and abs(a - b) > tolerance:
                return False
    return True
