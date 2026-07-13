"""Deterministic metrics for time to the first feasible CAD result."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional


@dataclass(frozen=True)
class FeasibilityResult:
    """One run measured up to its first valid result."""

    success: bool
    attempts: int
    solver_calls: int
    elapsed_seconds: Optional[float]


@dataclass(frozen=True)
class Percentiles:
    p50: Optional[float]
    p95: Optional[float]


@dataclass(frozen=True)
class FeasibilityAggregate:
    runs: int
    successes: int
    success_rate: float
    attempts: Percentiles
    solver_calls: Percentiles
    elapsed_seconds: Percentiles


class FeasibilityTracker:
    """Capture attempts, solver calls and injected elapsed time to first valid.

    ``clock`` is required deliberately: callers choose a monotonic production
    clock or a deterministic fake.  This module never reads wall time itself.
    """

    def __init__(self, clock: Callable[[], float]) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._started = float(clock())
        self._attempts = 0
        self._solver_calls = 0
        self._result: Optional[FeasibilityResult] = None

    def record_attempt(
        self, *, valid: bool, solver_calls: int = 0
    ) -> Optional[FeasibilityResult]:
        """Record an attempt and return the result when this is the first valid.

        Calls after feasibility has been reached are harmless and return the
        frozen first-valid result, preventing accidental metric inflation.
        """

        if isinstance(solver_calls, bool) or not isinstance(solver_calls, int):
            raise TypeError("solver_calls must be an integer")
        if solver_calls < 0:
            raise ValueError("solver_calls must be non-negative")
        if self._result is not None:
            return self._result
        self._attempts += 1
        self._solver_calls += solver_calls
        if not valid:
            return None
        elapsed = float(self._clock()) - self._started
        if elapsed < 0:
            raise ValueError("clock moved backwards")
        self._result = FeasibilityResult(
            True, self._attempts, self._solver_calls, elapsed
        )
        return self._result

    def snapshot(self) -> FeasibilityResult:
        """Current result; unsuccessful snapshots have no first-valid time."""

        return self._result or FeasibilityResult(
            False, self._attempts, self._solver_calls, None
        )


def aggregate_feasibility(
    results: Iterable[FeasibilityResult],
) -> FeasibilityAggregate:
    """Aggregate success plus nearest-rank p50/p95 over successful runs.

    Failed runs contribute to the denominator but not first-valid percentiles,
    because those values are right-censored rather than zero.
    """

    records = list(results)
    successful = [result for result in records if result.success]
    elapsed = [
        float(result.elapsed_seconds)
        for result in successful
        if result.elapsed_seconds is not None
    ]
    return FeasibilityAggregate(
        runs=len(records),
        successes=len(successful),
        success_rate=(len(successful) / len(records)) if records else 0.0,
        attempts=_percentiles([float(result.attempts) for result in successful]),
        solver_calls=_percentiles(
            [float(result.solver_calls) for result in successful]
        ),
        elapsed_seconds=_percentiles(elapsed),
    )


def _percentiles(values: list[float]) -> Percentiles:
    if not values:
        return Percentiles(None, None)
    ordered = sorted(values)
    return Percentiles(
        _nearest_rank(ordered, 0.50),
        _nearest_rank(ordered, 0.95),
    )


def _nearest_rank(ordered: list[float], probability: float) -> float:
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]
