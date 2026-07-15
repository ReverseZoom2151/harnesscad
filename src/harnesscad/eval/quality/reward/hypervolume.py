"""Hypervolume indicator and constraint-violation aggregation for design sets.

**BikeBench** (Regenwetter et al., 2025) evaluates a *set* of generated engineering
designs, not one design, on two summary metrics this module implements deterministically
(their Sec. 5.1):

* **Design Quality (Hypervolume).** "To quantify design quality, we calculate the
  hypervolume metric over any designs that simultaneously satisfy all constraints. The
  hypervolume metric is a staple of multi-objective optimization literature which
  calculates the overall multi-objective optimality of a set of designs." Hypervolume
  is the measure of objective space dominated by the (constraint-satisfying) set and
  bounded by a reference point -- higher is better and it rewards both quality and
  diversity in a single scalar. :func:`hypervolume` computes it exactly for any number
  of minimisation objectives via inclusion-exclusion over the non-dominated set.

* **Constraint Violation.** "We measure the mean number of constraints that the model
  violates per design." BikeBench ships "31 closed-form geometric constraint checks"
  (Sec. 4, feasibility): triangle-inequality, positive-dimension, non-overlap rules.
  :func:`mean_constraint_violation` and :func:`ConstraintSuite` aggregate a set of
  boolean/threshold checks into the per-design violation count and satisfaction rate.

Objectives are *minimised* (consistent with :mod:`harnesscad.eval.quality.reward.fitness`
and ``pareto``). Stdlib only, deterministic; the hypervolume uses the exact
inclusion-exclusion measure (fine for the small objective counts CAD design uses).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, Mapping, Sequence

__all__ = [
    "hypervolume",
    "non_dominated",
    "mean_constraint_violation",
    "constraint_satisfaction_rate",
    "feasible_designs",
    "ConstraintSuite",
    "positive_dimensions",
    "triangle_inequality_ok",
]

Number = float


def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Minimisation dominance: ``a`` no worse on every axis and better on at least one."""
    no_worse = all(x <= y for x, y in zip(a, b))
    strictly = any(x < y for x, y in zip(a, b))
    return no_worse and strictly


def non_dominated(points: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """The Pareto-minimal subset of objective vectors (duplicates collapsed)."""
    pts = [tuple(float(c) for c in p) for p in points]
    keep: list[tuple[float, ...]] = []
    for i, p in enumerate(pts):
        if any(j != i and _dominates(q, p) for j, q in enumerate(pts)):
            continue
        if p not in keep:
            keep.append(p)
    return tuple(keep)


def hypervolume(points: Sequence[Sequence[float]], reference: Sequence[float]) -> float:
    """Exact hypervolume dominated by a minimisation set, bounded above by ``reference``.

    Each point contributes the axis-aligned box ``[p_k, ref_k]``; the indicator is the
    volume of the union of those boxes, computed by inclusion-exclusion over the
    non-dominated points. Points not strictly below the reference on every axis are
    dropped (they enclose no volume). Returns 0.0 for an empty/degenerate set.
    """
    ref = tuple(float(c) for c in reference)
    dim = len(ref)
    boxes = [
        tuple(float(c) for c in p)
        for p in non_dominated(points)
        if len(p) == dim and all(p[k] < ref[k] for k in range(dim))
    ]
    if not boxes:
        return 0.0

    total = 0.0
    # Inclusion-exclusion: union volume = sum over non-empty subsets of
    #   (-1)^(|S|+1) * volume(intersection of boxes in S).
    # Each box is [p_k, ref_k]; intersection lower corner = elementwise max of p_k.
    n = len(boxes)
    for r in range(1, n + 1):
        sign = 1.0 if (r % 2 == 1) else -1.0
        for subset in combinations(range(n), r):
            vol = 1.0
            for k in range(dim):
                lo = max(boxes[i][k] for i in subset)
                edge = ref[k] - lo
                if edge <= 0.0:
                    vol = 0.0
                    break
                vol *= edge
            total += sign * vol
    return total


def mean_constraint_violation(
    designs: Iterable,
    checks: Sequence[Callable[[object], bool]],
) -> float:
    """Mean number of *violated* constraints per design (BikeBench, lower is better).

    Each check returns True when the design *satisfies* that constraint.
    """
    designs = list(designs)
    if not designs:
        return 0.0
    total_violations = 0
    for d in designs:
        total_violations += sum(0 if chk(d) else 1 for chk in checks)
    return total_violations / len(designs)


def constraint_satisfaction_rate(
    designs: Iterable,
    checks: Sequence[Callable[[object], bool]],
) -> float:
    """Fraction of (design, constraint) pairs satisfied (higher is better)."""
    designs = list(designs)
    if not designs or not checks:
        return 1.0
    satisfied = 0
    for d in designs:
        satisfied += sum(1 if chk(d) else 0 for chk in checks)
    return satisfied / (len(designs) * len(checks))


def feasible_designs(
    designs: Iterable,
    checks: Sequence[Callable[[object], bool]],
) -> list:
    """Designs satisfying *every* constraint (the set the hypervolume is scored over)."""
    return [d for d in designs if all(chk(d) for chk in checks)]


@dataclass(frozen=True)
class ConstraintSuite:
    """A named bundle of closed-form constraint checks (BikeBench feasibility set)."""

    checks: Mapping[str, Callable[[object], bool]]

    def violations(self, design) -> tuple[str, ...]:
        """Names of the constraints this design violates."""
        return tuple(name for name, chk in self.checks.items() if not chk(design))

    def is_feasible(self, design) -> bool:
        return not self.violations(design)

    def as_list(self) -> list[Callable[[object], bool]]:
        return list(self.checks.values())


# -- A couple of the closed-form geometric checks BikeBench enumerates (Sec. 4). --

def positive_dimensions(values: Sequence[float]) -> bool:
    """No parameter is negative (BikeBench "parts with negative dimensions")."""
    return all(v >= 0.0 for v in values)


def triangle_inequality_ok(a: float, b: float, c: float) -> bool:
    """The three lengths can close a triangle (BikeBench "violate the triangle inequality")."""
    return (a + b > c) and (a + c > b) and (b + c > a)
