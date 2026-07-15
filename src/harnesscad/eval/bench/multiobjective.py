"""Deterministic multi-objective design scoring (feasibility + hypervolume).

Mined from Zoo-adjacent BikeBench's scoring framework
(``src/bikebench/benchmarking/scoring.py``) and reduced to a deterministic,
stdlib-only core. BikeBench evaluates a *population* of candidate designs against
a set of **objectives** (minimised) and **constraints** (satisfied when ``<= 0``),
then scores the population's Pareto quality with hypervolume. The learned
surrogates and ``pygmo`` dependency are dropped; the scoring *bookkeeping* -- the
part that is pure arithmetic -- is kept:

*   **Feasibility** -- a design is feasible iff every constraint score is ``<= 0``
    (BikeBench's ``BinaryValidity`` / ``feas_mask = all(constraint_scores <= 0)``).
*   **Objective normalisation** -- objectives are minimised and normalised against
    a component-wise reference (nadir) point to the unit cube, clipping at the ref.
*   **Hypervolume** -- the dominated volume of the feasible, normalised objective
    set with reference point ``(1, 1, ...)``. An exact algorithm is provided for
    1 and 2 objectives (the common case); higher dimensions use a deterministic
    axis-aligned grid approximation with a configurable resolution.

Everything is deterministic: same rows in -> same scores out. No randomness, no
wall clock, no learned model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

__all__ = [
    "ScoreCard",
    "feasibility_mask",
    "feasibility_rate",
    "normalize_objectives",
    "hypervolume",
    "score_population",
]

Row = Sequence[float]


@dataclass(frozen=True)
class ScoreCard:
    """The deterministic scoring summary of a design population."""

    n_designs: int
    n_feasible: int
    feasibility_rate: float
    hypervolume: float
    mean_objectives_feasible: Tuple[float, ...]  # per-objective mean over feasible set


def feasibility_mask(constraint_scores: Sequence[Row]) -> List[bool]:
    """One bool per design: feasible iff every constraint score is ``<= 0``."""
    return [all(c <= 0.0 for c in row) for row in constraint_scores]


def feasibility_rate(constraint_scores: Sequence[Row]) -> float:
    """Fraction of designs that satisfy every constraint. Empty -> 0.0."""
    if not constraint_scores:
        return 0.0
    mask = feasibility_mask(constraint_scores)
    return sum(1 for f in mask if f) / len(mask)


def normalize_objectives(objectives: Sequence[Row], ref_point: Row) -> List[Tuple[float, ...]]:
    """Clip each objective to its reference and normalise to ``[0, 1]`` (0 = ideal).

    Objectives are minimised; ``ref_point`` is the component-wise nadir. A ref
    component of 0 is treated as 1 to avoid division by zero (a degenerate axis).
    """
    ref = [r if r != 0.0 else 1.0 for r in ref_point]
    out: List[Tuple[float, ...]] = []
    for row in objectives:
        if len(row) != len(ref):
            raise ValueError("objective row width does not match ref_point")
        out.append(tuple(min(v, ref[k]) / ref[k] for k, v in enumerate(row)))
    return out


def _pareto_front_min(points: Sequence[Row]) -> List[Tuple[float, ...]]:
    """Non-dominated points under minimisation, deduplicated, deterministically ordered."""
    pts = sorted(set(tuple(p) for p in points))
    front: List[Tuple[float, ...]] = []
    for p in pts:
        dominated = False
        for q in pts:
            if q == p:
                continue
            if all(qi <= pi for qi, pi in zip(q, p)) and any(qi < pi for qi, pi in zip(q, p)):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def hypervolume(points: Sequence[Row], ref_point: Row, *, grid: int = 64) -> float:
    """Dominated hypervolume of a minimisation set relative to ``ref_point``.

    Exact for 1 and 2 objectives; a deterministic axis-aligned grid approximation
    (``grid`` cells per axis) is used for 3+ objectives. Points are first clipped
    to the reference; points at or beyond the reference contribute nothing.
    Returns volume in the ORIGINAL objective units.
    """
    if not points:
        return 0.0
    dim = len(ref_point)
    clipped = [tuple(min(v, ref_point[k]) for k, v in enumerate(p)) for p in points]
    # Drop points that touch/exceed the reference on any axis (zero contribution).
    inside = [p for p in clipped if all(p[k] < ref_point[k] for k in range(dim))]
    if not inside:
        return 0.0
    front = _pareto_front_min(inside)

    if dim == 1:
        return ref_point[0] - min(p[0] for p in front)
    if dim == 2:
        # Minimisation staircase: on the Pareto front, sorting by x ascending
        # makes y strictly descending. Each point contributes a rectangle whose
        # width reaches the reference and whose height is the drop from the
        # previous y (ref on the first step).
        pts = sorted(front)
        area = 0.0
        prev_y = ref_point[1]
        for x, y in pts:
            area += (ref_point[0] - x) * (prev_y - y)
            prev_y = y
        return area
    # dim >= 3: deterministic grid inclusion test.
    lows = [min(p[k] for p in front) for k in range(dim)]
    steps = [(ref_point[k] - lows[k]) / grid for k in range(dim)]
    cell_vol = 1.0
    for s in steps:
        cell_vol *= s
    if cell_vol == 0.0:
        return 0.0
    dominated_cells = 0
    total = grid ** dim
    # Enumerate cell centres deterministically.
    idx = [0] * dim
    for _ in range(total):
        centre = [lows[k] + (idx[k] + 0.5) * steps[k] for k in range(dim)]
        if any(all(p[k] <= centre[k] for k in range(dim)) for p in front):
            dominated_cells += 1
        # increment odometer
        pos = 0
        while pos < dim:
            idx[pos] += 1
            if idx[pos] < grid:
                break
            idx[pos] = 0
            pos += 1
    return dominated_cells * cell_vol


def score_population(
    objectives: Sequence[Row],
    constraint_scores: Sequence[Row],
    ref_point: Row,
) -> ScoreCard:
    """Full deterministic scorecard for a population.

    Feasible designs (all constraints ``<= 0``) are normalised against
    ``ref_point`` and scored with :func:`hypervolume` in the unit cube (reference
    ``1...1``). ``mean_objectives_feasible`` is in normalised units.
    """
    if len(objectives) != len(constraint_scores):
        raise ValueError("objectives and constraint_scores must have equal length")
    n = len(objectives)
    mask = feasibility_mask(constraint_scores) if constraint_scores else [True] * n
    feasible_obj = [objectives[i] for i in range(n) if mask[i]]
    n_feasible = len(feasible_obj)

    if n_feasible == 0:
        width = len(ref_point)
        return ScoreCard(n, 0, 0.0, 0.0, tuple(0.0 for _ in range(width)))

    norm = normalize_objectives(feasible_obj, ref_point)
    unit_ref = tuple(1.0 for _ in ref_point)
    hv = hypervolume(norm, unit_ref)
    width = len(ref_point)
    means = tuple(sum(row[k] for row in norm) / n_feasible for k in range(width))
    return ScoreCard(
        n_designs=n,
        n_feasible=n_feasible,
        feasibility_rate=n_feasible / n if n else 0.0,
        hypervolume=hv,
        mean_objectives_feasible=means,
    )
