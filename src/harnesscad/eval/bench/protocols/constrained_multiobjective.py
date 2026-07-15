"""Constrained multi-objective engineering-design evaluation protocol
(Regenwetter et al., 2025, "BikeBench: A Bicycle Design Benchmark for Generative
Models with Objectives and Constraints").

BikeBench evaluates generative models not only on similarity to a dataset but on
whether generated designs *satisfy hard constraints* and *optimise multiple
real-world objectives* (aerodynamics, ergonomics, structure, ...). The bicycle
physics is out of scope, but the benchmark's scoring protocol is a deterministic
harness over per-design constraint values and objective values:

* A **constraint** is ``g(x) <= 0`` (satisfied when its value is <= tolerance). A
  design is *feasible* iff every constraint is satisfied.
* Population-level metrics mirror the paper: **feasibility rate** (fraction of
  feasible designs), **mean constraint-satisfaction rate** (average fraction of
  constraints each design satisfies), and per-constraint satisfaction rates.
* Objectives are aggregated only over feasible designs (an infeasible design is
  not a valid solution), via a weighted sum after min-max normalisation so
  heterogeneous units combine, matching the benchmark's "score" aggregation.

Deterministic, stdlib-only. Inputs are plain numeric dicts.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

__all__ = [
    "design_feasible",
    "constraint_satisfaction",
    "evaluate_population",
    "aggregate_objectives",
]


def design_feasible(constraints: Mapping[str, float], tol: float = 0.0) -> bool:
    """A design is feasible iff every constraint value ``g <= tol``."""
    return all(float(v) <= tol for v in constraints.values())


def constraint_satisfaction(constraints: Mapping[str, float], tol: float = 0.0) -> float:
    """Fraction of this design's constraints that are satisfied (0..1)."""
    if not constraints:
        return 1.0
    ok = sum(1 for v in constraints.values() if float(v) <= tol)
    return ok / len(constraints)


def evaluate_population(
    designs: Sequence[Mapping[str, float]], tol: float = 0.0
) -> Dict[str, object]:
    """Population-level constraint metrics over a list of per-design constraint maps.

    Returns ``feasibility_rate``, ``mean_satisfaction`` (mean per-design fraction
    satisfied), ``per_constraint`` (per-constraint satisfaction rate), and the
    list of feasible-design indices.
    """
    n = len(designs)
    if n == 0:
        raise ValueError("population must be non-empty")
    feasible = [i for i, d in enumerate(designs) if design_feasible(d, tol)]
    mean_sat = sum(constraint_satisfaction(d, tol) for d in designs) / n
    keys = sorted({k for d in designs for k in d})
    per_constraint = {}
    for k in keys:
        present = [d for d in designs if k in d]
        if present:
            per_constraint[k] = sum(
                1 for d in present if float(d[k]) <= tol
            ) / len(present)
    return {
        "feasibility_rate": len(feasible) / n,
        "mean_satisfaction": mean_sat,
        "per_constraint": per_constraint,
        "feasible_indices": feasible,
    }


def aggregate_objectives(
    objectives: Sequence[Mapping[str, float]],
    weights: Mapping[str, float],
    feasible_indices: Sequence[int],
    minimize: Sequence[str] = (),
) -> Dict[str, object]:
    """Weighted, min-max-normalised objective score over feasible designs.

    Each objective is normalised to ``[0, 1]`` across the feasible population;
    objectives named in ``minimize`` are inverted (``1 - norm``) so higher is
    always better. The design score is the weighted sum of normalised objectives.
    Returns ``{"scores": {idx: score}, "best": idx}``. With one feasible design,
    every normalised objective is 1.0 (degenerate range).
    """
    feas = list(feasible_indices)
    if not feas:
        return {"scores": {}, "best": None}
    minimize_set = set(minimize)
    keys = sorted(weights)
    ranges = {}
    for k in keys:
        vals = [float(objectives[i][k]) for i in feas]
        ranges[k] = (min(vals), max(vals))
    scores: Dict[int, float] = {}
    for i in feas:
        total = 0.0
        for k in keys:
            lo, hi = ranges[k]
            v = float(objectives[i][k])
            norm = 1.0 if hi == lo else (v - lo) / (hi - lo)
            if k in minimize_set:
                norm = 1.0 - norm
            total += float(weights[k]) * norm
        scores[i] = total
    best = max(scores, key=lambda i: (scores[i], -i))
    return {"scores": scores, "best": best}
