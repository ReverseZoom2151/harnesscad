"""BikeBench task taxonomy + population diversity/novelty metrics.

Ported from BikeBench (``src/bikebench/design_evaluation/design_evaluation.py``
and ``src/bikebench/benchmarking/scoring.py``).  BikeBench is a generative-design
benchmark: a *population* of candidate designs is scored against a fixed set of
**requirements**, each of which is either an *objective* (minimised) or a
*constraint* (satisfied when ``<= 0``), and some of which are *conditional* on
the design brief (e.g. rider dimensions, target aesthetic text).

This module contributes two things the harness's existing
``eval.bench.multiobjective`` (which already ports the feasibility/hypervolume
core) did not have:

1.  **The requirement taxonomy** (:data:`REQUIREMENTS`) -- the catalogue of what
    BikeBench actually measures, grouped by evaluator (Aero, FrameValidity,
    Structural, Aesthetics, Ergonomics, Validation), each tagged
    objective/constraint and conditional/unconditional.  This is the "task
    definition" an eval author needs to know which columns are goals vs. gates.
2.  **The population distribution metrics** that ``multiobjective`` omitted
    because they needed numpy/pygmo/sklearn: maximum-mean-discrepancy to a
    reference set (:func:`mmd_rbf`), nearest-reference novelty
    (:func:`average_novelty`), determinantal-point-process diversity
    (:func:`dpp_diversity`), and the per-requirement breakdowns
    (:func:`constraint_violation_rate`, :func:`mean_constraint_violation_magnitude`,
    :func:`average_constraint_violation`, :func:`min_objective`,
    :func:`mean_objective`).

All metrics are pure-stdlib and deterministic (a fixed reference standardiser,
median-heuristic RBF bandwidth, and an LU-based log-determinant instead of an
eigen-decomposition, so no numerical-library nondeterminism enters).

The learned surrogate models, the ``pygmo`` hypervolume, and all rendering are
intentionally dropped -- what remains is the benchmark's scoring bookkeeping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

__all__ = [
    "Requirement",
    "REQUIREMENTS",
    "evaluators",
    "objective_names",
    "constraint_names",
    "conditional_names",
    "requirements_for",
    "standardiser",
    "apply_standardiser",
    "rbf_gamma_median",
    "mmd_rbf",
    "average_novelty",
    "dpp_diversity",
    "average_constraint_violation",
    "constraint_violation_rate",
    "mean_constraint_violation_magnitude",
    "min_objective",
    "mean_objective",
]

Row = Sequence[float]


# ---------------------------------------------------------------------------
# Task taxonomy (mirrored from design_evaluation.py return_names/is_objective).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Requirement:
    """One scored quantity in the benchmark.

    ``kind`` is ``"objective"`` (minimised) or ``"constraint"`` (feasible when
    ``<= 0``).  ``conditional`` is True when the requirement depends on the
    design brief (rider / target text) rather than the geometry alone.
    """

    name: str
    evaluator: str
    kind: str
    conditional: bool

    @property
    def is_objective(self) -> bool:
        return self.kind == "objective"


REQUIREMENTS: Tuple[Requirement, ...] = (
    # Aero
    Requirement("Drag Force (N)", "Aero", "objective", True),
    # FrameValidity
    Requirement("Predicted Frame Validity", "FrameValidity", "constraint", False),
    # Structural
    Requirement("Mass (kg)", "Structural", "objective", False),
    Requirement("Planar Compliance Score", "Structural", "objective", False),
    Requirement("Transverse Compliance Score", "Structural", "objective", False),
    Requirement("Eccentric Compliance Score", "Structural", "objective", False),
    Requirement("Planar Safety Factor", "Structural", "constraint", False),
    Requirement("Eccentric Safety Factor", "Structural", "constraint", False),
    # Aesthetics
    Requirement("Cosine Distance To Text", "Aesthetics", "objective", True),
    # Ergonomics (default mode: 3 angle-error objectives + 6 fit constraints)
    Requirement("Knee Angle Error (deg.)", "Ergonomics", "objective", True),
    Requirement("Hip Angle Error (deg.)", "Ergonomics", "objective", True),
    Requirement("Arm Angle Error (deg.)", "Ergonomics", "objective", True),
    Requirement("Arm Too Long for Bike", "Ergonomics", "constraint", True),
    Requirement("Saddle Too Far From Handle", "Ergonomics", "constraint", True),
    Requirement("Torso Too Long for Bike", "Ergonomics", "constraint", True),
    Requirement("Saddle Too Far From Crank", "Ergonomics", "constraint", True),
    Requirement("Upper Leg Too Long for Bike", "Ergonomics", "constraint", True),
    Requirement("Lower Leg Too Long for Bike", "Ergonomics", "constraint", True),
)


def evaluators() -> List[str]:
    """Distinct evaluator names in declaration order."""
    seen: List[str] = []
    for r in REQUIREMENTS:
        if r.evaluator not in seen:
            seen.append(r.evaluator)
    return seen


def objective_names() -> List[str]:
    return [r.name for r in REQUIREMENTS if r.is_objective]


def constraint_names() -> List[str]:
    return [r.name for r in REQUIREMENTS if not r.is_objective]


def conditional_names() -> List[str]:
    return [r.name for r in REQUIREMENTS if r.conditional]


def requirements_for(evaluator: str) -> List[Requirement]:
    return [r for r in REQUIREMENTS if r.evaluator == evaluator]


# ---------------------------------------------------------------------------
# Standardisation (sklearn StandardScaler fit on a reference set).
# ---------------------------------------------------------------------------

def standardiser(reference: Sequence[Row]) -> Tuple[List[float], List[float]]:
    """Return ``(mean, std)`` per column over ``reference`` (population std).

    Mirrors ``StandardScaler().fit(reference)``.  Zero-variance columns get a
    std of 1.0 so :func:`apply_standardiser` leaves them centred but unscaled
    (sklearn's behaviour).
    """
    if not reference:
        raise ValueError("reference set is empty")
    n = len(reference)
    d = len(reference[0])
    mean = [0.0] * d
    for row in reference:
        for j in range(d):
            mean[j] += row[j]
    mean = [m / n for m in mean]
    var = [0.0] * d
    for row in reference:
        for j in range(d):
            diff = row[j] - mean[j]
            var[j] += diff * diff
    std = [math.sqrt(v / n) if v > 0 else 1.0 for v in var]
    return mean, std


def apply_standardiser(rows: Sequence[Row], mean: Sequence[float], std: Sequence[float]) -> List[List[float]]:
    return [[(row[j] - mean[j]) / std[j] for j in range(len(mean))] for row in rows]


# ---------------------------------------------------------------------------
# Distance / kernel helpers.
# ---------------------------------------------------------------------------

def _sqdist(a: Row, b: Row) -> float:
    return sum((a[j] - b[j]) ** 2 for j in range(len(a)))


def rbf_gamma_median(reference: Sequence[Row]) -> float:
    """Median-heuristic RBF bandwidth: ``1 / (2 * median pairwise sqdist)``.

    Mirrors ``MMD.compute_gamma``.  Uses all ordered pairs (including the zero
    self-distances), exactly like the reference which takes the median of the
    full ``n*n`` squared-distance matrix.
    """
    n = len(reference)
    if n == 0:
        raise ValueError("reference set is empty")
    dists: List[float] = []
    for i in range(n):
        for j in range(n):
            dists.append(_sqdist(reference[i], reference[j]))
    dists.sort()
    m = len(dists)
    if m % 2 == 1:
        med = dists[m // 2]
    else:
        med = 0.5 * (dists[m // 2 - 1] + dists[m // 2])
    return 1.0 / (2.0 * med) if med > 0 else 1.0


def _rbf_kernel_sum(A: Sequence[Row], B: Sequence[Row], gamma: float) -> float:
    total = 0.0
    for a in A:
        for b in B:
            total += math.exp(-gamma * _sqdist(a, b))
    return total


def mmd_rbf(generated: Sequence[Row], reference: Sequence[Row], gamma: float) -> float:
    """Squared MMD between two sets under an RBF kernel (mirror of ``MMD.mmd``).

    ``MMD = K_GG/n^2 + K_RR/m^2 - 2 K_GR/(n m)``.  Inputs are assumed already
    standardised by the caller (BikeBench standardises with the reference
    scaler first).
    """
    n = len(generated)
    m = len(reference)
    if n == 0 or m == 0:
        raise ValueError("both sets must be non-empty")
    k_gg = _rbf_kernel_sum(generated, generated, gamma)
    k_rr = _rbf_kernel_sum(reference, reference, gamma)
    k_gr = _rbf_kernel_sum(generated, reference, gamma)
    return k_gg / (n * n) + k_rr / (m * m) - 2.0 * k_gr / (n * m)


def average_novelty(generated: Sequence[Row], reference: Sequence[Row]) -> float:
    """Mean over generated designs of the nearest-reference Euclidean distance.

    Mirror of ``AverageNovelty.evaluate`` (higher = more novel).  Inputs assumed
    standardised.
    """
    if not generated:
        raise ValueError("generated set is empty")
    if not reference:
        raise ValueError("reference set is empty")
    total = 0.0
    for g in generated:
        best = min(_sqdist(g, r) for r in reference)
        total += math.sqrt(best)
    return total / len(generated)


# ---------------------------------------------------------------------------
# DPP diversity (log-determinant of an RBF similarity matrix).
# ---------------------------------------------------------------------------

def _logdet_spd(matrix: List[List[float]]) -> float:
    """log-determinant of a (numerically SPD) matrix via LU with partial pivot.

    Deterministic and stdlib-only.  Returns ``sum(log|U_ii|)``.  Row swaps do not
    affect the magnitude of the determinant, so we track only ``|U_ii|`` -- the
    determinant of a similarity (Gram) matrix is non-negative.
    """
    n = len(matrix)
    a = [list(row) for row in matrix]
    logdet = 0.0
    for col in range(n):
        # partial pivot: pick the largest-magnitude entry in this column
        pivot = col
        best = abs(a[col][col])
        for r in range(col + 1, n):
            if abs(a[r][col]) > best:
                best = abs(a[r][col])
                pivot = r
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        diag = a[col][col]
        if diag == 0.0:
            return -math.inf
        logdet += math.log(abs(diag))
        for r in range(col + 1, n):
            factor = a[r][col] / diag
            if factor != 0.0:
                for c in range(col, n):
                    a[r][c] -= factor * a[col][c]
    return logdet


def dpp_diversity(designs: Sequence[Row], mean: Sequence[float], std: Sequence[float]) -> float:
    """DPP diversity loss (lower = more diverse), mirror of ``DPPDiversity``.

    Deduplicates designs, standardises them, builds the RBF similarity matrix
    ``S = exp(-0.5 * D^2)`` where ``D`` is the dimension-normalised squared
    distance, adds a tiny diagonal jitter for PD stability, then returns
    ``-(1/n) * logdet(S)``.  The reference averages ``-log`` of the eigenvalues;
    since ``sum log eig = logdet`` for SPD ``S`` this is identical (we use the
    deterministic LU log-determinant instead of an eigensolver).
    """
    # deduplicate, preserving first-seen order
    seen = set()
    uniq: List[Row] = []
    for row in designs:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            uniq.append(row)
    n = len(uniq)
    if n <= 1:
        return 0.0
    dim = len(uniq[0])
    xs = apply_standardiser(uniq, mean, std)
    # dimension-normalised squared distances (D / dim^2 in the reference)
    S = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d2 = _sqdist(xs[i], xs[j]) / (dim * dim)
            S[i][j] = math.exp(-0.5 * (d2 * d2))
    for i in range(n):
        S[i][i] += 1e-12
    logdet = _logdet_spd(S)
    return -logdet / n


# ---------------------------------------------------------------------------
# Per-requirement breakdown metrics (mirror the Detailed scorers).
# ---------------------------------------------------------------------------

def _feasible_rows(objective_scores: Sequence[Row], constraint_scores: Sequence[Row]) -> List[Row]:
    out = []
    for objs, cons in zip(objective_scores, constraint_scores):
        if all(c <= 0 for c in cons):
            out.append(objs)
    return out


def average_constraint_violation(constraint_scores: Sequence[Row]) -> float:
    """Mean number of violated constraints per design (``AverageConstraintViolation``)."""
    if not constraint_scores:
        return 0.0
    return sum(sum(1 for c in row if c > 0) for row in constraint_scores) / len(constraint_scores)


def constraint_violation_rate(constraint_scores: Sequence[Row]) -> List[float]:
    """Per-constraint fraction of designs that violate it (``ConstraintViolationRate``)."""
    if not constraint_scores:
        return []
    d = len(constraint_scores[0])
    n = len(constraint_scores)
    return [sum(1 for row in constraint_scores if row[j] > 0) / n for j in range(d)]


def mean_constraint_violation_magnitude(constraint_scores: Sequence[Row]) -> List[float]:
    """Per-constraint mean of ``max(score, 0)`` (``MeanConstraintViolationMagnitude``)."""
    if not constraint_scores:
        return []
    d = len(constraint_scores[0])
    n = len(constraint_scores)
    return [sum(max(row[j], 0.0) for row in constraint_scores) / n for j in range(d)]


def min_objective(
    objective_scores: Sequence[Row],
    constraint_scores: Sequence[Row],
    ref_point: Row,
) -> List[float]:
    """Per-objective minimum over feasible designs (``MinimumObjective``).

    If no design is feasible, returns the reference point (BikeBench's fallback).
    """
    feas = _feasible_rows(objective_scores, constraint_scores)
    if not feas:
        return list(ref_point)
    d = len(feas[0])
    return [min(row[j] for row in feas) for j in range(d)]


def mean_objective(
    objective_scores: Sequence[Row],
    constraint_scores: Sequence[Row],
    ref_point: Row,
) -> List[float]:
    """Per-objective mean over feasible designs (``MeanObjective``)."""
    feas = _feasible_rows(objective_scores, constraint_scores)
    if not feas:
        return list(ref_point)
    d = len(feas[0])
    n = len(feas)
    return [sum(row[j] for row in feas) / n for j in range(d)]
