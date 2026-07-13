"""Topology-consistency metrics for generated vs target 3D shapes.

Topology-Aware LDM (Hu, Fei et al., 2024) generates shapes conditioned on a
*target topology* -- a Betti number (Sec. 4.2, "Betti number") or a target
persistence diagram (Sec. 4.3) -- and evaluates whether the generated shape
actually *realises* that topology.  Sec. 6 states the guiding rule: as a PD
point is drawn toward the diagonal its feature's lifespan shortens and vanishes,
and "if all points were positioned on the diagonal, then the shape would ...
[be] a genus zero object."

This module scores that consistency deterministically:

  * ``betti_match`` / ``betti_vector_distance`` -- does the generated Betti
    vector ``(beta_0, beta_1, beta_2)`` equal / how far is it from the target
    (via ``geometry/topodiff_betti_voxel.py``).
  * ``genus_match`` / ``component_match`` / ``cavity_match`` -- per-invariant
    agreement.
  * ``persistence_diagram_distance`` -- a 1-Wasserstein-style optimal matching
    (greedy, with diagonal projection) between two PDs from
    ``numeric/topodiff_cubical_persistence.py``; a small distance means the two
    shapes share topological scale-structure.
  * ``implies_genus_zero`` -- Sec. 6 rule: a PD whose every point is within
    ``eps`` of the diagonal has no persistent loops -> genus-zero.
  * ``topology_consistency_report`` -- dataset-level aggregate (Betti-match %,
    mean genus error, mean PD distance).

Complements ``bench/evocad_topology_metrics.py`` (single-scalar ``chi`` error on
meshes): here consistency is measured over the *full Betti vector* of solid
voxel shapes and over *persistence diagrams*, neither of which evocad addresses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.volumes.topodiff_betti_voxel import betti_numbers

Voxel = Tuple[int, int, int]
Pair = Tuple[float, float]


# --------------------------------------------------------------------------
# Betti-vector consistency
# --------------------------------------------------------------------------

def betti_vector_distance(
    a: Tuple[int, int, int], b: Tuple[int, int, int]
) -> int:
    """L1 distance ``|da0|+|da1|+|da2|`` between two Betti vectors."""
    return sum(abs(a[i] - b[i]) for i in range(3))


def betti_match(
    generated: Iterable[Voxel], target: Iterable[Voxel]
) -> int:
    """``1`` iff the two voxel shapes share ``(beta_0, beta_1, beta_2)``."""
    g = betti_numbers(generated).vector
    t = betti_numbers(target).vector
    return 1 if g == t else 0


def betti_match_vectors(
    generated: Tuple[int, int, int], target: Tuple[int, int, int]
) -> int:
    """``1`` iff two precomputed Betti vectors agree."""
    return 1 if tuple(generated) == tuple(target) else 0


def genus_match(generated: Iterable[Voxel], target: Iterable[Voxel]) -> int:
    """``1`` iff the generated and target shapes have equal handle count."""
    return 1 if betti_numbers(generated).b1 == betti_numbers(target).b1 else 0


def component_match(generated: Iterable[Voxel], target: Iterable[Voxel]) -> int:
    """``1`` iff ``beta_0`` (connected components) agree."""
    return 1 if betti_numbers(generated).b0 == betti_numbers(target).b0 else 0


def cavity_match(generated: Iterable[Voxel], target: Iterable[Voxel]) -> int:
    """``1`` iff ``beta_2`` (enclosed voids) agree."""
    return 1 if betti_numbers(generated).b2 == betti_numbers(target).b2 else 0


@dataclass(frozen=True)
class ConsistencyResult:
    """Per-sample topology consistency of a generated shape vs its target."""

    generated: Tuple[int, int, int]
    target: Tuple[int, int, int]
    betti_match: int
    betti_l1: int
    genus_error: int
    component_error: int
    cavity_error: int


def topology_consistency(
    generated: Iterable[Voxel], target: Iterable[Voxel]
) -> ConsistencyResult:
    """Full Betti-vector consistency between a generated and target voxel shape."""
    g = betti_numbers(generated)
    t = betti_numbers(target)
    return ConsistencyResult(
        generated=g.vector,
        target=t.vector,
        betti_match=1 if g.vector == t.vector else 0,
        betti_l1=betti_vector_distance(g.vector, t.vector),
        genus_error=abs(g.b1 - t.b1),
        component_error=abs(g.b0 - t.b0),
        cavity_error=abs(g.b2 - t.b2),
    )


# --------------------------------------------------------------------------
# Persistence-diagram consistency
# --------------------------------------------------------------------------

def _diag_dist(p: Pair) -> float:
    """Chebyshev distance from birth-death point ``p`` to the diagonal."""
    b, d = p
    return abs(d - b) / 2.0


def _pt_dist(p: Pair, q: Pair) -> float:
    """Chebyshev (L-inf) distance between two birth-death points."""
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


def persistence_diagram_distance(
    diagram_a: Sequence[Pair],
    diagram_b: Sequence[Pair],
) -> float:
    """Greedy 1-Wasserstein-style matching distance between two finite PDs.

    Off-diagonal points of each diagram are matched to the nearest available
    point of the other (or projected to the diagonal), summing L-inf costs.
    A deterministic greedy assignment: repeatedly take the globally cheapest
    remaining real-to-real pair, then send every leftover point to the diagonal.
    This is a stable, order-independent approximation of the optimal bottleneck /
    Wasserstein transport used to compare PD topology.
    """
    a = [p for p in diagram_a if not math.isinf(p[1])]
    b = [p for p in diagram_b if not math.isinf(p[1])]
    used_a = [False] * len(a)
    used_b = [False] * len(b)
    # Candidate real-real matches, cheapest first (deterministic tie-break).
    candidates: List[Tuple[float, int, int]] = []
    for i, pa in enumerate(a):
        for j, pb in enumerate(b):
            candidates.append((_pt_dist(pa, pb), i, j))
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    total = 0.0
    for cost, i, j in candidates:
        if used_a[i] or used_b[j]:
            continue
        # Only match if cheaper than sending both to the diagonal.
        if cost <= _diag_dist(a[i]) + _diag_dist(b[j]):
            used_a[i] = used_b[j] = True
            total += cost
    for i, pa in enumerate(a):
        if not used_a[i]:
            total += _diag_dist(pa)
    for j, pb in enumerate(b):
        if not used_b[j]:
            total += _diag_dist(pb)
    return total


def implies_genus_zero(
    diagram: Sequence[Pair], eps: float = 1e-9
) -> bool:
    """Sec. 6 rule: every finite PD point within ``eps`` of the diagonal.

    When all loop/void classes have (near-)zero persistence there are no lasting
    topological features, so the shape collapses to a genus-zero object.
    """
    for b, d in diagram:
        if math.isinf(d):
            continue
        if abs(d - b) > eps:
            return False
    return True


def collapse_to_diagonal(diagram: Sequence[Pair]) -> List[Pair]:
    """Project every finite point onto the diagonal (persistence -> 0).

    Models the paper's topology edit that removes a feature by drawing its PD
    point to the diagonal; the result ``implies_genus_zero``.
    """
    out: List[Pair] = []
    for b, d in diagram:
        if math.isinf(d):
            out.append((b, d))
        else:
            mid = (b + d) / 2.0
            out.append((mid, mid))
    return out


def significant_features(
    diagram: Sequence[Pair], min_persistence: float
) -> int:
    """Count finite PD points whose persistence exceeds ``min_persistence``.

    Near-diagonal points are treated as topological noise (Sec. 3, last para).
    """
    n = 0
    for b, d in diagram:
        if math.isinf(d):
            continue
        if (d - b) > min_persistence:
            n += 1
    return n


# --------------------------------------------------------------------------
# Dataset-level aggregate
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TopologyConsistencyReport:
    samples: int
    betti_match_pct: Optional[float]
    mean_genus_error: Optional[float]
    mean_betti_l1: Optional[float]


def topology_consistency_report(
    pairs: Sequence[Tuple[Iterable[Voxel], Iterable[Voxel]]]
) -> TopologyConsistencyReport:
    """Aggregate consistency over ``(generated, target)`` voxel-shape pairs."""
    n = len(pairs)
    if n == 0:
        return TopologyConsistencyReport(0, None, None, None)
    results = [topology_consistency(g, t) for g, t in pairs]
    match_pct = sum(r.betti_match for r in results) / n * 100.0
    mean_genus = sum(r.genus_error for r in results) / n
    mean_l1 = sum(r.betti_l1 for r in results) / n
    return TopologyConsistencyReport(n, match_pct, mean_genus, mean_l1)
