"""Deterministic generation metrics from LION (Zeng et al., NeurIPS 2022).

LION reports point-cloud generation quality with **1-NNA** (1-nearest-neighbour
accuracy), following PointFlow / Yang et al. It is a two-sample test that
measures the distributional similarity between a set of *generated* shapes and a
set of *reference* (validation) shapes, capturing both quality and diversity in
a single number. The paper also uses a Jensen-Shannon divergence computed over
the **voxel occupancy** of the pooled point sets.

Both quantities are fully deterministic once a shape-to-shape distance is fixed
(LION uses Chamfer distance and Earth-Mover distance, both already implemented
elsewhere in this repo -- ``geometry.dreamcad_metrics.chamfer_distance`` and
``reconstruction.gaussiancad_emd.mean_emd``). This module implements the
combinatorial metric machinery that consumes such a distance; it does NOT
re-implement the distances themselves, and it deliberately avoids the generic
coverage/MMD helper in ``bench.generative_brep_metrics``.

1-NNA definition (leave-one-out 1-NN classifier over ``S_g union S_r``)::

    For every sample s in S = S_g U S_r, find its nearest neighbour in
    S \\ {s}. Label the sample "generated" or "reference" by its true set, and
    predict the label of that nearest neighbour. 1-NNA is the *accuracy* of this
    classifier. A perfect generator makes the two sets indistinguishable, so the
    optimal (best) value is 0.5 (50%): the nearest neighbour is equally likely
    to come from either set. Values near 1.0 mean the sets are easily separated
    (poor generation); values near 0.0 mean strong over-/under-fitting.

Pure stdlib, deterministic (ties broken by lowest index).
"""

from __future__ import annotations

from math import log
from typing import Callable, Dict, List, Sequence, Tuple

# A "shape" here is opaque; the caller supplies a distance between two shapes.
Distance = Callable[[object, object], float]


def pairwise_distance_matrix(
    shapes: Sequence[object], distance: Distance
) -> List[List[float]]:
    """Symmetric all-pairs distance matrix (diagonal is 0.0).

    ``distance`` is assumed symmetric; it is evaluated once per unordered pair
    and mirrored, so an expensive Chamfer/EMD call runs ``n*(n-1)/2`` times.
    """
    n = len(shapes)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = float(distance(shapes[i], shapes[j]))
            mat[i][j] = d
            mat[j][i] = d
    return mat


def _nearest_other(mat: Sequence[Sequence[float]], i: int) -> int:
    """Index of the closest row to ``i`` excluding ``i`` (ties: lowest index)."""
    n = len(mat)
    best_j, best_d = -1, float("inf")
    for j in range(n):
        if j == i:
            continue
        if mat[i][j] < best_d:
            best_d, best_j = mat[i][j], j
    return best_j


def one_nna(
    generated: Sequence[object],
    reference: Sequence[object],
    distance: Distance,
) -> float:
    """1-NNA accuracy over ``generated`` (label 1) and ``reference`` (label 0).

    Returns a fraction in ``[0, 1]``; the optimal generator scores ``0.5``.
    Requires at least one shape in each set (and at least two shapes total).
    """
    ng, nr = len(generated), len(reference)
    if ng == 0 or nr == 0:
        raise ValueError("both generated and reference sets must be non-empty")
    if ng + nr < 2:
        raise ValueError("need at least two shapes in total")
    shapes = list(generated) + list(reference)
    labels = [1] * ng + [0] * nr
    mat = pairwise_distance_matrix(shapes, distance)
    correct = 0
    for i in range(len(shapes)):
        j = _nearest_other(mat, i)
        if labels[j] == labels[i]:
            correct += 1
    return correct / len(shapes)


def one_nna_from_matrix(
    matrix: Sequence[Sequence[float]], n_generated: int
) -> float:
    """1-NNA when the pooled distance matrix is precomputed.

    ``matrix`` is the square all-pairs matrix over ``generated ++ reference``
    (generated shapes occupy the first ``n_generated`` rows/columns). Useful
    when Chamfer *and* EMD variants are needed from one distance computation.
    """
    n = len(matrix)
    if not 0 < n_generated < n:
        raise ValueError("n_generated must be in (0, n)")
    labels = [1 if i < n_generated else 0 for i in range(n)]
    correct = 0
    for i in range(n):
        j = _nearest_other(matrix, i)
        if labels[j] == labels[i]:
            correct += 1
    return correct / n


def voxel_index(
    point: Sequence[float],
    grid: int,
    bounds: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[int, int, int]:
    """Map a 3D point to an integer voxel cell of a ``grid^3`` lattice.

    Points are clamped to ``bounds`` (LION normalises shapes into ``[-1, 1]``),
    so out-of-range coordinates fall into the boundary cells rather than raising.
    """
    lo, hi = bounds
    if hi <= lo:
        raise ValueError("bounds must be increasing")
    if grid < 1:
        raise ValueError("grid must be >= 1")
    span = hi - lo
    out = []
    for c in point[:3]:
        frac = (float(c) - lo) / span
        idx = int(frac * grid)
        if idx < 0:
            idx = 0
        elif idx >= grid:
            idx = grid - 1
        out.append(idx)
    return (out[0], out[1], out[2])


def voxel_occupancy_distribution(
    clouds: Sequence[Sequence[Sequence[float]]],
    grid: int = 28,
    bounds: Tuple[float, float] = (-1.0, 1.0),
) -> Dict[Tuple[int, int, int], int]:
    """Pooled voxel-occupancy histogram over a collection of point clouds.

    Every point of every cloud is dropped into the shared ``grid^3`` lattice and
    counted. The returned count dict is exactly the form consumed by
    ``bench.generative_brep_metrics.jsd`` -- pass the generated-set and
    reference-set histograms to that function to obtain LION's voxel JSD.
    """
    hist: Dict[Tuple[int, int, int], int] = {}
    for cloud in clouds:
        for pt in cloud:
            cell = voxel_index(pt, grid, bounds)
            hist[cell] = hist.get(cell, 0) + 1
    return hist


def voxel_jsd(
    generated: Sequence[Sequence[Sequence[float]]],
    reference: Sequence[Sequence[Sequence[float]]],
    grid: int = 28,
    bounds: Tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Jensen-Shannon divergence (base 2) between two pooled voxel occupancies.

    Self-contained so callers need not wire the histogram helper to ``jsd``; the
    result is identical to feeding both occupancy dicts through the shared JSD.
    Returns a value in ``[0, 1]`` (0 = identical occupancy).
    """
    a = voxel_occupancy_distribution(generated, grid, bounds)
    b = voxel_occupancy_distribution(reference, grid, bounds)
    keys = set(a) | set(b)
    sa = sum(a.values())
    sb = sum(b.values())
    if not sa or not sb:
        raise ValueError("both cloud collections must contain at least one point")
    p = {k: a.get(k, 0) / sa for k in keys}
    q = {k: b.get(k, 0) / sb for k in keys}
    m = {k: (p[k] + q[k]) / 2.0 for k in keys}

    def _kl(x: Dict) -> float:
        return sum(v * log(v / m[k], 2) for k, v in x.items() if v > 0)

    return (_kl(p) + _kl(q)) / 2.0
