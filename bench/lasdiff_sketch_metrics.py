"""Evaluation metrics from LAS-Diffusion's sketch-conditioned experiments.

From "Locally Attentional SDF Diffusion for Controllable 3D Shape Generation"
(Zheng et al., ACM TOG 2023), Sections 4.1 and 5. The paper adapts several
deterministic metrics for sketch-conditioned probabilistic generation:

  * **Sketch-CD** (Section 4.1): "treat the non-white pixels of sketches as 2D
    points and measure the 2D Chamfer distance between I and G". We rasterise a
    sketch into its non-background pixel coordinates and compute the symmetric
    2D Chamfer distance between two such point sets.
  * **1-NNA** (Section 5): 1-nearest-neighbour accuracy over the union of the
    generated and reference sets. For each sample the nearest *other* sample is
    found (leave-one-out) and we ask whether it shares the sample's label; the
    accuracy is reported -- a value near 50% means the two distributions are
    indistinguishable (best), so we also report the absolute gap to 50%.
  * **SquareRoot binning** (Fig. 20): the paper draws its Chamfer-distance
    histogram with the square-root rule ``bins = ceil(sqrt(n))``.

FID / Frechet-Gaussian distance and COV/MMD are already provided by
``bench.gencad_fid`` and ``bench.generative_brep_metrics`` respectively, so they
are deliberately not re-implemented here. Stdlib-only, deterministic.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Point = Tuple[float, float]


def sketch_points(raster: Sequence[Sequence[float]], white: float = 1.0,
                  tol: float = 1e-9) -> List[Point]:
    """Non-white pixels of a sketch raster as ``(x, y)`` points.

    ``raster[row][col]`` is a scalar intensity; a pixel is a *stroke* point when
    it differs from the ``white`` background by more than ``tol``. ``x`` is the
    column and ``y`` is the row, matching image convention.
    """
    pts: List[Point] = []
    for r, rowvals in enumerate(raster):
        for c, v in enumerate(rowvals):
            if abs(v - white) > tol:
                pts.append((float(c), float(r)))
    return pts


def _nearest_sq(p: Point, others: Sequence[Point]) -> float:
    px, py = p
    best = math.inf
    for qx, qy in others:
        d = (px - qx) ** 2 + (py - qy) ** 2
        if d < best:
            best = d
    return best


def chamfer_2d(a: Sequence[Point], b: Sequence[Point], squared: bool = False) -> float:
    """Symmetric 2D Chamfer distance between two point sets.

    ``(1/|A|) sum_a min_b d(a,b) + (1/|B|) sum_b min_a d(b,a)``, using Euclidean
    distance (or squared Euclidean when ``squared=True``). Raises on empty input.
    """
    if not a or not b:
        raise ValueError("both point sets must be non-empty")
    def term(src, dst):
        s = 0.0
        for p in src:
            d2 = _nearest_sq(p, dst)
            s += d2 if squared else math.sqrt(d2)
        return s / len(src)
    return term(a, b) + term(b, a)


def sketch_cd(raster_i: Sequence[Sequence[float]], raster_g: Sequence[Sequence[float]],
              white: float = 1.0) -> float:
    """Sketch-CD: 2D Chamfer distance between two rasterised sketches."""
    return chamfer_2d(sketch_points(raster_i, white), sketch_points(raster_g, white))


def one_nearest_neighbor_accuracy(dist: Sequence[Sequence[float]],
                                  labels: Sequence[int]) -> float:
    """1-NNA over a full pairwise distance matrix with ``{0, 1}`` labels.

    For each sample ``i`` find ``argmin_{j != i} dist[i][j]`` and count a hit when
    ``labels[j] == labels[i]``. Returns the accuracy in ``[0, 1]``. The diagonal
    is ignored. Ties break to the lowest index.
    """
    n = len(labels)
    if n < 2:
        raise ValueError("need at least two samples")
    if len(dist) != n or any(len(row) != n for row in dist):
        raise ValueError("distance matrix must be square and match labels")
    hits = 0
    for i in range(n):
        best_j, best_d = -1, math.inf
        for j in range(n):
            if j == i:
                continue
            if dist[i][j] < best_d:
                best_d, best_j = dist[i][j], j
        if labels[best_j] == labels[i]:
            hits += 1
    return hits / n


def nna_gap_to_half(accuracy: float) -> float:
    """Absolute distance of a 1-NNA accuracy from the ideal 50% (lower better)."""
    return abs(accuracy - 0.5)


def sqrt_bin_count(n: int) -> int:
    """Square-root binning rule ``ceil(sqrt(n))`` used for the CD histogram."""
    if n <= 0:
        raise ValueError("n must be positive")
    return max(1, math.ceil(math.sqrt(n)))


def histogram(values: Sequence[float], bins: int) -> List[int]:
    """Equal-width histogram counts over ``[min, max]`` with ``bins`` buckets.

    The maximum value falls in the last bucket. A degenerate range (all equal)
    puts every value in bucket 0.
    """
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not values:
        raise ValueError("values must be non-empty")
    lo, hi = min(values), max(values)
    counts = [0] * bins
    if hi == lo:
        counts[0] = len(values)
        return counts
    width = (hi - lo) / bins
    for v in values:
        idx = int((v - lo) / width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    return counts
