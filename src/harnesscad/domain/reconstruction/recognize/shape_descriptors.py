"""Deterministic rotation-invariant shape descriptors for 3D object retrieval.

Van den Herrewegen et al., *Fine-Tuning 3D Foundation Models for Geometric
Object Retrieval* (Computers & Graphics 2024) fine-tune a learned PointBERT
encoder and compare it against classic *axiomatic* descriptors (Tangelder &
Veltkamp survey [1], harmonic decompositions [13]) that "were defined in a
purely axiomatic manner". The learned foundation-model encoder is
research-heavy/external; the axiomatic descriptors it competes with are
closed-form and fully buildable.

This module provides three rotation- and translation-invariant global shape
signatures that turn an unordered point cloud into a fixed-length vector for
content-based retrieval:

* :func:`d2_shape_distribution` -- Osada et al. D2 shape distribution: a
  histogram of Euclidean distances between random pairs of surface points.
* :func:`radial_shell_signature` -- a spherical-harmonic-lite descriptor: the
  distribution of point distances from the centroid binned into concentric
  shells (invariant to rotation about the centroid).
* :func:`bounding_volume_signature` -- a PCA-aligned bounding-box signature
  (principal extents, aspect ratios, compactness) that is invariant to the
  object's arbitrary world orientation.

All descriptors are deterministic (any sampling uses ``random.Random(seed)``),
stdlib-only, and normalised so two clouds differing only by a rigid motion (and,
for the shells, a chosen scale normalisation) produce identical vectors.

The distance-histogram bin edges and the PCA sign convention are fixed so the
output is byte-reproducible.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

from harnesscad.domain.geometry.transforms.umeyama import jacobi_eigen

Point = Sequence[float]


def _as_points(points: Sequence[Point]) -> List[Tuple[float, float, float]]:
    out = []
    for p in points:
        vals = tuple(float(x) for x in p)
        if len(vals) != 3:
            raise ValueError("each point must have exactly 3 coordinates")
        out.append(vals)
    return out


def _centroid(pts: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    n = len(pts)
    return tuple(sum(p[d] for p in pts) / n for d in range(3))  # type: ignore[return-value]


def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _histogram(values: Sequence[float], bins: int, hi: float) -> List[float]:
    """L1-normalised histogram of ``values`` over ``[0, hi]`` with ``bins`` bins.

    Values >= ``hi`` fall in the last bin. Returns a probability vector that sums
    to 1 (or all zeros if there are no values / ``hi`` is 0).
    """
    counts = [0.0] * bins
    if hi <= 0.0 or not values:
        return counts
    width = hi / bins
    for v in values:
        idx = int(v / width)
        if idx >= bins:
            idx = bins - 1
        elif idx < 0:
            idx = 0
        counts[idx] += 1.0
    total = sum(counts)
    if total == 0.0:
        return counts
    return [c / total for c in counts]


def d2_shape_distribution(points: Sequence[Point], *, bins: int = 16,
                          samples: int = 1024, seed: int = 0) -> List[float]:
    """Osada D2 shape distribution: histogram of random pairwise distances.

    Draws ``samples`` random ordered pairs ``(i, j)``, ``i != j``, and histograms
    their Euclidean distance. Distances are normalised by the mean pairwise
    distance so the descriptor is scale-invariant (and rotation/translation
    invariant by construction), following the standard D2 normalisation. The
    histogram range is fixed to ``[0, 3]`` mean-distance units (a distance of 3x
    the mean is extremely rare) so descriptors of different clouds are
    directly comparable bin-for-bin.
    """
    if bins <= 0:
        raise ValueError("bins must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")
    pts = _as_points(points)
    n = len(pts)
    if n < 2:
        return [0.0] * bins
    rng = random.Random(seed)
    dists = []
    for _ in range(samples):
        i = rng.randrange(n)
        j = rng.randrange(n)
        while j == i:
            j = rng.randrange(n)
        dists.append(_dist(pts[i], pts[j]))
    mean = sum(dists) / len(dists)
    if mean <= 0.0:
        return [0.0] * bins
    normed = [d / mean for d in dists]
    return _histogram(normed, bins, 3.0)


def radial_shell_signature(points: Sequence[Point], *, bins: int = 16) -> List[float]:
    """Spherical-harmonic-lite descriptor: distances from centroid, binned.

    Centres the cloud at its centroid, then histograms every point's distance to
    the centroid over ``bins`` concentric shells spanning ``[0, r_max]``. Because
    it depends only on radii, the descriptor is invariant to rotation about the
    centroid (and to translation). Scale-normalised by the maximum radius so the
    range is fixed to ``[0, 1]``. This captures the same rotation-invariant radial
    energy that a truncated spherical-harmonic power spectrum would summarise.
    """
    if bins <= 0:
        raise ValueError("bins must be positive")
    pts = _as_points(points)
    if not pts:
        return [0.0] * bins
    c = _centroid(pts)
    radii = [_dist(p, c) for p in pts]
    r_max = max(radii)
    return _histogram(radii, bins, r_max)


def _covariance(pts: Sequence[Tuple[float, float, float]],
                c: Tuple[float, float, float]) -> List[List[float]]:
    cov = [[0.0, 0.0, 0.0] for _ in range(3)]
    for p in pts:
        d = (p[0] - c[0], p[1] - c[1], p[2] - c[2])
        for a in range(3):
            for b in range(3):
                cov[a][b] += d[a] * d[b]
    n = len(pts)
    for a in range(3):
        for b in range(3):
            cov[a][b] /= n
    return cov


def pca_extents(points: Sequence[Point]) -> Tuple[float, float, float]:
    """Principal-axis extents (span along each PCA eigenvector), descending.

    Diagonalises the point covariance (via the shared Jacobi eigensolver),
    projects the centred points onto the eigenvectors, and returns the
    max-minus-min span along each principal axis, sorted descending. Rotation-
    invariant because the eigenvectors rotate with the object.
    """
    pts = _as_points(points)
    if not pts:
        return (0.0, 0.0, 0.0)
    c = _centroid(pts)
    cov = _covariance(pts, c)
    _eigvals, vecs = jacobi_eigen(cov)
    # vecs columns are eigenvectors (descending eigenvalue order).
    spans = []
    for axis in range(3):
        ax = [vecs[r][axis] for r in range(3)]
        projs = [sum((p[d] - c[d]) * ax[d] for d in range(3)) for p in pts]
        spans.append(max(projs) - min(projs))
    return tuple(sorted(spans, reverse=True))  # type: ignore[return-value]


def bounding_volume_signature(points: Sequence[Point]) -> dict:
    """PCA-aligned bounding-volume signature (rotation/translation invariant).

    Returns a dict of orientation-independent shape ratios derived from the
    principal-axis extents ``(e1 >= e2 >= e3)``:

      * ``extents``      -- the three principal spans, scale-normalised by ``e1``.
      * ``elongation``   -- ``e2 / e1`` (0 = needle-like, 1 = not elongated).
      * ``flatness``     -- ``e3 / e2`` (0 = flat/planar, 1 = not flat).
      * ``compactness``  -- ``(4/3 pi r^3) / bbox_volume`` where ``r`` is the mean
        distance from the centroid: how sphere-like the mass distribution is.
      * ``anisotropy``   -- ``(e1 - e3) / e1``.

    All fields are dimensionless and invariant to rigid motion.
    """
    pts = _as_points(points)
    e1, e2, e3 = pca_extents(pts)
    eps = 1e-12
    d1 = e1 if e1 > eps else eps
    d2 = e2 if e2 > eps else eps
    if pts:
        c = _centroid(pts)
        r_mean = sum(_dist(p, c) for p in pts) / len(pts)
    else:
        r_mean = 0.0
    bbox_vol = e1 * e2 * e3
    sphere_vol = (4.0 / 3.0) * math.pi * (r_mean ** 3)
    compactness = sphere_vol / bbox_vol if bbox_vol > eps else 0.0
    return {
        "extents": (e1 / d1, e2 / d1, e3 / d1),
        "elongation": e2 / d1,
        "flatness": e3 / d2,
        "compactness": compactness,
        "anisotropy": (e1 - e3) / d1,
    }


def descriptor_vector(points: Sequence[Point], *, d2_bins: int = 16,
                      shell_bins: int = 16, samples: int = 1024,
                      seed: int = 0) -> List[float]:
    """Concatenated rotation-invariant retrieval descriptor.

    Stacks the D2 shape distribution, the radial-shell signature, and the scalar
    bounding-volume ratios into one fixed-length feature vector suitable as a
    hand-crafted baseline embedding for the retrieval protocol in
    :mod:`bench.geomretr_eval`.
    """
    d2 = d2_shape_distribution(points, bins=d2_bins, samples=samples, seed=seed)
    shells = radial_shell_signature(points, bins=shell_bins)
    bv = bounding_volume_signature(points)
    scalars = [bv["elongation"], bv["flatness"], bv["compactness"], bv["anisotropy"]]
    return list(d2) + list(shells) + scalars
