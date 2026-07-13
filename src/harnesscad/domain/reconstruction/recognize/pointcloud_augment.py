"""Seeded point-cloud augmentation pipeline for self-supervised retrieval pairs.

Van den Herrewegen et al., *Fine-Tuning 3D Foundation Models for Geometric
Object Retrieval* (2024), Section 3.3.3, generate VICReg positive pairs by
applying randomised augmentations *twice* to one normalised object. The two
resulting clouds share a label of "similar". The exact stochastic transforms
listed in the paper (point cloud on the unit sphere) are:

1. subsampling (e.g. 16000 -> 8192 points),
2. rotation around all axes or a single axis (dataset-dependent alignment),
3. anisotropic scaling with per-axis factors ``t ~ U(0.7, 1.25)``,
4. isotropic scaling with a single factor ``s ~ U(0.7, 1.25)``,
5. jittering each point with ``d ~ N(0, 0.005)`` clamped to ``|d| <= 0.05``,
6. global translation with ``v ~ U(-0.07, 0.07)``.

The learned VICReg encoder is external; this augmentation pipeline is pure,
deterministic given ``random.Random(seed)``, and stdlib-only. It feeds both the
VICReg loss surrogate in :mod:`bench.geomretr_losses` and any descriptor-based
similarity experiment.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

Point = Sequence[float]
Cloud = List[Tuple[float, float, float]]


def _as_cloud(points: Sequence[Point]) -> Cloud:
    out = []
    for p in points:
        vals = tuple(float(x) for x in p)
        if len(vals) != 3:
            raise ValueError("each point must have exactly 3 coordinates")
        out.append(vals)
    return out


def normalize_unit_sphere(points: Sequence[Point]) -> Cloud:
    """Centre at the centroid and scale so the farthest point is at radius 1."""
    pts = _as_cloud(points)
    if not pts:
        return []
    n = len(pts)
    c = tuple(sum(p[d] for p in pts) / n for d in range(3))
    centred = [(p[0] - c[0], p[1] - c[1], p[2] - c[2]) for p in pts]
    r = max(math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in centred)
    if r <= 0.0:
        return centred
    return [(p[0] / r, p[1] / r, p[2] / r) for p in centred]


def subsample(points: Sequence[Point], k: int, rng: random.Random) -> Cloud:
    """Random subsample to ``k`` points without replacement (order preserved)."""
    pts = _as_cloud(points)
    if k >= len(pts):
        return list(pts)
    if k < 0:
        raise ValueError("k must be non-negative")
    idx = sorted(rng.sample(range(len(pts)), k))
    return [pts[i] for i in idx]


def _rotation_matrix(ax: float, ay: float, az: float) -> List[List[float]]:
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    # Rz @ Ry @ Rx
    return [
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy, sx * cy, cx * cy],
    ]


def rotate(points: Sequence[Point], rng: random.Random, *,
           single_axis: bool = False) -> Cloud:
    """Rotate by uniform random angles. If ``single_axis`` only about z (gravity)."""
    pts = _as_cloud(points)
    az = rng.uniform(-math.pi, math.pi)
    if single_axis:
        ax = ay = 0.0
    else:
        ax = rng.uniform(-math.pi, math.pi)
        ay = rng.uniform(-math.pi, math.pi)
    r = _rotation_matrix(ax, ay, az)
    return [tuple(sum(r[a][b] * p[b] for b in range(3)) for a in range(3)) for p in pts]  # type: ignore[misc]


def anisotropic_scale(points: Sequence[Point], rng: random.Random, *,
                      lo: float = 0.7, hi: float = 1.25) -> Cloud:
    """Scale each axis independently by ``U(lo, hi)`` (paper item 3)."""
    pts = _as_cloud(points)
    s = (rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi))
    return [(p[0] * s[0], p[1] * s[1], p[2] * s[2]) for p in pts]


def isotropic_scale(points: Sequence[Point], rng: random.Random, *,
                    lo: float = 0.7, hi: float = 1.25) -> Cloud:
    """Scale all axes by one shared factor ``U(lo, hi)`` (paper item 4)."""
    pts = _as_cloud(points)
    s = rng.uniform(lo, hi)
    return [(p[0] * s, p[1] * s, p[2] * s) for p in pts]


def jitter(points: Sequence[Point], rng: random.Random, *,
           sigma: float = 0.005, clip: float = 0.05) -> Cloud:
    """Add Gaussian per-coordinate jitter ``N(0, sigma)`` clamped to ``+/-clip``."""
    pts = _as_cloud(points)
    def _j(x: float) -> float:
        d = rng.gauss(0.0, sigma)
        d = max(-clip, min(clip, d))
        return x + d
    return [(_j(p[0]), _j(p[1]), _j(p[2])) for p in pts]


def translate(points: Sequence[Point], rng: random.Random, *,
              lo: float = -0.07, hi: float = 0.07) -> Cloud:
    """Add a single global displacement vector with each component ``U(lo, hi)``."""
    pts = _as_cloud(points)
    v = (rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(lo, hi))
    return [(p[0] + v[0], p[1] + v[1], p[2] + v[2]) for p in pts]


def augment(points: Sequence[Point], rng: random.Random, *,
            subsample_to: int = None, single_axis: bool = False,
            rotate_enabled: bool = True) -> Cloud:
    """Apply the paper's full augmentation chain once, in listed order.

    Assumes ``points`` is already unit-sphere normalised. ``rotate_enabled=False``
    reproduces the paper's choice to skip rotation on canonically aligned
    datasets (e.g. ModelNet40). Deterministic given ``rng``.
    """
    cloud = _as_cloud(points)
    if subsample_to is not None:
        cloud = subsample(cloud, subsample_to, rng)
    if rotate_enabled:
        cloud = rotate(cloud, rng, single_axis=single_axis)
    cloud = anisotropic_scale(cloud, rng)
    cloud = isotropic_scale(cloud, rng)
    cloud = jitter(cloud, rng)
    cloud = translate(cloud, rng)
    return cloud


def positive_pair(points: Sequence[Point], seed: int, *,
                  subsample_to: int = None, single_axis: bool = False,
                  rotate_enabled: bool = True) -> Tuple[Cloud, Cloud]:
    """Two independently augmented views of one object -> a VICReg positive pair.

    Normalises to the unit sphere, then draws two augmentations from one seeded
    RNG stream. The two views encode the same object but differ, exactly the
    similar-pair construction of Section 3.3.3. Fully reproducible from ``seed``.
    """
    base = normalize_unit_sphere(points)
    rng = random.Random(seed)
    view_a = augment(base, rng, subsample_to=subsample_to,
                     single_axis=single_axis, rotate_enabled=rotate_enabled)
    view_b = augment(base, rng, subsample_to=subsample_to,
                     single_axis=single_axis, rotate_enabled=rotate_enabled)
    return view_a, view_b
