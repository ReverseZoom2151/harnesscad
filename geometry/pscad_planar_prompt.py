"""PS-CAD planar prompting: candidate-plane detection and prompt encoding.

Implements the deterministic "PlaneDetection" step of PS-CAD (Yang et al. 2024,
Sec. 4).  Extrusion cylinders have a planar bottom and top surface, so the paper
detects a set of candidate planes from a point cloud with an off-the-shelf
RANSAC method and encodes each as a *planar prompt* by "randomly sampling 64
inlier points from a plane detected by RANSAC".  These planes are the potential
cross-sections from which the next CAD modelling step (a sketch-extrude) can be
started.

This module provides:

  * a seeded RANSAC plane detector (deterministic via ``random.Random(seed)``);
  * the local surface descriptor at a difference region -- surface type
    (``"plane"``) plus fitted normal and offset;
  * planar-prompt encoding (a fixed-size inlier sample, as in the paper), and
  * the prompt boundary curve: the 2D convex-hull polygon of the inliers in the
    plane's local coordinate frame (the "cross-section" a sketch would lie on).

The learned single-step reconstruction network is out of scope.  Everything
here is closed-form geometry, stdlib-only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import dist, sqrt


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return sqrt(_dot(a, a))


def fit_plane(p0, p1, p2):
    """Fit the plane through three points as ``(unit_normal, offset)``.

    The plane is the set ``{x : normal . x == offset}``.  Returns ``None`` when
    the three points are collinear (degenerate normal).
    """
    normal = _cross(_sub(p1, p0), _sub(p2, p0))
    length = _norm(normal)
    if length == 0.0:
        return None
    unit = tuple(c / length for c in normal)
    # Canonical sign: first non-zero component positive, so the descriptor is
    # deterministic regardless of point ordering.
    for c in unit:
        if abs(c) > 1e-12:
            if c < 0:
                unit = tuple(-x for x in unit)
            break
    return unit, _dot(unit, p0)


def point_plane_distance(point, plane):
    """Absolute distance from ``point`` to ``plane = (normal, offset)``."""
    normal, offset = plane
    return abs(_dot(normal, point) - offset)


@dataclass(frozen=True)
class DetectedPlane:
    """A RANSAC-detected candidate plane and its inliers."""

    normal: tuple
    offset: float
    inlier_indices: tuple

    @property
    def support(self):
        return len(self.inlier_indices)


def ransac_planes(points, *, threshold, min_inliers, iterations=200, seed=0,
                  max_planes=8):
    """Greedy multi-plane RANSAC (deterministic).

    ``threshold`` is ``t`` in the paper -- the point-to-plane distance below
    which a point is an inlier.  ``min_inliers`` is ``d`` -- the number of
    inliers required to accept a plane.  Detected planes are removed greedily so
    that later planes cover the remaining geometry.  Returns planes sorted by
    descending support (ties broken by the canonical plane descriptor).
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if min_inliers < 3:
        raise ValueError("min_inliers must be at least 3")
    pts = [tuple(p) for p in points]
    remaining = set(range(len(pts)))
    rng = random.Random(seed)
    planes = []
    while len(planes) < max_planes and len(remaining) >= min_inliers:
        pool = sorted(remaining)
        best_plane = None
        best_inliers = ()
        for _ in range(iterations):
            trio = rng.sample(pool, 3)
            plane = fit_plane(pts[trio[0]], pts[trio[1]], pts[trio[2]])
            if plane is None:
                continue
            inliers = tuple(i for i in pool
                            if point_plane_distance(pts[i], plane) <= threshold)
            if len(inliers) > len(best_inliers):
                best_plane, best_inliers = plane, inliers
        if best_plane is None or len(best_inliers) < min_inliers:
            break
        normal, offset = best_plane
        planes.append(DetectedPlane(normal, float(offset), best_inliers))
        remaining -= set(best_inliers)
    planes.sort(key=lambda p: (-p.support, p.normal, p.offset))
    return tuple(planes)


def _plane_basis(normal):
    """A deterministic orthonormal in-plane basis ``(u, v)`` for ``normal``."""
    seed_axis = (1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = _cross(normal, seed_axis)
    lu = _norm(u)
    u = tuple(c / lu for c in u)
    v = _cross(normal, u)
    lv = _norm(v)
    v = tuple(c / lv for c in v)
    return u, v


def project_to_plane(point, plane, origin):
    """2D coordinates of ``point`` in the local frame of ``plane`` at ``origin``."""
    normal, _ = plane
    u, v = _plane_basis(normal)
    rel = _sub(point, origin)
    return (_dot(rel, u), _dot(rel, v))


def _convex_hull_2d(points):
    """Andrew's monotone-chain convex hull of 2D points (CCW, no repeat)."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return tuple(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return tuple(lower[:-1] + upper[:-1])


@dataclass(frozen=True)
class PlanarPrompt:
    """A PS-CAD planar prompt: sampled inliers plus the local surface descriptor.

    ``surface_type`` is always ``"plane"`` (the candidate cross-section for the
    next extrusion).  ``boundary`` is the 2D convex-hull polygon of the inliers
    in the plane's local frame -- the sketch region an extrusion would start on.
    """

    surface_type: str
    normal: tuple
    offset: float
    origin: tuple
    sample: tuple        # sampled 3D inlier points (the prompt point cloud)
    boundary: tuple      # 2D hull polygon in the local plane frame

    @property
    def sample_size(self):
        return len(self.sample)


def encode_planar_prompt(points, plane, *, sample_count=64, seed=0):
    """Encode a :class:`DetectedPlane` as a planar prompt.

    Randomly samples ``sample_count`` inlier points (with replacement only when
    there are too few, matching the fixed-size prompt in the paper) and computes
    the surface descriptor and 2D boundary curve.  Sampling is seeded and
    therefore deterministic.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    pts = [tuple(p) for p in points]
    inliers = [pts[i] for i in plane.inlier_indices]
    if not inliers:
        raise ValueError("plane has no inliers to sample")
    rng = random.Random(seed)
    if len(inliers) >= sample_count:
        idx = sorted(rng.sample(range(len(inliers)), sample_count))
        sample = tuple(inliers[i] for i in idx)
    else:
        sample = tuple(rng.choice(inliers) for _ in range(sample_count))
    plane_tuple = (plane.normal, plane.offset)
    origin = inliers[0]
    projected = [project_to_plane(p, plane_tuple, origin) for p in inliers]
    boundary = _convex_hull_2d(projected)
    return PlanarPrompt("plane", plane.normal, plane.offset, origin,
                        sample, boundary)


def extract_prompts(points, *, threshold, min_inliers, iterations=200, seed=0,
                    max_planes=8, sample_count=64):
    """Full PS-CAD prompt extraction: detect planes then encode each as a prompt.

    Returns a tuple of :class:`PlanarPrompt` -- the candidate ``pr_i`` that the
    single-step reconstruction module would be run on, one per detected plane.
    """
    planes = ransac_planes(points, threshold=threshold, min_inliers=min_inliers,
                           iterations=iterations, seed=seed, max_planes=max_planes)
    return tuple(
        encode_planar_prompt(points, plane, sample_count=sample_count,
                             seed=seed + rank)
        for rank, plane in enumerate(planes)
    )
