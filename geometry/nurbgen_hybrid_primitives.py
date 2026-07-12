"""Hybrid analytic primitives and Chamfer-distance fidelity fallback (NURBGen).

Usama, Khan, Stricker & Afzal, *NURBGen: High-Fidelity Text-to-CAD Generation
through LLM-Driven NURBS Modeling* (AAAI 2026), Sec. "CAD Representation",
Eq. 4, and Appendix "CAD Representation".

NURBGen's key robustness idea is a *hybrid* representation: ~70% of faces are
kept as untrimmed NURBS, but thin/hole-adjacent faces where NURBS fitting
produces artifacts fall back to **analytic primitives** (lines, circles/arcs,
ellipses).  The fallback decision is a fidelity gate (paper Eq. 4):

    CD(f_n, f_gt) <= epsilon      (epsilon = 6e-4)

where ``f_n`` is the reconstructed (NURBS) surface's sampled point cloud and
``f_gt`` the ground-truth face's sampled point cloud, and CD is the Chamfer
distance ("average squared distance from points in one set to their nearest
neighbours in another").  If the NURBS reconstruction is within ``epsilon`` it is
kept; otherwise the analytic primitive is retained.

This module implements:
  * deterministic samplers for the Appendix analytic primitives
    (:func:`sample_line`, :func:`sample_circle`, :func:`sample_ellipse`),
    parameterised exactly as the paper's JSON schema (center/normal/radius,
    first/last angles, etc.);
  * :func:`chamfer_distance` (Eq. 4) and its directional form;
  * :func:`accept_nurbs` / :func:`choose_representation` -- the hybrid gate.

Pure-Python stdlib, deterministic.  Points are 3-tuples of floats.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Point = Tuple[float, float, float]

# Paper's empirical Chamfer-distance threshold (Sec. CAD Representation).
EPSILON = 6e-4


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    return math.sqrt(_dot(a, a))


def _normalize(a):
    n = _norm(a)
    if n < 1e-14:
        raise ValueError("cannot normalize a zero-length vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def plane_basis(normal: Point) -> Tuple[Point, Point]:
    """Return two orthonormal in-plane axes (u, v) perpendicular to ``normal``.

    Deterministic: picks the world axis least aligned with ``normal`` as a seed,
    so ``(u, v, normal)`` forms a right-handed frame.
    """
    n = _normalize(normal)
    # Seed with the axis least parallel to n.
    ax, ay, az = abs(n[0]), abs(n[1]), abs(n[2])
    if ax <= ay and ax <= az:
        seed = (1.0, 0.0, 0.0)
    elif ay <= az:
        seed = (0.0, 1.0, 0.0)
    else:
        seed = (0.0, 0.0, 1.0)
    # Gram-Schmidt: project the seed onto the plane so axis-aligned normals
    # yield the conventional in-plane axes (e.g. normal +z -> u=+x, v=+y).
    proj = _sub(seed, _scale(n, _dot(seed, n)))
    u = _normalize(proj)
    v = _cross(n, u)
    return u, v


# ---------------------------------------------------------------------------
# Analytic primitive samplers (Appendix "CAD Representation")
# ---------------------------------------------------------------------------

def sample_line(start: Point, end: Point, samples: int = 16) -> List[Point]:
    """Sample the line segment ``start -> end`` into ``samples + 1`` points."""
    if samples < 1:
        raise ValueError("samples must be >= 1")
    d = _sub(end, start)
    return [_add(start, _scale(d, k / samples)) for k in range(samples + 1)]


def sample_circle(center: Point, normal: Point, radius: float,
                  first: float = 0.0, last: float = 2.0 * math.pi,
                  samples: int = 32) -> List[Point]:
    """Sample a circle/arc (Appendix: center, normal, radius, first/last angle).

    ``first`` and ``last`` are start/end angles in radians (``0, pi`` for a
    semicircle; ``0, 2*pi`` for a full circle).
    """
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    u, v = plane_basis(normal)
    pts: List[Point] = []
    for k in range(samples + 1):
        t = first + (last - first) * (k / samples)
        offset = _add(_scale(u, radius * math.cos(t)),
                      _scale(v, radius * math.sin(t)))
        pts.append(_add(center, offset))
    return pts


def sample_ellipse(center: Point, normal: Point, major_radius: float,
                   minor_radius: float, first: float = 0.0,
                   last: float = 2.0 * math.pi, samples: int = 32
                   ) -> List[Point]:
    """Sample an ellipse/arc (Appendix: center, normal, major/minor, first/last).

    The major axis lies along the first in-plane basis vector, the minor axis
    along the second.
    """
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if major_radius <= 0.0 or minor_radius <= 0.0:
        raise ValueError("radii must be positive")
    u, v = plane_basis(normal)
    pts: List[Point] = []
    for k in range(samples + 1):
        t = first + (last - first) * (k / samples)
        offset = _add(_scale(u, major_radius * math.cos(t)),
                      _scale(v, minor_radius * math.sin(t)))
        pts.append(_add(center, offset))
    return pts


def sample_primitive(spec: Dict, samples: int = 32) -> List[Point]:
    """Dispatch on ``spec['type']`` to the matching sampler (JSON schema).

    Supports ``line``, ``circle``, ``ellipse`` -- the analytic families NURBGen
    falls back to.  ``spec`` mirrors the paper's JSON fields.
    """
    kind = spec.get("type")
    if kind == "line":
        return sample_line(tuple(spec["start"]), tuple(spec["end"]), samples)
    if kind == "circle":
        return sample_circle(
            tuple(spec["center"]), tuple(spec["normal"]), spec["radius"],
            spec.get("first", 0.0), spec.get("last", 2.0 * math.pi), samples)
    if kind == "ellipse":
        return sample_ellipse(
            tuple(spec["center"]), tuple(spec["normal"]),
            spec["major_radius"], spec["minor_radius"],
            spec.get("first", 0.0), spec.get("last", 2.0 * math.pi), samples)
    raise ValueError("unknown primitive type %r" % kind)


# ---------------------------------------------------------------------------
# Chamfer distance (paper Eq. 4)
# ---------------------------------------------------------------------------

def _nearest_sq(p: Point, cloud: Sequence[Point]) -> float:
    best = float("inf")
    for q in cloud:
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        dz = p[2] - q[2]
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < best:
            best = d2
    return best


def directional_chamfer(a: Sequence[Point], b: Sequence[Point]) -> float:
    """Mean squared nearest-neighbour distance from ``a`` to ``b``."""
    if not a or not b:
        raise ValueError("both point sets must be non-empty")
    return sum(_nearest_sq(p, b) for p in a) / len(a)


def chamfer_distance(a: Sequence[Point], b: Sequence[Point]) -> float:
    """Symmetric Chamfer distance (paper Eq. 4).

    Average of the two directional mean-squared nearest-neighbour distances.
    Zero iff the two clouds coincide (as sets, at the sampling resolution).
    """
    return 0.5 * (directional_chamfer(a, b) + directional_chamfer(b, a))


# ---------------------------------------------------------------------------
# Hybrid representation gate (Eq. 4: CD <= epsilon -> keep NURBS)
# ---------------------------------------------------------------------------

def accept_nurbs(reconstructed: Sequence[Point], ground_truth: Sequence[Point],
                 epsilon: float = EPSILON) -> bool:
    """Return True if the NURBS reconstruction is within ``epsilon`` (Eq. 4).

    ``True`` means "keep the NURBS surface"; ``False`` means "fall back to the
    analytic primitive" -- exactly the paper's per-face decision.
    """
    return chamfer_distance(reconstructed, ground_truth) <= epsilon


def choose_representation(reconstructed: Sequence[Point],
                          ground_truth: Sequence[Point],
                          epsilon: float = EPSILON) -> str:
    """Return ``'nurbs'`` or ``'analytic'`` per the hybrid fidelity gate."""
    return "nurbs" if accept_nurbs(reconstructed, ground_truth, epsilon) \
        else "analytic"


def hybrid_stats(decisions: Sequence[str]) -> Dict[str, float]:
    """Summarise a batch of per-face decisions.

    Returns counts and the NURBS fraction -- the paper reports ~70% NURBS /
    ~30% analytic across a model (Sec. CAD Representation).
    """
    n = len(decisions)
    n_nurbs = sum(1 for d in decisions if d == "nurbs")
    n_analytic = sum(1 for d in decisions if d == "analytic")
    if n_nurbs + n_analytic != n:
        raise ValueError("decisions must be 'nurbs' or 'analytic'")
    return {
        "n": n,
        "n_nurbs": n_nurbs,
        "n_analytic": n_analytic,
        "nurbs_fraction": (n_nurbs / n) if n else 0.0,
    }
