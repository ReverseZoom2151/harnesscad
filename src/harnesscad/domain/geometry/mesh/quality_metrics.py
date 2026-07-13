"""Block- and mesh-quality metrics for RL block decomposition.

From *Reinforcement Learning for Block Decomposition of CAD Models* (DiPrete
et al., AAAI-2022). The reward the agent receives is driven by the *quality of
the resulting parts* -- "reduced complexity, low aspect ratio" (Sec.
"Methodology") -- and the decomposed blocks are ultimately meshed by mapping a
regular mesh of a unit cube onto each block, so classical structured-mesh
quality measures apply. This module collects the deterministic quality metrics
that underlie the paper's reward and its valid-decomposition test.

Metrics provided:

  * ``aspect_ratio`` -- longest bounding-box side / shortest (the paper's
    ``R_i``; minimum 1 for a square, Sec. "Reward Function");
  * ``scaled_jacobian`` -- the standard structured-mesh corner quality (min over
    corners of the normalised edge cross product); 1 for a rectangle, 0 for a
    degenerate/collapsed corner, negative if inverted;
  * ``orthogonality`` -- worst corner-angle deviation from 90 degrees;
  * ``area_variance_ratio`` -- normalised spread of part areas (0 when equal),
    the quantity the reward penalises;
  * ``quad_fraction`` -- N_q / N, fraction of parts that are quads;
  * ``is_valid_decomposition`` -- parts are non-overlapping, cover the domain
    exactly, and each is connected;
  * ``all_quads`` -- the episode-termination test (every part is a rectangle).

Pure stdlib; deterministic (no randomness, no wall clock).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from harnesscad.domain.geometry.mesh.block_domain import Shape, Vec2

_EPS = 1e-9


def aspect_ratio(shape: Shape) -> float:
    """Longest bounding-box side / shortest side (>= 1)."""
    return shape.aspect_ratio()


def quad_corners(shape: Shape) -> Tuple[Vec2, Vec2, Vec2, Vec2]:
    """The four bounding-box corners CCW (for a rectangular block)."""
    x0, y0, x1, y1 = shape.bbox()
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def scaled_jacobian(corners: Sequence[Vec2]) -> float:
    """Scaled Jacobian of a quad from its 4 CCW corner points.

    Minimum over the four corners of the normalised cross product of the two
    incident edge vectors. Equals 1 for a rectangle/square, decreases toward 0
    for skewed or collapsed corners, and is negative for an inverted corner.
    """
    if len(corners) != 4:
        raise ValueError("scaled_jacobian expects exactly 4 corners")
    worst = float("inf")
    for k in range(4):
        p = corners[k]
        a = corners[(k - 1) % 4]
        b = corners[(k + 1) % 4]
        e1 = (b[0] - p[0], b[1] - p[1])
        e2 = (a[0] - p[0], a[1] - p[1])
        n1 = math.hypot(*e1)
        n2 = math.hypot(*e2)
        if n1 < _EPS or n2 < _EPS:
            return 0.0
        j = (e1[0] * e2[1] - e1[1] * e2[0]) / (n1 * n2)
        worst = min(worst, j)
    return worst


def block_scaled_jacobian(shape: Shape) -> float:
    """Scaled Jacobian of a rectangular block (its bounding-box corners)."""
    return scaled_jacobian(quad_corners(shape))


def orthogonality(corners: Sequence[Vec2]) -> float:
    """Worst deviation (degrees) of a quad's corner angles from 90."""
    if len(corners) != 4:
        raise ValueError("orthogonality expects exactly 4 corners")
    worst = 0.0
    for k in range(4):
        p = corners[k]
        a = corners[(k - 1) % 4]
        b = corners[(k + 1) % 4]
        e1 = (a[0] - p[0], a[1] - p[1])
        e2 = (b[0] - p[0], b[1] - p[1])
        n1 = math.hypot(*e1)
        n2 = math.hypot(*e2)
        if n1 < _EPS or n2 < _EPS:
            return 90.0
        dot = max(-1.0, min(1.0, (e1[0] * e2[0] + e1[1] * e2[1]) / (n1 * n2)))
        ang = math.degrees(math.acos(dot))
        worst = max(worst, abs(ang - 90.0))
    return worst


def area_variance_ratio(parts: Sequence[Shape]) -> float:
    """Normalised area spread: sqrt(sum (A_i - Abar)^2) / sum A_i, >= 0.

    Zero when all parts have equal area (the reward's ideal), matching the
    variance term of the paper's reward (Eq. 1). Returns 0 for a single part.
    """
    areas = [p.area() for p in parts]
    if not areas:
        return 0.0
    total = sum(areas)
    if total < _EPS:
        return 0.0
    mean = total / len(areas)
    ss = sum((a - mean) ** 2 for a in areas)
    return math.sqrt(ss) / total


def quad_fraction(parts: Sequence[Shape]) -> float:
    """Fraction N_q / N of parts that are quadrilateral (rectangular) blocks."""
    if not parts:
        return 0.0
    nq = sum(1 for p in parts if p.is_quad())
    return nq / len(parts)


def all_quads(parts: Sequence[Shape]) -> bool:
    """Episode-termination test: every part is a rectangular block."""
    return bool(parts) and all(p.is_quad() for p in parts)


def is_valid_decomposition(parts: Sequence[Shape], domain: Shape) -> bool:
    """True iff ``parts`` partition ``domain``: disjoint, covering, connected."""
    if not parts:
        return False
    for p in parts:
        if p.xs != domain.xs or p.ys != domain.ys:
            return False
        if not p.cells or not p.is_connected():
            return False
    seen = set()
    for p in parts:
        for c in p.cells:
            if c in seen:
                return False  # overlap
            seen.add(c)
    return seen == set(domain.cells)


def mean_aspect_ratio(parts: Sequence[Shape]) -> float:
    """Mean aspect ratio across parts (lower is better)."""
    if not parts:
        return 0.0
    return sum(p.aspect_ratio() for p in parts) / len(parts)
