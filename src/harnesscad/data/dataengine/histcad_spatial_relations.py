"""HistCAD inter-part spatial-relation analysis (AMHistCAD, Algorithm 2).

The HistCAD annotation module derives, for every part, a 3D oriented bounding
box (OBB) and then classifies pairwise spatial relations between parts using
the Separating Axis Theorem (SAT), plus directional labels from centroid
offsets. This module implements that deterministic geometry (no LLM):

  * :class:`OBB` — centre, half-extents, and an orthonormal axis frame;
  * :func:`sat_overlap` — SAT test for two OBBs (returns overlap + per-axis gap);
  * :func:`classify_contact` — separate / touch / intersect / contain /
    contained (Algorithm 2 rel_type);
  * :func:`relative_position_labels` — left/right, below/above, back/front
    from centroid offset (Algorithm 2 rel_pos);
  * :func:`analyze_parts` — full pairwise relation table over a list of OBBs.

Stdlib-only, deterministic. Vectors are plain 3-tuples of floats.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Vec = Tuple[float, float, float]

_EPS = 1e-9


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _normalize(a: Vec) -> Vec:
    n = math.sqrt(_dot(a, a))
    if n < _EPS:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


@dataclass(frozen=True)
class OBB:
    """Oriented bounding box: centre, half-extents, orthonormal axes."""

    center: Vec
    half: Vec  # half-extents along each local axis
    axes: Tuple[Vec, Vec, Vec] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    @staticmethod
    def from_aabb(minc: Vec, maxc: Vec) -> "OBB":
        center = tuple((minc[i] + maxc[i]) / 2.0 for i in range(3))
        half = tuple((maxc[i] - minc[i]) / 2.0 for i in range(3))
        return OBB(center, half)  # axis-aligned frame

    def corners(self) -> List[Vec]:
        out: List[Vec] = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    p = tuple(
                        self.center[i]
                        + sx * self.half[0] * self.axes[0][i]
                        + sy * self.half[1] * self.axes[1][i]
                        + sz * self.half[2] * self.axes[2][i]
                        for i in range(3)
                    )
                    out.append(p)
        return out


def _project_radius(o: OBB, axis: Vec) -> float:
    return (abs(_dot(o.axes[0], axis)) * o.half[0]
            + abs(_dot(o.axes[1], axis)) * o.half[1]
            + abs(_dot(o.axes[2], axis)) * o.half[2])


def sat_overlap(a: OBB, b: OBB, tol: float = 1e-9):
    """SAT test for two OBBs.

    Returns ``(collides, min_gap, separating_axes)`` where ``min_gap`` is the
    largest positive separation over all 15 candidate axes (<= 0 means overlap)
    and ``separating_axes`` lists axes with gap ~= 0 (touching contact).
    """
    t = _sub(b.center, a.center)
    candidates: List[Vec] = list(a.axes) + list(b.axes)
    for i in range(3):
        for j in range(3):
            c = (
                a.axes[i][1] * b.axes[j][2] - a.axes[i][2] * b.axes[j][1],
                a.axes[i][2] * b.axes[j][0] - a.axes[i][0] * b.axes[j][2],
                a.axes[i][0] * b.axes[j][1] - a.axes[i][1] * b.axes[j][0],
            )
            if _dot(c, c) > _EPS:
                candidates.append(_normalize(c))

    max_gap = -math.inf
    touching: List[Vec] = []
    collides = True
    for axis in candidates:
        if _dot(axis, axis) < _EPS:
            continue
        dist = abs(_dot(t, axis))
        ra = _project_radius(a, axis)
        rb = _project_radius(b, axis)
        gap = dist - (ra + rb)
        if gap > max_gap:
            max_gap = gap
        if gap > tol:
            collides = False
        elif abs(gap) <= tol:
            touching.append(axis)
    return collides, max_gap, tuple(touching)


def _contains(a: OBB, b: OBB, tol: float = 1e-9) -> bool:
    """True if OBB ``b`` is fully inside OBB ``a``."""
    for corner in b.corners():
        local = _sub(corner, a.center)
        for k in range(3):
            if abs(_dot(local, a.axes[k])) > a.half[k] + tol:
                return False
    return True


def classify_contact(a: OBB, b: OBB, tol: float = 1e-9) -> str:
    """Classify the contact type between two parts (Algorithm 2 rel_type)."""
    collides, gap, touching = sat_overlap(a, b, tol)
    if not collides:
        return "separate"
    if _contains(a, b, tol):
        return "contain"      # a contains b
    if _contains(b, a, tol):
        return "contained"    # a is contained in b
    if touching and abs(gap) <= tol:
        return "touch"
    return "intersect"


def relative_position_labels(a: OBB, b: OBB, tol: float = 1e-9) -> Tuple[str, ...]:
    """Directional labels of ``b`` relative to ``a`` from centroid offset.

    Axes: x -> left/right, y -> below/above, z -> back/front. Offsets within
    ``tol`` produce no label on that axis (aligned).
    """
    off = _sub(b.center, a.center)
    labels: List[str] = []
    if off[0] > tol:
        labels.append("right")
    elif off[0] < -tol:
        labels.append("left")
    if off[1] > tol:
        labels.append("above")
    elif off[1] < -tol:
        labels.append("below")
    if off[2] > tol:
        labels.append("front")
    elif off[2] < -tol:
        labels.append("back")
    return tuple(labels)


@dataclass(frozen=True)
class PartRelation:
    i: int
    j: int
    rel_type: str
    rel_pos: Tuple[str, ...]


def analyze_parts(obbs: Sequence[OBB], tol: float = 1e-9) -> List[PartRelation]:
    """Pairwise spatial relations over parts (Algorithm 2), ordered i<j."""
    rels: List[PartRelation] = []
    n = len(obbs)
    for i in range(n):
        for j in range(i + 1, n):
            rt = classify_contact(obbs[i], obbs[j], tol)
            rp = relative_position_labels(obbs[i], obbs[j], tol)
            rels.append(PartRelation(i, j, rt, rp))
    return rels
