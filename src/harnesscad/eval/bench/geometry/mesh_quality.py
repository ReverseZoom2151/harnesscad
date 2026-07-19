"""Artist-like mesh quality metrics.

The premise is that artist-created meshes differ from iso-surfaced ones by
their *topology and tessellation*: well-structured faces, controlled face
counts, and regular edge flow rather than overly dense, bumpy tessellation.
This module provides deterministic geometric metrics for judging those
qualities on a triangle mesh:

  * **triangle / face counts** and quad-face share (common density knobs an
    artist-mesh generator conditions on),
  * **face-area distribution** (high-face-count meshes have smaller,
    more uniform faces),
  * **triangle aspect ratios** -- both the longest/shortest edge ratio and the
    circumradius-to-inradius radius ratio (1.0 for an equilateral triangle;
    large values flag slivers),
  * **vertex-valence regularity** -- artist meshes favour regular valence-6
    interior vertices,
  * generic **distribution summaries** (min/mean/median/std/percentiles) so the
    above can be reported as histograms.

Vertices are 3D ``(x, y, z)`` tuples. Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Vertex = Tuple[float, float, float]
Face = Sequence[int]


# --------------------------------------------------------------------------- #
# Counts
# --------------------------------------------------------------------------- #
def face_count(faces: Sequence[Face]) -> int:
    """Total number of faces."""
    return len(faces)


def triangle_count(faces: Sequence[Face]) -> int:
    """Number of triangular (3-vertex) faces."""
    return sum(1 for f in faces if len(f) == 3)


def quad_count(faces: Sequence[Face]) -> int:
    """Number of quadrilateral (4-vertex) faces."""
    return sum(1 for f in faces if len(f) == 4)


def quad_ratio(faces: Sequence[Face]) -> float:
    """Fraction of faces that are quads (a quad-dominance control)."""
    if not faces:
        return 0.0
    return quad_count(faces) / len(faces)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _sub(a: Sequence[float], b: Sequence[float]) -> Vertex:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vertex, b: Vertex) -> Vertex:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def triangle_area(a: Vertex, b: Vertex, c: Vertex) -> float:
    """Area of triangle ``a-b-c`` via half the cross-product magnitude."""
    return 0.5 * _norm(_cross(_sub(b, a), _sub(c, a)))


def face_area(vertices: Sequence[Vertex], face: Face) -> float:
    """Area of a face (fan-triangulated for polygons with >3 vertices)."""
    if len(face) < 3:
        raise ValueError("a face needs at least 3 vertices")
    total = 0.0
    v0 = vertices[face[0]]
    for i in range(1, len(face) - 1):
        total += triangle_area(v0, vertices[face[i]], vertices[face[i + 1]])
    return total


def face_areas(vertices: Sequence[Vertex], faces: Sequence[Face]) -> List[float]:
    """Area of every face."""
    return [face_area(vertices, f) for f in faces]


# --------------------------------------------------------------------------- #
# Triangle aspect ratios
# --------------------------------------------------------------------------- #
def _edge_lengths(a: Vertex, b: Vertex, c: Vertex) -> Tuple[float, float, float]:
    return (_norm(_sub(b, a)), _norm(_sub(c, b)), _norm(_sub(a, c)))


def edge_ratio(a: Vertex, b: Vertex, c: Vertex) -> float:
    """Longest-edge / shortest-edge ratio (>=1; 1 for equilateral).

    ``inf`` for a degenerate triangle with a zero-length edge.
    """
    lengths = _edge_lengths(a, b, c)
    lo = min(lengths)
    if lo == 0.0:
        return math.inf
    return max(lengths) / lo


def radius_ratio(a: Vertex, b: Vertex, c: Vertex) -> float:
    """Circumradius / (2 * inradius) aspect ratio (>=1; 1 for equilateral).

    ``inf`` for a degenerate (zero-area) triangle.
    """
    la, lb, lc = _edge_lengths(a, b, c)
    area = triangle_area(a, b, c)
    if area == 0.0:
        return math.inf
    perimeter = la + lb + lc
    inradius = 2.0 * area / perimeter
    circumradius = (la * lb * lc) / (4.0 * area)
    return circumradius / (2.0 * inradius)


def triangle_aspect_ratios(
    vertices: Sequence[Vertex], faces: Sequence[Face], metric: str = "radius"
) -> List[float]:
    """Aspect ratio of each triangular face (non-triangles skipped).

    ``metric`` is ``"radius"`` (circumradius/inradius) or ``"edge"``
    (longest/shortest edge).
    """
    if metric not in ("radius", "edge"):
        raise ValueError("metric must be 'radius' or 'edge'")
    fn = radius_ratio if metric == "radius" else edge_ratio
    out: List[float] = []
    for face in faces:
        if len(face) != 3:
            continue
        a, b, c = (vertices[i] for i in face)
        out.append(fn(a, b, c))
    return out


# --------------------------------------------------------------------------- #
# Vertex valence regularity
# --------------------------------------------------------------------------- #
def vertex_valences(
    vertex_count: int, faces: Sequence[Face]
) -> List[int]:
    """Number of distinct neighbours (edge degree) of each vertex index."""
    neighbours: List[set] = [set() for _ in range(vertex_count)]
    for face in faces:
        n = len(face)
        for i in range(n):
            a = face[i]
            b = face[(i + 1) % n]
            neighbours[a].add(b)
            neighbours[b].add(a)
    return [len(s) for s in neighbours]


def valence_regularity(
    vertex_count: int, faces: Sequence[Face], ideal: int = 6
) -> float:
    """Fraction of vertices with the ideal valence (default 6 for triangles)."""
    if vertex_count == 0:
        return 0.0
    valences = vertex_valences(vertex_count, faces)
    return sum(1 for v in valences if v == ideal) / vertex_count


# --------------------------------------------------------------------------- #
# Distribution summary
# --------------------------------------------------------------------------- #
def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile ``q`` in [0, 1] of a sorted list."""
    if not sorted_vals:
        raise ValueError("empty sequence")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def distribution_summary(values: Sequence[float]) -> Dict[str, float]:
    """Return count/min/max/mean/median/std/p25/p75 of ``values``.

    Infinite values (degenerate triangles) are excluded from the moment
    statistics but counted in ``n_infinite``.
    """
    finite = [v for v in values if math.isfinite(v)]
    n_inf = len(values) - len(finite)
    if not finite:
        return {"count": float(len(values)), "n_infinite": float(n_inf)}
    ordered = sorted(finite)
    n = len(ordered)
    mean = sum(ordered) / n
    var = sum((v - mean) ** 2 for v in ordered) / n
    return {
        "count": float(len(values)),
        "n_infinite": float(n_inf),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": mean,
        "median": _percentile(ordered, 0.5),
        "std": math.sqrt(var),
        "p25": _percentile(ordered, 0.25),
        "p75": _percentile(ordered, 0.75),
    }


def histogram(
    values: Sequence[float], bins: int
) -> Tuple[List[int], List[float]]:
    """Equal-width histogram over the finite values.

    Returns ``(counts, edges)`` with ``bins`` counts and ``bins + 1`` edges.
    """
    if bins <= 0:
        raise ValueError("bins must be positive")
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [0] * bins, []
    lo, hi = min(finite), max(finite)
    if hi == lo:
        counts = [0] * bins
        counts[0] = len(finite)
        return counts, [lo + i * 0.0 for i in range(bins + 1)]
    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for v in finite:
        idx = int((v - lo) / width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    return counts, edges


def mesh_quality_report(
    vertices: Sequence[Vertex], faces: Sequence[Face]
) -> Dict[str, object]:
    """One-shot artist-like quality report combining every metric above."""
    areas = face_areas(vertices, faces)
    aspects = triangle_aspect_ratios(vertices, faces, metric="radius")
    return {
        "face_count": face_count(faces),
        "triangle_count": triangle_count(faces),
        "quad_count": quad_count(faces),
        "quad_ratio": quad_ratio(faces),
        "valence_regularity": valence_regularity(len(vertices), faces),
        "face_area": distribution_summary(areas),
        "aspect_ratio": distribution_summary(aspects),
    }
