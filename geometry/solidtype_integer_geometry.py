"""Fixed-point integer geometry with shared-grid vertex welding (SolidType).

SolidType (``packages/core/src/num/integer-geometry.ts``) attacks the single
most persistent source of failure in a B-Rep boolean kernel: two faces that
*share* an edge or vertex end up with coordinates that differ in the last few
floating-point bits, so the shared entity silently becomes two nearly-coincident
entities and the resulting shell is non-manifold.  SolidType's fix is not a
tolerance -- it is a *representation* change:

* every coordinate is stored as an **integer** count of a fixed sub-unit
  (nanometres; ``NANO_PER_MM = 1_000_000``).  Two points are equal *iff* their
  integers are equal -- there is no epsilon;
* an intersection point is computed **once** in floating point and immediately
  **snapped** to the integer grid; every face that touches that intersection is
  handed back *the same integers*, so the shared vertex is bit-exact for all of
  them ("compute once, snap once");
* the combinatorial decisions that drive the intersection (are two segments
  parallel? does the crossing lie inside both segments?) are done with **exact
  integer determinants** and a **division-free** in-range test, so they never
  round to the wrong answer.

This module reimplements that fixed-point substrate in stdlib Python.  It is
deliberately *different* from :mod:`numeric.manifold_predicates` (Shewchuk-style
exact *sign* predicates): those decide the sign of a determinant on
floating-point inputs, whereas this module quantises the inputs themselves so
that near-coincident vertices collapse to one canonical grid point.  The two are
complementary -- predicates classify, quantisation welds.

The distinctive higher-level payoff is :class:`VertexRegistry`: interning a
snapped grid point returns a canonical integer id, so every producer of the same
grid coordinate is given the same id and the shared-vertex problem disappears at
the topology layer.

Deterministic: pure integer arithmetic for the decisions, ``round`` (banker's
rounding is fine -- it is applied identically to every producer of a point), no
clock, no randomness.

Public API
----------
Constants: ``NANO_PER_MM``, ``NANO_PER_M``, ``MAX_COORD``.
Conversions: :func:`mm_to_nano`, :func:`nano_to_mm`, :func:`vec_to_int`,
:func:`vec_to_float`.
Integer vectors: :func:`add_i`, :func:`sub_i`, :func:`dot_i`, :func:`cross_i`,
:func:`equals_i`, :func:`length_squared_i`.
Exact intersections: :func:`segment_intersection_2i`,
:func:`line_line_closest_points_3i`, :func:`plane_plane_intersection`,
:func:`clip_line_to_polygon_3i`.
Welding: :class:`VertexRegistry`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "NANO_PER_MM",
    "NANO_PER_M",
    "MAX_COORD",
    "CoordRangeError",
    "mm_to_nano",
    "nano_to_mm",
    "vec_to_int",
    "vec_to_float",
    "add_i",
    "sub_i",
    "dot_i",
    "cross_i",
    "equals_i",
    "length_squared_i",
    "segment_intersection_2i",
    "line_line_closest_points_3i",
    "plane_plane_intersection",
    "clip_line_to_polygon_3i",
    "VertexRegistry",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Nanometres per millimetre -- the fixed-point sub-unit.
NANO_PER_MM = 1_000_000

#: Nanometres per metre.
NANO_PER_M = 1_000_000_000

#: Largest safe integer coordinate.  JavaScript uses ``Number.MAX_SAFE_INTEGER``
#: (2**53 - 1); Python integers are unbounded, but keeping the same guard lets
#: exact integer products (dot/cross, ~coord**2) stay well inside the range
#: where a downstream float round-trip is lossless, and flags absurd inputs.
MAX_COORD = 2 ** 53 - 1


class CoordRangeError(ValueError):
    """Raised when a quantised coordinate exceeds :data:`MAX_COORD`."""


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def _check_range(value: int) -> int:
    if value > MAX_COORD or value < -MAX_COORD:
        raise CoordRangeError(
            f"coordinate {value} exceeds MAX_COORD ({MAX_COORD})"
        )
    return value


def mm_to_nano(mm: float) -> int:
    """Quantise a millimetre value to an integer nanometre grid coordinate."""
    return _check_range(int(round(mm * NANO_PER_MM)))


def nano_to_mm(nano: int) -> float:
    """Convert an integer nanometre coordinate back to floating millimetres."""
    return nano / NANO_PER_MM


def vec_to_int(v: Sequence[float]) -> Tuple[int, ...]:
    """Quantise a float vector (mm) to an integer tuple (nm)."""
    return tuple(mm_to_nano(c) for c in v)


def vec_to_float(v: Sequence[int]) -> Tuple[float, ...]:
    """Convert an integer vector (nm) back to a float tuple (mm)."""
    return tuple(nano_to_mm(c) for c in v)


# ---------------------------------------------------------------------------
# Integer vector operations (exact)
# ---------------------------------------------------------------------------

def add_i(a: Sequence[int], b: Sequence[int]) -> Tuple[int, ...]:
    """Component-wise integer sum."""
    return tuple(x + y for x, y in zip(a, b))


def sub_i(a: Sequence[int], b: Sequence[int]) -> Tuple[int, ...]:
    """Component-wise integer difference."""
    return tuple(x - y for x, y in zip(a, b))


def dot_i(a: Sequence[int], b: Sequence[int]) -> int:
    """Exact integer dot product (nm**2 scale)."""
    return sum(x * y for x, y in zip(a, b))


def cross_i(a: Sequence[int], b: Sequence[int]) -> Tuple[int, int, int]:
    """Exact integer 3D cross product (nm**2 scale)."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def equals_i(a: Sequence[int], b: Sequence[int]) -> bool:
    """Exact equality of two integer points (no tolerance)."""
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


def length_squared_i(v: Sequence[int]) -> int:
    """Exact squared length (avoids sqrt, stays in the integer domain)."""
    return sum(c * c for c in v)


# ---------------------------------------------------------------------------
# Exact 2D segment intersection with grid snapping
# ---------------------------------------------------------------------------

def segment_intersection_2i(
    p1: Sequence[int],
    p2: Sequence[int],
    p3: Sequence[int],
    p4: Sequence[int],
) -> Optional[Tuple[int, int]]:
    """Intersect two 2D integer segments, snapped to the integer grid.

    ``p1``-``p2`` is the first segment, ``p3``-``p4`` the second.  The parallel
    test and the "does the crossing lie inside both segments?" test are done
    with **exact integer determinants** and **no division**, so they are never
    fooled by rounding.  Only the final point is snapped to the grid with a
    single ``round`` -- the same operation every producer of this crossing
    applies, so the snapped vertex is shared bit-for-bit.

    Returns the snapped integer point, or ``None`` when the segments are
    parallel or do not cross within both spans.
    """
    d1x = p2[0] - p1[0]
    d1y = p2[1] - p1[1]
    d2x = p4[0] - p3[0]
    d2y = p4[1] - p3[1]

    # Exact integer determinant of the two directions.
    cross = d1x * d2y - d1y * d2x
    if cross == 0:
        return None  # parallel (or degenerate)

    dx = p3[0] - p1[0]
    dy = p3[1] - p1[1]

    # t = t_numer / cross  (parameter on segment 1)
    # s = s_numer / cross  (parameter on segment 2)
    t_numer = dx * d2y - dy * d2x
    s_numer = dx * d1y - dy * d1x

    # Division-free range test: t in [0, 1] and s in [0, 1].
    # Multiply the inequalities through by ``cross`` and branch on its sign so
    # the direction of each inequality stays correct.
    if cross > 0:
        if t_numer < 0 or t_numer > cross:
            return None
        if s_numer < 0 or s_numer > cross:
            return None
    else:
        if t_numer > 0 or t_numer < cross:
            return None
        if s_numer > 0 or s_numer < cross:
            return None

    t = t_numer / cross
    x = p1[0] + t * d1x
    y = p1[1] + t * d1y
    return (int(round(x)), int(round(y)))


# ---------------------------------------------------------------------------
# Exact 3D line closest points with grid snapping
# ---------------------------------------------------------------------------

def line_line_closest_points_3i(
    p1: Sequence[int],
    d1: Sequence[int],
    p2: Sequence[int],
    d2: Sequence[int],
) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Closest points of two 3D integer lines, snapped to a *single* grid point.

    Line 1 is ``p1 + t*d1``, line 2 is ``p2 + s*d2``.  The determinant
    ``|d1|**2 |d2|**2 - (d1.d2)**2`` is computed exactly; when it is zero the
    lines are parallel and ``None`` is returned.  The two closest points are
    averaged and snapped, so both lines are handed back *the same* integer
    vertex -- this is the mechanism that welds an intersection shared by two
    edges into one point.
    """
    w = sub_i(p1, p2)
    a = dot_i(d1, d1)
    b = dot_i(d1, d2)
    c = dot_i(d2, d2)
    d = dot_i(d1, w)
    e = dot_i(d2, w)

    denom = a * c - b * b  # exact
    if abs(denom) < 1:
        return None  # parallel

    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom

    p1x = p1[0] + t * d1[0]
    p1y = p1[1] + t * d1[1]
    p1z = p1[2] + t * d1[2]
    p2x = p2[0] + s * d2[0]
    p2y = p2[1] + s * d2[1]
    p2z = p2[2] + s * d2[2]

    snapped = (
        int(round((p1x + p2x) / 2)),
        int(round((p1y + p2y) / 2)),
        int(round((p1z + p2z) / 2)),
    )
    return (snapped, snapped)


# ---------------------------------------------------------------------------
# Exact plane-plane intersection line
# ---------------------------------------------------------------------------

def plane_plane_intersection(
    n1: Sequence[int],
    p1: Sequence[int],
    n2: Sequence[int],
    p2: Sequence[int],
) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Intersection line of two planes as ``(point, direction)``.

    Plane 1 is ``n1 . (p - p1) = 0``; plane 2 is ``n2 . (p - p2) = 0``.  The
    line direction is the exact integer cross product ``n1 x n2`` (``None`` when
    the planes are parallel).  The returned point is the point on the line
    closest to the origin, snapped to the grid.  The direction is left as the
    raw (unnormalised) integer cross product.
    """
    direction = cross_i(n1, n2)
    dir_len_sq = length_squared_i(direction)
    if dir_len_sq == 0:
        return None  # parallel planes

    d1 = dot_i(n1, p1)
    d2 = dot_i(n2, p2)

    n2_cross_dir = cross_i(n2, direction)
    dir_cross_n1 = cross_i(direction, n1)

    scale1 = d1 / dir_len_sq
    scale2 = d2 / dir_len_sq

    px = scale1 * n2_cross_dir[0] + scale2 * dir_cross_n1[0]
    py = scale1 * n2_cross_dir[1] + scale2 * dir_cross_n1[1]
    pz = scale1 * n2_cross_dir[2] + scale2 * dir_cross_n1[2]

    point = (int(round(px)), int(round(py)), int(round(pz)))
    return (point, direction)


# ---------------------------------------------------------------------------
# Clip a line to a 3D polygon
# ---------------------------------------------------------------------------

def clip_line_to_polygon_3i(
    line_point: Sequence[int],
    line_dir: Sequence[int],
    polygon: Sequence[Sequence[int]],
) -> List[Tuple[float, float, Tuple[int, int, int]]]:
    """Clip an infinite integer line against a closed 3D polygon.

    Returns a list of ``(t_start, t_end, start_point, end_point)`` spans in
    which the line lies "inside" the polygon boundary, each endpoint being a
    grid-snapped crossing shared with the polygon edge.  Every crossing is
    produced by :func:`line_line_closest_points_3i`, so the endpoints are welded
    to the same integers a neighbouring face would compute.

    The return is flattened to a list of 4-tuples
    ``(t_start, t_end, start, end)`` for span pairs.
    """
    if len(polygon) < 3:
        return []

    crossings: List[Tuple[float, Tuple[int, int, int]]] = []
    line_len_sq = length_squared_i(line_dir)
    if line_len_sq == 0:
        return []

    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        edge_dir = (b[0] - a[0], b[1] - a[1], b[2] - a[2])

        result = line_line_closest_points_3i(line_point, line_dir, a, edge_dir)
        if result is None:
            continue  # parallel to the edge
        point_on_line, point_on_edge = result

        edge_len_sq = length_squared_i(edge_dir)
        if edge_len_sq == 0:
            continue

        to_point = (
            point_on_edge[0] - a[0],
            point_on_edge[1] - a[1],
            point_on_edge[2] - a[2],
        )
        s = dot_i(to_point, edge_dir) / edge_len_sq
        tol = 1e-9
        if s < -tol or s > 1 + tol:
            continue

        to_point_on_line = (
            point_on_line[0] - line_point[0],
            point_on_line[1] - line_point[1],
            point_on_line[2] - line_point[2],
        )
        t = dot_i(to_point_on_line, line_dir) / line_len_sq
        crossings.append((t, point_on_line))

    if len(crossings) < 2:
        return []

    crossings.sort(key=lambda c: c[0])

    spans: List[Tuple[float, float, Tuple[int, int, int]]] = []
    for i in range(0, len(crossings) - 1, 2):
        t_start, start = crossings[i]
        t_end, end = crossings[i + 1]
        spans.append((t_start, t_end, start, end))
    return spans


# ---------------------------------------------------------------------------
# Shared-grid vertex welding
# ---------------------------------------------------------------------------

class VertexRegistry:
    """Intern grid points to canonical integer ids so coincident vertices weld.

    This operationalises SolidType's "both faces reference the same integers"
    principle at the topology layer: whenever a producer snaps a point to the
    grid and interns it, the registry returns a stable id; every later producer
    of the *same* integer point receives *the same id*.  Because the key is the
    exact integer tuple there is no tolerance and no order dependence -- two
    vertices are the same vertex iff their grid coordinates are identical.

    ``coords`` preserves the canonical point for each id in insertion order.
    """

    def __init__(self) -> None:
        self._ids: Dict[Tuple[int, ...], int] = {}
        self._coords: List[Tuple[int, ...]] = []

    def intern(self, point: Sequence[int]) -> int:
        """Return the canonical id for ``point``, allocating one if new."""
        key = tuple(int(c) for c in point)
        existing = self._ids.get(key)
        if existing is not None:
            return existing
        new_id = len(self._coords)
        self._ids[key] = new_id
        self._coords.append(key)
        return new_id

    def intern_mm(self, point_mm: Sequence[float]) -> int:
        """Quantise a millimetre point then intern it (convenience)."""
        return self.intern(vec_to_int(point_mm))

    def get(self, point: Sequence[int]) -> Optional[int]:
        """Return the id for ``point`` if already interned, else ``None``."""
        return self._ids.get(tuple(int(c) for c in point))

    def coord(self, vertex_id: int) -> Tuple[int, ...]:
        """Return the canonical integer coordinate of ``vertex_id``."""
        return self._coords[vertex_id]

    def __len__(self) -> int:
        return len(self._coords)

    def __contains__(self, point: Sequence[int]) -> bool:
        return tuple(int(c) for c in point) in self._ids
