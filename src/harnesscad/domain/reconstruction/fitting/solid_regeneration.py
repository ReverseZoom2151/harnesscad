"""3D solid regeneration from extracted 2D dimensions.

Implements the geometric heart of the CAD-regeneration step: a calibrated 2D
contour with a uniform cross-section is turned into a 3D solid by **linear
extrusion**. The approach is limited to 3D objects with uniform cross-sections,
which is exactly what a prism extrusion produces.

Given

  * a closed contour of metric (x, y) points (from
    :mod:`vision.cvcad_pixel_calibration`), and
  * an extrusion depth (the object thickness, itself a measured dimension),

this builds a prism mesh (vertices + polygonal faces) and reports the volume,
surface area and axis-aligned bounding box. The simple cubical case
(two endpoints of a rectangular surface) is handled by :func:`regenerate_box`.

The CATScript/CATIA macro emission and the actual OCCT B-rep are *external*; this
module produces the neutral geometry those steps would consume.

Stdlib-only, deterministic, no wall clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]
Face = Tuple[int, ...]


def polygon_area(points: Sequence[Point2]) -> float:
    """Signed shoelace area; positive for counter-clockwise ordering."""
    n = len(points)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def polygon_perimeter(points: Sequence[Point2], *, closed: bool = True) -> float:
    n = len(points)
    if n < 2:
        return 0.0
    total = 0.0
    last = n if closed else n - 1
    for i in range(last):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def is_closed(points: Sequence[Point2], tol: float = 1e-9) -> bool:
    if len(points) < 2:
        return False
    a, b = points[0], points[-1]
    return math.hypot(b[0] - a[0], b[1] - a[1]) <= tol


def close_contour(points: Sequence[Point2], tol: float) -> List[Point2]:
    """Snap a nearly-closed open contour shut.

    Regenerated sketches often have a few open contours requiring minor
    geometry adjustments. If the last point is within ``tol`` of the
    first, drop the duplicate; otherwise the caller's contour is treated as
    already implicitly closed (first != last kept as-is).
    """
    pts = list(points)
    if len(pts) >= 2 and math.hypot(pts[-1][0] - pts[0][0],
                                    pts[-1][1] - pts[0][1]) <= tol:
        pts = pts[:-1]
    return pts


@dataclass(frozen=True)
class Solid:
    """A prism solid: vertices, faces (index tuples) and derived properties."""

    vertices: List[Point3]
    faces: List[Face]
    volume: float
    surface_area: float
    bounding_box: Tuple[Point3, Point3]

    def to_dict(self) -> dict:
        return {
            "num_vertices": len(self.vertices),
            "num_faces": len(self.faces),
            "volume": self.volume,
            "surface_area": self.surface_area,
            "bounding_box": self.bounding_box,
        }


def _bbox3(vertices: Sequence[Point3]) -> Tuple[Point3, Point3]:
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


def extrude_contour(contour: Sequence[Point2], depth: float,
                    *, close_tol: float = 1e-9) -> Solid:
    """Linear-extrude a metric contour by ``depth`` along +Z into a prism.

    Faces: one bottom cap (z=0), one top cap (z=depth) and one quad per contour
    edge. Volume = |cross-section area| * depth; surface area = 2*|area| +
    perimeter*depth.
    """
    if depth <= 0.0:
        raise ValueError("depth must be positive")
    pts = close_contour(contour, close_tol)
    n = len(pts)
    if n < 3:
        raise ValueError("contour needs at least 3 distinct points")

    area = abs(polygon_area(pts))
    if area <= 0.0:
        raise ValueError("degenerate contour (zero area)")
    perim = polygon_perimeter(pts, closed=True)

    vertices: List[Point3] = [(x, y, 0.0) for (x, y) in pts]
    vertices += [(x, y, depth) for (x, y) in pts]

    faces: List[Face] = []
    faces.append(tuple(range(n)))                       # bottom cap
    faces.append(tuple(range(n, 2 * n)))                # top cap
    for i in range(n):                                  # side quads
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))

    volume = area * depth
    surface_area = 2.0 * area + perim * depth
    return Solid(vertices, faces, volume, surface_area, _bbox3(vertices))


def regenerate_box(width: float, height: float, depth: float) -> Solid:
    """The simple cubical case: a rectangular surface extruded to depth.

    ``width`` x ``height`` is the measured rectangular cross-section (from the
    two corner endpoints), ``depth`` the measured thickness.
    """
    if min(width, height, depth) <= 0.0:
        raise ValueError("box dimensions must be positive")
    contour = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    return extrude_contour(contour, depth)


def box_from_corners(p0: Point2, p1: Point2, depth: float) -> Solid:
    """Build a box from two opposite corners of the rectangular surface."""
    width = abs(p1[0] - p0[0])
    height = abs(p1[1] - p0[1])
    return regenerate_box(width, height, depth)
