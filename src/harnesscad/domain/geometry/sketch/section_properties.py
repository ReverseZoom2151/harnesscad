"""Planar section properties of closed polygons (Open CAD Studio ``MASSPROP``).

**Open CAD Studio** is a Rust 2-D/3-D CAD application (LibreCAD lineage) whose
``MASSPROP`` command reports "area, perimeter, centroid of selected entities".
This module reimplements and extends that inquiry command as a deterministic
computation over closed polygonal regions -- the analytic area moments an
engineer needs to size a beam or plate section, not just area/centroid.

Everything derives from the polygon's ordered vertices via the shoelace family
of formulas (Green's theorem), so the result is exact for any simple polygon and
needs no meshing:

* :func:`signed_area` / :func:`perimeter` / :func:`centroid`;
* :func:`area_moments` -- the second moments of area about the centroid,
  ``Ixx``, ``Iyy`` and the product moment ``Ixy`` (units: length^4);
* :func:`principal_moments` -- the principal second moments ``I1 >= I2`` and the
  rotation ``theta`` (radians) of the principal axes, from the 2x2 moment tensor;
* :func:`section_properties` -- the full report for an outer boundary with any
  number of holes (holes subtract, via signed-area superposition), including the
  bounding box, radii of gyration and elastic section moduli.

A polygon may be given open (first != last) or closed; winding may be either
direction -- areas are taken as magnitudes and the centroid/moment signs are
handled internally. Holes are supplied as separate loops and subtracted.

Distinct from :mod:`harnesscad.domain.geometry.mesh.polyhedron` (3-D solid mass
properties) and from point-cloud principal axes: this is the classical 2-D
*section* analysis on a polygonal region. Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]

__all__ = [
    "SectionError",
    "signed_area",
    "polygon_area",
    "perimeter",
    "centroid",
    "area_moments",
    "principal_moments",
    "SectionProperties",
    "section_properties",
]


class SectionError(ValueError):
    """A section is degenerate (fewer than 3 vertices or zero area)."""


def _clean(points: Sequence[Point]) -> List[Point]:
    """Drop a duplicated closing vertex if present; validate count."""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) >= 2 and abs(pts[0][0] - pts[-1][0]) < 1e-12 and abs(pts[0][1] - pts[-1][1]) < 1e-12:
        pts = pts[:-1]
    if len(pts) < 3:
        raise SectionError("a polygon needs at least 3 distinct vertices")
    return pts


def signed_area(points: Sequence[Point]) -> float:
    """Signed area (positive for counter-clockwise winding)."""
    pts = _clean(points)
    total = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def polygon_area(points: Sequence[Point]) -> float:
    """Unsigned area."""
    return abs(signed_area(points))


def perimeter(points: Sequence[Point]) -> float:
    """Total edge length of the closed polygon."""
    pts = _clean(points)
    n = len(pts)
    total = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def centroid(points: Sequence[Point]) -> Point:
    """Area centroid (Green's theorem)."""
    pts = _clean(points)
    a = signed_area(pts)
    if abs(a) < 1e-15:
        raise SectionError("degenerate polygon (zero area) has no centroid")
    n = len(pts)
    cx = cy = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    return (cx / (6.0 * a), cy / (6.0 * a))


def area_moments(points: Sequence[Point], about: Optional[Point] = None) -> Tuple[float, float, float]:
    """Second moments of area ``(Ixx, Iyy, Ixy)`` about ``about``.

    Defaults to the centroid. Uses the exact polygon second-moment formulas
    (Green's theorem). ``Ixx`` integrates y^2 dA, ``Iyy`` integrates x^2 dA,
    ``Ixy`` integrates x*y dA -- always non-negative area weighting, signed by
    winding then normalised to a positive area.
    """
    pts = _clean(points)
    a_signed = signed_area(pts)
    if abs(a_signed) < 1e-15:
        raise SectionError("degenerate polygon (zero area) has no moments")
    if about is None:
        about = centroid(pts)
    ax, ay = about
    n = len(pts)
    sxx = syy = sxy = 0.0
    for i in range(n):
        x0, y0 = pts[i][0] - ax, pts[i][1] - ay
        x1, y1 = pts[(i + 1) % n][0] - ax, pts[(i + 1) % n][1] - ay
        cross = x0 * y1 - x1 * y0
        sxx += (y0 * y0 + y0 * y1 + y1 * y1) * cross
        syy += (x0 * x0 + x0 * x1 + x1 * x1) * cross
        sxy += (x0 * y1 + 2 * x0 * y0 + 2 * x1 * y1 + x1 * y0) * cross
    ixx = sxx / 12.0
    iyy = syy / 12.0
    ixy = sxy / 24.0
    # Normalise sign: a clockwise polygon produces negative area & moments.
    if a_signed < 0:
        ixx, iyy, ixy = -ixx, -iyy, -ixy
    return (ixx, iyy, ixy)


def principal_moments(ixx: float, iyy: float, ixy: float) -> Tuple[float, float, float]:
    """Principal second moments ``(I1, I2, theta)`` with ``I1 >= I2``.

    ``theta`` (radians) is the angle from the x-axis to the axis of ``I1``.
    """
    avg = (ixx + iyy) / 2.0
    diff = (ixx - iyy) / 2.0
    radius = math.hypot(diff, ixy)
    i1 = avg + radius
    i2 = avg - radius
    theta = 0.5 * math.atan2(-2.0 * ixy, ixx - iyy)
    return (i1, i2, theta)


@dataclass(frozen=True)
class SectionProperties:
    """The full planar section report."""

    area: float
    perimeter: float
    centroid: Point
    ixx: float
    iyy: float
    ixy: float
    i1: float
    i2: float
    principal_angle: float
    bbox: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    rg_x: float  # radius of gyration about centroidal x-axis
    rg_y: float

    def section_modulus_x(self) -> float:
        """Elastic section modulus about the centroidal x-axis (Ixx / c_y)."""
        xmin, ymin, xmax, ymax = self.bbox
        cy = max(self.centroid[1] - ymin, ymax - self.centroid[1])
        return self.ixx / cy if cy > 0 else 0.0

    def section_modulus_y(self) -> float:
        xmin, ymin, xmax, ymax = self.bbox
        cx = max(self.centroid[0] - xmin, xmax - self.centroid[0])
        return self.iyy / cx if cx > 0 else 0.0


def _bbox(loops: Sequence[Sequence[Point]]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for loop in loops:
        for x, y in loop:
            xs.append(float(x))
            ys.append(float(y))
    return (min(xs), min(ys), max(xs), max(ys))


def section_properties(
    outer: Sequence[Point], holes: Optional[Sequence[Sequence[Point]]] = None
) -> SectionProperties:
    """Full section properties of ``outer`` minus any ``holes``.

    Holes subtract by signed-area superposition: net area, net first moment
    (centroid) and net second moment are computed by treating each hole as a
    negative-area region. Moments are reported about the *net* centroid.
    """
    holes = list(holes or [])

    # Net area and centroid via first moments about the origin.
    regions: List[Tuple[float, Point]] = []
    a_outer = polygon_area(outer)
    regions.append((a_outer, centroid(outer)))
    for hole in holes:
        regions.append((-polygon_area(hole), centroid(hole)))

    net_area = sum(a for a, _ in regions)
    if net_area <= 1e-12:
        raise SectionError("net section area is non-positive (holes exceed outer)")
    cx = sum(a * c[0] for a, c in regions) / net_area
    cy = sum(a * c[1] for a, c in regions) / net_area
    net_centroid = (cx, cy)

    # Net second moments about the net centroid (parallel-axis per region).
    ixx = iyy = ixy = 0.0
    for loop, sign in [(outer, 1.0)] + [(h, -1.0) for h in holes]:
        a = polygon_area(loop) * sign
        lc = centroid(loop)
        lixx, liyy, lixy = area_moments(loop)  # about loop's own centroid
        dx = lc[0] - cx
        dy = lc[1] - cy
        ixx += sign * (abs(lixx) + polygon_area(loop) * dy * dy)
        iyy += sign * (abs(liyy) + polygon_area(loop) * dx * dx)
        ixy += sign * (_signed_product(lixy) + polygon_area(loop) * dx * dy)

    i1, i2, theta = principal_moments(ixx, iyy, ixy)
    bbox = _bbox([outer])
    per = perimeter(outer) + sum(perimeter(h) for h in holes)
    rg_x = math.sqrt(ixx / net_area) if ixx > 0 else 0.0
    rg_y = math.sqrt(iyy / net_area) if iyy > 0 else 0.0

    return SectionProperties(
        area=net_area,
        perimeter=per,
        centroid=net_centroid,
        ixx=ixx,
        iyy=iyy,
        ixy=ixy,
        i1=i1,
        i2=i2,
        principal_angle=theta,
        bbox=bbox,
        rg_x=rg_x,
        rg_y=rg_y,
    )


def _signed_product(ixy: float) -> float:
    """area_moments already returns a positively-oriented Ixy; pass through."""
    return ixy
