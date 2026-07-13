"""Solid-of-revolution geometry for RLCAD's revolution operation.

RLCAD (Yin, Lu, Shen et al., "RLCAD: Reinforcement Learning Training Gym for
Revolution Involved CAD Command Sequence Generation") extends the usual
sketch+extrude CAD command vocabulary with a **revolution** (revolve) operation:
a 2D profile is rotated around a coplanar axis line, under a rotation angle, to
form a 3D solid of revolution (Sec. 4.2, Fig. 4). Most CAD datasets are
extrude-only; this module supplies the deterministic geometry the revolve op
needs.

Canonical representation
------------------------
The profile is a closed polygon in the half-plane ``(r, z)`` where ``r >= 0`` is
the distance from the rotation axis and ``z`` runs along the axis. This mirrors
the paper's construction: sampling points are projected onto the rotation axis
(the red points of Fig. 4) and connected to form the profile. A general 3D
axis+profile is reduced to this canonical frame by :func:`project_to_profile`.

Volume and surface area follow from **Pappus's (Guldinus) theorems**, which are
exact for a polygonal profile:

* First theorem (surface): the area swept by a plane curve revolved about a
  coplanar axis equals the curve length times the distance travelled by its
  centroid: ``S = angle * r_curve_centroid * L``.
* Second theorem (volume): the volume swept by a plane region equals the region
  area times the distance travelled by its centroid: ``V = angle * r_area_centroid * A``.

For a full revolution ``angle = 2*pi`` these reduce to the textbook
``S = 2*pi*r*L`` and ``V = 2*pi*r*A``. Partial revolutions scale linearly with
the sweep angle. A cylinder (rectangle profile) and a cone (right-triangle
profile) come out exactly, which the tests check.

Validity: the profile must not cross the axis (Sec. 4, command-sequence
validity) -- if part of the region lies at ``r < 0`` the swept solid
self-intersects. Touching the axis (``r == 0``) is allowed (e.g. a sphere
profile). Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

Point2 = Tuple[float, float]   # (r, z)
Point3 = Tuple[float, float, float]

FULL_TURN = 2.0 * math.pi
_EPS = 1e-12


def _closed(points: Sequence[Point2]) -> list[Point2]:
    """Return the polygon vertex list without a duplicated closing vertex."""
    pts = [(float(r), float(z)) for r, z in points]
    if len(pts) >= 2 and abs(pts[0][0] - pts[-1][0]) < _EPS and abs(
            pts[0][1] - pts[-1][1]) < _EPS:
        pts = pts[:-1]
    if len(pts) < 3:
        raise ValueError("a profile polygon needs at least 3 distinct vertices")
    return pts


def signed_area(points: Sequence[Point2]) -> float:
    """Signed polygon area via the shoelace formula (+ for CCW in the r-z plane)."""
    pts = _closed(points)
    total = 0.0
    n = len(pts)
    for i in range(n):
        r0, z0 = pts[i]
        r1, z1 = pts[(i + 1) % n]
        total += r0 * z1 - r1 * z0
    return total / 2.0


def profile_area(points: Sequence[Point2]) -> float:
    """Unsigned area of the profile region."""
    return abs(signed_area(points))


def area_centroid(points: Sequence[Point2]) -> Point2:
    """Centroid ``(r_c, z_c)`` of the profile *region* (area-weighted)."""
    pts = _closed(points)
    a = signed_area(pts)
    if abs(a) < _EPS:
        raise ValueError("degenerate (zero-area) profile has no area centroid")
    cr = 0.0
    cz = 0.0
    n = len(pts)
    for i in range(n):
        r0, z0 = pts[i]
        r1, z1 = pts[(i + 1) % n]
        cross = r0 * z1 - r1 * z0
        cr += (r0 + r1) * cross
        cz += (z0 + z1) * cross
    cr /= (6.0 * a)
    cz /= (6.0 * a)
    return (cr, cz)


def perimeter(points: Sequence[Point2]) -> float:
    """Total length of the closed boundary curve."""
    pts = _closed(points)
    n = len(pts)
    total = 0.0
    for i in range(n):
        r0, z0 = pts[i]
        r1, z1 = pts[(i + 1) % n]
        total += math.hypot(r1 - r0, z1 - z0)
    return total


def curve_centroid(points: Sequence[Point2]) -> Point2:
    """Centroid ``(r_c, z_c)`` of the boundary *curve* (arc-length-weighted)."""
    pts = _closed(points)
    n = len(pts)
    length = 0.0
    cr = 0.0
    cz = 0.0
    for i in range(n):
        r0, z0 = pts[i]
        r1, z1 = pts[(i + 1) % n]
        seg = math.hypot(r1 - r0, z1 - z0)
        length += seg
        cr += seg * (r0 + r1) / 2.0
        cz += seg * (z0 + z1) / 2.0
    if length < _EPS:
        raise ValueError("degenerate profile boundary has zero length")
    return (cr / length, cz / length)


def crosses_axis(points: Sequence[Point2], tol: float = 1e-9) -> bool:
    """True if any part of the profile lies on the negative-``r`` side of the axis.

    A profile that crosses the axis produces a self-intersecting solid; touching
    the axis (``r == 0``) is permitted.
    """
    return any(r < -tol for r, _ in points)


def _normalize(vec: Point3) -> Point3:
    n = math.sqrt(sum(c * c for c in vec))
    if n < _EPS:
        raise ValueError("axis direction must be a non-zero vector")
    return (vec[0] / n, vec[1] / n, vec[2] / n)


def project_to_profile(points3d: Sequence[Point3], axis_point: Point3,
                       axis_dir: Point3) -> list[Point2]:
    """Reduce 3D sampling points + a rotation axis to canonical ``(r, z)`` pairs.

    Mirrors Fig. 4: each point is projected onto the axis to get its axial
    coordinate ``z`` (the red projection point), and its perpendicular distance
    to the axis gives ``r``. ``z`` is measured from ``axis_point`` along the unit
    axis direction.
    """
    ax = _normalize(axis_dir)
    out: list[Point2] = []
    for p in points3d:
        d = (p[0] - axis_point[0], p[1] - axis_point[1], p[2] - axis_point[2])
        z = d[0] * ax[0] + d[1] * ax[1] + d[2] * ax[2]
        perp = (d[0] - z * ax[0], d[1] - z * ax[1], d[2] - z * ax[2])
        r = math.sqrt(sum(c * c for c in perp))
        out.append((r, z))
    return out


def pappus_volume(points: Sequence[Point2], angle: float = FULL_TURN) -> float:
    """Volume swept by revolving the profile region (Pappus's second theorem).

    ``V = angle * r_c * A`` where ``r_c`` is the region centroid's axis distance
    and ``A`` the region area. Raises if the profile crosses the axis.
    """
    if angle <= 0.0:
        raise ValueError("revolution angle must be positive")
    if crosses_axis(points):
        raise ValueError("profile crosses the rotation axis; revolve is invalid")
    a = profile_area(points)
    rc, _ = area_centroid(points)
    return angle * rc * a


def pappus_surface_area(points: Sequence[Point2], angle: float = FULL_TURN,
                        include_caps: bool = True) -> float:
    """Surface area of the solid of revolution (Pappus's first theorem).

    The revolved boundary curve contributes ``angle * r_c * L``. For a *partial*
    revolution (``angle < 2*pi``) two planar cap faces (the profile at the start
    and end angles) close the solid; ``include_caps`` adds their ``2 * A``.
    """
    if angle <= 0.0:
        raise ValueError("revolution angle must be positive")
    if crosses_axis(points):
        raise ValueError("profile crosses the rotation axis; revolve is invalid")
    rc, _ = curve_centroid(points)
    lateral = angle * rc * perimeter(points)
    if include_caps and angle < FULL_TURN - 1e-9:
        lateral += 2.0 * profile_area(points)
    return lateral


@dataclass(frozen=True)
class RevolveSolid:
    """A solid of revolution derived from a profile, axis and sweep angle."""

    profile: Tuple[Point2, ...]
    angle: float = FULL_TURN

    def __post_init__(self):
        # Validate eagerly so an invalid revolve never becomes a usable solid.
        _closed(self.profile)
        if self.angle <= 0.0 or self.angle > FULL_TURN + 1e-9:
            raise ValueError("revolution angle must be in (0, 2*pi]")
        if crosses_axis(self.profile):
            raise ValueError("profile crosses the rotation axis; revolve is invalid")

    @property
    def is_full(self) -> bool:
        return abs(self.angle - FULL_TURN) < 1e-9

    @property
    def volume(self) -> float:
        return pappus_volume(self.profile, self.angle)

    @property
    def surface_area(self) -> float:
        return pappus_surface_area(self.profile, self.angle)

    def profile_bounds(self) -> Tuple[float, float, float, float]:
        """``(r_min, r_max, z_min, z_max)`` of the profile."""
        rs = [r for r, _ in self.profile]
        zs = [z for _, z in self.profile]
        return (min(rs), max(rs), min(zs), max(zs))

    def bounding_cylinder_volume(self) -> float:
        """Volume of the smallest axis-aligned cylinder containing the full solid."""
        r_min, r_max, z_min, z_max = self.profile_bounds()
        return math.pi * r_max * r_max * (z_max - z_min)
