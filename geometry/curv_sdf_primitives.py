"""Exact signed-distance-field primitives, ported from Curv's shape library.

Curv (Doug Moen, https://github.com/curv3d/curv) is a functional language that
represents every 3D/2D shape as a *signed distance field* (SDF): a function
``f: R^n -> R`` whose sign tells inside (``f < 0``) from outside (``f > 0``) and
whose zero level set is the boundary.  A field is **exact** (Euclidean) when
``|f(p)|`` equals the true Euclidean distance to the boundary and ``|grad f|``
is 1 almost everywhere (the Eikonal / 1-Lipschitz property); it is **mitred**
when it is 1-Lipschitz but underestimates distance near reflex features (edges
are preserved rather than rounded); and **approximate** otherwise.

This module reimplements Curv's primitive constructors (``lib/curv/std.curv``)
in stdlib-only Python.  Each returns a plain ``float`` distance.  The exact
primitives (sphere, box.exact, cylinder.exact, cone.exact, capped cone,
capsule, torus, plane/half-space, 2D circle/rect.exact) satisfy the true
Euclidean-distance property; ``box.mitred`` / ``rect.mitred`` /
``regular_polygon`` are the cheaper mitred fields; ``ellipsoid`` is a bounded
approximation (anisotropic stretch of the unit sphere).

Credits inside Curv: several exact fields (cone, capped_cone, capsule, polygon)
are due to Inigo Quilez / MERCURY (hg_sdf); reimplemented here from the maths.

Conventions: points are ``(x, y, z)`` tuples in 3D and ``(x, y)`` in 2D.
Diameters ``d`` follow Curv (a ``sphere d`` has diameter ``d``, radius ``d/2``).
stdlib-only, deterministic, no randomness, no wall clock.
"""

from __future__ import annotations

from math import atan2, cos, fmod, hypot, sin, sqrt, tau
from typing import Sequence, Tuple

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
# small vector helpers                                                        #
# --------------------------------------------------------------------------- #
def _mag2(x: float, y: float) -> float:
    return hypot(x, y)


def _mag3(x: float, y: float, z: float) -> float:
    return sqrt(x * x + y * y + z * z)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# --------------------------------------------------------------------------- #
# 3D primitives                                                               #
# --------------------------------------------------------------------------- #
def sphere(p: Sequence[float], d: float = 2.0) -> float:
    """Sphere of diameter ``d`` centred on origin.  Exact distance field."""
    r = d / 2.0
    return _mag3(p[0], p[1], p[2]) - r


def box_mitred(p: Sequence[float], size: Sequence[float]) -> float:
    """Axis-aligned box, full dimensions ``size``.  *Mitred* field (fast).

    ``max(|p| - r)`` where ``r = size/2``; 1-Lipschitz but underestimates the
    true distance in the exterior near edges/corners (edges preserved).
    """
    rx, ry, rz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    dx, dy, dz = abs(p[0]) - rx, abs(p[1]) - ry, abs(p[2]) - rz
    return max(dx, dy, dz)


def box_exact(p: Sequence[float], size: Sequence[float]) -> float:
    """Axis-aligned box, full dimensions ``size``.  *Exact* Euclidean field.

    ``min(max(d), 0) + |max(d, 0)|`` with ``d = |p| - r``.  The interior term
    ``min(max(d),0)`` is the negative distance inside; the exterior term is the
    Euclidean length of the positive part.
    """
    rx, ry, rz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    dx, dy, dz = abs(p[0]) - rx, abs(p[1]) - ry, abs(p[2]) - rz
    inside = min(max(dx, dy, dz), 0.0)
    outside = _mag3(max(dx, 0.0), max(dy, 0.0), max(dz, 0.0))
    return inside + outside


def rounded_box(p: Sequence[float], size: Sequence[float], radius: float) -> float:
    """Box with rounded edges/corners of radius ``radius``.

    Exact field: ``box.exact(p, size) - radius`` (Minkowski sum with a ball of
    ``radius``); ``size`` is the dimension of the un-rounded core.
    """
    return box_exact(p, size) - radius


def cylinder(p: Sequence[float], d: float = 2.0, h: float = 2.0) -> float:
    """Z-axis cylinder, diameter ``d``, height ``h``.  Exact field.

    Curv builds this as ``extrude.exact h (circle d)``; the closed form is the
    2D union of the radial slab ``mag(x,y)-R`` and the axial slab ``|z|-h/2``.
    """
    r = d / 2.0
    hh = h / 2.0
    dr = _mag2(p[0], p[1]) - r
    dz = abs(p[2]) - hh
    inside = min(max(dr, dz), 0.0)
    outside = _mag2(max(dr, 0.0), max(dz, 0.0))
    return inside + outside


def cone(p: Sequence[float], d: float, h: float) -> float:
    """Cone: base diameter ``d`` in the XY plane, apex at ``(0,0,h)``.

    Exact Euclidean field (credit: MERCURY hg_sdf, via Curv ``cone.exact``).
    """
    radius = d / 2.0
    qx = _mag2(p[0], p[1])
    qy = p[2]
    # apex-relative
    ax, ay = qx - 0.0, qy - h
    mlen = _mag2(h, radius)
    mdx, mdy = h / mlen, radius / mlen  # normalize[h, radius]
    mantle = ax * mdx + ay * mdy
    projected = ax * mdy + ay * (-mdx)
    dist = max(mantle, -qy)
    if qy > h and projected < 0.0:
        dist = max(dist, _mag2(ax, ay))
    if qx > radius and projected > _mag2(h, radius):
        dist = max(dist, _mag2(qx - radius, qy))  # distance to base ring, q-[radius,0]
    return dist


def capped_cone(p: Sequence[float], h: float, top: float, bottom: float) -> float:
    """Capped cone (truncated), Z axis, centred on origin.  Exact field.

    ``h`` full height, ``top``/``bottom`` end-cap diameters (credit: IQ).
    """
    hh = h / 2.0
    r1 = bottom / 2.0
    r2 = top / 2.0
    k1 = (r2, hh)
    k2 = (r2 - r1, h)
    qx = _mag2(p[0], p[1])
    qy = p[2]
    cax = qx - min(qx, r1 if qy < 0.0 else r2)
    cay = abs(qy) - hh
    denom = k2[0] * k2[0] + k2[1] * k2[1]
    tclamp = _clamp(((k1[0] - qx) * k2[0] + (k1[1] - qy) * k2[1]) / denom, 0.0, 1.0)
    cbx = qx - k1[0] + k2[0] * tclamp
    cby = qy - k1[1] + k2[1] * tclamp
    s = -1.0 if (cbx < 0.0 and cay < 0.0) else 1.0
    return s * sqrt(min(cax * cax + cay * cay, cbx * cbx + cby * cby))


def capsule(p: Sequence[float], a: Sequence[float], b: Sequence[float], d: float) -> float:
    """Capsule: cylinder of diameter ``d`` from ``a`` to ``b`` with hemi caps.

    Exact field (IQ line-segment distance).  ``a``/``b`` are 3D points.
    """
    r = d / 2.0
    pax, pay, paz = p[0] - a[0], p[1] - a[1], p[2] - a[2]
    bax, bay, baz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    denom = bax * bax + bay * bay + baz * baz
    hh = _clamp((pax * bax + pay * bay + paz * baz) / denom, 0.0, 1.0) if denom > 0.0 else 0.0
    dx, dy, dz = pax - bax * hh, pay - bay * hh, paz - baz * hh
    return _mag3(dx, dy, dz) - r


def torus(p: Sequence[float], major: float, minor: float) -> float:
    """Torus, Z axis.  ``major``/``minor`` are diameters.  Exact field.

    ``major`` measured centre-of-tube to centre-of-tube through the origin.
    """
    rmaj = major / 2.0
    rmin = minor / 2.0
    q = _mag2(p[0], p[1]) - rmaj
    return _mag2(q, p[2]) - rmin


def ellipsoid(p: Sequence[float], size: Sequence[float]) -> float:
    """Axis-aligned ellipsoid, full dimensions ``size``.  *Bounded* field.

    Curv's ``ellipsoid`` is the unit sphere stretched anisotropically:
    ``(mag(x/dx, y/dy, z/dz) - 1/2) * min(size)``.  Multiplying by the smallest
    axis keeps the field 1-Lipschitz (``|grad| <= 1``) so it never *over*
    estimates distance -- a safe lower bound for sphere tracing, but not exact.
    """
    sx, sy, sz = size[0], size[1], size[2]
    m = _mag3(p[0] / sx, p[1] / sy, p[2] / sz) - 0.5
    return m * min(sx, sy, sz)


def plane(p: Sequence[float], normal: Sequence[float], offset: float = 0.0) -> float:
    """Half-space with unit ``normal`` whose boundary is ``offset`` from origin.

    ``dot(normal, p) - offset``.  Exact for a unit normal.  Inside (``< 0``) is
    the side the normal points away from when ``offset >= 0``.
    """
    return normal[0] * p[0] + normal[1] * p[1] + normal[2] * p[2] - offset


# --------------------------------------------------------------------------- #
# 2D primitives                                                               #
# --------------------------------------------------------------------------- #
def circle(p: Sequence[float], d: float = 2.0) -> float:
    """2D circle of diameter ``d``.  Exact field."""
    return _mag2(p[0], p[1]) - d / 2.0


def rect_mitred(p: Sequence[float], size: Sequence[float]) -> float:
    """2D rectangle, full dimensions ``size``.  Mitred field."""
    rx, ry = size[0] / 2.0, size[1] / 2.0
    return max(abs(p[0]) - rx, abs(p[1]) - ry)


def rect_exact(p: Sequence[float], size: Sequence[float]) -> float:
    """2D rectangle, full dimensions ``size``.  Exact Euclidean field."""
    rx, ry = size[0] / 2.0, size[1] / 2.0
    dx, dy = abs(p[0]) - rx, abs(p[1]) - ry
    inside = min(max(dx, dy), 0.0)
    outside = _mag2(max(dx, 0.0), max(dy, 0.0))
    return inside + outside


def half_plane(p: Sequence[float], normal: Sequence[float], offset: float = 0.0) -> float:
    """2D half-plane with unit ``normal``, edge ``offset`` from origin.  Exact."""
    return normal[0] * p[0] + normal[1] * p[1] - offset


def regular_polygon(p: Sequence[float], n: int, d: float = 2.0) -> float:
    """Regular ``n``-gon, incircle diameter ``d``, bottom edge parallel to X.

    Curv builds this by radially repeating a half-plane (``mitred`` field).
    The point is folded into one of ``n`` angular wedges, then clipped by the
    edge line at apothem ``d/2``.  1-Lipschitz mitred field (corners sharp).
    """
    if n < 3:
        raise ValueError("regular_polygon needs n >= 3")
    apothem = d / 2.0
    angle = tau / n
    ashift = tau / 4.0 + angle / 2.0
    a = atan2(p[1], p[0]) + ashift
    r = _mag2(p[0], p[1])
    a2 = fmod(a, angle)
    if a2 < 0.0:
        a2 += angle
    a2 -= ashift
    # y coordinate after folding into the reference wedge
    fy = sin(a2) * r
    return -fy - apothem


# --------------------------------------------------------------------------- #
# 2D -> 3D lifts                                                              #
# --------------------------------------------------------------------------- #
def extrude(dist2d: float, z: float, h: float) -> float:
    """Extrude a 2D field value to 3D height ``h`` (exact ``extrude.exact``).

    Given the 2D distance ``dist2d`` at ``(x, y)`` and the height ``z``, returns
    the exact 3D distance of the prism of full height ``h``.
    """
    hh = h / 2.0
    dz = abs(z) - hh
    inside = min(max(dz, dist2d), 0.0)
    outside = _mag2(max(dz, 0.0), max(dist2d, 0.0))
    return inside + outside


def revolve(shape2d, p: Sequence[float]) -> float:
    """Revolve a 2D shape ``shape2d(x, y) -> float`` around the Z axis.

    Curv: ``revolve = perimeter_extrude (circle 0)``; the 3D field is
    ``shape2d(mag(x, y), z)`` -- the 2D shape's local X becomes the radius and
    its local Y becomes world Z.
    """
    return shape2d(_mag2(p[0], p[1]), p[2])
