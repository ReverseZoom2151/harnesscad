"""Winding-number point-in-mesh test and Kahan mass properties.

The robust way to decide whether a point is inside a closed triangle mesh -- the
one a mesh boolean relies on to classify a triangle of A as inside/outside B --
is the **generalised winding number**: sum the signed solid angle each triangle
subtends at the query point and divide by ``4*pi``.  For a closed, outward-
oriented 2-manifold the sum is an integer: the number of times the surface wraps
the point (``1`` strictly inside a simple solid, ``0`` outside), and it is
robust to the exact ray-vs-vertex degeneracies that break naive even/odd ray
casting.  A mesh-boolean kernel uses the same solid-angle idea and computes its
mass properties with Kahan-compensated summation for numerical stability.

This module provides, in stdlib Python:

* :func:`solid_angle` -- the signed solid angle of a
  triangle seen from a point (the numerically stable ``atan2`` form);
* :func:`winding_number` -- the generalised winding number of a point w.r.t. a
  triangle mesh (sum of solid angles / ``4*pi``);
* :func:`is_inside` -- the boolean inside test (rounded winding number != 0),
  the exact classification a mesh boolean needs;
* :func:`signed_volume` / :func:`surface_area` -- Kahan-compensated mass
  properties over a triangle mesh.

Overlap with the harness: ``geometry.angelcad_polyhedron`` computes a *plain-sum*
signed volume and area of an explicit polyhedron, and ``bench.cgb_mesh_betti``
does an **even/odd ray-cast** parity inside test on a triangle soup.  Neither
implements a **winding-number (solid-angle) inside test** -- the degeneracy-
robust classifier the boolean needs -- and neither uses Kahan-compensated
summation.  This module fills that gap; the winding-number test is the primary
new contribution.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

__all__ = [
    "solid_angle",
    "winding_number",
    "is_inside",
    "signed_volume",
    "surface_area",
]

Vec3 = Tuple[float, float, float]
Tri = Tuple[int, int, int]

_FOUR_PI = 4.0 * math.pi


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def solid_angle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> float:
    """Signed solid angle subtended by triangle ``(a, b, c)`` at point ``p``.

    Uses the signed solid-angle formula:

        tan(Omega / 2) = det[A, B, C]
                         / (|A||B||C| + (A.B)|C| + (B.C)|A| + (C.A)|B|)

    where ``A, B, C`` are the vectors from ``p`` to the triangle vertices.  The
    sign follows the triangle's winding (positive for a CCW triangle seen from
    the outside), and the result lies in ``(-2*pi, 2*pi)``.
    """
    A = _sub(a, p)
    B = _sub(b, p)
    C = _sub(c, p)
    la = _norm(A)
    lb = _norm(B)
    lc = _norm(C)
    if la == 0.0 or lb == 0.0 or lc == 0.0:
        return 0.0
    numer = _dot(A, _cross(B, C))
    denom = (la * lb * lc + _dot(A, B) * lc + _dot(B, C) * la + _dot(C, A) * lb)
    return 2.0 * math.atan2(numer, denom)


def winding_number(p: Vec3, vertices: Sequence[Vec3], tris: Sequence[Tri]) -> float:
    """Generalised winding number of ``p`` w.r.t. the triangle mesh.

    For a closed, outward-oriented 2-manifold this is (to floating precision) an
    integer: ``1`` strictly inside a simple solid, ``0`` outside, higher for
    nested or multiply-wound shells.  Robust to ray-vs-vertex degeneracies.
    """
    total = 0.0
    comp = 0.0  # Kahan compensation
    for (i, j, k) in tris:
        omega = solid_angle(p, vertices[i], vertices[j], vertices[k])
        y = omega - comp
        t = total + y
        comp = (t - total) - y
        total = t
    return total / _FOUR_PI


def is_inside(p: Vec3, vertices: Sequence[Vec3], tris: Sequence[Tri]) -> bool:
    """True iff ``p`` is inside the closed mesh.

    A point strictly inside a simple solid has winding number ~1 and outside
    ~0, so the threshold at ``0.5`` classifies robustly.  Points exactly on the
    surface (winding ~0.5) are ambiguous by construction.
    """
    return abs(winding_number(p, vertices, tris)) > 0.5


def signed_volume(vertices: Sequence[Vec3], tris: Sequence[Tri]) -> float:
    """Signed volume of the mesh via Kahan-compensated tetrahedron summation.

    Positive when the triangles are wound outward (CCW seen from outside).
    Each triangle contributes
    ``dot(v0, cross(v1, v0 -> ... )) / 6`` with Kahan compensation.
    """
    value = 0.0
    comp = 0.0
    for (i, j, k) in tris:
        v0 = vertices[i]
        cp = _cross(_sub(vertices[j], v0), _sub(vertices[k], v0))
        contrib = _dot(cp, v0) / 6.0
        y = contrib - comp
        t = value + y
        comp = (t - value) - y
        value = t
    return value


def surface_area(vertices: Sequence[Vec3], tris: Sequence[Tri]) -> float:
    """Surface area via Kahan-compensated triangle-area summation."""
    value = 0.0
    comp = 0.0
    for (i, j, k) in tris:
        v0 = vertices[i]
        cp = _cross(_sub(vertices[j], v0), _sub(vertices[k], v0))
        contrib = _norm(cp) / 2.0
        y = contrib - comp
        t = value + y
        comp = (t - value) - y
        value = t
    return value
