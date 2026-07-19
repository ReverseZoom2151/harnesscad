"""Robust adaptive geometric orientation predicates.

A mesh boolean is built on *exact-sign* geometric predicates: the
combinatorial decisions (which side of a plane a vertex lies on, whether a
point is inside a circumscribed sphere) must be sign-correct even when the
floating-point determinant rounds to a value of the wrong sign, otherwise the
half-edge arrangement produced by the boolean becomes non-manifold.  A robust
kernel therefore provides adaptive ``orient2d``/``orient3d``/``incircle``/
``insphere`` for exactly this reason.

This module implements those four predicates in stdlib Python with a
guaranteed-correct sign.  The strategy is the standard two-tier one:

* a fast floating-point evaluation of the determinant with a conservative
  a-priori error bound;  when the magnitude of the float result comfortably
  exceeds that bound the sign is trusted and returned immediately;
* otherwise fall back to an *exact* recomputation.  Python ``float`` values are
  binary fractions, so :class:`fractions.Fraction` represents each input with
  zero error and the exact determinant sign is recovered with no rounding at
  all -- this is stronger than the standard staged expansions, which only refine
  until the sign is certain.

The returned sign convention matches the standard convention:

* :func:`orient2d` ``> 0`` when ``a, b, c`` are counter-clockwise;
* :func:`orient3d` ``> 0`` when ``d`` is below the plane ``a, b, c`` oriented
  by the right-hand rule (i.e. ``d`` on the negative side of the CCW triangle);
* :func:`incircle` ``> 0`` when ``d`` is inside the circle through ``a, b, c``
  (a, b, c CCW);
* :func:`insphere` ``> 0`` when ``e`` is inside the sphere through
  ``a, b, c, d`` (a, b, c, d positively oriented, ``orient3d > 0``).

The harness ``geometry.euclid_validity`` has only an inexact 2D ``orient``
helper (a raw cross product with an absolute ``1e-12`` threshold); no exact 3D
orientation, in-circle or in-sphere test existed, and these are the foundation
every downstream mesh-boolean decision rests on.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence, Tuple

__all__ = [
    "orient2d",
    "orient3d",
    "incircle",
    "insphere",
    "sign",
]

Point2 = Sequence[float]
Point3 = Sequence[float]

# Error-bound coefficients.  eps is the IEEE-754 double unit round-off (2**-53).
# The constants are the standard conservative a-priori relative error bounds for
# each determinant expansion; multiplying by the sum of magnitudes of the
# permanent gives an absolute bound below which the float sign is not trusted.
_EPS = 2.0 ** -53
_O2_BOUND = (3.0 + 16.0 * _EPS) * _EPS
_O3_BOUND = (7.0 + 56.0 * _EPS) * _EPS
_INCIRCLE_BOUND = (10.0 + 96.0 * _EPS) * _EPS
_INSPHERE_BOUND = (16.0 + 224.0 * _EPS) * _EPS


def sign(x: float) -> int:
    """Return -1, 0 or +1 for the sign of ``x``."""
    return (x > 0) - (x < 0)


# --------------------------------------------------------------------------
# orient2d
# --------------------------------------------------------------------------


def _orient2d_exact(a: Point2, b: Point2, c: Point2) -> int:
    ax, ay = Fraction(a[0]), Fraction(a[1])
    bx, by = Fraction(b[0]), Fraction(b[1])
    cx, cy = Fraction(c[0]), Fraction(c[1])
    det = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
    return (det > 0) - (det < 0)


def orient2d(a: Point2, b: Point2, c: Point2) -> int:
    """Exact sign of the 2D orientation determinant.  +1 iff CCW."""
    detleft = (a[0] - c[0]) * (b[1] - c[1])
    detright = (a[1] - c[1]) * (b[0] - c[0])
    det = detleft - detright
    # Only bother with the error bound when the two products could cancel.
    if (detleft > 0.0 and detright <= 0.0) or (detleft < 0.0 and detright >= 0.0) or detleft == 0.0:
        return (det > 0.0) - (det < 0.0)
    summ = abs(detleft) + abs(detright)
    errbound = _O2_BOUND * summ
    if det > errbound or -det > errbound:
        return (det > 0.0) - (det < 0.0)
    return _orient2d_exact(a, b, c)


# --------------------------------------------------------------------------
# orient3d
# --------------------------------------------------------------------------


def _orient3d_exact(a: Point3, b: Point3, c: Point3, d: Point3) -> int:
    ax, ay, az = Fraction(a[0]), Fraction(a[1]), Fraction(a[2])
    bx, by, bz = Fraction(b[0]), Fraction(b[1]), Fraction(b[2])
    cx, cy, cz = Fraction(c[0]), Fraction(c[1]), Fraction(c[2])
    dx, dy, dz = Fraction(d[0]), Fraction(d[1]), Fraction(d[2])
    adx, ady, adz = ax - dx, ay - dy, az - dz
    bdx, bdy, bdz = bx - dx, by - dy, bz - dz
    cdx, cdy, cdz = cx - dx, cy - dy, cz - dz
    det = (
        adx * (bdy * cdz - bdz * cdy)
        - bdx * (ady * cdz - adz * cdy)
        + cdx * (ady * bdz - adz * bdy)
    )
    return (det > 0) - (det < 0)


def orient3d(a: Point3, b: Point3, c: Point3, d: Point3) -> int:
    """Exact sign of the 3D orientation determinant.

    Positive iff ``d`` lies on the negative side of the plane through
    ``a, b, c`` taken counter-clockwise (right-hand rule) -- equivalently, the
    signed volume of the tetrahedron ``(a, b, c, d)`` has the opposite sign of
    the returned value's convention: it matches the standard ``orient3d``.
    """
    adx, ady, adz = a[0] - d[0], a[1] - d[1], a[2] - d[2]
    bdx, bdy, bdz = b[0] - d[0], b[1] - d[1], b[2] - d[2]
    cdx, cdy, cdz = c[0] - d[0], c[1] - d[1], c[2] - d[2]

    bdxcdy = bdx * cdy
    cdxbdy = cdx * bdy
    cdxady = cdx * ady
    adxcdy = adx * cdy
    adxbdy = adx * bdy
    bdxady = bdx * ady

    det = (
        adz * (bdxcdy - cdxbdy)
        + bdz * (cdxady - adxcdy)
        + cdz * (adxbdy - bdxady)
    )
    permanent = (
        (abs(bdxcdy) + abs(cdxbdy)) * abs(adz)
        + (abs(cdxady) + abs(adxcdy)) * abs(bdz)
        + (abs(adxbdy) + abs(bdxady)) * abs(cdz)
    )
    errbound = _O3_BOUND * permanent
    if det > errbound or -det > errbound:
        return (det > 0.0) - (det < 0.0)
    return _orient3d_exact(a, b, c, d)


# --------------------------------------------------------------------------
# incircle
# --------------------------------------------------------------------------


def _incircle_exact(a: Point2, b: Point2, c: Point2, d: Point2) -> int:
    ax, ay = Fraction(a[0]) - Fraction(d[0]), Fraction(a[1]) - Fraction(d[1])
    bx, by = Fraction(b[0]) - Fraction(d[0]), Fraction(b[1]) - Fraction(d[1])
    cx, cy = Fraction(c[0]) - Fraction(d[0]), Fraction(c[1]) - Fraction(d[1])
    alift = ax * ax + ay * ay
    blift = bx * bx + by * by
    clift = cx * cx + cy * cy
    det = (
        alift * (bx * cy - cx * by)
        - blift * (ax * cy - cx * ay)
        + clift * (ax * by - bx * ay)
    )
    return (det > 0) - (det < 0)


def incircle(a: Point2, b: Point2, c: Point2, d: Point2) -> int:
    """Exact in-circle test.  +1 iff ``d`` is inside the circumcircle of the
    CCW triangle ``a, b, c``."""
    adx, ady = a[0] - d[0], a[1] - d[1]
    bdx, bdy = b[0] - d[0], b[1] - d[1]
    cdx, cdy = c[0] - d[0], c[1] - d[1]

    bdxcdy = bdx * cdy
    cdxbdy = cdx * bdy
    alift = adx * adx + ady * ady
    cdxady = cdx * ady
    adxcdy = adx * cdy
    blift = bdx * bdx + bdy * bdy
    adxbdy = adx * bdy
    bdxady = bdx * ady
    clift = cdx * cdx + cdy * cdy

    det = (
        alift * (bdxcdy - cdxbdy)
        + blift * (cdxady - adxcdy)
        + clift * (adxbdy - bdxady)
    )
    permanent = (
        (abs(bdxcdy) + abs(cdxbdy)) * alift
        + (abs(cdxady) + abs(adxcdy)) * blift
        + (abs(adxbdy) + abs(bdxady)) * clift
    )
    errbound = _INCIRCLE_BOUND * permanent
    if det > errbound or -det > errbound:
        return (det > 0.0) - (det < 0.0)
    return _incircle_exact(a, b, c, d)


# --------------------------------------------------------------------------
# insphere
# --------------------------------------------------------------------------


def _insphere_exact(a: Point3, b: Point3, c: Point3, d: Point3, e: Point3) -> int:
    # Mirror the float formula exactly with rational arithmetic so the sign
    # convention is identical to the fast path.
    def sub(p, q):
        return (Fraction(p[0]) - Fraction(q[0]),
                Fraction(p[1]) - Fraction(q[1]),
                Fraction(p[2]) - Fraction(q[2]))

    aex, aey, aez = sub(a, e)
    bex, bey, bez = sub(b, e)
    cex, cey, cez = sub(c, e)
    dex, dey, dez = sub(d, e)

    ab = aex * bey - bex * aey
    bc = bex * cey - cex * bey
    cd = cex * dey - dex * cey
    da = dex * aey - aex * dey
    ac = aex * cey - cex * aey
    bd = bex * dey - dex * bey

    abc = aez * bc - bez * ac + cez * ab
    bcd = bez * cd - cez * bd + dez * bc
    cda = cez * da + dez * ac + aez * cd
    dab = dez * ab + aez * bd + bez * da

    alift = aex * aex + aey * aey + aez * aez
    blift = bex * bex + bey * bey + bez * bez
    clift = cex * cex + cey * cey + cez * cez
    dlift = dex * dex + dey * dey + dez * dez

    det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd)
    return (det > 0) - (det < 0)


def insphere(a: Point3, b: Point3, c: Point3, d: Point3, e: Point3) -> int:
    """Exact in-sphere test.

    Returns +1 iff ``e`` lies inside the sphere through ``a, b, c, d`` when
    those four are positively oriented (``orient3d(a, b, c, d) > 0``).  The
    float fast path uses a conservative permanent bound; ties fall back to the
    exact Fraction determinant.
    """
    aex, aey, aez = a[0] - e[0], a[1] - e[1], a[2] - e[2]
    bex, bey, bez = b[0] - e[0], b[1] - e[1], b[2] - e[2]
    cex, cey, cez = c[0] - e[0], c[1] - e[1], c[2] - e[2]
    dex, dey, dez = d[0] - e[0], d[1] - e[1], d[2] - e[2]

    ab = aex * bey - bex * aey
    bc = bex * cey - cex * bey
    cd = cex * dey - dex * cey
    da = dex * aey - aex * dey
    ac = aex * cey - cex * aey
    bd = bex * dey - dex * bey

    abc = aez * bc - bez * ac + cez * ab
    bcd = bez * cd - cez * bd + dez * bc
    cda = cez * da + dez * ac + aez * cd
    dab = dez * ab + aez * bd + bez * da

    alift = aex * aex + aey * aey + aez * aez
    blift = bex * bex + bey * bey + bez * bez
    clift = cex * cex + cey * cey + cez * cez
    dlift = dex * dex + dey * dey + dez * dez

    det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd)

    permanent = (abs(abc) + abs(bcd) + abs(cda) + abs(dab)) * (
        alift + blift + clift + dlift
    )
    errbound = _INSPHERE_BOUND * permanent
    if det > errbound or -det > errbound:
        return (det > 0.0) - (det < 0.0)
    return _insphere_exact(a, b, c, d, e)
