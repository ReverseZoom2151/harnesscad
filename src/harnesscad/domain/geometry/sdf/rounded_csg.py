"""Circular-arc rounded Boolean CSG.

Rounded CSG combines shapes with a *rounded* max/min rather than the plain
``max``/``min`` of hard CSG.  Where ``max(x, y) = 0`` has a sharp square corner,
``rmax r x y`` replaces that corner with a **quarter of a circle
of radius r** -- an exact circular fillet -- and ``rmin`` does the same for the
concave union corner. The ``R`` denotes a rounding radius.

The closed forms, for ``|x - y| < r``, are::

    rmax r x y = y - r*sin(pi/4 - asin((x - y)/(r*sqrt 2))) + r
    rmin r x y = y + r*sin(pi/4 + asin((x - y)/(r*sqrt 2))) - r

and plain ``max``/``min`` otherwise (and when ``r == 0``).

Why add this when there are already smooth blends
-------------------------------------------------
:mod:`combinators` already carries the *polynomial* / *exponential* /
*power* smooth minima and a 45-degree *chamfer*.  The ``rmax``/``rmin``
operators are none of those: the blend profile is a true **circular arc** of a specified
radius (the fillet radius is a real geometric radius, not a soft "k" parameter),
which is what a CAD ``fillet r`` / ``round r`` on a Boolean edge actually means.
It sits between the polynomial smin (cheap, radius-inexact) and the chamfer
(straight bevel): the fillet radius ``r`` is exactly the arc radius.

Caveats:

* ``rmax`` / ``rmin`` are **not associative**, so the n-ary forms
  (:func:`rmaximum` / :func:`rminimum`) sort the arguments and round only the
  extreme *pair*, leaving the rest as a hard extremum.
* Only valid for ``r >= 0``.  ``r == 0`` degrades to hard CSG.
* Like every fillet blend it perturbs the Eikonal property inside the blend band
  (``|grad|`` may slightly exceed 1 there); away from the band it is unchanged.

All functions take/return plain floats; deterministic, stdlib-only.
"""

from __future__ import annotations

from math import asin, pi, sin, sqrt
from typing import Sequence

__all__ = [
    "rmax",
    "rmin",
    "rmaximum",
    "rminimum",
    "rounded_union",
    "rounded_intersection",
    "rounded_difference",
    "rounded_complement",
]

_SQRT2 = sqrt(2.0)
_PI_4 = pi / 4.0


def rmax(r: float, x: float, y: float) -> float:
    """Rounded maximum: ``max(x, y)`` with a radius-``r`` circular fillet.

    Replaces the sharp corner of ``max`` with a quarter circle of radius ``r``
    when the two fields are within ``r`` of each other.  ``r == 0`` (or a gap
    ``>= r``) falls back to the exact hard ``max``.  Not associative.
    """
    if r < 0.0:
        raise ValueError("rounding radius must be non-negative")
    if r != 0.0 and abs(x - y) < r:
        return y - r * sin(_PI_4 - asin((x - y) / (r * _SQRT2))) + r
    return x if x > y else y


def rmin(r: float, x: float, y: float) -> float:
    """Rounded minimum: ``min(x, y)`` with a radius-``r`` circular fillet.

    The concave-corner analogue of :func:`rmax`; used for the rounded *union*.
    ``r == 0`` (or a gap ``>= r``) falls back to the exact hard ``min``.  Not
    associative.
    """
    if r < 0.0:
        raise ValueError("rounding radius must be non-negative")
    if r != 0.0 and abs(x - y) < r:
        return y + r * sin(_PI_4 + asin((x - y) / (r * _SQRT2))) - r
    return x if x < y else y


def rmaximum(r: float, values: Sequence[float]) -> float:
    """N-ary rounded maximum.

    Rounds only the two largest arguments and leaves the rest hard, because
    :func:`rmax` is not associative so only the dominant pair is rounded.
    Empty -> ``0.0`` (by convention); single -> itself.
    """
    vs = list(values)
    if not vs:
        return 0.0
    if len(vs) == 1:
        return vs[0]
    a, b = sorted(vs, reverse=True)[:2]
    return rmax(r, a, b)


def rminimum(r: float, values: Sequence[float]) -> float:
    """N-ary rounded minimum.

    Rounds only the two smallest arguments and leaves the rest hard.  Empty ->
    ``0.0``; single -> itself.
    """
    vs = list(values)
    if not vs:
        return 0.0
    if len(vs) == 1:
        return vs[0]
    a, b = sorted(vs)[:2]
    return rmin(r, a, b)


# --------------------------------------------------------------------------- #
# named Boolean operators (rounded union / intersection / difference)         #
# --------------------------------------------------------------------------- #
def rounded_union(r: float, a: float, b: float) -> float:
    """Filleted union: ``rmin(r, a, b)``."""
    return rmin(r, a, b)


def rounded_intersection(r: float, a: float, b: float) -> float:
    """Filleted intersection: ``rmax(r, a, b)``."""
    return rmax(r, a, b)


def rounded_complement(a: float) -> float:
    """Reverse inside/outside: ``-a`` (boundary unchanged)."""
    return -a


def rounded_difference(r: float, a: float, b: float) -> float:
    """Filleted difference ``a - b``: ``rmax(r, a, -b)``.

    The subtracted edge is rounded with a concave fillet of radius ``r`` -- the
    CAD "round the inside corner left by a cut" operation.
    """
    return rmax(r, a, -b)
