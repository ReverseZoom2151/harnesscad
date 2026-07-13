"""SDF Boolean combinators and smooth blends, ported from Curv.

Curv composes shapes by combining their distance fields (``lib/curv/std.curv``
and ``lib/curv/lib/blend.curv``).  The hard set operations are the classic
min/max algebra; the smooth ("blended") variants replace min/max with a smooth
minimum so that adjacent shapes join with a fillet instead of a crease.

Hard operators (scalar distances ``a``, ``b`` of the argument fields):

* ``union``        = ``min(a, b)``
* ``intersection`` = ``max(a, b)``
* ``difference``   = ``max(a, -b)`` (i.e. ``intersection(a, complement b)``)
* ``complement``   = ``-a``

Smooth operators use a *smooth minimum* ``smin`` with radius/parameter ``k``:

* polynomial (quadratic) smin -- Curv's default "elliptic blend" (IQ);
* exponential smin (IQ) -- smoothly blends any number of fields;
* power smin (IQ) -- requires positive arguments.

Distance-field classes / Lipschitz caveats (from Curv's docs):

* Hard ``union`` of exact fields: the *exterior* stays exact but the *interior*
  becomes only approximate (``max`` of two exact interiors is exact, ``min`` of
  two exact exteriors is exact -- but the mixed regions degrade).  In practice
  ``min``/``max`` are exactly 1-Lipschitz (they never increase the gradient).
* Smooth blends are only *approximate* fields: the smoothing term
  ``- k*h*(1-h)`` slightly violates the Eikonal equation inside the blend band,
  so ``|grad|`` can exceed 1 (Curv recommends ``lipschitz`` compensation before
  meshing).  The bound is small for the polynomial blend and the maximum
  under-shoot never exceeds ``k/4`` (``smooth k .union [s, s] == offset (k/4) s``).
* ``smooth_union <= hard_union`` everywhere (the blend only ever *adds*
  material), and both blends are continuous.

All functions take/return plain floats; deterministic, stdlib-only.
"""

from __future__ import annotations

from math import exp, log


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# --------------------------------------------------------------------------- #
# hard Boolean operators                                                       #
# --------------------------------------------------------------------------- #
def union(a: float, b: float) -> float:
    """Set union: ``min(a, b)``."""
    return a if a < b else b


def intersection(a: float, b: float) -> float:
    """Set intersection: ``max(a, b)``."""
    return a if a > b else b


def complement(a: float) -> float:
    """Reverse inside/outside: ``-a`` (boundary unchanged)."""
    return -a


def difference(a: float, b: float) -> float:
    """Subtract ``b`` from ``a``: ``max(a, -b)``."""
    return intersection(a, complement(b))


def union_all(values) -> float:
    """N-ary union; identity is ``+inf`` (Curv's ``nothing``)."""
    result = float("inf")
    for v in values:
        if v < result:
            result = v
    return result


def intersection_all(values) -> float:
    """N-ary intersection; identity is ``-inf`` (Curv's ``everything``)."""
    result = float("-inf")
    for v in values:
        if v > result:
            result = v
    return result


# --------------------------------------------------------------------------- #
# smooth minima                                                                #
# --------------------------------------------------------------------------- #
def smooth_min_poly(a: float, b: float, k: float) -> float:
    """Polynomial (quadratic) smooth minimum -- Curv's ``smooth_min`` (IQ).

    ``h = clamp(0.5 + 0.5*(b - a)/k, 0, 1);  lerp(b, a, h) - k*h*(1 - h)``.
    ``k > 0`` is the blend radius.  As ``k -> 0`` it converges to ``min(a, b)``.
    Reference: iquilezles.org/articles/smin.
    """
    if k <= 0.0:
        return a if a < b else b
    if a == float("inf"):
        return b
    if b == float("inf"):
        return a
    h = _clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return _lerp(b, a, h) - k * h * (1.0 - h)


def smooth_max_poly(a: float, b: float, k: float) -> float:
    """Polynomial smooth maximum: ``-smooth_min_poly(-a, -b, k)``."""
    return -smooth_min_poly(-a, -b, k)


def smooth_min_exp(a: float, b: float, k: float) -> float:
    """Exponential smooth minimum (IQ).

    ``-log(exp(-a/k) + exp(-b/k)) * k`` with ``k > 0`` the smoothing amount.
    Unlike the polynomial form this generalises to N arguments associatively.
    """
    if k <= 0.0:
        return a if a < b else b
    # shift by the true min for numerical stability (does not change result)
    m = a if a < b else b
    return m - k * log(exp(-(a - m) / k) + exp(-(b - m) / k))


def smooth_min_power(a: float, b: float, k: float) -> float:
    """Power smooth minimum (IQ).  Requires ``a > 0`` and ``b > 0``.

    ``pow(a, k) ...`` blends only where both fields are positive; used mostly
    for the *exterior*.  ``k > 1`` sharpens.  Raises for non-positive inputs.
    """
    if k <= 0.0:
        return a if a < b else b
    if a <= 0.0 or b <= 0.0:
        raise ValueError("power smooth-min requires strictly positive arguments")
    ak = a ** k
    bk = b ** k
    return (ak * bk / (ak + bk)) ** (1.0 / k)


# --------------------------------------------------------------------------- #
# smooth Boolean operators (the "smooth r" blending kernel)                    #
# --------------------------------------------------------------------------- #
def smooth_union(a: float, b: float, k: float) -> float:
    """Blended union (elliptic blend).  ``<=`` hard union everywhere."""
    return smooth_min_poly(a, b, k)


def smooth_intersection(a: float, b: float, k: float) -> float:
    """Blended intersection: ``smooth_max_poly(a, b, k)``."""
    return smooth_max_poly(a, b, k)


def smooth_difference(a: float, b: float, k: float) -> float:
    """Blended difference: ``smooth_intersection(a, complement b, k)``."""
    return smooth_intersection(a, complement(b), k)


def chamfer_min(a: float, b: float, r: float) -> float:
    """Chamfer minimum (Curv/MERCURY): a 45-degree bevel of size ``r``.

    ``min(a, b) - 0.5*max(r - |a - b|, 0)``.
    """
    e = max(r - abs(a - b), 0.0)
    return min(a, b) - e * 0.5


def chamfer_union(a: float, b: float, r: float) -> float:
    """Chamfered union."""
    return chamfer_min(a, b, r)


def chamfer_intersection(a: float, b: float, r: float) -> float:
    """Chamfered intersection: ``-chamfer_min(-a, -b, r)``."""
    return -chamfer_min(-a, -b, r)
