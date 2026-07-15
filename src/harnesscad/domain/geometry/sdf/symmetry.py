"""Arbitrary-hyperplane mirror and geometric-mean scale, from ImplicitCAD.

Two domain/field operators taken from ImplicitCAD's shared object handling
(``Graphics/Implicit/ObjectUtil/GetImplicitShared.hs`` and ``MathUtil.hs``) that
the harness's :mod:`field_transforms` does not already provide.

**Arbitrary-hyperplane reflection.**  :mod:`field_transforms` only mirrors across
the axis-aligned YZ plane (``mirror_x`` / ``reflect_x``).  ImplicitCAD's
``Mirror v`` reflects across a hyperplane through the origin with an *arbitrary*
normal ``v`` using the Householder reflection
``reflect a v = v - 2*(v . a / a . a)*a`` (``MathUtil.reflect``).  A reflection is
an isometry, so the reflected field ``f(reflect(n, p))`` is a valid SDF with no
distance compensation needed.

**Geometric-mean anisotropic scale.**  :mod:`field_transforms` scales
anisotropically with ``f(p/v) * min(v)`` (a conservative 1-Lipschitz *lower*
bound).  ImplicitCAD's ``Scale s`` instead multiplies the field by
``normalize s = |prod(s_i)|**(1/n)`` -- the **geometric mean** of the per-axis
factors (``GetImplicitShared.normalize``).  The geometric mean tracks the
volumetric scale of the transform, so a shape scaled by ``s`` reports distances
consistent with its new size on average rather than being clamped to the
smallest axis; it is the exact Euclidean factor when the scale is isotropic.
Both variants are approximate for a genuinely anisotropic ``s`` (no single
scalar can restore the Eikonal property under a non-uniform stretch); they
differ only in which scalar they choose, and this module offers ImplicitCAD's.

All functions are pure, operate on point tuples / field callables, stdlib-only,
deterministic.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple


def reflect_point(normal: Sequence[float], p: Sequence[float]) -> Tuple[float, ...]:
    """Reflect point ``p`` across the origin hyperplane with the given ``normal``.

    Householder reflection ``p - 2*(p . n / n . n)*n`` (ImplicitCAD
    ``MathUtil.reflect``).  ``normal`` need not be unit length.  Raises for a
    zero normal (no hyperplane is defined).
    """
    nn = sum(c * c for c in normal)
    if nn == 0.0:
        raise ValueError("mirror normal must be non-zero")
    dot = sum(p[i] * normal[i] for i in range(len(normal)))
    k = 2.0 * dot / nn
    return tuple(p[i] - k * normal[i] for i in range(len(normal)))


def mirror(f: Callable, normal: Sequence[float]):
    """Return the field ``f`` mirrored across the origin hyperplane ``normal``.

    ``p -> f(reflect_point(normal, p))``.  A reflection is an isometry, so the
    result is a valid SDF with the same distance class as ``f`` (no
    compensation).  Reproduces ImplicitCAD's ``Mirror`` for an arbitrary normal.
    """
    n = tuple(float(c) for c in normal)

    def g(p):
        return f(reflect_point(n, p))

    return g


def geometric_mean_scale(factors: Sequence[float]) -> float:
    """ImplicitCAD's ``normalize``: geometric-mean magnitude of ``factors``.

    ``|prod(factors)|**(1/n)``.  This is the scalar ImplicitCAD multiplies a
    field by after an anisotropic ``Scale``; it equals the common factor when the
    scale is isotropic.  Raises for an empty ``factors``.
    """
    n = len(factors)
    if n == 0:
        raise ValueError("need at least one scale factor")
    prod = 1.0
    for c in factors:
        prod *= c
    return abs(prod) ** (1.0 / n)


def scale_geometric(f: Callable, factors: Sequence[float]):
    """Anisotropic scale with ImplicitCAD's geometric-mean compensation.

    ``p -> normalize(factors) * f(p_i / factors_i)`` where ``normalize`` is the
    geometric mean (:func:`geometric_mean_scale`).  Distinct from
    :func:`field_transforms.stretch`, which compensates with ``min(factors)``.
    Every factor must be non-zero.
    """
    v = tuple(float(c) for c in factors)
    if any(c == 0.0 for c in v):
        raise ValueError("scale factors must be non-zero")
    comp = geometric_mean_scale(v)

    def g(p):
        return comp * f(tuple(p[i] / v[i] for i in range(len(v))))

    return g
