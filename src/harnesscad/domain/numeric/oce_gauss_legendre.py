"""Gauss-Legendre quadrature nodes and weights (OCCT ``math`` package).

Open CASCADE Technology (OCE / OCCT), ``src/FoundationClasses/TKMath/math/
math.cxx``.  OCCT's ``math`` toolkit ships a hard-coded table of Gauss-Legendre
quadrature abscissae (``Point[]``) and weights (``Weight[]``) for every order
``N = 1 .. 61`` (``math::GaussPointsMax() == 61``).  These are the reference
sample points OCCT uses throughout its integration machinery
(``math_GaussSingleIntegration``, ``math_GaussMultipleIntegration``,
``BSplCLib`` surface-area / mass-property integrals, etc.).  Exploiting the
symmetry of the Legendre roots about the origin, the table stores only the
non-negative half of each rule.

The harness has extensive NURBS / Bernstein machinery (``numeric.nurbs_basis``,
``geometry.dreamcad_rational_bezier``, ``geometry.nurbgen_*``) but **no**
numerical-quadrature primitive at all -- no way to integrate a function, and no
Gauss-Legendre node/weight table.  This module fills that gap.

Rather than transcribe OCCT's 61-order static table verbatim, the nodes and
weights are *generated* deterministically -- the roots of the degree-``n``
Legendre polynomial found by Newton's method, with the classic weight formula

    w_i = 2 / ((1 - x_i^2) * P_n'(x_i)^2)

The generator reproduces OCCT's published table to full double precision (the
unit tests check the low-order rows against the exact values copied out of
``math.cxx``).  Everything here is pure-Python stdlib and deterministic (a
fixed initial guess ``x = cos(pi * (k - 1/4) / (n + 1/2))`` -- no randomness,
no wall clock).

Public API
----------
  * :func:`legendre_p`         -- P_n(x) and P_n'(x) via the three-term recurrence.
  * :func:`nodes_and_weights`  -- the ``n``-point rule on the reference [-1, 1].
  * :func:`gauss_points_max`   -- OCCT's cap (61); rules above it still generate.
  * :func:`integrate`          -- definite integral of ``f`` on ``[a, b]``.
  * :func:`integrate_2d`       -- tensor-product rule on a rectangle.
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

# OCCT math::GaussPointsMax() -- the largest tabulated rule in math.cxx.
_OCCT_GAUSS_POINTS_MAX = 61


def gauss_points_max() -> int:
    """Return OCCT's tabulated maximum rule order (``math::GaussPointsMax()``).

    This module can generate rules of any positive order; the value is exposed
    only to mirror OCCT's documented limit.
    """
    return _OCCT_GAUSS_POINTS_MAX


def legendre_p(n: int, x: float) -> Tuple[float, float]:
    """Evaluate the Legendre polynomial ``P_n(x)`` and its derivative.

    Uses the numerically stable three-term recurrence

        (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)

    and the derivative identity

        (1 - x^2) P_n'(x) = n (P_{n-1}(x) - x P_n(x)).

    Returns ``(P_n(x), P_n'(x))``.
    """
    if n < 0:
        raise ValueError("degree n must be non-negative")
    if n == 0:
        return 1.0, 0.0
    p_prev = 1.0        # P_0
    p_curr = x          # P_1
    for k in range(1, n):
        p_next = ((2 * k + 1) * x * p_curr - k * p_prev) / (k + 1)
        p_prev, p_curr = p_curr, p_next
    # derivative from the closed-form identity (avoids the 1-x^2 singularity
    # blow-up because callers only evaluate at interior roots).
    denom = x * x - 1.0
    if denom == 0.0:
        # x == +-1: P_n'(+-1) = (+-1)^(n-1) * n (n+1) / 2
        dp = (n * (n + 1) / 2.0) * (1.0 if (x > 0 or n % 2 == 1) else -1.0)
    else:
        dp = n * (x * p_curr - p_prev) / denom
    return p_curr, dp


def nodes_and_weights(n: int) -> Tuple[List[float], List[float]]:
    """Return the ``n``-point Gauss-Legendre rule on the reference ``[-1, 1]``.

    Produces ``(nodes, weights)``, each a length-``n`` list, with ``nodes``
    sorted ascending.  The rule integrates polynomials up to degree ``2n - 1``
    exactly.  Equivalent to OCCT's ``Point``/``Weight`` rows (which store only
    the non-negative half by symmetry).
    """
    if n < 1:
        raise ValueError("number of points n must be >= 1")
    if n == 1:
        return [0.0], [2.0]

    nodes = [0.0] * n
    weights = [0.0] * n
    m = (n + 1) // 2  # only half the roots need solving; the rest mirror.
    for i in range(m):
        # Initial guess: asymptotic location of the (i+1)-th root.
        x = math.cos(math.pi * (i + 0.75) / (n + 0.5))
        # Newton iteration on P_n(x) = 0.
        for _ in range(100):
            p, dp = legendre_p(n, x)
            dx = -p / dp
            x += dx
            if abs(dx) <= 1e-15 * (abs(x) + 1e-300):
                break
        _, dp = legendre_p(n, x)
        w = 2.0 / ((1.0 - x * x) * dp * dp)
        # Roots are symmetric about 0: place +x and -x.
        nodes[i] = -x
        nodes[n - 1 - i] = x
        weights[i] = w
        weights[n - 1 - i] = w
    if n % 2 == 1:
        nodes[m - 1] = 0.0  # exact centre node for odd rules
    return nodes, weights


def integrate(func: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Definite integral of ``func`` on ``[a, b]`` by the ``n``-point rule.

    Maps the reference rule from ``[-1, 1]`` onto ``[a, b]`` with the affine
    change of variable ``x = (b-a)/2 * t + (a+b)/2`` (Jacobian ``(b-a)/2``).
    Exact for polynomials of degree up to ``2n - 1``.
    """
    nodes, weights = nodes_and_weights(n)
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    total = 0.0
    for t, w in zip(nodes, weights):
        total += w * func(half * t + mid)
    return half * total


def integrate_2d(
    func: Callable[[float, float], float],
    ax: float,
    bx: float,
    ay: float,
    by: float,
    nx: int,
    ny: int,
) -> float:
    """Integrate ``func(x, y)`` over the rectangle ``[ax,bx] x [ay,by]``.

    Tensor product of an ``nx``-point rule in ``x`` and an ``ny``-point rule in
    ``y`` -- the same construction OCCT's ``math_GaussMultipleIntegration``
    uses for separable domains.
    """
    xn, xw = nodes_and_weights(nx)
    yn, yw = nodes_and_weights(ny)
    hx, mx = 0.5 * (bx - ax), 0.5 * (ax + bx)
    hy, my = 0.5 * (by - ay), 0.5 * (ay + by)
    total = 0.0
    for tx, wx in zip(xn, xw):
        x = hx * tx + mx
        row = 0.0
        for ty, wy in zip(yn, yw):
            row += wy * func(x, hy * ty + my)
        total += wx * row
    return hx * hy * total
