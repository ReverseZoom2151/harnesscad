"""Finite-difference gradient / Hessian / mixed-stencil for a sampled SDF.

Deterministic differential operators distilled from **FlatCAD: Fast Curvature
Regularization of Neural SDFs for CAD Models** (Yin, Plocharski, Wlodarczyk,
Kida & Musialski, 2024/2025).  FlatCAD's *learned* contribution is a SIREN
network trained with an off-diagonal Weingarten penalty -- that training is
research-heavy and lives outside this repo.  What is fully deterministic is the
closed-form finite-difference machinery the paper uses to *evaluate* the
mixed second derivative of any signed-distance field ``f: R^3 -> R`` without
assembling a full Hessian.

This module implements those samplers for an arbitrary callable ``f``:

* ``central_gradient``   -- second-order central difference of grad f (Eq. for
  the unit-gradient / Eikonal quantity ||grad f||).
* ``central_hessian``    -- the full symmetric 3x3 Hessian by central
  differences (diagonal three-point stencil, off-diagonal four-point stencil).
* ``mixed_stencil_uv``   -- FlatCAD Section 4.1 symmetric mixed-difference
  stencil ``D^(c)_uv`` (Eq. 9): six SDF queries plus ``f00`` that converge to
  ``u^T H_f v`` with **O(h^2)** truncation error (the odd powers cancel).
* ``forward_mixed_uv`` / ``backward_mixed_uv`` -- the one-sided ``O(h)`` halves
  whose average is the symmetric stencil.

All quantities are validated against analytic SDFs (a sphere ``|x|-r`` and a
plane) in the tests.  stdlib-only, no randomness, no wall clock.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Vec3, Vec3, Vec3]
SDF = Callable[[float, float, float], float]


def _add(a: Sequence[float], b: Sequence[float], s: float = 1.0) -> Vec3:
    return (a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2])


def central_gradient(f: SDF, x: Sequence[float], h: float = 1e-4) -> Vec3:
    """Second-order central-difference gradient of ``f`` at ``x``.

    ``(f(x + h e_i) - f(x - h e_i)) / (2 h)`` per axis; truncation ``O(h^2)``.
    """
    if h <= 0.0:
        raise ValueError("step h must be positive")
    x = (float(x[0]), float(x[1]), float(x[2]))
    g = []
    for i in range(3):
        fp = f(*_add(x, _unit(i), h))
        fm = f(*_add(x, _unit(i), -h))
        g.append((fp - fm) / (2.0 * h))
    return (g[0], g[1], g[2])


def _unit(i: int) -> Vec3:
    return (1.0 if i == 0 else 0.0, 1.0 if i == 1 else 0.0, 1.0 if i == 2 else 0.0)


def central_hessian(f: SDF, x: Sequence[float], h: float = 1e-3) -> Mat3:
    """Full symmetric 3x3 Hessian of ``f`` by central differences.

    Diagonal: three-point stencil ``(f(x+h) - 2 f(x) + f(x-h)) / h^2``.
    Off-diagonal: the standard four-point mixed stencil
    ``(f(++) - f(+-) - f(-+) + f(--)) / (4 h^2)``.  Both are ``O(h^2)``.
    The returned matrix is symmetrised exactly (H[i][j] == H[j][i]).
    """
    if h <= 0.0:
        raise ValueError("step h must be positive")
    x = (float(x[0]), float(x[1]), float(x[2]))
    f0 = f(*x)
    hh = h * h
    H = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for i in range(3):
        fp = f(*_add(x, _unit(i), h))
        fm = f(*_add(x, _unit(i), -h))
        H[i][i] = (fp - 2.0 * f0 + fm) / hh
    for i in range(3):
        for j in range(i + 1, 3):
            ei, ej = _unit(i), _unit(j)
            fpp = f(*_add(_add(x, ei, h), ej, h))
            fpm = f(*_add(_add(x, ei, h), ej, -h))
            fmp = f(*_add(_add(x, ei, -h), ej, h))
            fmm = f(*_add(_add(x, ei, -h), ej, -h))
            val = (fpp - fpm - fmp + fmm) / (4.0 * hh)
            H[i][j] = val
            H[j][i] = val
    return ((H[0][0], H[0][1], H[0][2]),
            (H[1][0], H[1][1], H[1][2]),
            (H[2][0], H[2][1], H[2][2]))


def forward_mixed_uv(f: SDF, x: Sequence[float], u: Vec3, v: Vec3,
                     h: float = 1e-3, f00: float | None = None) -> float:
    """One-sided forward mixed difference ``D^(+)_uv`` (FlatCAD Sec. 4.1).

    ``(f(x+h(u+v)) - f(x+hu) - f(x+hv) + f00) / h^2 = u^T H v + O(h)``.
    """
    if h <= 0.0:
        raise ValueError("step h must be positive")
    x = (float(x[0]), float(x[1]), float(x[2]))
    if f00 is None:
        f00 = f(*x)
    fpu = f(*_add(x, u, h))
    fpv = f(*_add(x, v, h))
    fpuv = f(*_add(_add(x, u, h), v, h))
    return (fpuv - fpu - fpv + f00) / (h * h)


def backward_mixed_uv(f: SDF, x: Sequence[float], u: Vec3, v: Vec3,
                      h: float = 1e-3, f00: float | None = None) -> float:
    """One-sided backward mixed difference ``D^(-)_uv`` (FlatCAD Sec. 4.1)."""
    if h <= 0.0:
        raise ValueError("step h must be positive")
    x = (float(x[0]), float(x[1]), float(x[2]))
    if f00 is None:
        f00 = f(*x)
    fmu = f(*_add(x, u, -h))
    fmv = f(*_add(x, v, -h))
    fmuv = f(*_add(_add(x, u, -h), v, -h))
    return (fmuv - fmu - fmv + f00) / (h * h)


def mixed_stencil_uv(f: SDF, x: Sequence[float], u: Vec3, v: Vec3,
                     h: float = 1e-3) -> float:
    """FlatCAD symmetric mixed stencil ``D^(c)_uv`` (Eq. 9).

    ``0.5 (D^(+)_uv + D^(-)_uv) = u^T H_f(x) v + O(h^2)``.  Uses six SDF
    queries plus one shared ``f00`` -- seven total -- and no second-order
    autodiff graph.  The symmetric average cancels the odd-power error terms,
    upgrading the accuracy from ``O(h)`` to ``O(h^2)``.
    """
    x = (float(x[0]), float(x[1]), float(x[2]))
    f00 = f(*x)
    dp = forward_mixed_uv(f, x, u, v, h, f00)
    dm = backward_mixed_uv(f, x, u, v, h, f00)
    return 0.5 * (dp + dm)
