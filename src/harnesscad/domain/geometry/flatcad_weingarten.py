"""Weingarten shape operator & curvature of an implicit SDF surface.

Deterministic differential geometry from **FlatCAD: Fast Curvature
Regularization of Neural SDFs for CAD Models** (Yin, Plocharski, Wlodarczyk,
Kida & Musialski).  FlatCAD trains a neural SDF -- that is research-heavy and
external -- but the *curvature math* it builds on is closed-form and fully
deterministic.  This module implements that math for any signed-distance field
represented by its gradient ``g = grad f`` and Hessian ``H = H_f``:

* **Shape operator / Weingarten map** (paper Sec. 3.1): projecting the Hessian
  into an orthonormal tangent frame ``(u, v)`` and dividing by ``||grad f||``
  gives the symmetric 2x2 operator ``S`` whose eigenvalues are the principal
  curvatures ``k1, k2``.
* **Principal / mean / Gaussian curvature** via the closed-form implicit-surface
  (Goldman) formulas -- Gaussian ``K = g . adj(H) g / |g|^4`` and mean
  ``H_mean = (|g|^2 tr H - g^T H g) / (2 |g|^3)`` -- so no tangent frame is
  needed.  Validated: sphere ``|x|-r`` gives ``H_mean = 1/r``, ``K = 1/r^2``;
  plane gives both zero.
* **Off-diagonal Weingarten term** ``S12 = u^T H v / |grad f|`` (Eq. 1): the
  "curvature gap" warp measure, ``S12(theta) = 1/2 (k2 - k1) sin 2 theta``.
* **Random / deterministic tangent frames** orthogonal to the surface normal
  (the resampled frame angle ``theta ~ U[0, 2 pi)`` of the paper's Monte-Carlo
  estimator), seeded via ``random.Random`` for reproducibility.
* **ODW loss** (Eqs. 8): the mean absolute (L1) or squared (L2) off-diagonal
  Weingarten term over a batch of samples, plus the closed-form expectations
  ``E[S12^2] = (k2-k1)^2 / 8`` (Eq. 2) and ``E[|S12|] = |k2-k1| / pi`` (Eq. 3).
* **Curvature-regime classification** (Fig. 2): planar / parabolic / elliptic /
  hyperbolic / umbilic from the principal-curvature pair.

stdlib-only, deterministic.  The finite-difference route to ``S12`` from raw
SDF queries lives in :mod:`numeric.flatcad_sdf_derivatives`.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Sequence[float], Sequence[float], Sequence[float]]


# --------------------------------------------------------------------------- #
# small vector helpers
# --------------------------------------------------------------------------- #
def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _normalize(a: Sequence[float]) -> Vec3:
    n = _norm(a)
    if n == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def _matvec(H: Mat3, v: Sequence[float]) -> Vec3:
    return (H[0][0] * v[0] + H[0][1] * v[1] + H[0][2] * v[2],
            H[1][0] * v[0] + H[1][1] * v[1] + H[1][2] * v[2],
            H[2][0] * v[0] + H[2][1] * v[1] + H[2][2] * v[2])


# --------------------------------------------------------------------------- #
# tangent frames
# --------------------------------------------------------------------------- #
def orthonormal_tangent_frame(normal: Sequence[float]) -> Tuple[Vec3, Vec3]:
    """A deterministic orthonormal tangent basis ``(u, v)`` orthogonal to
    ``normal`` (``normal`` need not be unit).  ``u, v, n`` is right-handed."""
    n = _normalize(normal)
    # pick a reference axis least parallel to n
    ax = min(range(3), key=lambda i: abs(n[i]))
    ref = (1.0 if ax == 0 else 0.0, 1.0 if ax == 1 else 0.0, 1.0 if ax == 2 else 0.0)
    u = _normalize(_cross(n, ref))
    v = _cross(n, u)
    return u, v


def rotate_frame(u: Vec3, v: Vec3, theta: float) -> Tuple[Vec3, Vec3]:
    """Rotate the tangent basis ``(u, v)`` by angle ``theta`` in-plane."""
    c, s = math.cos(theta), math.sin(theta)
    ur = (c * u[0] + s * v[0], c * u[1] + s * v[1], c * u[2] + s * v[2])
    vr = (-s * u[0] + c * v[0], -s * u[1] + c * v[1], -s * u[2] + c * v[2])
    return ur, vr


def random_tangent_frame(normal: Sequence[float],
                         rng: random.Random) -> Tuple[Vec3, Vec3]:
    """Random orthonormal tangent frame (angle ``theta ~ U[0, 2 pi)``).

    Reproduces the paper's per-iteration fresh-frame sampling.  ``rng`` must be
    a ``random.Random`` instance so results are deterministic given a seed.
    """
    u0, v0 = orthonormal_tangent_frame(normal)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return rotate_frame(u0, v0, theta)


# --------------------------------------------------------------------------- #
# shape operator & off-diagonal Weingarten term
# --------------------------------------------------------------------------- #
def shape_operator(grad: Sequence[float], hess: Mat3,
                   u: Vec3, v: Vec3) -> Tuple[Tuple[float, float],
                                              Tuple[float, float]]:
    """The 2x2 Weingarten map ``S = [[u^T H u, u^T H v],[v^T H u, v^T H v]]``
    divided by ``||grad f||`` (paper Sec. 3.1).  Symmetric because ``H`` is."""
    gn = _norm(grad)
    if gn == 0.0:
        raise ValueError("gradient norm is zero; not a valid SDF point")
    Hu = _matvec(hess, u)
    Hv = _matvec(hess, v)
    a = _dot(u, Hu) / gn
    b = _dot(u, Hv) / gn
    d = _dot(v, Hv) / gn
    return ((a, b), (b, d))


def off_diagonal_weingarten(grad: Sequence[float], hess: Mat3,
                            u: Vec3, v: Vec3) -> float:
    """``S12 = u^T H_f v / ||grad f||`` (Eq. 1): the curvature-gap warp term."""
    gn = _norm(grad)
    if gn == 0.0:
        raise ValueError("gradient norm is zero; not a valid SDF point")
    return _dot(u, _matvec(hess, v)) / gn


def s12_from_principal(k1: float, k2: float, theta: float) -> float:
    """Analytic off-diagonal entry in a frame rotated by ``theta`` from the
    principal frame: ``S12(theta) = 1/2 (k2 - k1) sin 2 theta`` (Eq. 1)."""
    return 0.5 * (k2 - k1) * math.sin(2.0 * theta)


def off_diagonal_weingarten_fd(f, x: Sequence[float], u: Vec3, v: Vec3,
                               h: float = 1e-3) -> float:
    """Finite-difference off-diagonal Weingarten term straight from an SDF
    callable ``f`` (FlatCAD Eq. 10): ``D^(c)_uv(x) / ||grad f(x)||``.

    Combines the symmetric mixed stencil and the central gradient from
    :mod:`numeric.flatcad_sdf_derivatives`; no autodiff, ``O(h^2)`` accurate.
    """
    from harnesscad.domain.numeric.flatcad_sdf_derivatives import (
        central_gradient, mixed_stencil_uv,
    )
    dc = mixed_stencil_uv(f, x, u, v, h)
    gn = _norm(central_gradient(f, x, h))
    if gn == 0.0:
        raise ValueError("gradient norm is zero; not a valid SDF point")
    return dc / gn


# --------------------------------------------------------------------------- #
# closed-form implicit-surface curvature (Goldman formulas)
# --------------------------------------------------------------------------- #
def _adjugate_sym(H: Mat3) -> Mat3:
    a, b, c = H[0][0], H[0][1], H[0][2]
    d, e = H[1][1], H[1][2]
    f = H[2][2]
    # cofactors of the symmetric matrix [[a,b,c],[b,d,e],[c,e,f]]
    c00 = d * f - e * e
    c01 = c * e - b * f
    c02 = b * e - d * c
    c11 = a * f - c * c
    c12 = b * c - a * e
    c22 = a * d - b * b
    return ((c00, c01, c02), (c01, c11, c12), (c02, c12, c22))


def gaussian_curvature(grad: Sequence[float], hess: Mat3) -> float:
    """Gaussian curvature ``K = g . adj(H) g / |g|^4`` (Goldman 2005).

    Sphere ``|x|-r`` -> ``1/r^2``; plane -> ``0``.
    """
    gn2 = _dot(grad, grad)
    if gn2 == 0.0:
        raise ValueError("gradient norm is zero; not a valid SDF point")
    adj = _adjugate_sym(hess)
    q = _dot(grad, _matvec(adj, grad))
    return q / (gn2 * gn2)


def mean_curvature(grad: Sequence[float], hess: Mat3) -> float:
    """Mean curvature ``H = (|g|^2 tr H - g^T H g) / (2 |g|^3)``.

    Signed so an outward-pointing SDF gradient yields a *positive* value for a
    convex surface: sphere ``|x|-r`` -> ``1/r``; plane -> ``0``.
    """
    gn2 = _dot(grad, grad)
    if gn2 == 0.0:
        raise ValueError("gradient norm is zero; not a valid SDF point")
    gn = math.sqrt(gn2)
    trH = hess[0][0] + hess[1][1] + hess[2][2]
    gHg = _dot(grad, _matvec(hess, grad))
    return (gn2 * trH - gHg) / (2.0 * gn2 * gn)


def principal_curvatures(grad: Sequence[float], hess: Mat3) -> Tuple[float, float]:
    """Principal curvatures ``(k1, k2)`` with ``k1 <= k2``.

    Derived from the mean ``H`` and Gaussian ``K`` curvatures via
    ``k = H +/- sqrt(max(H^2 - K, 0))``.
    """
    H = mean_curvature(grad, hess)
    K = gaussian_curvature(grad, hess)
    disc = max(H * H - K, 0.0)
    r = math.sqrt(disc)
    return (H - r, H + r)


# --------------------------------------------------------------------------- #
# ODW loss and its expectations
# --------------------------------------------------------------------------- #
def odw_loss_l1(samples: Sequence[Tuple[Sequence[float], Mat3, Vec3, Vec3]]) -> float:
    """Mean ``|S12|`` over samples (the paper's default L1 ODW loss, Eq. 8).

    Each sample is ``(grad, hess, u, v)``.
    """
    if not samples:
        raise ValueError("need at least one sample")
    return sum(abs(off_diagonal_weingarten(g, H, u, v))
               for g, H, u, v in samples) / len(samples)


def odw_loss_l2(samples: Sequence[Tuple[Sequence[float], Mat3, Vec3, Vec3]]) -> float:
    """Mean ``S12^2`` over samples (the L2 ODW loss variant)."""
    if not samples:
        raise ValueError("need at least one sample")
    return sum(off_diagonal_weingarten(g, H, u, v) ** 2
               for g, H, u, v in samples) / len(samples)


def expected_s12_squared(k1: float, k2: float) -> float:
    """``E_theta[S12^2] = (k2 - k1)^2 / 8`` (Eq. 2)."""
    return (k2 - k1) ** 2 / 8.0


def expected_abs_s12(k1: float, k2: float) -> float:
    """``E_theta[|S12|] = |k2 - k1| / pi`` (Eq. 3)."""
    return abs(k2 - k1) / math.pi


# --------------------------------------------------------------------------- #
# curvature-regime classification (Fig. 2)
# --------------------------------------------------------------------------- #
def classify_curvature(k1: float, k2: float, tol: float = 1e-6) -> str:
    """Classify a point by its principal curvatures.

    Returns one of ``"planar"`` (both ~0), ``"parabolic"`` (exactly one ~0),
    ``"umbilic"`` (equal non-zero -> spherical), ``"elliptic"`` (same sign,
    distinct), or ``"hyperbolic"`` (opposite signs -> saddle).
    """
    z1 = abs(k1) <= tol
    z2 = abs(k2) <= tol
    if z1 and z2:
        return "planar"
    if z1 or z2:
        return "parabolic"
    if abs(k1 - k2) <= tol:
        return "umbilic"
    if k1 * k2 < 0.0:
        return "hyperbolic"
    return "elliptic"
