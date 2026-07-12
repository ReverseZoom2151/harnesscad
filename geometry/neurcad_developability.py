"""Zero-Gaussian-curvature developability energy for CAD SDFs.

Deterministic differential geometry from **NeurCADRecon: Neural Representation
for Reconstructing CAD Surfaces by Enforcing Zero Gaussian Curvature** (Dong,
Xu, Wang, Chen, Xin, Jia, Wang & Tu, ACM TOG 2024).  The neural SDF *training*
is external, but the geometric machinery the paper builds on is closed-form:

* **Developability energy** (Eqs. 5, 7): a CAD surface is piecewise smooth with
  each patch *approximately developable*, and developability is equivalent to
  *zero Gaussian curvature*.  The paper therefore minimises the overall absolute
  Gaussian curvature ``L_Gauss = (1/|Omega|) sum |k_Gauss(x)|`` over a batch of
  samples.  This is distinct from FlatCAD's off-diagonal-Weingarten flatness
  term (which penalises the *warp* ``S12``, not ``K`` itself).
* **Double-trough function** (Eqs. 8, 9): the Gaussian curvature at a *tip /
  corner* point is non-zero (approximately ``pi/2``), so a universal ``K=0``
  constraint bulges those points.  The paper maps ``K`` through a quartic
  ``DT(t)`` with troughs at ``0`` and ``pi/2`` and a peak at ``pi/4``, so the
  desired ``K`` polarises toward ``0`` **or** ``~pi/2``.  ``DT`` is fixed by five
  interpolation conditions (Eq. 8) and solved here from that linear system.
* **Annealing factor** (Sec. 4.1): the developability weight ``tau`` holds at 1
  for the first 20% of iterations, decays linearly to ``1e-4`` over 20%-50%, and
  drops to 0 by the end -- gradually releasing the prior so non-developable
  patches (e.g. a sphere) keep fidelity.
* **Surface projection** (Eq. 12): the Newton pull ``x' = x - (grad f/|grad f|)
  * f(x)`` that projects a query onto the current zero level-set for dynamic
  sampling.

The Gaussian-curvature evaluation itself is *reused* from
:mod:`geometry.flatcad_weingarten` (Goldman formula) -- not reimplemented.

stdlib-only, deterministic (no wall clock, no global RNG state).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from geometry.flatcad_weingarten import gaussian_curvature

Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Sequence[float], Sequence[float], Sequence[float]]
Sample = Tuple[Sequence[float], Mat3]  # (grad, hess)


# --------------------------------------------------------------------------- #
# developability energy (Eqs. 5, 7)
# --------------------------------------------------------------------------- #
def point_developability_defect(grad: Sequence[float], hess: Mat3) -> float:
    """Absolute Gaussian curvature ``|k_Gauss(x)|`` at one point.

    Zero exactly when the point is (locally) developable.  Uses the Goldman
    formula from :mod:`geometry.flatcad_weingarten`.
    """
    return abs(gaussian_curvature(grad, hess))


def developability_energy(samples: Sequence[Sample]) -> float:
    """Overall absolute Gaussian curvature ``(1/N) sum |k_Gauss(x)|`` (Eq. 5/7).

    Each sample is ``(grad, hess)``.  This is NeurCADRecon's core developability
    loss: it is minimised toward 0 on piecewise-developable CAD surfaces.
    """
    if not samples:
        raise ValueError("need at least one sample")
    return sum(point_developability_defect(g, H) for g, H in samples) / len(samples)


def developability_energy_squared(samples: Sequence[Sample]) -> float:
    """Mean squared Gaussian curvature ``(1/N) sum k_Gauss^2`` (L2 variant)."""
    if not samples:
        raise ValueError("need at least one sample")
    return sum(gaussian_curvature(g, H) ** 2 for g, H in samples) / len(samples)


# --------------------------------------------------------------------------- #
# double-trough function (Eqs. 8, 9)
# --------------------------------------------------------------------------- #
def _solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve ``A x = b`` for a small dense system via Gaussian elimination with
    partial pivoting.  Deterministic; used to fit the quartic coefficients."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-15:
            raise ValueError("singular system")
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pv
        for r in range(n):
            if r != col:
                factor = m[r][col]
                for j in range(col, n + 1):
                    m[r][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def double_trough_coeffs(a: float = 0.25) -> Tuple[float, float, float, float]:
    """Coefficients ``(c1, c2, c3, c4)`` of the double-trough quartic
    ``DT(t) = c1 t + c2 t^2 + c3 t^3 + c4 t^4`` (``c0 = 0`` since ``DT(0)=0``).

    Fixed by the four remaining interpolation conditions of Eq. 8:
    ``DT(pi/4)=pi/4``, ``DT'(pi/4)=0``, ``DT(pi/2)=a``, ``DT'(pi/2)=0``.
    ``a`` is the tolerated trough height at ``pi/2`` (default ``1/4``).
    """
    t1 = math.pi / 4.0
    t2 = math.pi / 2.0
    # rows: value/deriv at t1, value/deriv at t2, in unknowns c1..c4
    A = [
        [t1, t1 ** 2, t1 ** 3, t1 ** 4],
        [1.0, 2.0 * t1, 3.0 * t1 ** 2, 4.0 * t1 ** 3],
        [t2, t2 ** 2, t2 ** 3, t2 ** 4],
        [1.0, 2.0 * t2, 3.0 * t2 ** 2, 4.0 * t2 ** 3],
    ]
    rhs = [t1, 0.0, a, 0.0]
    c1, c2, c3, c4 = _solve_linear(A, rhs)
    return c1, c2, c3, c4


def double_trough(t: float, a: float = 0.25) -> float:
    """Double-trough map ``DT(t)`` (Eq. 9).  ``t`` is a Gaussian curvature value.

    Troughs (minima) at ``t=0`` and ``t=pi/2`` with a peak at ``t=pi/4``: it
    polarises curvature toward developable (``0``) or corner (``~pi/2``).
    """
    c1, c2, c3, c4 = double_trough_coeffs(a)
    return c1 * t + c2 * t * t + c3 * t ** 3 + c4 * t ** 4


def double_trough_deriv(t: float, a: float = 0.25) -> float:
    """Derivative ``DT'(t)`` of the double-trough quartic."""
    c1, c2, c3, c4 = double_trough_coeffs(a)
    return c1 + 2.0 * c2 * t + 3.0 * c3 * t * t + 4.0 * c4 * t ** 3


def developability_energy_double_trough(samples: Sequence[Sample],
                                        a: float = 0.25) -> float:
    """Tip-tolerant developability energy ``(1/N) sum DT(|k_Gauss(x)|)`` (Eq. 10).

    Applies the double-trough curve to the absolute Gaussian curvature so corner
    points (``|K| ~ pi/2``) are tolerated rather than forced flat.
    """
    if not samples:
        raise ValueError("need at least one sample")
    return sum(double_trough(point_developability_defect(g, H), a)
               for g, H in samples) / len(samples)


# --------------------------------------------------------------------------- #
# annealing schedule (Sec. 4.1)
# --------------------------------------------------------------------------- #
def annealing_factor(progress: float,
                     tau_hold: float = 1.0,
                     tau_mid: float = 1e-4,
                     hold_end: float = 0.2,
                     decay_end: float = 0.5) -> float:
    """Annealing factor ``tau`` for the developability weight (Sec. 4.1).

    ``progress`` is the fraction of training completed in ``[0, 1]``:

    * ``[0, hold_end]``      -> ``tau_hold`` (default 1.0),
    * ``[hold_end, decay_end]`` -> linear ``tau_hold`` -> ``tau_mid`` (1e-4),
    * ``[decay_end, 1]``     -> linear ``tau_mid`` -> ``0``.
    """
    p = min(max(progress, 0.0), 1.0)
    if p <= hold_end:
        return tau_hold
    if p <= decay_end:
        frac = (p - hold_end) / (decay_end - hold_end)
        return tau_hold + frac * (tau_mid - tau_hold)
    frac = (p - decay_end) / (1.0 - decay_end)
    return tau_mid + frac * (0.0 - tau_mid)


def annealed_developability_weight(progress: float,
                                   lambda_gauss: float = 10.0,
                                   **kw) -> float:
    """Effective developability weight ``tau * lambda_Gauss`` at ``progress``
    (Eq. 11).  ``lambda_gauss`` defaults to the paper's value of 10."""
    return annealing_factor(progress, **kw) * lambda_gauss


# --------------------------------------------------------------------------- #
# dynamic-sampling surface projection (Eq. 12)
# --------------------------------------------------------------------------- #
def surface_projection(point: Sequence[float],
                       grad: Sequence[float],
                       value: float) -> Vec3:
    """Newton pull of ``point`` onto the zero level-set (Eq. 12):

    ``x' = x - (grad f / ||grad f||) * f(x)``.

    Used by NeurCADRecon's dynamic sampling to place fresh developability
    samples near the current surface.  Exact for a true SDF (``||grad f||=1``).
    """
    gn = math.sqrt(grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2)
    if gn == 0.0:
        raise ValueError("gradient norm is zero; cannot project")
    step = value / gn
    return (point[0] - grad[0] / gn * step,
            point[1] - grad[1] / gn * step,
            point[2] - grad[2] / gn * step)
