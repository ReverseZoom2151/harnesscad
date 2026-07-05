"""Deterministic forward math for a single 3D Gaussian (GaussianCAD / 3DGS).

GaussianCAD (Zhou et al.) represents a reconstructed CAD model as a set of 3D
Gaussians and renders them with 3D Gaussian Splatting (3DGS, Kerbl et al. 2023).
The *training* of the Gaussians is a learned optimisation (out of scope), but the
per-Gaussian *forward* operations are pure, deterministic geometry:

  * a Gaussian's world-space covariance from an anisotropic scale and a rotation
    quaternion,   Sigma = R S S^T R^T   (Kerbl et al. Eq. 6);
  * evaluating the (normalised or unnormalised) Gaussian density at a point;
  * projecting the Gaussian to a camera/view plane by a linear 2x3 projection
    P (for an orthographic view P selects the two in-plane axes), giving the
    marginal 2D Gaussian  (mu2d = P mu,  Sigma2d = P Sigma P^T);
  * the axis-aligned 2D *footprint* (splat bounding box) of the projected
    Gaussian at a chosen sigma radius, plus 2D density evaluation for the alpha
    contribution inside that footprint.

Everything here is closed-form (3x3 / 2x2 inverse and eigen), so it is exact and
reproducible. No learned model, no optimisation, no wall clock, no randomness.

Vectors are plain tuples; matrices are row-major tuples of tuples.
"""

from __future__ import annotations

from math import cos, exp, pi, sin, sqrt
from typing import Sequence, Tuple

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]
Mat3 = Tuple[Tuple[float, float, float], ...]
Mat2 = Tuple[Tuple[float, float], ...]


# --------------------------------------------------------------------------- #
# Quaternion -> rotation
# --------------------------------------------------------------------------- #
def normalize_quaternion(q: Sequence[float]) -> Tuple[float, float, float, float]:
    """Return the unit quaternion ``(w, x, y, z)`` (raises on zero norm)."""
    w, x, y, z = (float(c) for c in q)
    n = sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("quaternion has zero norm")
    return (w / n, x / n, y / n, z / n)


def quaternion_to_matrix(q: Sequence[float]) -> Mat3:
    """Rotation matrix of quaternion ``(w, x, y, z)`` (normalised internally)."""
    w, x, y, z = normalize_quaternion(q)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


# --------------------------------------------------------------------------- #
# Small linear-algebra helpers
# --------------------------------------------------------------------------- #
def mat3_mul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def mat3_transpose(a: Mat3) -> Mat3:
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))


def mat3_vec(a: Mat3, v: Sequence[float]) -> Vec3:
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))


def mat3_det(a: Mat3) -> float:
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def mat3_inverse(a: Mat3) -> Mat3:
    """Inverse of a 3x3 matrix (raises if singular)."""
    det = mat3_det(a)
    if abs(det) < 1e-18:
        raise ValueError("matrix is singular")
    c = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            i0, i1 = (k for k in range(3) if k != i)
            j0, j1 = (k for k in range(3) if k != j)
            minor = a[i0][j0] * a[i1][j1] - a[i0][j1] * a[i1][j0]
            c[j][i] = ((-1) ** (i + j)) * minor / det  # transpose of cofactors
    return tuple(tuple(row) for row in c)


def mat2_det(a: Mat2) -> float:
    return a[0][0] * a[1][1] - a[0][1] * a[1][0]


def mat2_inverse(a: Mat2) -> Mat2:
    det = mat2_det(a)
    if abs(det) < 1e-18:
        raise ValueError("matrix is singular")
    return ((a[1][1] / det, -a[0][1] / det), (-a[1][0] / det, a[0][0] / det))


# --------------------------------------------------------------------------- #
# 3D Gaussian: covariance from scale + rotation, and density
# --------------------------------------------------------------------------- #
def covariance_from_scale_rotation(scale: Sequence[float], quat: Sequence[float]) -> Mat3:
    """World covariance ``Sigma = R S S^T R^T`` (3DGS Eq. 6).

    ``scale`` are the per-axis standard deviations (must be positive) and ``quat``
    is the orientation quaternion. ``S`` is ``diag(scale)`` so ``S S^T`` is
    ``diag(scale**2)``; the result is symmetric positive-definite.
    """
    s = [float(v) for v in scale]
    if len(s) != 3 or any(v <= 0.0 for v in s):
        raise ValueError("scale must be three positive numbers")
    r = quaternion_to_matrix(quat)
    ss = ((s[0] * s[0], 0.0, 0.0), (0.0, s[1] * s[1], 0.0), (0.0, 0.0, s[2] * s[2]))
    return mat3_mul(mat3_mul(r, ss), mat3_transpose(r))


def mahalanobis_sq(point: Sequence[float], mean: Sequence[float], cov: Mat3) -> float:
    """Squared Mahalanobis distance ``d^T Sigma^-1 d`` (``d = point - mean``)."""
    d = tuple(float(point[i]) - float(mean[i]) for i in range(3))
    inv = mat3_inverse(cov)
    return sum(d[i] * inv[i][j] * d[j] for i in range(3) for j in range(3))


def evaluate_gaussian_3d(point: Sequence[float], mean: Sequence[float], cov: Mat3,
                         normalized: bool = False) -> float:
    """Evaluate the 3D Gaussian at ``point``.

    ``normalized=False`` (default) returns the *unnormalised* kernel
    ``exp(-1/2 d^T Sigma^-1 d)`` in ``[0, 1]`` — the shape 3DGS actually splats.
    ``normalized=True`` multiplies by ``1/((2 pi)^{3/2} sqrt(det Sigma))`` to give
    a proper probability density that integrates to one.
    """
    m = mahalanobis_sq(point, mean, cov)
    kernel = exp(-0.5 * m)
    if not normalized:
        return kernel
    det = mat3_det(cov)
    if det <= 0.0:
        raise ValueError("covariance must be positive-definite")
    return kernel / (((2 * pi) ** 1.5) * sqrt(det))


# --------------------------------------------------------------------------- #
# Projection of a Gaussian to a 2D view
# --------------------------------------------------------------------------- #
# Orthographic view selection matrices P (2x3): the two rows pick the in-plane
# axes. Conventions match drawings.creft_projection (Z-up, third-angle).
ORTHO_PROJECTIONS = {
    "front": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # h=X, v=Z (look along -Y)
    "top": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),    # h=X, v=Y (look along -Z)
    "side": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),   # h=Y, v=Z (look along -X)
}


def project_gaussian(mean: Sequence[float], cov: Mat3,
                     proj: Sequence[Sequence[float]]) -> Tuple[Vec2, Mat2]:
    """Project a 3D Gaussian through a linear 2x3 map ``P`` to a 2D Gaussian.

    Returns ``(mu2d, Sigma2d)`` with ``mu2d = P mu`` and ``Sigma2d = P Sigma P^T``
    — the exact marginal for a linear (orthographic) projection.
    """
    p = [[float(proj[i][j]) for j in range(3)] for i in range(2)]
    mu2d = tuple(sum(p[i][k] * float(mean[k]) for k in range(3)) for i in range(2))
    # Sigma2d = P Sigma P^T
    ps = [[sum(p[i][k] * cov[k][j] for k in range(3)) for j in range(3)] for i in range(2)]
    cov2d = tuple(
        tuple(sum(ps[i][k] * p[j][k] for k in range(3)) for j in range(2))
        for i in range(2)
    )
    return mu2d, cov2d


def project_gaussian_orthographic(mean: Sequence[float], cov: Mat3,
                                  view: str) -> Tuple[Vec2, Mat2]:
    """Project to one of the named orthographic views (front/top/side)."""
    if view not in ORTHO_PROJECTIONS:
        raise ValueError("unknown view %r" % (view,))
    return project_gaussian(mean, cov, ORTHO_PROJECTIONS[view])


def evaluate_gaussian_2d(point: Sequence[float], mean: Sequence[float], cov: Mat2,
                         normalized: bool = False) -> float:
    """Evaluate a 2D Gaussian (unnormalised kernel by default)."""
    d = (float(point[0]) - float(mean[0]), float(point[1]) - float(mean[1]))
    inv = mat2_inverse(cov)
    m = d[0] * inv[0][0] * d[0] + d[0] * inv[0][1] * d[1] \
        + d[1] * inv[1][0] * d[0] + d[1] * inv[1][1] * d[1]
    kernel = exp(-0.5 * m)
    if not normalized:
        return kernel
    det = mat2_det(cov)
    if det <= 0.0:
        raise ValueError("covariance must be positive-definite")
    return kernel / (2 * pi * sqrt(det))


# --------------------------------------------------------------------------- #
# 2D footprint (splat bounding box)
# --------------------------------------------------------------------------- #
def covariance_eigenvalues_2d(cov: Mat2) -> Tuple[float, float]:
    """Closed-form eigenvalues (ascending) of a symmetric 2x2 covariance."""
    a, b, c = cov[0][0], cov[0][1], cov[1][1]
    tr = a + c
    disc = sqrt(max(0.0, (a - c) * (a - c) + 4.0 * b * b))
    return ((tr - disc) / 2.0, (tr + disc) / 2.0)


def footprint_bbox(mean: Sequence[float], cov: Mat2,
                   sigma: float = 3.0) -> Tuple[float, float, float, float]:
    """Axis-aligned bounding box of the ``sigma``-radius splat of a 2D Gaussian.

    Returns ``(u_min, v_min, u_max, v_max)``. The half-extent along each axis is
    ``sigma * sqrt(Sigma_ii)`` (the exact bounding box of the sigma-level ellipse
    projected onto the axes), so a 3-sigma footprint captures ~99.7% of the mass
    per axis — the standard tile-culling radius 3DGS uses.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    hu = sigma * sqrt(max(0.0, cov[0][0]))
    hv = sigma * sqrt(max(0.0, cov[1][1]))
    return (mean[0] - hu, mean[1] - hv, mean[0] + hu, mean[1] + hv)
