"""Frechet latent distance (FID) for image-conditional CAD generation.

Alam & Ahmed, *GenCAD* (2024), Section 4.3.3 / 5.4.

GenCAD quantifies how well generated CAD programs align with the input CAD
images by the **FID score** between the distribution of generated CAD latents
and the distribution of ground-truth (test) CAD latents.  If the ground-truth
embeddings and the generated embeddings are modelled as Gaussians
``N(mu_S, Sigma_S)`` and ``N(mu_G, Sigma_G)`` respectively, then

    FID = || mu_S - mu_G ||^2  +  tr( Sigma_S + Sigma_G - 2 (Sigma_S Sigma_G)^{1/2} )

(the Frechet-2 / Wasserstein-2 distance between two Gaussians).  Lower is
better; a value of ``0`` means the two latent distributions coincide.

This is distinct from everything already in the repository:

  * ``bench/generative_brep_metrics.py`` implements COV / MMD / JSD (point-cloud
    set-distance metrics), NOT a Gaussian Frechet distance over latents.
  * ``geometry/dreamcad_metrics.py`` only *mentions* FID in prose; it implements
    Chamfer / Hausdorff point-cloud distances.

The tricky term ``tr((Sigma_S Sigma_G)^{1/2})`` needs the trace of a matrix
square root.  For symmetric positive semi-definite (SPSD) covariances the
product ``Sigma_S Sigma_G`` has real non-negative eigenvalues, and

    tr((Sigma_S Sigma_G)^{1/2}) = sum_i sqrt(lambda_i),

where the ``lambda_i`` are the eigenvalues of the *symmetric* matrix
``Sigma_S^{1/2} Sigma_G Sigma_S^{1/2}`` (which is similar to ``Sigma_S Sigma_G``
but symmetric, so a Jacobi eigensolver applies).  This avoids ever forming a
non-symmetric square root.

Pure stdlib, fully deterministic (a classical cyclic Jacobi eigenvalue sweep).
"""

from __future__ import annotations

from math import sqrt
from typing import List, Sequence

Matrix = List[List[float]]
Vector = List[float]


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def mean_vector(samples: Sequence[Sequence[float]]) -> Vector:
    """Column-wise mean of a set of latent vectors."""
    if not samples:
        raise ValueError("need at least one sample")
    d = len(samples[0])
    if any(len(s) != d for s in samples):
        raise ValueError("all samples must share the same dimension")
    n = len(samples)
    return [sum(s[j] for s in samples) / n for j in range(d)]


def covariance_matrix(samples: Sequence[Sequence[float]], ddof: int = 1) -> Matrix:
    """Sample covariance matrix (``ddof=1`` -> unbiased, matching ``numpy.cov``).

    A single sample (or ``n <= ddof``) yields the zero matrix.
    """
    if not samples:
        raise ValueError("need at least one sample")
    d = len(samples[0])
    if any(len(s) != d for s in samples):
        raise ValueError("all samples must share the same dimension")
    n = len(samples)
    mu = mean_vector(samples)
    cov = [[0.0] * d for _ in range(d)]
    for s in samples:
        diff = [s[j] - mu[j] for j in range(d)]
        for i in range(d):
            di = diff[i]
            row = cov[i]
            for j in range(i, d):
                row[j] += di * diff[j]
    denom = n - ddof
    if denom <= 0:
        return [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            cov[i][j] /= denom
            cov[j][i] = cov[i][j]
    return cov


# --------------------------------------------------------------------------- #
# Symmetric eigendecomposition (cyclic Jacobi) and matrix square root
# --------------------------------------------------------------------------- #
def _identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def jacobi_eigen(a: Matrix, max_sweeps: int = 100, tol: float = 1e-12):
    """Eigenvalues/vectors of a real symmetric matrix via cyclic Jacobi rotations.

    Returns ``(eigenvalues, eigenvectors)`` where ``eigenvectors[k]`` is the k-th
    eigenvector (a column of the accumulated rotation matrix).  Deterministic:
    the sweep order is fixed and rotations depend only on matrix entries.
    """
    n = len(a)
    for row in a:
        if len(row) != n:
            raise ValueError("matrix must be square")
    # Work on a mutable copy.
    m = [list(row) for row in a]
    v = _identity(n)
    for _ in range(max_sweeps):
        # Off-diagonal Frobenius norm.
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += m[p][q] * m[p][q]
        if off <= tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = m[p][q]
                if abs(apq) <= 1e-300:
                    continue
                app = m[p][p]
                aqq = m[q][q]
                phi = (aqq - app) / (2.0 * apq)
                # t = sign(phi) / (|phi| + sqrt(phi^2 + 1)), stable form.
                if phi >= 0.0:
                    t = 1.0 / (phi + sqrt(phi * phi + 1.0))
                else:
                    t = -1.0 / (-phi + sqrt(phi * phi + 1.0))
                c = 1.0 / sqrt(t * t + 1.0)
                s = t * c
                # Rotate rows/columns p, q.
                for k in range(n):
                    mkp = m[k][p]
                    mkq = m[k][q]
                    m[k][p] = c * mkp - s * mkq
                    m[k][q] = s * mkp + c * mkq
                for k in range(n):
                    mpk = m[p][k]
                    mqk = m[q][k]
                    m[p][k] = c * mpk - s * mqk
                    m[q][k] = s * mpk + c * mqk
                for k in range(n):
                    vkp = v[k][p]
                    vkq = v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    eigenvalues = [m[i][i] for i in range(n)]
    eigenvectors = [[v[r][i] for r in range(n)] for i in range(n)]
    return eigenvalues, eigenvectors


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    n = len(a)
    k = len(b)
    p = len(b[0]) if b else 0
    out = [[0.0] * p for _ in range(n)]
    for i in range(n):
        ai = a[i]
        oi = out[i]
        for t in range(k):
            ait = ai[t]
            if ait == 0.0:
                continue
            bt = b[t]
            for j in range(p):
                oi[j] += ait * bt[j]
    return out


def symmetric_sqrt(a: Matrix, eps: float = 0.0) -> Matrix:
    """Principal square root of a symmetric positive-semi-definite matrix.

    Negative eigenvalues (numerical noise) are clamped to ``eps``.
    """
    vals, vecs = jacobi_eigen(a)
    n = len(a)
    # Reconstruct Q diag(sqrt(w)) Q^T.  vecs[k] is the k-th eigenvector.
    root = [[0.0] * n for _ in range(n)]
    for k in range(n):
        w = vals[k]
        sw = sqrt(w) if w > eps else (sqrt(eps) if eps > 0.0 else 0.0)
        vk = vecs[k]
        for i in range(n):
            vki = vk[i]
            if vki == 0.0:
                continue
            ri = root[i]
            for j in range(n):
                ri[j] += sw * vki * vk[j]
    return root


# --------------------------------------------------------------------------- #
# FID / Frechet-Gaussian distance
# --------------------------------------------------------------------------- #
def frechet_gaussian_distance(mu_s: Sequence[float], sigma_s: Matrix,
                              mu_g: Sequence[float], sigma_g: Matrix) -> float:
    """FID between Gaussians ``N(mu_s, sigma_s)`` and ``N(mu_g, sigma_g)``.

    ``FID = ||mu_s - mu_g||^2 + tr(sigma_s + sigma_g - 2 (sigma_s sigma_g)^{1/2})``.
    Clamped at ``0.0`` (the exact distance is non-negative; tiny negatives from
    finite-precision eigenvalues are rounded up).
    """
    d = len(mu_s)
    if len(mu_g) != d:
        raise ValueError("mean vectors must share the same dimension")
    mean_term = sum((mu_s[j] - mu_g[j]) ** 2 for j in range(d))
    trace_s = sum(sigma_s[i][i] for i in range(d))
    trace_g = sum(sigma_g[i][i] for i in range(d))
    # tr((sigma_s sigma_g)^{1/2}) via the SPD matrix sigma_s^{1/2} sigma_g sigma_s^{1/2}.
    root_s = symmetric_sqrt(sigma_s)
    middle = _matmul(_matmul(root_s, sigma_g), root_s)
    # Symmetrise to kill round-off asymmetry before the eigensolver.
    for i in range(d):
        for j in range(i + 1, d):
            avg = 0.5 * (middle[i][j] + middle[j][i])
            middle[i][j] = avg
            middle[j][i] = avg
    eigs, _ = jacobi_eigen(middle)
    trace_cross = sum(sqrt(e) if e > 0.0 else 0.0 for e in eigs)
    fid = mean_term + trace_s + trace_g - 2.0 * trace_cross
    return fid if fid > 0.0 else 0.0


def fid_score(real_latents: Sequence[Sequence[float]],
              generated_latents: Sequence[Sequence[float]],
              ddof: int = 1) -> float:
    """FID between two sets of CAD latent vectors (GenCAD Sec. 5.4).

    Estimates each set's Gaussian ``(mean, covariance)`` then returns the
    Frechet distance.  Lower means the generated latents better match the real
    (test-image-aligned) distribution.
    """
    mu_s = mean_vector(real_latents)
    mu_g = mean_vector(generated_latents)
    sigma_s = covariance_matrix(real_latents, ddof=ddof)
    sigma_g = covariance_matrix(generated_latents, ddof=ddof)
    return frechet_gaussian_distance(mu_s, sigma_s, mu_g, sigma_g)
