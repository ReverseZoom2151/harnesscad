"""Multimodal latent-space alignment between a geometry modality and CAD programs.

Yu, Alam, Hart & Ahmed, *CAD Program Generation using GenCAD-3D* (J. Mech. Des.
2026), Sections 3.1-3.2.

GenCAD-3D aligns a modality-specific geometry encoder's latent ``z_M`` with the
frozen CAD-program latent ``z_C`` so that "embeddings corresponding to the same
geometry across modalities are pulled together while embeddings of different
geometries are pushed apart" (Sec. 3.1). The learned pieces are a contrastive
loss (InfoNCE, Eq. 3 -- already deterministically covered by
``bench/contrastcad_contrastive``) and a diffusion prior mapping ``z_M -> z_C``
(Eq. 4). This module provides the *deterministic* alignment surrogates and the
alignment-quality diagnostics that do not require training:

* :func:`fit_linear_alignment` -- a closed-form least-squares linear map ``W``
  (with ridge regularisation) that projects geometry latents into the CAD latent
  space, ``z_M W ~= z_C`` -- a deterministic stand-in for the diffusion prior
  ``p(z_C | z_M)``. Solved with a stdlib Gaussian-elimination linear solver.
* :func:`alignment_quality` -- how well two paired latent sets are aligned:
  mean paired cosine similarity, the paired-vs-cross **alignment margin**, and
  the mean reciprocal rank / top-1 accuracy of the true cross-modal match over
  the *full* library (a deterministic complement to ``bench/gencad_retrieval``'s
  stochastic bootstrap R_B estimator).
* :func:`cross_modal_topk_accuracy` -- deterministic full-library Top-k retrieval
  accuracy averaged over every query.

Reuses ``bench.geomretr_embedding.cosine_similarity``. Pure stdlib, deterministic
(no randomness, no wall clock).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from bench.geomretr_embedding import cosine_similarity

Vector = List[float]
Matrix = List[List[float]]


# =====================================================================
# Linear algebra helpers (stdlib Gaussian elimination)
# =====================================================================

def _transpose(a: Sequence[Sequence[float]]) -> Matrix:
    return [list(col) for col in zip(*a)] if a else []


def _matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix:
    bt = _transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def _solve(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix:
    """Solve the linear system ``A X = B`` for X via Gaussian elimination with
    partial pivoting. ``A`` is ``n x n`` (square, symmetric-PD in our use); ``B``
    is ``n x k``. Returns X (``n x k``). Deterministic.
    """
    n = len(a)
    # Build augmented matrix [A | B].
    m = [list(a[i]) + list(b[i]) for i in range(n)]
    k = len(b[0]) if b and b[0] else 0
    for col in range(n):
        # Partial pivot: largest magnitude in this column at/below the diagonal.
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular system; increase ridge regularisation")
        m[col], m[pivot] = m[pivot], m[col]
        piv = m[col][col]
        inv = 1.0 / piv
        m[col] = [v * inv for v in m[col]]
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor != 0.0:
                m[r] = [rv - factor * cv for rv, cv in zip(m[r], m[col])]
    return [row[n:n + k] for row in m]


# =====================================================================
# Linear alignment (deterministic prior surrogate)
# =====================================================================

@dataclass
class LinearAlignment:
    """A fitted linear map ``W`` projecting geometry latents into CAD-latent space."""

    weight: Matrix                  # d_in x d_out
    d_in: int
    d_out: int
    ridge: float

    def apply(self, z_m: Sequence[float]) -> Vector:
        """Project one geometry latent ``z_M`` into the CAD latent space."""
        if len(z_m) != self.d_in:
            raise ValueError("latent dimension mismatch")
        return [sum(z_m[i] * self.weight[i][j] for i in range(self.d_in))
                for j in range(self.d_out)]

    def apply_all(self, zs: Sequence[Sequence[float]]) -> Matrix:
        return [self.apply(z) for z in zs]


def fit_linear_alignment(z_geometry: Sequence[Sequence[float]],
                         z_cad: Sequence[Sequence[float]],
                         ridge: float = 1e-6) -> LinearAlignment:
    """Fit ``W`` minimising ``||Z_M W - Z_C||^2 + ridge*||W||^2`` (normal equations).

    ``z_geometry[i]`` and ``z_cad[i]`` are the paired latents of example ``i``.
    Solves ``(Z_M^T Z_M + ridge I) W = Z_M^T Z_C``. Deterministic closed form; the
    ridge term keeps the Gram matrix invertible. This is a linear stand-in for the
    learned diffusion prior ``p(z_C | z_M)`` (paper Eq. 4).
    """
    n = len(z_geometry)
    if n == 0:
        raise ValueError("need at least one paired example")
    if len(z_cad) != n:
        raise ValueError("geometry and CAD latent sets must be paired (equal length)")
    d_in = len(z_geometry[0])
    d_out = len(z_cad[0])
    zt = _transpose(z_geometry)                       # d_in x n
    gram = _matmul(zt, z_geometry)                    # d_in x d_in
    for i in range(d_in):
        gram[i][i] += ridge
    rhs = _matmul(zt, z_cad)                           # d_in x d_out
    weight = _solve(gram, rhs)                         # d_in x d_out
    return LinearAlignment(weight=weight, d_in=d_in, d_out=d_out, ridge=ridge)


# =====================================================================
# Alignment quality diagnostics
# =====================================================================

@dataclass
class AlignmentQuality:
    """Diagnostics for how well two paired latent sets are aligned."""

    n: int
    mean_paired_cosine: float
    mean_cross_cosine: float
    top1_accuracy: float
    mean_reciprocal_rank: float

    @property
    def margin(self) -> float:
        """Alignment margin: paired similarity minus mean off-diagonal similarity.

        Positive and large means matched pairs sit far closer than mismatched
        ones -- a well-aligned latent space.
        """
        return self.mean_paired_cosine - self.mean_cross_cosine

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_paired_cosine": self.mean_paired_cosine,
            "mean_cross_cosine": self.mean_cross_cosine,
            "margin": self.margin,
            "top1_accuracy": self.top1_accuracy,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
        }


def _rank_of_match(query: Vector, library: Sequence[Vector], true_idx: int) -> int:
    """1-based rank of ``library[true_idx]`` when scoring ``library`` by cosine
    similarity to ``query`` (rank 1 = closest). Ties count against the match
    (an equally-similar decoy ranks ahead), a conservative deterministic choice.
    """
    target = cosine_similarity(query, library[true_idx])
    better = sum(1 for j, cand in enumerate(library)
                 if j != true_idx and cosine_similarity(query, cand) >= target)
    return better + 1


def alignment_quality(z_geometry: Sequence[Sequence[float]],
                      z_cad: Sequence[Sequence[float]]) -> AlignmentQuality:
    """Alignment diagnostics for paired geometry/CAD latents over the full library.

    For each example ``i`` the geometry latent queries the whole CAD library:
    top-1 accuracy counts how often the nearest CAD latent is the true match, and
    the reciprocal rank rewards near-misses. Also reports the mean paired cosine
    and the mean cross-pair cosine whose difference is the alignment margin.
    """
    n = len(z_geometry)
    if n == 0:
        raise ValueError("need at least one paired example")
    if len(z_cad) != n:
        raise ValueError("geometry and CAD latent sets must be paired (equal length)")

    paired = [cosine_similarity(z_geometry[i], z_cad[i]) for i in range(n)]
    cross_sum = 0.0
    cross_cnt = 0
    hits = 0
    rr_sum = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                cross_sum += cosine_similarity(z_geometry[i], z_cad[j])
                cross_cnt += 1
        rank = _rank_of_match(z_geometry[i], z_cad, i)
        if rank == 1:
            hits += 1
        rr_sum += 1.0 / rank
    return AlignmentQuality(
        n=n,
        mean_paired_cosine=sum(paired) / n,
        mean_cross_cosine=(cross_sum / cross_cnt) if cross_cnt else 0.0,
        top1_accuracy=hits / n,
        mean_reciprocal_rank=rr_sum / n,
    )


def cross_modal_topk_accuracy(z_geometry: Sequence[Sequence[float]],
                              z_cad: Sequence[Sequence[float]],
                              k: int = 1) -> float:
    """Deterministic full-library Top-k cross-modal retrieval accuracy.

    Each geometry latent queries every CAD latent; a hit is counted when the true
    match falls within the top ``k`` by cosine similarity. Averaged over all
    queries. Complements ``bench.gencad_retrieval.retrieval_accuracy`` (which
    bootstraps random library batches to estimate R_B).
    """
    n = len(z_geometry)
    if n == 0:
        raise ValueError("need at least one paired example")
    if len(z_cad) != n:
        raise ValueError("geometry and CAD latent sets must be paired (equal length)")
    if not 1 <= k <= n:
        raise ValueError("k must be in [1, n]")
    hits = 0
    for i in range(n):
        if _rank_of_match(z_geometry[i], z_cad, i) <= k:
            hits += 1
    return hits / n


def alignment_improvement(z_geometry: Sequence[Sequence[float]],
                          z_cad: Sequence[Sequence[float]],
                          ridge: float = 1e-6) -> dict:
    """Fit a linear alignment and report before/after alignment quality.

    Returns a dict with ``before`` and ``after`` :class:`AlignmentQuality` dicts
    and the ``margin``/``top1`` deltas, quantifying how much the deterministic
    linear map improves cross-modal alignment.
    """
    before = alignment_quality(z_geometry, z_cad)
    model = fit_linear_alignment(z_geometry, z_cad, ridge=ridge)
    projected = model.apply_all(z_geometry)
    after = alignment_quality(projected, z_cad)
    return {
        "before": before.to_dict(),
        "after": after.to_dict(),
        "margin_delta": after.margin - before.margin,
        "top1_delta": after.top1_accuracy - before.top1_accuracy,
    }
