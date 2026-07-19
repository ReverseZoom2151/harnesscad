"""Embedding post-processing for 3D object-retrieval systems.

A common pipeline stage for a fine-tuned geometric encoder: the encoder maps
every object to a fixed-length latent vector, and retrieval is done by
nearest-neighbour search under **cosine distance**. Cosine search is exactly
L2-normalisation followed by an inner-product ranking.

The learned encoder is research-heavy/external. The vector post-processing that
turns raw embeddings into retrieval-ready features is deterministic and
buildable. This module provides:

* :func:`l2_normalize` / :func:`l2_normalize_all` -- unit-norm projection so dot
  product equals cosine similarity.
* :func:`pca_whiten` -- centre + PCA whitening (decorrelate and equalise
  variance) using the shared Jacobi eigensolver; a classic retrieval trick that
  removes the dominant "burstiness" directions before cosine search.
* :func:`average_query_expansion` -- re-rank by folding the top-k retrieved
  vectors back into the query (AQE), a standard training-free retrieval boost.

Stdlib-only, deterministic, no numpy.
"""

from __future__ import annotations

import math
from typing import List, Sequence

from harnesscad.domain.geometry.transforms.umeyama import jacobi_eigen

Vector = Sequence[float]


def l2_normalize(vec: Vector, *, eps: float = 1e-12) -> List[float]:
    """Return ``vec / ||vec||_2``. A zero vector is returned unchanged (as zeros)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < eps:
        return [0.0 for _ in vec]
    return [x / norm for x in vec]


def l2_normalize_all(vectors: Sequence[Vector]) -> List[List[float]]:
    """L2-normalise every row (so inner product == cosine similarity)."""
    return [l2_normalize(v) for v in vectors]


def _mean(vectors: Sequence[Vector]) -> List[float]:
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[d] for v in vectors) / n for d in range(dim)]


def _covariance(vectors: Sequence[Vector], mean: Sequence[float]) -> List[List[float]]:
    n = len(vectors)
    dim = len(mean)
    cov = [[0.0] * dim for _ in range(dim)]
    for v in vectors:
        diff = [v[d] - mean[d] for d in range(dim)]
        for a in range(dim):
            da = diff[a]
            row = cov[a]
            for b in range(dim):
                row[b] += da * diff[b]
    denom = max(1, n - 1)
    for a in range(dim):
        for b in range(dim):
            cov[a][b] /= denom
    return cov


def pca_whiten(vectors: Sequence[Vector], *, eps: float = 1e-8):
    """Fit a PCA-whitening transform on ``vectors`` and apply it.

    Returns ``(whitened_rows, transform)`` where ``transform`` is a dict with the
    fitted ``mean``, eigenvector matrix ``components`` (columns), and per-axis
    ``scales`` (``1/sqrt(eigenvalue + eps)``). The whitened vectors have (near)
    identity covariance: dimensions are decorrelated and equalised. Apply the same
    transform to new vectors with :func:`apply_whiten`.
    """
    if not vectors:
        return [], {"mean": [], "components": [], "scales": []}
    mean = _mean(vectors)
    cov = _covariance(vectors, mean)
    eigvals, vecs = jacobi_eigen(cov)
    scales = [1.0 / math.sqrt(max(ev, 0.0) + eps) for ev in eigvals]
    transform = {"mean": mean, "components": vecs, "scales": scales}
    whitened = [apply_whiten(v, transform) for v in vectors]
    return whitened, transform


def apply_whiten(vec: Vector, transform: dict) -> List[float]:
    """Apply a fitted :func:`pca_whiten` transform to one vector."""
    mean = transform["mean"]
    vecs = transform["components"]
    scales = transform["scales"]
    dim = len(mean)
    diff = [vec[d] - mean[d] for d in range(dim)]
    out = []
    for axis in range(len(scales)):
        proj = sum(diff[d] * vecs[d][axis] for d in range(dim))
        out.append(proj * scales[axis])
    return out


def average_query_expansion(query: Vector, gallery: Sequence[Vector],
                            ranked_indices: Sequence[int], *, top_k: int = 5,
                            include_query: bool = True) -> List[float]:
    """Average-query-expansion: fold the top-k retrieved vectors into the query.

    Given an initial ranking (``ranked_indices`` of ``gallery`` best-first), form a
    new query as the L2-normalised mean of the original query and the top-``k``
    gallery vectors, then it can be re-searched for a refined ranking. This is the
    standard training-free AQE re-ranking step. Deterministic.
    """
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    dim = len(query)
    members: List[Vector] = []
    if include_query:
        members.append(l2_normalize(query))
    for idx in list(ranked_indices)[:top_k]:
        members.append(l2_normalize(gallery[idx]))
    if not members:
        return l2_normalize(query)
    mean = [sum(m[d] for m in members) / len(members) for d in range(dim)]
    return l2_normalize(mean)


def cosine_similarity(u: Vector, v: Vector, *, eps: float = 1e-12) -> float:
    """Cosine similarity ``u . v / (||u|| ||v||)`` clamped to ``[-1, 1]``."""
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu < eps or nv < eps:
        return 0.0
    dot = sum(a * b for a, b in zip(u, v))
    return max(-1.0, min(1.0, dot / (nu * nv)))
