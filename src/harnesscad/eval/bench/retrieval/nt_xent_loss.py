"""Deterministic contrastive-learning maths for ContrastCAD.

Jung, Kim & Kim, *ContrastCAD* (2024), Section 4.2.2 / 4.2.3.

ContrastCAD builds positive pairs with the SimCSE dropout trick (Gao et al.):
each latent vector ``z_proj_i`` is passed **twice** through a dropout masking
layer with two *different* masks, producing views ``d_i`` and ``d_j`` that encode
the same CAD model but differ slightly -- a positive pair. Every other view in the
batch is a negative. The model is trained with the NT-Xent / InfoNCE loss over the
``2m`` views of a size-``m`` batch (Eq. 5)::

    l_cont = -log( exp(SIM(d_i, d_j) / tau)
                   / sum_{k != i} exp(SIM(d_i, d_k) / tau) )

with cosine similarity ``SIM(u, v) = u . v / (||u|| ||v||)`` (Eq. 4) and
temperature ``tau``.

The learned encoder / projection weights are out of scope (research-heavy). What
this module provides is the fully deterministic part: cosine similarity, seeded
dropout-mask view construction, the per-anchor and mean NT-Xent losses, and the
similarity matrix that feeds them. Given fixed embeddings and a fixed seed the
loss is byte-reproducible. Stdlib only, no numpy, no wall clock.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

Vector = Sequence[float]


def cosine_similarity(u: Vector, v: Vector) -> float:
    """SIM(u, v) = u . v / (||u|| ||v||) (Eq. 4), clamped to [-1, 1]."""
    if len(u) != len(v):
        raise ValueError("vectors must have equal dimension")
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu == 0.0 or nv == 0.0:
        raise ValueError("cannot take cosine similarity of a zero vector")
    dot = sum(a * b for a, b in zip(u, v))
    return max(-1.0, min(1.0, dot / (nu * nv)))


def dropout_view(vector: Vector, prob: float, rng: random.Random) -> List[float]:
    """Standard inverted dropout: zero each coordinate with probability ``prob``
    and scale survivors by ``1 / (1 - prob)`` (expectation-preserving).

    Two calls with different RNG state give the two masks that form a positive
    pair (paper Section 4.2.2). Deterministic given the RNG.
    """
    if not 0.0 <= prob < 1.0:
        raise ValueError("dropout prob must be in [0, 1)")
    scale = 1.0 / (1.0 - prob) if prob < 1.0 else 0.0
    return [0.0 if rng.random() < prob else x * scale for x in vector]


def make_positive_pairs(latents: Sequence[Vector], prob: float,
                        seed) -> List[List[float]]:
    """Build the ``2m`` dropout views of a size-``m`` batch (paper Section 4.2.3).

    Returns a list of length ``2 * len(latents)`` laid out as
    ``[d_0, ..., d_{m-1}, d_m, ..., d_{2m-1}]`` where view ``i`` and view ``m + i``
    are the two dropout masks of ``latents[i]`` and thus a positive pair. Masks are
    drawn from one seeded stream, so the whole batch is deterministic.
    """
    rng = random.Random(seed)
    m = len(latents)
    first = [dropout_view(z, prob, rng) for z in latents]
    second = [dropout_view(z, prob, rng) for z in latents]
    return first + second


def positive_index(i: int, batch_size: int) -> int:
    """Index of the positive partner of view ``i`` in a ``2m`` view list.

    Views ``i`` and ``m + i`` are partners; the mapping is its own inverse.
    """
    m = batch_size
    if not 0 <= i < 2 * m:
        raise ValueError("view index out of range")
    return i + m if i < m else i - m


def nt_xent_anchor_loss(views: Sequence[Vector], anchor: int,
                        temperature: float = 0.07) -> float:
    """NT-Xent / InfoNCE loss for a single anchor view (Eq. 5).

    ``views`` is the ``2m`` list from :func:`make_positive_pairs`; the positive of
    ``anchor`` is ``positive_index(anchor, m)``. The denominator sums over all
    views except the anchor itself (the ``I[k != i]`` indicator). Temperature
    ``tau`` defaults to the paper's 0.07.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    total = len(views)
    if total % 2 != 0:
        raise ValueError("views must contain an even number (2m) of vectors")
    m = total // 2
    pos = positive_index(anchor, m)
    num = math.exp(cosine_similarity(views[anchor], views[pos]) / temperature)
    denom = 0.0
    for k in range(total):
        if k == anchor:
            continue
        denom += math.exp(cosine_similarity(views[anchor], views[k]) / temperature)
    if denom == 0.0:
        raise ValueError("degenerate denominator in NT-Xent loss")
    return -math.log(num / denom)


def nt_xent_loss(views: Sequence[Vector], temperature: float = 0.07) -> float:
    """Mean NT-Xent loss over all ``2m`` anchors (the batch contrastive loss).

    Averages :func:`nt_xent_anchor_loss` across every view, which is the standard
    symmetric NT-Xent objective used to train ContrastCAD's projection head.
    """
    total = len(views)
    if total == 0:
        raise ValueError("no views supplied")
    return sum(nt_xent_anchor_loss(views, i, temperature)
               for i in range(total)) / total


def contrastive_loss(latents: Sequence[Vector], prob: float, seed,
                     temperature: float = 0.07) -> float:
    """End-to-end deterministic ``l_cont`` for a batch of latent vectors.

    Builds the ``2m`` dropout views (seeded) then returns the mean NT-Xent loss.
    This is the fully reproducible surrogate for ContrastCAD's contrastive term
    given fixed latent vectors.
    """
    views = make_positive_pairs(latents, prob, seed)
    return nt_xent_loss(views, temperature)


def similarity_matrix(views: Sequence[Vector]) -> List[List[float]]:
    """Full pairwise cosine-similarity matrix of a set of views (symmetric)."""
    n = len(views)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = 1.0 if i == j else cosine_similarity(views[i], views[j])
            matrix[i][j] = s
            matrix[j][i] = s
    return matrix
