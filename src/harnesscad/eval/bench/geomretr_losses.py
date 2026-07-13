"""Deterministic metric-learning losses for 3D retrieval fine-tuning.

Van den Herrewegen et al., *Fine-Tuning 3D Foundation Models for Geometric
Object Retrieval* (2024) fine-tune the encoder with two label-free objectives.
The *training* (backprop, encoder weights) is external, but each loss is a
closed-form function of a batch of embeddings and is therefore fully
reproducible. This module implements them exactly as written in the paper.

**VICReg** (Section 3.3, Bardes et al. [31]) -- three components:

  * Variance (Eq. 2): ``L_var = (1/D) sum_d max(1 - std(z_d), 0)`` pushes each
    embedding dimension's batch std toward 1, preventing collapse.
  * Invariance: ``L_inv`` = mean squared Euclidean distance between the two views
    of each pair.
  * Covariance: ``L_cov = (1/D) sum_{a != b} C[a][b]^2``, the sum of squared
    off-diagonal covariances, decorrelating the dimensions.
  * Total: weighted sum, following the paper's relative weight of 25 on the
    invariance and variance terms.

**MMCL** (Section 3.2.2, Eq. 1) -- the multi-modal contrastive loss aligning
point-cloud embeddings with image and text embeddings via a symmetric InfoNCE
over cosine similarities.

The existing :mod:`bench.contrastcad_contrastive` covers the single-modality
NT-Xent loss; VICReg and the four-term multi-modal MMCL are new here. Stdlib
only, no numpy.
"""

from __future__ import annotations

import math
from typing import List, Sequence

Vector = Sequence[float]


def _check_batch(batch: Sequence[Vector]) -> int:
    if not batch:
        raise ValueError("empty batch")
    dim = len(batch[0])
    if dim == 0:
        raise ValueError("zero-dimensional vectors")
    for v in batch:
        if len(v) != dim:
            raise ValueError("all vectors must share the same dimension")
    return dim


# ---------------------------------------------------------------------------
# VICReg (Section 3.3)
# ---------------------------------------------------------------------------
def variance_loss(batch: Sequence[Vector], *, gamma: float = 1.0,
                  eps: float = 1e-4) -> float:
    """VICReg variance term (Eq. 2): ``(1/D) sum_d max(gamma - std_d, 0)``.

    ``std_d`` is the (biased-corrected) standard deviation of dimension ``d``
    across the batch, with ``eps`` added under the root for stability, as in the
    reference VICReg implementation. ``gamma`` is the target std (1 in the paper).
    """
    dim = _check_batch(batch)
    n = len(batch)
    if n < 2:
        return gamma * 1.0  # cannot estimate variance; maximal penalty
    means = [sum(v[d] for v in batch) / n for d in range(dim)]
    total = 0.0
    for d in range(dim):
        var = sum((v[d] - means[d]) ** 2 for v in batch) / (n - 1)
        std = math.sqrt(var + eps)
        total += max(gamma - std, 0.0)
    return total / dim


def covariance_loss(batch: Sequence[Vector]) -> float:
    """VICReg covariance term: ``(1/D) sum_{a != b} C[a][b]^2``.

    ``C`` is the batch covariance matrix; the loss sums the squared off-diagonal
    entries (dividing by ``D``), encouraging decorrelated dimensions.
    """
    dim = _check_batch(batch)
    n = len(batch)
    if n < 2:
        return 0.0
    means = [sum(v[d] for v in batch) / n for d in range(dim)]
    cov = [[0.0] * dim for _ in range(dim)]
    for v in batch:
        diff = [v[d] - means[d] for d in range(dim)]
        for a in range(dim):
            da = diff[a]
            row = cov[a]
            for b in range(dim):
                row[b] += da * diff[b]
    denom = n - 1
    total = 0.0
    for a in range(dim):
        for b in range(dim):
            if a != b:
                total += (cov[a][b] / denom) ** 2
    return total / dim


def invariance_loss(view_a: Sequence[Vector], view_b: Sequence[Vector]) -> float:
    """VICReg invariance term: mean squared Euclidean distance between paired views."""
    if len(view_a) != len(view_b):
        raise ValueError("paired views must have equal length")
    if not view_a:
        raise ValueError("empty batch")
    total = 0.0
    for a, b in zip(view_a, view_b):
        if len(a) != len(b):
            raise ValueError("paired vectors must share dimension")
        total += sum((x - y) ** 2 for x, y in zip(a, b))
    return total / len(view_a)


def vicreg_loss(view_a: Sequence[Vector], view_b: Sequence[Vector], *,
                sim_weight: float = 25.0, var_weight: float = 25.0,
                cov_weight: float = 1.0) -> dict:
    """Full VICReg loss over a batch of paired embeddings.

    Returns a dict with the ``invariance``, ``variance``, ``covariance`` terms and
    the weighted ``total``. Default weights follow the paper: relative weight 25 on
    the Euclidean (invariance) and variance losses, 1 on covariance. The variance
    and covariance terms are computed on the concatenation of both views (both
    branches regularised jointly, as in the reference implementation averaged).
    """
    inv = invariance_loss(view_a, view_b)
    var = 0.5 * (variance_loss(view_a) + variance_loss(view_b))
    cov = 0.5 * (covariance_loss(view_a) + covariance_loss(view_b))
    total = sim_weight * inv + var_weight * var + cov_weight * cov
    return {"invariance": inv, "variance": var, "covariance": cov, "total": total}


# ---------------------------------------------------------------------------
# Multi-modal contrastive loss (Section 3.2.2, Eq. 1)
# ---------------------------------------------------------------------------
def _dot(u: Vector, v: Vector) -> float:
    return sum(a * b for a, b in zip(u, v))


def _l2(v: Vector) -> List[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return [x / n for x in v]


def _infonce_direction(anchors: Sequence[Vector], targets: Sequence[Vector]) -> float:
    """Mean cross-entropy that anchor_i's positive is target_i, over a batch.

    ``-(1/N) sum_i log( exp(a_i . t_i) / sum_j exp(a_i . t_j) )`` with cosine
    similarity (vectors are L2-normalised internally).
    """
    a = [_l2(x) for x in anchors]
    t = [_l2(x) for x in targets]
    n = len(a)
    total = 0.0
    for i in range(n):
        sims = [math.exp(_dot(a[i], t[j])) for j in range(n)]
        denom = sum(sims)
        total += -math.log(sims[i] / denom)
    return total / n


def mmcl_loss(z_point: Sequence[Vector], z_text: Sequence[Vector],
              z_image: Sequence[Vector]) -> dict:
    """Multi-modal contrastive loss ``L_MMCL`` (Eq. 1).

    Four symmetric InfoNCE terms pairing the point-cloud modality with text and
    image in both directions (P->T, T->P, P->I, I->P), averaged with the paper's
    ``1/4`` factor. Each batch entry ``i`` is a positive triplet; all other
    entries in the batch are negatives. Returns the per-direction terms and the
    combined ``total``.
    """
    n = len(z_point)
    if not (len(z_text) == n and len(z_image) == n):
        raise ValueError("all three modality batches must have equal length")
    if n == 0:
        raise ValueError("empty batch")
    pt = _infonce_direction(z_point, z_text)
    tp = _infonce_direction(z_text, z_point)
    pi = _infonce_direction(z_point, z_image)
    ip = _infonce_direction(z_image, z_point)
    total = 0.25 * (pt + tp + pi + ip)
    return {"pt": pt, "tp": tp, "pi": pi, "ip": ip, "total": total}
