"""Silhouette IoU agreement losses from Magic3DSketch (Zang et al., Neurocomputing 2024).

Magic3DSketch trains a sketch->mesh network end-to-end by rendering the
generated mesh to a silhouette and comparing it against the input sketch with a
soft intersection-over-union (IoU) loss (paper Eq. 1):

    L_iou(S1, S2) = 1 - |S1 (x) S2|_1 / |S1 (+) S2 - S1 (x) S2|_1

where ``(x)`` is the element-wise product (soft intersection) and ``(+)`` the
element-wise sum, both reduced by the L1 norm (sum of entries).  The denominator
``sum(S1) + sum(S2) - sum(S1*S2)`` is the soft union.  For *binary* masks this
reduces to the classical |A AND B| / |A OR B| Jaccard index, so the loss is
1 - Jaccard.  Note: with binary masks the soft-IoU is 0 (loss 1.0) exactly when
the two masks are disjoint, and 1 (loss 0.0) when identical.

To improve efficiency Magic3DSketch progressively increases silhouette
resolution and forms a multi-scale mIoU loss (paper Eq. 2):

    L_ms = sum_i lambda_i * L_iou^i

evaluated on a pyramid of downsampled masks.  This module implements the soft
IoU, the loss, average-pool downsampling, and the weighted multi-scale
aggregation.  The differentiable renderer (SoftRas) and the CLIP / encoder-
decoder network that *produce* the silhouettes are learned and external.

Everything here is stdlib-only and deterministic.  Masks are rectangular grids
of floats (rows of columns), values conventionally in ``[0, 1]``.
"""

from __future__ import annotations

from typing import List, Sequence

Mask = Sequence[Sequence[float]]


def _dims(mask: Mask) -> tuple:
    rows = len(mask)
    if rows == 0:
        raise ValueError("mask must have at least one row")
    cols = len(mask[0])
    if cols == 0:
        raise ValueError("mask must have at least one column")
    for r in mask:
        if len(r) != cols:
            raise ValueError("mask must be rectangular")
    return rows, cols


def _check_same_shape(s1: Mask, s2: Mask) -> tuple:
    d1 = _dims(s1)
    d2 = _dims(s2)
    if d1 != d2:
        raise ValueError("masks must have identical shape, got %r and %r" % (d1, d2))
    return d1


def soft_intersection(s1: Mask, s2: Mask) -> float:
    """Sum of the element-wise product ``sum(S1 * S2)`` (soft |S1 AND S2|)."""
    _check_same_shape(s1, s2)
    total = 0.0
    for r1, r2 in zip(s1, s2):
        for a, b in zip(r1, r2):
            total += a * b
    return total


def soft_union(s1: Mask, s2: Mask) -> float:
    """Soft union ``sum(S1) + sum(S2) - sum(S1 * S2)`` (paper Eq. 1 denominator)."""
    _check_same_shape(s1, s2)
    sum1 = 0.0
    sum2 = 0.0
    inter = 0.0
    for r1, r2 in zip(s1, s2):
        for a, b in zip(r1, r2):
            sum1 += a
            sum2 += b
            inter += a * b
    return sum1 + sum2 - inter


def soft_iou(s1: Mask, s2: Mask, *, eps: float = 1e-12) -> float:
    """Soft intersection-over-union of two masks, in ``[0, 1]``.

    ``eps`` guards the degenerate case of two all-zero masks, for which the
    union is 0; there we define the IoU as 1.0 (perfect agreement of "empty").
    """
    inter = soft_intersection(s1, s2)
    union = soft_union(s1, s2)
    if union <= eps:
        return 1.0
    return inter / union


def iou_loss(s1: Mask, s2: Mask, *, eps: float = 1e-12) -> float:
    """Silhouette IoU loss ``1 - soft_iou`` (paper Eq. 1), in ``[0, 1]``."""
    return 1.0 - soft_iou(s1, s2, eps=eps)


def downsample(mask: Mask, factor: int) -> List[List[float]]:
    """Average-pool ``mask`` by an integer ``factor`` (block mean).

    The mask is partitioned into ``factor x factor`` blocks; each output cell is
    the mean of its block.  A trailing partial block (when a dimension is not a
    multiple of ``factor``) is averaged over its actual, smaller area so no data
    is dropped.  ``factor == 1`` returns a copy.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    rows, cols = _dims(mask)
    if factor == 1:
        return [list(r) for r in mask]
    out_rows = (rows + factor - 1) // factor
    out_cols = (cols + factor - 1) // factor
    result: List[List[float]] = []
    for oi in range(out_rows):
        r0 = oi * factor
        r1 = min(r0 + factor, rows)
        out_row: List[float] = []
        for oj in range(out_cols):
            c0 = oj * factor
            c1 = min(c0 + factor, cols)
            acc = 0.0
            for i in range(r0, r1):
                row = mask[i]
                for j in range(c0, c1):
                    acc += row[j]
            area = (r1 - r0) * (c1 - c0)
            out_row.append(acc / area)
        result.append(out_row)
    return result


def multiscale_iou_loss(
    s1: Mask,
    s2: Mask,
    factors: Sequence[int],
    weights: Sequence[float],
    *,
    eps: float = 1e-12,
) -> float:
    """Multi-scale mIoU loss ``sum_i weights[i] * L_iou(down_i(S1), down_i(S2))``.

    ``factors`` gives the pooling factor at each scale (e.g. ``(1, 2, 4)`` for a
    3-level pyramid) and ``weights`` the corresponding lambda_i (paper Eq. 2).
    """
    if len(factors) != len(weights):
        raise ValueError("factors and weights must have equal length")
    if not factors:
        raise ValueError("need at least one scale")
    total = 0.0
    for f, w in zip(factors, weights):
        d1 = downsample(s1, f)
        d2 = downsample(s2, f)
        total += w * iou_loss(d1, d2, eps=eps)
    return total
