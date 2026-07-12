"""picasso_metrics -- image-based evaluation metrics for CAD sketch rendering.

For quantitative evaluation PICASSO reports **image-based metrics** computed "on
the explicit rendering of predicted primitive sequences" (Sec. 5, Sec. 9): a
normalised pixel-wise Mean Squared Error (ImgMSE) and a bidirectional Chamfer
Distance (CD).  These are the inference-time metrics that judge a predicted
parameterisation purely through its raster, and they are the natural companions
of the rendering self-supervision losses -- they need no parameter labels either,
only the two explicit renderings.

This module implements those two metrics exactly as defined in the supplementary
(Sec. 9), plus a couple of convenience aggregates.

ImgMSE (Eq. 5)::

    ImgMSE = 1/(2 N_F) * sum_k 1[X_k=1] (Xhat_k - X_k)^2
           + 1/(2 w h) * sum_k       (Xhat_k - X_k)^2

i.e. the average of a *foreground-restricted* MSE (over the ``N_F`` ink pixels of
the target) and a *global* MSE over all ``w*h`` pixels.  The foreground term
penalises missing ink; the global term penalises spurious ink.

CD (Eq. 6): the bidirectional Chamfer Distance between the sets of foreground
pixel coordinates of the two explicit renderings::

    CD = 1/(2 Nhat_F) sum_n min_k ||zhat_n - z_k||^2
       + 1/(2 N_F)    sum_n min_k ||z_n - zhat_k||^2

All functions are pure stdlib and deterministic.  Distances are computed in pixel
units on the integer grid.
"""

from __future__ import annotations

import math

Image = list[list[float]]


def _shape(image: Image) -> tuple[int, int]:
    if not image or not image[0]:
        raise ValueError("image must be non-empty")
    h = len(image)
    w = len(image[0])
    for row in image:
        if len(row) != w:
            raise ValueError("image rows must all have equal length")
    return h, w


def _check_same(a: Image, b: Image) -> tuple[int, int]:
    ha, wa = _shape(a)
    hb, wb = _shape(b)
    if (ha, wa) != (hb, wb):
        raise ValueError(f"images differ in shape: {(ha, wa)} vs {(hb, wb)}")
    return ha, wa


def _foreground(image: Image, threshold: float) -> list[tuple[int, int]]:
    return [
        (y, x)
        for y, row in enumerate(image)
        for x, v in enumerate(row)
        if v >= threshold
    ]


# ---------------------------------------------------------------------------
# ImgMSE (Eq. 5).
# ---------------------------------------------------------------------------


def img_mse(pred: Image, target: Image, threshold: float = 0.5) -> float:
    """PICASSO ImgMSE (Eq. 5): mean of foreground-restricted and global MSE.

    ``target`` foreground (ink) pixels are those ``>= threshold``.  When the
    target has no foreground pixels the foreground term is taken as zero.
    """

    h, w = _check_same(pred, target)
    n_f = 0
    fg_sum = 0.0
    global_sum = 0.0
    for pr, tr in zip(pred, target):
        for p, t in zip(pr, tr):
            d = p - t
            d2 = d * d
            global_sum += d2
            if t >= threshold:
                n_f += 1
                fg_sum += d2
    fg_term = 0.0 if n_f == 0 else fg_sum / (2.0 * n_f)
    global_term = global_sum / (2.0 * w * h)
    return fg_term + global_term


# ---------------------------------------------------------------------------
# Bidirectional Chamfer Distance (Eq. 6).
# ---------------------------------------------------------------------------


def _directed_chamfer(
    src: list[tuple[int, int]], dst: list[tuple[int, int]]
) -> float:
    """Mean over ``src`` of the nearest squared distance to ``dst``."""

    if not src:
        return 0.0
    total = 0.0
    for sy, sx in src:
        best = math.inf
        for dy, dx in dst:
            ddy = sy - dy
            ddx = sx - dx
            d2 = ddy * ddy + ddx * ddx
            if d2 < best:
                best = d2
                if best == 0:
                    break
        total += best
    return total / len(src)


def chamfer_distance(
    pred: Image, target: Image, threshold: float = 0.5
) -> float:
    """PICASSO bidirectional Chamfer Distance (Eq. 6) on foreground pixels.

    Averages the two directed Chamfer terms (pred->target and target->pred),
    each halved as in Eq. 6.  If either foreground set is empty and the other is
    not, returns ``inf`` (no correspondence possible); if both are empty returns
    ``0.0``.
    """

    _check_same(pred, target)
    zp = _foreground(pred, threshold)
    zt = _foreground(target, threshold)
    if not zp and not zt:
        return 0.0
    if not zp or not zt:
        return math.inf
    forward = _directed_chamfer(zp, zt)
    backward = _directed_chamfer(zt, zp)
    return 0.5 * forward + 0.5 * backward


# ---------------------------------------------------------------------------
# Convenience aggregates.
# ---------------------------------------------------------------------------


def pixel_accuracy(
    pred: Image, target: Image, threshold: float = 0.5
) -> float:
    """Fraction of pixels whose binarised ink label agrees."""

    h, w = _check_same(pred, target)
    agree = 0
    for pr, tr in zip(pred, target):
        for p, t in zip(pr, tr):
            if (p >= threshold) == (t >= threshold):
                agree += 1
    return agree / (h * w)


def foreground_iou(
    pred: Image, target: Image, threshold: float = 0.5
) -> float:
    """Hard IoU of the binarised foreground masks (1.0 if both empty)."""

    _check_same(pred, target)
    inter = 0
    union = 0
    for pr, tr in zip(pred, target):
        for p, t in zip(pr, tr):
            pi = p >= threshold
            ti = t >= threshold
            if pi and ti:
                inter += 1
            if pi or ti:
                union += 1
    if union == 0:
        return 1.0
    return inter / union


def render_eval(
    pred: Image, target: Image, threshold: float = 0.5
) -> dict[str, float]:
    """Bundle the image-based metrics into a single report dict."""

    return {
        "img_mse": img_mse(pred, target, threshold),
        "chamfer": chamfer_distance(pred, target, threshold),
        "pixel_accuracy": pixel_accuracy(pred, target, threshold),
        "foreground_iou": foreground_iou(pred, target, threshold),
    }
