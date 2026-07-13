"""picasso_render_loss -- rendering self-supervision losses for CAD sketches.

The core idea of PICASSO is *rendering self-supervision*: instead of comparing
predicted primitive parameters to ground-truth parameters (which are often
unavailable), the predicted primitives are rendered to a raster image and
compared to the target sketch image with an **image-level loss** (Sec. 4.3).
Because the comparison lives entirely in image space, no parameter labels are
needed.

The paper's central training objective is the **multiscale l2 loss** (Eq. 1)::

    L_ml2 = sum_{s in S} || d_s(render(pred)) - d_s(target) ||_2^2

where ``d_s`` is a downsampling to pyramid level ``s`` (5 levels in the paper).
The multiscale construction is what "ensures that if the rendered sketch only
partially overlaps with the input raster image at a higher resolution, coarser
resolutions can produce informative gradients" (Sec. 4.3).

This module implements the deterministic image-consistency machinery around that
idea, operating on grayscale rasters produced by
:mod:`drawings.picasso_rasterizer`:

* :func:`downsample` / :func:`image_pyramid` -- 2x2 average-pool pyramids.
* :func:`l2_loss` -- single-scale squared-L2 image loss.
* :func:`multiscale_l2_loss` -- the PICASSO Eq. 1 objective.
* :func:`bce_loss` -- the binary-cross-entropy alternative ablated in Table 4.
* :func:`raster_iou` / :func:`iou_loss` -- soft intersection-over-union between
  rendered prediction and target (a scale-free consistency signal).
* :func:`distance_field_l2` -- L2 between the two rasters' distance transforms
  (a smoother, longer-range consistency term than raw pixel L2).

All functions are pure stdlib and deterministic.
"""

from __future__ import annotations

import math

Image = list[list[float]]


# ---------------------------------------------------------------------------
# Shape / validation helpers.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Multiscale image pyramids.
# ---------------------------------------------------------------------------


def downsample(image: Image) -> Image:
    """Halve resolution by 2x2 average pooling (odd rows/cols dropped)."""

    h, w = _shape(image)
    hh, hw = h // 2, w // 2
    if hh == 0 or hw == 0:
        raise ValueError("image too small to downsample")
    out: Image = []
    for y in range(hh):
        row: list[float] = []
        for x in range(hw):
            s = (
                image[2 * y][2 * x]
                + image[2 * y][2 * x + 1]
                + image[2 * y + 1][2 * x]
                + image[2 * y + 1][2 * x + 1]
            )
            row.append(s / 4.0)
        out.append(row)
    return out


def image_pyramid(image: Image, levels: int = 5) -> list[Image]:
    """Return ``levels`` images: full resolution then successive 2x pool-downs.

    Stops early if the image can no longer be halved.
    """

    if levels < 1:
        raise ValueError("levels must be >= 1")
    pyramid = [image]
    cur = image
    for _ in range(levels - 1):
        h, w = _shape(cur)
        if h // 2 == 0 or w // 2 == 0:
            break
        cur = downsample(cur)
        pyramid.append(cur)
    return pyramid


# ---------------------------------------------------------------------------
# Single-scale losses.
# ---------------------------------------------------------------------------


def l2_loss(pred: Image, target: Image) -> float:
    """Sum of squared per-pixel differences (unnormalised squared L2)."""

    _check_same(pred, target)
    total = 0.0
    for pr, tr in zip(pred, target):
        for p, t in zip(pr, tr):
            d = p - t
            total += d * d
    return total


def mse(pred: Image, target: Image) -> float:
    """Mean squared error over all pixels."""

    h, w = _check_same(pred, target)
    return l2_loss(pred, target) / (h * w)


def bce_loss(pred: Image, target: Image, eps: float = 1e-7) -> float:
    """Mean binary cross-entropy (Table 4 ablation baseline).

    ``pred`` values are clamped into ``[eps, 1-eps]``; ``target`` treated as a
    soft label in ``[0, 1]``.
    """

    h, w = _check_same(pred, target)
    total = 0.0
    for pr, tr in zip(pred, target):
        for p, t in zip(pr, tr):
            pc = min(max(p, eps), 1.0 - eps)
            total += -(t * math.log(pc) + (1.0 - t) * math.log(1.0 - pc))
    return total / (h * w)


# ---------------------------------------------------------------------------
# Multiscale l2 loss (PICASSO Eq. 1).
# ---------------------------------------------------------------------------


def multiscale_l2_loss(
    pred: Image, target: Image, levels: int = 5
) -> float:
    """PICASSO's multiscale squared-L2 loss (Eq. 1).

    Builds matching ``levels``-deep pyramids of ``pred`` and ``target`` and sums
    the squared-L2 loss at every pyramid level.
    """

    _check_same(pred, target)
    pyr_p = image_pyramid(pred, levels)
    pyr_t = image_pyramid(target, levels)
    n = min(len(pyr_p), len(pyr_t))
    return sum(l2_loss(pyr_p[i], pyr_t[i]) for i in range(n))


# ---------------------------------------------------------------------------
# Intersection-over-union consistency.
# ---------------------------------------------------------------------------


def raster_iou(pred: Image, target: Image) -> float:
    """Soft IoU between two grayscale rasters.

    Uses the fuzzy-set form: intersection = sum(min), union = sum(max).  Returns
    ``1.0`` when both images are empty (perfect trivial agreement).
    """

    _check_same(pred, target)
    inter = 0.0
    union = 0.0
    for pr, tr in zip(pred, target):
        for p, t in zip(pr, tr):
            inter += min(p, t)
            union += max(p, t)
    if union <= 0.0:
        return 1.0
    return inter / union


def iou_loss(pred: Image, target: Image) -> float:
    """``1 - soft-IoU``; zero when the rasters coincide."""

    return 1.0 - raster_iou(pred, target)


# ---------------------------------------------------------------------------
# Distance-field consistency.
# ---------------------------------------------------------------------------


def distance_transform(
    image: Image, threshold: float = 0.5
) -> list[list[float]]:
    """Euclidean distance from each pixel to the nearest foreground pixel.

    Foreground = pixels ``>= threshold``.  Uses the exact two-pass chamfer-free
    brute check via a separable lower bound is avoided for determinism; instead a
    simple exact BFS-like expansion on the integer grid with true Euclidean
    distance computed against collected foreground coordinates.  If there are no
    foreground pixels every distance is the grid diagonal.
    """

    h, w = _shape(image)
    fg = [
        (y, x)
        for y in range(h)
        for x in range(w)
        if image[y][x] >= threshold
    ]
    diag = math.hypot(h, w)
    if not fg:
        return [[diag for _ in range(w)] for _ in range(h)]
    out = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if image[y][x] >= threshold:
                out[y][x] = 0.0
                continue
            best = diag
            for fy, fx in fg:
                d = math.hypot(y - fy, x - fx)
                if d < best:
                    best = d
            out[y][x] = best
    return out


def distance_field_l2(
    pred: Image, target: Image, threshold: float = 0.5, normalize: bool = True
) -> float:
    """Squared-L2 distance between the distance transforms of two rasters.

    A smoother, longer-range consistency signal than raw pixel L2: partially
    overlapping strokes still receive a graded penalty proportional to how far
    apart their ink is.  ``normalize`` divides by pixel count (mean).
    """

    h, w = _check_same(pred, target)
    dp = distance_transform(pred, threshold)
    dt = distance_transform(target, threshold)
    total = 0.0
    for py, ty in zip(dp, dt):
        for p, t in zip(py, ty):
            d = p - t
            total += d * d
    return total / (h * w) if normalize else total
