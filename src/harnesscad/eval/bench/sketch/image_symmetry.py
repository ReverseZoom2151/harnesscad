"""Image symmetry metric for VQ-CAD generated sketches (Sec. 4.1.3, Eq. 13).

VQ-CAD (Wang et al., CAGD 2024) argues its diffusion model preserves the *implicit*
regularity of CAD sketches better than augmentation-trained baselines, and measures
this with a dedicated **symmetry** metric (Eq. 13) that no other module in the repo
implements (the existing symmetry helpers operate on 2D loop coordinates, not raster
images). Given a rendered sketch image ``I`` of height ``h`` and width ``w``, with
intensity centroid ``O = (O_x, O_y)``, the metric flips the image about its centroid
both horizontally and vertically and averages the absolute pixel differences:

    Symmetry(I) = 1/N * sum_i sum_j | I(i, j) - I(i, 2*O_y - j) |
                + 1/N * sum_i sum_j | I(i, j) - I(2*O_x - i, j) |

where ``i`` indexes rows (0..h-1), ``j`` indexes columns (0..w-1), and ``N = h * w``
is the total pixel count. ``O_x`` is the (row) centroid over ``i`` and ``O_y`` the
(column) centroid over ``j``, both intensity-weighted. Lower values mean a more
symmetric sketch (a perfectly mirror-symmetric image about its centroid scores 0).

The reflected coordinate ``2*O - k`` is a real number; it is rounded to the nearest
integer pixel. A reflection that lands outside the image is treated as background
(intensity ``0.0``), so edge asymmetry is still penalised. When the image carries no
intensity at all the centroid falls back to the geometric centre.

Images are rectangular ``list[list[float]]`` grids. Pure stdlib, deterministic.
"""

from __future__ import annotations

from typing import List, Sequence

Image = Sequence[Sequence[float]]


def _validate(image: Image) -> tuple[int, int]:
    h = len(image)
    if h == 0:
        raise ValueError("image must have at least one row")
    w = len(image[0])
    if w == 0:
        raise ValueError("image rows must be non-empty")
    if any(len(row) != w for row in image):
        raise ValueError("image must be rectangular")
    return h, w


def intensity_centroid(image: Image) -> tuple[float, float]:
    """Intensity-weighted centroid ``(O_x, O_y)`` = (row centroid, column centroid).

    Falls back to the geometric centre ``((h-1)/2, (w-1)/2)`` when total intensity is
    zero. Negative intensities are not allowed (a rendered sketch is non-negative).
    """
    h, w = _validate(image)
    total = 0.0
    sum_i = 0.0
    sum_j = 0.0
    for i in range(h):
        row = image[i]
        for j in range(w):
            v = row[j]
            if v < 0.0:
                raise ValueError("image intensities must be non-negative")
            total += v
            sum_i += v * i
            sum_j += v * j
    if total <= 0.0:
        return ((h - 1) / 2.0, (w - 1) / 2.0)
    return (sum_i / total, sum_j / total)


def _reflect(idx: int, center: float, size: int) -> int | None:
    """Nearest-integer reflection ``round(2*center - idx)``; ``None`` if out of range."""
    r = int(round(2.0 * center - idx))
    if 0 <= r < size:
        return r
    return None


def _pixel(image: Image, i: int | None, j: int | None) -> float:
    """Pixel value with out-of-range reflections treated as background 0.0."""
    if i is None or j is None:
        return 0.0
    return image[i][j]


def horizontal_symmetry(image: Image) -> float:
    """Mean absolute difference under a horizontal (column) flip about ``O_y``."""
    h, w = _validate(image)
    o_x, o_y = intensity_centroid(image)
    n = h * w
    acc = 0.0
    for i in range(h):
        row = image[i]
        for j in range(w):
            rj = _reflect(j, o_y, w)
            acc += abs(row[j] - _pixel(image, i, rj))
    return acc / n


def vertical_symmetry(image: Image) -> float:
    """Mean absolute difference under a vertical (row) flip about ``O_x``."""
    h, w = _validate(image)
    o_x, o_y = intensity_centroid(image)
    n = h * w
    acc = 0.0
    for i in range(h):
        for j in range(w):
            ri = _reflect(i, o_x, h)
            acc += abs(image[i][j] - _pixel(image, ri, j))
    return acc / n


def symmetry_score(image: Image) -> float:
    """The full Eq. 13 metric: horizontal + vertical flip discrepancy. Lower = better."""
    return horizontal_symmetry(image) + vertical_symmetry(image)


def is_more_symmetric(image_a: Image, image_b: Image) -> bool:
    """True when ``image_a`` scores strictly lower (more symmetric) than ``image_b``.

    Mirrors the paper's comparison (Sec. 4.4.1): a generated sketch closer to the
    ground-truth symmetry has the lower :func:`symmetry_score`.
    """
    return symmetry_score(image_a) < symmetry_score(image_b)
