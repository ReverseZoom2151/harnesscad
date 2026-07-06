"""First-derivative and Sobel edge extraction for 2D disk-view wheel images.

Deterministic image-processing pieces of the 3D-CAD-automation stage
(Section 4.4.1) of:

    Yoo et al., "Integrating deep learning into CAD/CAE system: generative design
    and evaluation of 3D conceptual wheel", Struct. Multidisc. Optim. 64 (2021)
    2725-2747.

Before a 2D wheel image can be turned into CAD splines, its edges must be
extracted.  The paper computes an edge (gradient) map from a brightness image
``f`` and thresholds it.  This module implements those closed-form operators:

    * First-derivative differences (Section 4.4.1)::

          Gx = f(x+1, y) - f(x, y)
          Gy = f(x, y+1) - f(x, y)

    * Gradient magnitude (equation 3)::

          |grad G| = sqrt(Gx^2 + Gy^2)

    * The Sobel operator (Kanopoulos et al. 1988), which the paper prefers
      because it "can extract diagonal edges well".  The 3x3 Sobel kernels are::

          Sx = [[-1, 0, 1],    Sy = [[-1, -2, -1],
                [-2, 0, 2],           [ 0,  0,  0],
                [-1, 0, 1]]           [ 1,  2,  1]]

    * A threshold step turning the gradient magnitude into a binary edge map,
      "where the boundary is an area with a large change rate in brightness".

Images are represented as a rectangular list-of-rows (row-major) of numeric
brightness values.  All functions are deterministic and stdlib-only (``math``).
"""

from __future__ import annotations

import math
from typing import List, Sequence


Image = Sequence[Sequence[float]]


def _dims(image: Image) -> tuple:
    if not image or not image[0]:
        raise ValueError("image must be a non-empty 2D grid")
    h = len(image)
    w = len(image[0])
    for row in image:
        if len(row) != w:
            raise ValueError("image must be rectangular")
    return h, w


def gradient_x(image: Image) -> List[List[float]]:
    """Forward horizontal difference ``Gx = f(x+1, y) - f(x, y)``.

    The last column replicates its neighbour's difference (edge padding) so the
    output has the same shape as the input.
    """
    h, w = _dims(image)
    out: List[List[float]] = []
    for y in range(h):
        row: List[float] = []
        for x in range(w):
            x1 = x + 1 if x + 1 < w else x
            row.append(float(image[y][x1]) - float(image[y][x]))
        out.append(row)
    return out


def gradient_y(image: Image) -> List[List[float]]:
    """Forward vertical difference ``Gy = f(x, y+1) - f(x, y)``.

    The last row replicates its neighbour's difference (edge padding).
    """
    h, w = _dims(image)
    out: List[List[float]] = []
    for y in range(h):
        y1 = y + 1 if y + 1 < h else y
        row: List[float] = []
        for x in range(w):
            row.append(float(image[y1][x]) - float(image[y][x]))
        out.append(row)
    return out


def gradient_magnitude(gx: Image, gy: Image) -> List[List[float]]:
    """Elementwise ``sqrt(Gx^2 + Gy^2)`` (equation 3)."""
    h, w = _dims(gx)
    hy, wy = _dims(gy)
    if (h, w) != (hy, wy):
        raise ValueError("gx and gy must have the same shape")
    out: List[List[float]] = []
    for y in range(h):
        row: List[float] = []
        for x in range(w):
            a = float(gx[y][x])
            b = float(gy[y][x])
            row.append(math.sqrt(a * a + b * b))
        out.append(row)
    return out


def first_derivative_edges(image: Image) -> List[List[float]]:
    """Gradient magnitude using the first-derivative differences (Section 4.4.1)."""
    return gradient_magnitude(gradient_x(image), gradient_y(image))


# 3x3 Sobel kernels.
_SOBEL_X = ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))
_SOBEL_Y = ((-1, -2, -1), (0, 0, 0), (1, 2, 1))


def _convolve3(image: Image, kernel) -> List[List[float]]:
    """3x3 convolution with edge (clamp) padding, same output shape."""
    h, w = _dims(image)
    out: List[List[float]] = []
    for y in range(h):
        row: List[float] = []
        for x in range(w):
            acc = 0.0
            for ky in range(3):
                yy = min(max(y + ky - 1, 0), h - 1)
                for kx in range(3):
                    xx = min(max(x + kx - 1, 0), w - 1)
                    acc += kernel[ky][kx] * float(image[yy][xx])
            row.append(acc)
        out.append(row)
    return out


def sobel_x(image: Image) -> List[List[float]]:
    """Horizontal Sobel response (clamp-padded, same shape)."""
    return _convolve3(image, _SOBEL_X)


def sobel_y(image: Image) -> List[List[float]]:
    """Vertical Sobel response (clamp-padded, same shape)."""
    return _convolve3(image, _SOBEL_Y)


def sobel_magnitude(image: Image) -> List[List[float]]:
    """Sobel gradient magnitude ``sqrt(Sx^2 + Sy^2)`` (Kanopoulos et al. 1988)."""
    return gradient_magnitude(sobel_x(image), sobel_y(image))


def threshold_edges(magnitude: Image, threshold: float) -> List[List[int]]:
    """Binary edge map: 1 where ``magnitude >= threshold`` else 0 (Section 4.4.1)."""
    h, w = _dims(magnitude)
    return [
        [1 if float(magnitude[y][x]) >= threshold else 0 for x in range(w)]
        for y in range(h)
    ]


def edge_coordinates(edge_map: Sequence[Sequence[int]]) -> List[tuple]:
    """Return ``(x, y)`` coordinates of every non-zero pixel in an edge map.

    Mirrors the paper's step of converting the detected edges into coordinate
    data.  Coordinates are returned in row-major (y-major) order.
    """
    h, w = _dims(edge_map)
    coords: List[tuple] = []
    for y in range(h):
        for x in range(w):
            if edge_map[y][x]:
                coords.append((x, y))
    return coords
