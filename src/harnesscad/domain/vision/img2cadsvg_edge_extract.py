"""img2cadsvg_edge_extract -- synthetic sketch / edge-map extraction for SVG.

To build its ABC-mono dataset (and to accept sketch/edge-map input), Img2CAD
extracts the *candidate strokes* -- the raw material of the Structured Visual
Geometry -- from rendered images with a classical Canny-style pipeline
(paper, Sec. V.A):

    "We first apply Gaussian smoothing filters to the images to reduce noise ...
    Then, the gradients and gradient directions of the images are calculated,
    followed by Non-Maximum Suppression (NMS) to eliminate non-edge pixels,
    retaining only some thin lines as the candidate strokes.  Finally, the edge
    detection process is completed by applying high and low thresholding and
    connecting edges."

Every stage is deterministic classical image processing, implemented here on a
plain grayscale image (a list-of-lists of floats):

1. :func:`gaussian_blur`   -- separable Gaussian smoothing;
2. :func:`sobel_gradients` -- gradient magnitude and direction (Sobel);
3. :func:`non_max_suppression` -- thin edges to 1px along the gradient;
4. :func:`hysteresis`      -- high/low double-threshold + connectivity linking.

:func:`extract_edges` runs the full pipeline and returns a binary edge map; the
retained pixels are the "candidate strokes".  Pure stdlib, deterministic.
"""

from __future__ import annotations

import math


Image = list[list[float]]


def _dims(img: Image) -> tuple[int, int]:
    h = len(img)
    if h == 0:
        raise ValueError("empty image")
    w = len(img[0])
    if w == 0 or any(len(row) != w for row in img):
        raise ValueError("image must be a non-empty rectangular grid")
    return h, w


def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def gaussian_kernel(sigma: float, radius: int | None = None) -> list[float]:
    """Normalised 1D Gaussian kernel (for separable blur)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if radius is None:
        radius = max(1, int(math.ceil(3.0 * sigma)))
    vals = [math.exp(-(x * x) / (2.0 * sigma * sigma)) for x in range(-radius, radius + 1)]
    s = sum(vals)
    return [v / s for v in vals]


def _convolve_axis(img: Image, kernel: list[float], horizontal: bool) -> Image:
    h, w = _dims(img)
    r = len(kernel) // 2
    out = [[0.0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for k, kv in enumerate(kernel):
                off = k - r
                if horizontal:
                    xx = _clamp(x + off, 0, w - 1)
                    acc += kv * img[y][xx]
                else:
                    yy = _clamp(y + off, 0, h - 1)
                    acc += kv * img[yy][x]
            out[y][x] = acc
    return out


def gaussian_blur(img: Image, sigma: float = 1.0) -> Image:
    """Separable Gaussian smoothing with edge-clamped borders."""
    kernel = gaussian_kernel(sigma)
    return _convolve_axis(_convolve_axis(img, kernel, True), kernel, False)


def sobel_gradients(img: Image) -> tuple[Image, Image]:
    """Return ``(magnitude, direction)`` via the Sobel operator.

    Direction is in radians in ``(-pi, pi]`` (``atan2(gy, gx)``).
    """
    h, w = _dims(img)
    mag = [[0.0] * w for _ in range(h)]
    ang = [[0.0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            gx = 0.0
            gy = 0.0
            for j in (-1, 0, 1):
                for i in (-1, 0, 1):
                    yy = _clamp(y + j, 0, h - 1)
                    xx = _clamp(x + i, 0, w - 1)
                    v = img[yy][xx]
                    # Sobel kernels
                    kx = i * (2 if j == 0 else 1)
                    ky = j * (2 if i == 0 else 1)
                    gx += kx * v
                    gy += ky * v
            mag[y][x] = math.hypot(gx, gy)
            ang[y][x] = math.atan2(gy, gx)
    return mag, ang


def _quantize_dir(angle: float) -> int:
    """Quantise a gradient angle to one of 4 NMS neighbour directions (0..3)."""
    deg = math.degrees(angle) % 180.0
    if deg < 22.5 or deg >= 157.5:
        return 0  # horizontal gradient -> compare left/right
    if deg < 67.5:
        return 1  # diagonal /
    if deg < 112.5:
        return 2  # vertical gradient -> compare up/down
    return 3  # diagonal \


_OFFSETS = {0: (0, 1), 1: (-1, 1), 2: (1, 0), 3: (1, 1)}


def non_max_suppression(mag: Image, ang: Image) -> Image:
    """Thin edges: keep a pixel only if it is a local max along the gradient."""
    h, w = _dims(mag)
    out = [[0.0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            q = _quantize_dir(ang[y][x])
            dy, dx = _OFFSETS[q]
            m = mag[y][x]
            n1 = mag[y + dy][x + dx] if 0 <= y + dy < h and 0 <= x + dx < w else 0.0
            n2 = mag[y - dy][x - dx] if 0 <= y - dy < h and 0 <= x - dx < w else 0.0
            if m >= n1 and m >= n2:
                out[y][x] = m
    return out


def hysteresis(nms: Image, low: float, high: float) -> list[list[int]]:
    """Double-threshold + 8-connectivity linking of strong/weak edges.

    Pixels ``>= high`` are strong; pixels in ``[low, high)`` become edges only if
    connected (8-neighbourhood) to a strong edge.  Returns a 0/1 edge map.
    """
    if low < 0 or high < low:
        raise ValueError("require 0 <= low <= high")
    h, w = _dims(nms)
    strong = [[nms[y][x] >= high for x in range(w)] for y in range(h)]
    weak = [[low <= nms[y][x] < high for x in range(w)] for y in range(h)]
    out = [[1 if strong[y][x] else 0 for x in range(w)] for y in range(h)]
    # BFS from strong pixels through weak pixels
    stack = [(y, x) for y in range(h) for x in range(w) if strong[y][x]]
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and weak[ny][nx] and not out[ny][nx]:
                    out[ny][nx] = 1
                    stack.append((ny, nx))
    return out


def extract_edges(
    img: Image, sigma: float = 1.0, low: float = 0.1, high: float = 0.3
) -> list[list[int]]:
    """Full Canny-style pipeline: blur -> gradients -> NMS -> hysteresis.

    Returns a 0/1 edge map; the 1-pixels are the paper's "candidate strokes".
    """
    blurred = gaussian_blur(img, sigma)
    mag, ang = sobel_gradients(blurred)
    thin = non_max_suppression(mag, ang)
    return hysteresis(thin, low, high)


def edge_pixels(edge_map: list[list[int]]) -> list[tuple[int, int]]:
    """List ``(y, x)`` coordinates of edge pixels, row-major (deterministic)."""
    return [
        (y, x)
        for y, row in enumerate(edge_map)
        for x, v in enumerate(row)
        if v
    ]
