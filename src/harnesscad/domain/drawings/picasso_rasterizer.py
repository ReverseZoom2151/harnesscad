"""picasso_rasterizer -- deterministic explicit rasteriser for CAD sketch primitives.

PICASSO (Karadeniz et al., "PICASSO: A Feed-Forward Framework for Parametric
Inference of CAD Sketches via Rendering Self-Supervision") relies on rendering a
set of parametric primitives to a raster image so an image-level loss can drive
learning *without* parameter labels.  The paper's Sketch Rendering Network (SRN)
learns a **neural, differentiable** renderer; but it explicitly contrasts this
with the "explicit rendering", i.e. the "direct rasterization of parametric
primitives", which it notes is deterministic and non-differentiable (Sec. 4.1).
The evaluation of the paper also reports metrics "on the explicit rendering of
predicted primitive sequences" (Sec. 5 / Sec. 9).

This module implements exactly that deterministic explicit rasteriser.  It draws
the four PICASSO primitive types -- ``line``, ``circle``, ``arc`` and ``point``
(Sec. 3) -- onto a pixel grid using an **anti-aliased signed-distance field**:
every pixel takes the maximum coverage over all primitives, where coverage is a
smooth falloff of the Euclidean distance from the pixel centre to the primitive.
The result is a grayscale image in ``[0, 1]`` (1 = ink, 0 = background) that is a
smooth, sub-pixel-accurate stand-in for the SRN rendering and can be fed directly
into the rendering-consistency losses in :mod:`drawings.picasso_render_loss`.

Coordinates are expressed in a normalised ``[0, 1] x [0, 1]`` sketch canvas.  The
canvas maps to pixel centres so that ``(0, 0)`` is the centre of the top-left
pixel and ``(1, 1)`` the centre of the bottom-right pixel (row / y grows
downward, as in image space).  Stroke width and anti-alias falloff are given in
pixels.  Pure stdlib, fully deterministic (no randomness, no wall clock).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


Point = tuple[float, float]

# ---------------------------------------------------------------------------
# Primitive representation (PICASSO Sec. 3).  These are the *rendering* view of
# the primitives; a line by its two endpoints, a circle by centre + radius, an
# arc by three points on it, a point by its coordinate.  All coordinates are in
# the normalised [0, 1] canvas.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Line:
    """A line segment from ``start`` to ``end``."""

    start: Point
    end: Point


@dataclass(frozen=True)
class Circle:
    """A full circle of ``radius`` about ``center`` (radius in canvas units)."""

    center: Point
    radius: float


@dataclass(frozen=True)
class Arc:
    """A circular arc passing through ``start``, ``mid`` and ``end`` in order."""

    start: Point
    mid: Point
    end: Point


@dataclass(frozen=True)
class Dot:
    """A single point primitive at ``pos`` (PICASSO's ``point`` type)."""

    pos: Point


Primitive = Line | Circle | Arc | Dot


# ---------------------------------------------------------------------------
# Distance helpers (all in pixel space).
# ---------------------------------------------------------------------------


def _dist_point_point(px: float, py: float, qx: float, qy: float) -> float:
    return math.hypot(px - qx, py - qy)


def _dist_point_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Euclidean distance from ``(px, py)`` to segment ``a``->``b``."""

    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    seg_len2 = vx * vx + vy * vy
    if seg_len2 <= 1e-12:
        return _dist_point_point(px, py, ax, ay)
    t = (wx * vx + wy * vy) / seg_len2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx, cy = ax + t * vx, ay + t * vy
    return _dist_point_point(px, py, cx, cy)


def circumcircle(a: Point, b: Point, c: Point) -> tuple[Point, float] | None:
    """Return ``(center, radius)`` of the circle through 3 points, or ``None``.

    Returns ``None`` when the points are (near-)collinear, in which case no
    finite circle exists.
    """

    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    return (ux, uy), r


def _angle(cx: float, cy: float, px: float, py: float) -> float:
    """Angle of ``(px, py)`` about centre ``(cx, cy)`` in ``[0, 2*pi)``."""

    ang = math.atan2(py - cy, px - cx)
    if ang < 0.0:
        ang += 2.0 * math.pi
    return ang


def _arc_contains_angle(
    start_ang: float, mid_ang: float, end_ang: float, theta: float
) -> bool:
    """Whether ``theta`` lies on the arc start->mid->end (traversal direction).

    The arc goes from ``start_ang`` to ``end_ang`` in whichever rotational
    direction passes through ``mid_ang``.
    """

    def _ccw_span(frm: float, to: float) -> float:
        s = to - frm
        while s < 0.0:
            s += 2.0 * math.pi
        while s >= 2.0 * math.pi:
            s -= 2.0 * math.pi
        return s

    # Decide direction by checking whether mid is reached before end going CCW.
    span_end_ccw = _ccw_span(start_ang, end_ang)
    span_mid_ccw = _ccw_span(start_ang, mid_ang)
    if span_mid_ccw <= span_end_ccw:
        # CCW traversal: theta is inside if its CCW offset <= end offset.
        return _ccw_span(start_ang, theta) <= span_end_ccw + 1e-9
    # CW traversal: mirror by measuring CW spans.
    span_end_cw = 2.0 * math.pi - span_end_ccw
    return (2.0 * math.pi - _ccw_span(start_ang, theta)) <= span_end_cw + 1e-9


def _dist_point_arc(
    px: float,
    py: float,
    start: Point,
    mid: Point,
    end: Point,
) -> float:
    """Distance from ``(px, py)`` to the circular arc start->mid->end."""

    cc = circumcircle(start, mid, end)
    if cc is None:
        # Collinear: degrade to two segments start->mid->end.
        d1 = _dist_point_segment(px, py, start[0], start[1], mid[0], mid[1])
        d2 = _dist_point_segment(px, py, mid[0], mid[1], end[0], end[1])
        return min(d1, d2)
    (cx, cy), r = cc
    start_ang = _angle(cx, cy, start[0], start[1])
    mid_ang = _angle(cx, cy, mid[0], mid[1])
    end_ang = _angle(cx, cy, end[0], end[1])
    theta = _angle(cx, cy, px, py)
    if _arc_contains_angle(start_ang, mid_ang, end_ang, theta):
        # Radial distance to the circle.
        return abs(_dist_point_point(px, py, cx, cy) - r)
    # Nearest endpoint otherwise.
    de1 = _dist_point_point(px, py, start[0], start[1])
    de2 = _dist_point_point(px, py, end[0], end[1])
    return min(de1, de2)


# ---------------------------------------------------------------------------
# Rasterisation.
# ---------------------------------------------------------------------------


def _canvas_to_pixel(x: float, y: float, width: int, height: int) -> Point:
    """Map a normalised canvas coordinate to pixel-centre coordinates."""

    return x * (width - 1), y * (height - 1)


def _primitive_pixel_distance(
    prim: Primitive, px: float, py: float, width: int, height: int
) -> float:
    if isinstance(prim, Line):
        ax, ay = _canvas_to_pixel(*prim.start, width, height)
        bx, by = _canvas_to_pixel(*prim.end, width, height)
        return _dist_point_segment(px, py, ax, ay, bx, by)
    if isinstance(prim, Dot):
        qx, qy = _canvas_to_pixel(*prim.pos, width, height)
        return _dist_point_point(px, py, qx, qy)
    if isinstance(prim, Circle):
        cx, cy = _canvas_to_pixel(*prim.center, width, height)
        # Radius scales with the average pixel extent.
        r = prim.radius * ((width - 1) + (height - 1)) / 2.0
        return abs(_dist_point_point(px, py, cx, cy) - r)
    if isinstance(prim, Arc):
        s = _canvas_to_pixel(*prim.start, width, height)
        m = _canvas_to_pixel(*prim.mid, width, height)
        e = _canvas_to_pixel(*prim.end, width, height)
        return _dist_point_arc(px, py, s, m, e)
    raise TypeError(f"unknown primitive type: {type(prim)!r}")


def _coverage(dist: float, half_width: float, aa: float) -> float:
    """Anti-aliased coverage for a distance-field stroke.

    Full ink within ``half_width`` pixels of the primitive; linearly fades to
    zero across an ``aa``-pixel band; zero beyond.
    """

    if dist <= half_width:
        return 1.0
    if aa <= 0.0:
        return 0.0
    if dist >= half_width + aa:
        return 0.0
    return 1.0 - (dist - half_width) / aa


def rasterize(
    primitives: list[Primitive],
    width: int = 128,
    height: int = 128,
    stroke_width: float = 1.5,
    aa: float = 1.0,
) -> list[list[float]]:
    """Rasterise ``primitives`` onto a ``height`` x ``width`` grayscale grid.

    Returns a row-major list of rows; each pixel is a coverage in ``[0, 1]``
    (1 = ink).  ``stroke_width`` is the *full* stroke thickness in pixels
    (half-width = ``stroke_width / 2``) and ``aa`` the anti-alias band width in
    pixels.  Coverage combines across primitives with ``max`` (opaque union).
    """

    if width < 2 or height < 2:
        raise ValueError("width and height must both be >= 2")
    half_width = stroke_width / 2.0
    reach = half_width + max(aa, 0.0)
    img = [[0.0 for _ in range(width)] for _ in range(height)]
    if not primitives:
        return img
    for py in range(height):
        row = img[py]
        for px in range(width):
            best = 0.0
            for prim in primitives:
                dist = _primitive_pixel_distance(
                    prim, float(px), float(py), width, height
                )
                if dist >= reach:
                    continue
                cov = _coverage(dist, half_width, aa)
                if cov > best:
                    best = cov
                    if best >= 1.0:
                        break
            row[px] = best
    return img


def binarize(
    image: list[list[float]], threshold: float = 0.5
) -> list[list[int]]:
    """Threshold a grayscale raster to a ``{0, 1}`` binary image."""

    return [[1 if v >= threshold else 0 for v in row] for row in image]


def foreground_pixels(
    image: list[list[float]], threshold: float = 0.5
) -> list[tuple[int, int]]:
    """Return ``(row, col)`` coordinates of foreground (ink) pixels."""

    out: list[tuple[int, int]] = []
    for y, row in enumerate(image):
        for x, v in enumerate(row):
            if v >= threshold:
                out.append((y, x))
    return out
