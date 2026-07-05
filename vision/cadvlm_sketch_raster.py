"""Deterministic sketch-image rasteriser for CadVLM's vision branch.

CadVLM (Wu et al.) feeds each engineering sketch to the model in *two* modalities:
the primitive token sequence (handled by :mod:`ingest.cadvlm_codec`) and a rendered
raster image ``I`` (the paper renders every sketch at ``224 x 224`` and fine-tunes a
ViT-MAE on it). The learned encoder is out of scope, but the *rendering* itself is a
pure, deterministic function of the sketch geometry -- and nothing in the repository
actually turns sketch primitives into pixels (``quality.sketch_crossmodal`` only
consumes a caller-supplied ``rasterizer`` callback). This module is that missing
concrete rasteriser.

It rasterises the three CadVLM entity types -- lines (start/end), arcs
(start/mid/end) and circles (four circumference points or centre+radius) -- onto a
square pixel grid, mapping the paper's quantised ``[1, 64]`` coordinate space (or any
supplied ``coord_range``) to the ``[0, resolution)`` pixel range. Lines use a
Bresenham traversal, circles the integer midpoint-circle algorithm, and arcs are
sampled by angle and Bresenham-connected so the stroke stays 8-connected regardless
of curvature. Everything is integer arithmetic over a fixed grid, so the same sketch
always renders to the same pixel set.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, cos, hypot, pi, sin


TWO_PI = 2.0 * pi


@dataclass(frozen=True)
class RasterImage:
    """A rasterised sketch: the set of lit pixels on a ``resolution`` square grid."""

    resolution: int
    pixels: frozenset

    @property
    def occupancy(self) -> int:
        """Number of lit pixels."""
        return len(self.pixels)

    def to_grid(self) -> tuple:
        """Dense ``resolution x resolution`` grid of 0/1 ints (row-major, y then x)."""
        n = self.resolution
        return tuple(
            tuple(1 if (x, y) in self.pixels else 0 for x in range(n))
            for y in range(n)
        )


def _to_pixel(point, resolution: int, low: float, high: float) -> tuple:
    """Map a coordinate to an integer pixel, clamped into ``[0, resolution)``."""
    span = high - low if high != low else 1.0
    px = round((point[0] - low) / span * (resolution - 1))
    py = round((point[1] - low) / span * (resolution - 1))
    px = min(resolution - 1, max(0, px))
    py = min(resolution - 1, max(0, py))
    return (px, py)


def _bresenham(a, b):
    """8-connected integer line from pixel ``a`` to pixel ``b`` (inclusive)."""
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    out = []
    while True:
        out.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return out


def _midpoint_circle(center, radius: int):
    """Integer midpoint-circle rasterisation returning the 8-symmetric pixel set."""
    cx, cy = center
    if radius <= 0:
        return {(cx, cy)}
    x = radius
    y = 0
    err = 1 - radius
    pts = set()
    while x >= y:
        for px, py in (
            (cx + x, cy + y), (cx - x, cy + y), (cx + x, cy - y), (cx - x, cy - y),
            (cx + y, cy + x), (cx - y, cy + x), (cx + y, cy - x), (cx - y, cy - x),
        ):
            pts.add((px, py))
        y += 1
        if err < 0:
            err += 2 * y + 1
        else:
            x -= 1
            err += 2 * (y - x) + 1
    return pts


def _circumcenter(p0, pm, p1):
    """Centre of the circle through three points, or ``None`` if collinear."""
    ax, ay = p0
    bx, by = pm
    cx, cy = p1
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return (ux, uy)


def _arc_polyline(p0, pm, p1, resolution, low, high):
    """Sample the arc start->mid->end into coordinate-space points."""
    center = _circumcenter(p0, pm, p1)
    if center is None:
        return (p0, pm, p1)
    r = hypot(p0[0] - center[0], p0[1] - center[1])
    a0 = atan2(p0[1] - center[1], p0[0] - center[0])
    am = atan2(pm[1] - center[1], pm[0] - center[0])
    a1 = atan2(p1[1] - center[1], p1[0] - center[0])
    ccw_span = (a1 - a0) % TWO_PI
    mid_ccw = (am - a0) % TWO_PI
    if mid_ccw <= ccw_span:            # mid lies on the ccw sweep
        span = ccw_span
    else:                              # otherwise sweep clockwise
        span = ccw_span - TWO_PI
    # radius in pixels, to pick a step count that keeps the stroke connected.
    r_px = r / ((high - low) if high != low else 1.0) * (resolution - 1)
    steps = max(4, int(ceil(abs(span) * max(r_px, 1.0))))
    return tuple(
        (center[0] + r * cos(a0 + span * t / steps),
         center[1] + r * sin(a0 + span * t / steps))
        for t in range(steps + 1)
    )


def _entity_points(entity):
    """Coordinate points a CadVLM entity is defined by (codec dict format)."""
    kind = entity["type"]
    if kind == "line":
        return (entity["start"], entity["end"])
    if kind == "arc":
        return (entity["start"], entity["mid"], entity["end"])
    if kind == "circle":
        if "points" in entity:
            return tuple(entity["points"])
        return (entity["center"], entity["radius"])  # centre + radius form
    raise ValueError(f"unknown entity type {kind!r}")


def rasterize_entity(entity, resolution: int = 224,
                     coord_range: tuple = (1.0, 64.0)) -> frozenset:
    """Rasterise one entity to a pixel set on a ``resolution`` square grid."""
    low, high = float(coord_range[0]), float(coord_range[1])
    kind = entity["type"]
    pix = set()
    if kind == "line":
        a = _to_pixel(entity["start"], resolution, low, high)
        b = _to_pixel(entity["end"], resolution, low, high)
        pix.update(_bresenham(a, b))
    elif kind == "arc":
        poly = _arc_polyline(entity["start"], entity["mid"], entity["end"],
                             resolution, low, high)
        prev = _to_pixel(poly[0], resolution, low, high)
        for point in poly[1:]:
            cur = _to_pixel(point, resolution, low, high)
            pix.update(_bresenham(prev, cur))
            prev = cur
    elif kind == "circle":
        if "points" in entity:
            pts = tuple(entity["points"])
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            r = sum(hypot(p[0] - cx, p[1] - cy) for p in pts) / len(pts)
        else:
            cx, cy = entity["center"]
            r = float(entity["radius"])
        center = _to_pixel((cx, cy), resolution, low, high)
        edge = _to_pixel((cx + r, cy), resolution, low, high)
        r_px = abs(edge[0] - center[0])
        pix.update(
            (px, py) for px, py in _midpoint_circle(center, r_px)
            if 0 <= px < resolution and 0 <= py < resolution
        )
    else:
        raise ValueError(f"unknown entity type {kind!r}")
    return frozenset(pix)


def rasterize_sketch(entities, resolution: int = 224,
                     coord_range: tuple = (1.0, 64.0)) -> RasterImage:
    """Rasterise a whole sketch (iterable of entity dicts) into a :class:`RasterImage`."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    pixels = set()
    for entity in entities:
        pixels |= rasterize_entity(entity, resolution, coord_range)
    return RasterImage(resolution=resolution, pixels=frozenset(pixels))
