"""img2cadsvg_hat_field -- closed-form 4D geometric vector field for wireframes.

Img2CAD encodes line segments with the **Holistic Attention Transformer (HAT)
field** through a "closed-form 4D geometric vector field, which generates dense
sets of line segments and extracts endpoint proposals from heatmaps" (paper,
Sec. II / IV, following the Holistically-Attracted Wireframe Parsing line of work
of Xue et al.).  The neural regressor that *predicts* the field from an image is
learned and out of scope, but the **closed-form geometry** -- the invertible map
between a line segment and its 4D attraction vector at a pixel -- is deterministic
and is exactly what "generates dense sets of line segments" from a dense field.

For a pixel ``p`` that is *attracted* to a line segment with endpoints
``x1, x2``, we encode the segment relative to ``p`` by a 4-vector::

    (d, phi, t1, t2)

* drop the perpendicular from ``p`` to the *infinite* line through the segment;
  ``d >= 0`` is the perpendicular distance and ``phi`` the angle of the unit
  normal ``n = (cos phi, sin phi)`` that points from ``p`` toward the foot;
* the foot is ``c = p + d * n``; along the line direction
  ``u = (-sin phi, cos phi)`` the two endpoints sit at signed offsets ``t1, t2``
  so that ``x1 = c + t1 * u`` and ``x2 = c + t2 * u``.

This 4D vector is a *closed-form* function of ``(p, x1, x2)`` and is **exactly
invertible**: from ``(p, d, phi, t1, t2)`` the endpoints are recovered.  A dense
field assigns every pixel its nearest segment's 4-vector; decoding each pixel
regenerates the segment, and identical segments from many pixels collapse under
:func:`decode_field` -- reproducing the paper's "dense sets of line segments"
step.  Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


Point = tuple[float, float]
Seg = tuple[Point, Point]


@dataclass(frozen=True)
class HatVector:
    d: float
    phi: float
    t1: float
    t2: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.d, self.phi, self.t1, self.t2)


def encode(p: Point, seg: Seg) -> HatVector:
    """Closed-form 4D attraction vector of ``seg`` at pixel ``p``.

    Raises ``ValueError`` for a degenerate (zero-length) segment.
    """
    (x1, y1), (x2, y2) = seg
    ux, uy = x2 - x1, y2 - y1
    length = math.hypot(ux, uy)
    if length == 0.0:
        raise ValueError("cannot encode a zero-length segment")
    ux, uy = ux / length, uy / length  # unit line direction
    px, py = p
    # signed offsets of the endpoints along the line, from the foot of p.
    # foot parameter s0 = projection of p onto the line (origin x1):
    s0 = (px - x1) * ux + (py - y1) * uy
    fx, fy = x1 + s0 * ux, y1 + s0 * uy  # foot point c
    # perpendicular from p to foot
    nx, ny = fx - px, fy - py
    d = math.hypot(nx, ny)
    if d == 0.0:
        # p lies on the line; pick the left normal of u as a canonical phi
        nx, ny = -uy, ux
    else:
        nx, ny = nx / d, ny / d
    phi = math.atan2(ny, nx)
    # Measure the endpoint offsets along the SAME direction convention decode
    # uses -- u = left rotation of the normal n = (-sin phi, cos phi) -- so the
    # round-trip is exact regardless of the segment's stored orientation.
    du_x, du_y = -ny, nx
    t1 = (x1 - fx) * du_x + (y1 - fy) * du_y
    t2 = (x2 - fx) * du_x + (y2 - fy) * du_y
    return HatVector(d=d, phi=phi, t1=t1, t2=t2)


def decode(p: Point, vec: HatVector) -> Seg:
    """Invert :func:`encode`: recover the segment endpoints from ``(p, vec)``."""
    px, py = p
    nx, ny = math.cos(vec.phi), math.sin(vec.phi)
    # line direction u is the left rotation of the normal n
    ux, uy = -ny, nx
    cx, cy = px + vec.d * nx, py + vec.d * ny  # foot
    x1 = (cx + vec.t1 * ux, cy + vec.t1 * uy)
    x2 = (cx + vec.t2 * ux, cy + vec.t2 * uy)
    return (x1, x2)


def _dist2(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def point_segment_distance2(p: Point, seg: Seg) -> float:
    """Squared distance from ``p`` to the *segment* (clamped to endpoints)."""
    (x1, y1), (x2, y2) = seg
    ux, uy = x2 - x1, y2 - y1
    L2 = ux * ux + uy * uy
    if L2 == 0.0:
        return _dist2(p, (x1, y1))
    t = ((p[0] - x1) * ux + (p[1] - y1) * uy) / L2
    t = max(0.0, min(1.0, t))
    foot = (x1 + t * ux, y1 + t * uy)
    return _dist2(p, foot)


def build_field(
    pixels: list[Point], segments: list[Seg]
) -> list[HatVector]:
    """Dense field: each pixel encodes its *nearest* segment's 4-vector.

    Nearest is by clamped point-to-segment distance (ties -> lowest index).
    """
    if not segments:
        raise ValueError("need at least one segment to build a field")
    out: list[HatVector] = []
    for p in pixels:
        best_i, best_d = 0, math.inf
        for i, seg in enumerate(segments):
            dd = point_segment_distance2(p, seg)
            if dd < best_d:
                best_d, best_i = dd, i
        out.append(encode(p, segments[best_i]))
    return out


def _round_seg(seg: Seg, ndigits: int) -> tuple[float, float, float, float]:
    (x1, y1), (x2, y2) = seg
    a = (round(x1, ndigits), round(y1, ndigits))
    b = (round(x2, ndigits), round(y2, ndigits))
    # undirected canonical order
    return (a + b) if a <= b else (b + a)


def decode_field(
    pixels: list[Point], field: list[HatVector], ndigits: int = 6
) -> list[Seg]:
    """Decode a dense field back to the unique set of line segments.

    Each pixel proposes a segment via :func:`decode`; identical proposals
    (rounded to ``ndigits``) collapse, reproducing the paper's step that
    "generates dense sets of line segments" from the field.  Output order is
    first-seen deterministic.
    """
    if len(pixels) != len(field):
        raise ValueError("pixels and field length mismatch")
    seen: dict[tuple[float, float, float, float], Seg] = {}
    order: list[tuple[float, float, float, float]] = []
    for p, v in zip(pixels, field):
        seg = decode(p, v)
        key = _round_seg(seg, ndigits)
        if key not in seen:
            seen[key] = seg
            order.append(key)
    return [seen[k] for k in order]
