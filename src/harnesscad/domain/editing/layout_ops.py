"""Align and distribute objects by axis-aligned bounding box.

A CAD host's automation layer typically exposes an align operation (LEFT /
RIGHT / CENTER) and a distribute operation (even spacing along X) that read
each entity's reference point / geometric extents through the host and then
move it. The host does the moving, but the *arrangement arithmetic* --
computing, from a set of bounding boxes, the translation that snaps each to a
common edge or spreads them at a chosen pitch -- is pure deterministic geometry
usable anywhere.

This module implements that arithmetic on plain ``(minx, miny, maxx, maxy)``
boxes and adds the natural completions such host automation layers lack (TOP /
BOTTOM / MIDDLE alignment on Y, and even *gap* distribution as well as even
*centre* distribution). Each function returns the per-box translation
``(dx, dy)`` so the caller can apply it however it stores geometry.
Stdlib-only, deterministic.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Sequence, Tuple

BBox = Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
Vec = Tuple[float, float]


class Align(Enum):
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"   # horizontal centre (X)
    TOP = "top"
    BOTTOM = "bottom"
    MIDDLE = "middle"   # vertical centre (Y)


def _cx(b: BBox) -> float:
    return (b[0] + b[2]) / 2.0


def _cy(b: BBox) -> float:
    return (b[1] + b[3]) / 2.0


def align(boxes: Sequence[BBox], mode: Align) -> List[Vec]:
    """Return per-box ``(dx, dy)`` translations snapping boxes to a common line.

    LEFT/RIGHT/CENTER move along X; TOP/BOTTOM/MIDDLE along Y. An empty input
    yields an empty list.
    """
    if not boxes:
        return []
    if mode == Align.LEFT:
        target = min(b[0] for b in boxes)
        return [(target - b[0], 0.0) for b in boxes]
    if mode == Align.RIGHT:
        target = max(b[2] for b in boxes)
        return [(target - b[2], 0.0) for b in boxes]
    if mode == Align.CENTER:
        lo = min(_cx(b) for b in boxes)
        hi = max(_cx(b) for b in boxes)
        target = (lo + hi) / 2.0
        return [(target - _cx(b), 0.0) for b in boxes]
    if mode == Align.BOTTOM:
        target = min(b[1] for b in boxes)
        return [(0.0, target - b[1]) for b in boxes]
    if mode == Align.TOP:
        target = max(b[3] for b in boxes)
        return [(0.0, target - b[3]) for b in boxes]
    if mode == Align.MIDDLE:
        lo = min(_cy(b) for b in boxes)
        hi = max(_cy(b) for b in boxes)
        target = (lo + hi) / 2.0
        return [(0.0, target - _cy(b)) for b in boxes]
    raise ValueError(f"unknown alignment {mode!r}")


def distribute_centers(boxes: Sequence[BBox], axis: str = "x") -> List[Vec]:
    """Even-*centre* distribution: spread box centres uniformly.

    The extreme boxes (lowest / highest centre) stay put and the interior boxes
    are placed at equal centre-to-centre spacing between them. Returns per-box
    translations in original order.
    """
    n = len(boxes)
    if n <= 2:
        return [(0.0, 0.0)] * n
    horiz = axis == "x"
    key = _cx if horiz else _cy
    order = sorted(range(n), key=lambda i: key(boxes[i]))
    lo = key(boxes[order[0]])
    hi = key(boxes[order[-1]])
    step = (hi - lo) / (n - 1)
    out: List[Vec] = [(0.0, 0.0)] * n
    for rank, idx in enumerate(order):
        target = lo + rank * step
        delta = target - key(boxes[idx])
        out[idx] = (delta, 0.0) if horiz else (0.0, delta)
    return out


def distribute_gaps(boxes: Sequence[BBox], spacing: float,
                    axis: str = "x") -> List[Vec]:
    """Fixed-*gap* distribution: place boxes edge-to-edge with ``spacing`` gap.

    Mirrors the COM ``distribute_objects``: sort by position, keep the first box
    fixed, then lay each following box so its leading edge sits ``spacing`` past
    the previous box's trailing edge. Returns per-box translations in original
    order.
    """
    n = len(boxes)
    if n == 0:
        return []
    horiz = axis == "x"
    lo_i = 0 if horiz else 1
    hi_i = 2 if horiz else 3
    order = sorted(range(n), key=lambda i: boxes[i][lo_i])
    out: List[Vec] = [(0.0, 0.0)] * n
    cursor = boxes[order[0]][hi_i]  # trailing edge of the anchor box
    for rank in range(1, n):
        idx = order[rank]
        b = boxes[idx]
        new_lo = cursor + spacing
        delta = new_lo - b[lo_i]
        out[idx] = (delta, 0.0) if horiz else (0.0, delta)
        cursor = new_lo + (b[hi_i] - b[lo_i])  # advance by this box's size
    return out
