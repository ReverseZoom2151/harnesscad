"""autocad_array -- deterministic array (repetition) placement patterns.

Generalises ``AutoCAD.py``'s ``repeat_block_horizontally`` (which tiles a block
along +X, ``num = total_length // block_length`` copies) into the full family of
CAD "array" tools that produce copy transforms without any host:

  * **linear** -- ``count`` copies stepped by a constant vector (the original
    horizontal-repeat, plus arbitrary direction);
  * **fit-linear** -- fill a run of ``total_length`` with copies of a given
    pitch (the exact ``//`` behaviour of ``repeat_block_horizontally``);
  * **rectangular** -- a rows x cols grid with independent row/col steps;
  * **polar** -- ``count`` copies evenly around a centre, each rotated to face
    outward, over a given sweep angle.

Every function returns a list of placement transforms ``(x, y, rotation)`` in
deterministic order. Stdlib-only, no wall clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class Placement:
    """A single array copy: position plus rotation (radians)."""

    x: float
    y: float
    rotation: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.rotation)


def linear_array(base: Point, count: int, step: Point) -> List[Placement]:
    """``count`` copies of ``base`` displaced by ``k * step`` for k in 0..count-1."""
    if count < 0:
        raise ValueError("count must be non-negative")
    return [Placement(base[0] + k * step[0], base[1] + k * step[1])
            for k in range(count)]


def fit_linear_array(base: Point, total_length: float, pitch: float,
                     direction: Point = (1.0, 0.0)) -> List[Placement]:
    """Fill ``total_length`` along ``direction`` with copies at ``pitch`` spacing.

    Reproduces ``repeat_block_horizontally`` exactly: the number of copies is
    ``floor(total_length / pitch)`` (integer division), each stepped by ``pitch``
    along the unit ``direction``.
    """
    if pitch <= 0.0:
        raise ValueError("pitch must be positive")
    n = int(total_length // pitch)
    dx, dy = direction
    dnorm = math.hypot(dx, dy)
    if dnorm == 0.0:
        raise ValueError("direction must be non-zero")
    ux, uy = dx / dnorm, dy / dnorm
    return [Placement(base[0] + k * pitch * ux, base[1] + k * pitch * uy)
            for k in range(n)]


def rectangular_array(base: Point, rows: int, cols: int,
                      row_step: Point, col_step: Point) -> List[Placement]:
    """A ``rows`` x ``cols`` grid; row-major order (row 0 first)."""
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    out: List[Placement] = []
    for r in range(rows):
        for c in range(cols):
            x = base[0] + r * row_step[0] + c * col_step[0]
            y = base[1] + r * row_step[1] + c * col_step[1]
            out.append(Placement(x, y))
    return out


def polar_array(center: Point, radius: float, count: int,
                start_angle: float = 0.0, sweep: float = 2 * math.pi,
                rotate_items: bool = True) -> List[Placement]:
    """``count`` copies evenly spaced on a circle of ``radius`` around ``center``.

    ``sweep`` is the total angular span; if it is a full turn (2*pi) the copies
    are spaced by ``sweep / count`` (endpoint not duplicated), otherwise by
    ``sweep / (count - 1)`` so both ends are populated. When ``rotate_items`` is
    true each copy's rotation points radially outward.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return []
    full = abs((sweep % (2 * math.pi))) < 1e-12 and sweep != 0.0
    if count == 1:
        divisor = 1
    elif full:
        divisor = count
    else:
        divisor = count - 1
    out: List[Placement] = []
    for k in range(count):
        ang = start_angle + (sweep * k / divisor if divisor else 0.0)
        x = center[0] + radius * math.cos(ang)
        y = center[1] + radius * math.sin(ang)
        rot = ang if rotate_items else 0.0
        out.append(Placement(x, y, rot))
    return out
