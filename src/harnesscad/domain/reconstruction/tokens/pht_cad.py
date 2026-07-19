"""Efficient Hybrid Parametrization (EHP) for 2D engineering drawings.

EHP represents a 2D engineering drawing with four *atomic components* --
point, line, circle, arc -- under a compact, hybrid parametrization that merges
the point-based and implicit strategies while eliminating redundant fields:

    Point:  p = (xp, yp)
    Line:   l = (xstart, ystart, xend, yend, v)      # v = validity flag (solid/dashed)
    Circle: c = (xc, yc, r)
    Arc:    a = (xa, ya, r, theta_start, theta_end)

EHP's three modifications over an *over-parameterized* baseline (which carries
BOTH relative constraints -- direction vector + reference point + length/sweep --
AND explicit key points):

  1. Lines and arcs drop the direction vector; direction is *inferred* from the
     start/end coordinates (line) or the start/end angles (arc).
  2. Circles and arcs use centre + radius (+ angles), not discrete boundary points.
  3. All coordinates are normalised into the range ``[0, 1000)`` so spatial scale
     is consistent across images of differing resolution.

This module is the canonical primitive record. It provides construction
with validation, direction/geometry inference, coordinate normalisation, a flat
token vector (the target the regression heads predict), and a *parametrization
efficiency* metric quantifying how many scalar fields EHP saves versus the
over-parameterized baseline. All arithmetic is deterministic and stdlib-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# EHP normalisation range: coordinates are scaled into [0, NORM_RANGE).
NORM_RANGE = 1000.0

# Scalar-field counts per component under the over-parameterized baseline vs EHP.
# Baseline (implicit + point-based) fields:
#   point : x, y                                            -> 2
#   line  : dir_x, dir_y, ref_x, ref_y, length, start(2), end(2), validity -> 10
#   circle: dir_x, dir_y, ref_x, ref_y, r, p_top(2), p_bottom(2)           -> 9
#   arc   : dir_x, dir_y, ref_x, ref_y, r, mid_angle, start(2), end(2), sweep -> 11
_BASELINE_FIELDS = {"point": 2, "line": 10, "circle": 9, "arc": 11}
_EHP_FIELDS = {"point": 2, "line": 5, "circle": 3, "arc": 5}


def normalise_coord(value: float, lo: float, hi: float) -> float:
    """Map ``value`` from source range ``[lo, hi]`` into ``[0, NORM_RANGE)``.

    A degenerate range (``hi == lo``) maps everything to ``0.0``.
    """
    if hi <= lo:
        return 0.0
    frac = (value - lo) / (hi - lo)
    scaled = frac * NORM_RANGE
    # Clamp into [0, NORM_RANGE); the upper bound is exclusive.
    if scaled < 0.0:
        return 0.0
    if scaled >= NORM_RANGE:
        return math.nextafter(NORM_RANGE, 0.0)
    return scaled


def _wrap_angle(deg: float) -> float:
    """Normalise an angle in degrees into ``[0, 360)``."""
    return deg % 360.0


@dataclass(frozen=True)
class Point:
    """EHP point primitive ``p = (xp, yp)``."""

    x: float
    y: float
    kind: str = "point"

    def tokens(self) -> list[float]:
        return [self.x, self.y]


@dataclass(frozen=True)
class Line:
    """EHP line ``l = (xstart, ystart, xend, yend, v)``.

    ``v`` is a binary validity/style flag (1 = solid, 0 = dashed). The direction
    is not stored; it is inferred from the endpoints via :meth:`direction_deg`.
    """

    xstart: float
    ystart: float
    xend: float
    yend: float
    v: int = 1
    kind: str = "line"

    def direction_deg(self) -> float:
        """Inferred heading (degrees, ``[0, 360)``) from start toward end."""
        return _wrap_angle(math.degrees(math.atan2(self.yend - self.ystart,
                                                    self.xend - self.xstart)))

    def length(self) -> float:
        return math.hypot(self.xend - self.xstart, self.yend - self.ystart)

    def tokens(self) -> list[float]:
        return [self.xstart, self.ystart, self.xend, self.yend, float(self.v)]


@dataclass(frozen=True)
class Circle:
    """EHP circle ``c = (xc, yc, r)``."""

    xc: float
    yc: float
    r: float
    kind: str = "circle"

    def tokens(self) -> list[float]:
        return [self.xc, self.yc, self.r]


@dataclass(frozen=True)
class Arc:
    """EHP arc ``a = (xa, ya, r, theta_start, theta_end)`` (angles in degrees)."""

    xc: float
    yc: float
    r: float
    theta_start: float
    theta_end: float
    kind: str = "arc"

    def sweep_deg(self) -> float:
        """Counter-clockwise sweep in degrees, ``(0, 360]``."""
        d = (self.theta_end - self.theta_start) % 360.0
        return 360.0 if d == 0.0 else d

    def endpoint_start(self) -> tuple[float, float]:
        a = math.radians(self.theta_start)
        return (self.xc + self.r * math.cos(a), self.yc + self.r * math.sin(a))

    def endpoint_end(self) -> tuple[float, float]:
        a = math.radians(self.theta_end)
        return (self.xc + self.r * math.cos(a), self.yc + self.r * math.sin(a))

    def tokens(self) -> list[float]:
        return [self.xc, self.yc, self.r,
                _wrap_angle(self.theta_start), _wrap_angle(self.theta_end)]


def make_line(xstart: float, ystart: float, xend: float, yend: float,
              v: int = 1) -> Line:
    """Build a validated :class:`Line`; rejects zero-length lines and bad flags."""
    if v not in (0, 1):
        raise ValueError(f"validity flag v must be 0 or 1, got {v!r}")
    if xstart == xend and ystart == yend:
        raise ValueError("line start and end coincide (zero length)")
    return Line(float(xstart), float(ystart), float(xend), float(yend), int(v))


def make_circle(xc: float, yc: float, r: float) -> Circle:
    """Build a validated :class:`Circle`; requires positive radius."""
    if r <= 0:
        raise ValueError(f"circle radius must be positive, got {r!r}")
    return Circle(float(xc), float(yc), float(r))


def make_arc(xc: float, yc: float, r: float,
             theta_start: float, theta_end: float) -> Arc:
    """Build a validated :class:`Arc`; requires positive radius, non-empty sweep."""
    if r <= 0:
        raise ValueError(f"arc radius must be positive, got {r!r}")
    if _wrap_angle(theta_start) == _wrap_angle(theta_end):
        raise ValueError("arc start and end angle coincide (empty sweep)")
    return Arc(float(xc), float(yc), float(r),
               float(theta_start), float(theta_end))


def field_count(primitives: list[object]) -> int:
    """Total EHP scalar-field count across ``primitives``."""
    return sum(_EHP_FIELDS[p.kind] for p in primitives)


def baseline_field_count(primitives: list[object]) -> int:
    """Total over-parameterized scalar-field count across ``primitives``."""
    return sum(_BASELINE_FIELDS[p.kind] for p in primitives)


@dataclass(frozen=True)
class EfficiencyReport:
    """Parametrization efficiency of EHP vs the over-parameterized baseline."""

    ehp_fields: int
    baseline_fields: int
    saved_fields: int
    compression_ratio: float   # ehp / baseline (lower is more efficient)
    reduction: float           # fraction of baseline fields eliminated, [0, 1]


def efficiency(primitives: list[object]) -> EfficiencyReport:
    """Quantify how compactly EHP encodes ``primitives`` vs the baseline.

    ``reduction`` is the headline "efficiency" number: the fraction of the
    over-parameterized scalar fields that EHP eliminates.
    """
    ehp = field_count(primitives)
    base = baseline_field_count(primitives)
    saved = base - ehp
    ratio = (ehp / base) if base else 0.0
    reduction = (saved / base) if base else 0.0
    return EfficiencyReport(ehp, base, saved, ratio, reduction)


def to_tokens(primitives: list[object]) -> list[float]:
    """Flatten primitives into the ordered scalar target the heads regress."""
    out: list[float] = []
    for p in primitives:
        out.extend(p.tokens())
    return out
