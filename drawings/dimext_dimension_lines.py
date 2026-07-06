"""dimext_dimension_lines — deterministic dimension-line / extension-line detection.

From Bhandari & Manandhar (Machines 2023, 11, 1083). The paper extracts object
dimensions from engineering drawings and calibrated images. Given the *segment
geometry* of a drawing (produced upstream by an edge detector, which is
external), a linear dimension in a technical drawing is defined by:

  * a **dimension line** — the segment carrying the measurement, drawn parallel
    to the measured direction and terminated by arrowheads; and
  * two **extension lines** — short segments perpendicular to the dimension line
    that project out from the two feature endpoints being measured.

This module recovers those groupings purely geometrically: for each candidate
dimension line it finds two roughly perpendicular extension lines whose endpoints
coincide (within tolerance) with the dimension line's two ends. The measured
length is the dimension line's length, optionally converted to metric units with
a :class:`vision.cvcad_pixel_calibration.Calibration`.

The learned pieces (Canny edge detection, arrowhead / dimension-text OCR) are
NOT implemented; this consumes segment lists and returns measurements.

Stdlib-only, deterministic, no wall clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Segment = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


def _length(seg: Segment) -> float:
    return math.hypot(seg[2] - seg[0], seg[3] - seg[1])


def _endpoints(seg: Segment) -> Tuple[Point, Point]:
    return ((seg[0], seg[1]), (seg[2], seg[3]))


def _direction(seg: Segment) -> Point:
    dx, dy = seg[2] - seg[0], seg[3] - seg[1]
    n = math.hypot(dx, dy)
    if n == 0.0:
        return (0.0, 0.0)
    return (dx / n, dy / n)


def angle_between(a: Segment, b: Segment) -> float:
    """Unsigned acute angle (degrees) between two segments' orientations."""
    da, db = _direction(a), _direction(b)
    dot = max(-1.0, min(1.0, abs(da[0] * db[0] + da[1] * db[1])))
    return math.degrees(math.acos(dot))


def _dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _point_segment_distance(pt: Point, seg: Segment) -> float:
    """Shortest distance from ``pt`` to the segment ``seg``."""
    ax, ay = seg[0], seg[1]
    bx, by = seg[2], seg[3]
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return _dist(pt, (ax, ay))
    t = ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    proj = (ax + t * dx, ay + t * dy)
    return _dist(pt, proj)


def _touches(pt: Point, seg: Segment, tol: float) -> bool:
    """True if ``pt`` lies within ``tol`` of the extension segment's span.

    Extension lines usually overshoot the dimension line, so the dimension
    endpoint meets the extension line somewhere along its length, not at an
    extension endpoint.
    """
    return _point_segment_distance(pt, seg) <= tol


@dataclass(frozen=True)
class DimensionMeasurement:
    """A recovered linear dimension."""

    dimension_line: Segment
    extension_a: Segment
    extension_b: Segment
    length_pixels: float
    length_metric: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "dimension_line": self.dimension_line,
            "extension_a": self.extension_a,
            "extension_b": self.extension_b,
            "length_pixels": self.length_pixels,
            "length_metric": self.length_metric,
        }


def detect_dimensions(segments: Sequence[Segment],
                      *,
                      touch_tol: float = 3.0,
                      perpendicular_tol_deg: float = 15.0,
                      min_dimension_length: float = 1.0,
                      calibration=None) -> List[DimensionMeasurement]:
    """Group segments into linear dimensions.

    A candidate dimension line D is accepted when there exist two *distinct*
    extension segments E_a, E_b such that:

      * E_a is perpendicular to D (within ``perpendicular_tol_deg``) and one of
        its endpoints lies within ``touch_tol`` of D's first endpoint;
      * E_b is perpendicular to D and touches D's second endpoint.

    Returns one :class:`DimensionMeasurement` per accepted dimension line, sorted
    by descending pixel length so the largest (object) dimension comes first.

    If ``calibration`` (a ``cvcad_pixel_calibration.Calibration``) is supplied the
    metric length is filled in.
    """
    segs = list(segments)
    results: List[DimensionMeasurement] = []

    for i, dline in enumerate(segs):
        dlen = _length(dline)
        if dlen < min_dimension_length:
            continue
        a_end, b_end = _endpoints(dline)

        ext_a = _find_extension(segs, i, a_end, dline, touch_tol,
                                perpendicular_tol_deg)
        if ext_a is None:
            continue
        ext_b = _find_extension(segs, i, b_end, dline, touch_tol,
                                perpendicular_tol_deg, exclude=ext_a)
        if ext_b is None:
            continue

        metric = None
        if calibration is not None:
            metric = dlen * calibration.mm_per_pixel
        results.append(DimensionMeasurement(
            dimension_line=dline,
            extension_a=segs[ext_a],
            extension_b=segs[ext_b],
            length_pixels=dlen,
            length_metric=metric,
        ))

    results.sort(key=lambda m: m.length_pixels, reverse=True)
    return results


def _find_extension(segs: Sequence[Segment], dline_idx: int, end: Point,
                    dline: Segment, touch_tol: float,
                    perpendicular_tol_deg: float,
                    exclude: Optional[int] = None) -> Optional[int]:
    for j, e in enumerate(segs):
        if j == dline_idx or j == exclude:
            continue
        if angle_between(dline, e) < (90.0 - perpendicular_tol_deg):
            continue
        if _touches(end, e, touch_tol):
            return j
    return None


def measurements_to_metric(measurements: Sequence[DimensionMeasurement],
                           calibration) -> List[DimensionMeasurement]:
    """Return copies of ``measurements`` with metric lengths filled from cal."""
    out: List[DimensionMeasurement] = []
    for m in measurements:
        out.append(DimensionMeasurement(
            dimension_line=m.dimension_line,
            extension_a=m.extension_a,
            extension_b=m.extension_b,
            length_pixels=m.length_pixels,
            length_metric=m.length_pixels * calibration.mm_per_pixel,
        ))
    return out
