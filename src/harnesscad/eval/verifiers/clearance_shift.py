"""Clearance-shift suggester for overlapping axis-aligned bounding boxes.

The interference gate answers *whether* two placed parts clash. This
module answers the follow-up: *how do I clear the clash with the least
motion?* Given the two parts' AABBs it computes the cheapest single-axis
translation that pushes part A clear of part B with a requested
clearance, and reports it as a signed shift along one of X/Y/Z:

    plate clips cbeam -> shift +Y by 1.35 mm to clear (1 mm clearance)

The algorithm is pure, deterministic bounding-box arithmetic (no CAD
kernel): for each axis it finds the smaller-magnitude of the two moves
that separate A's interval from B's interval (push A below B, or push A
above B) with clearance, then picks the axis whose required move is
smallest. That axis is the cheapest to clear because it is the direction
of shallowest penetration. Ties (equal magnitudes on the min axis, or a
symmetric interval) resolve by pushing A away from B's centre, which is
stable and reproducible.

This is the deterministic "suggested fix vector" that turns an
interference finding into an actionable edit. It is intentionally
separate from the overlap-detection verifier: detection needs solids or
AABBs and may call the kernel; this needs only the two AABBs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# AABB = (xmin, ymin, zmin, xmax, ymax, zmax)
BBox = Tuple[float, float, float, float, float, float]
_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class ClearanceShift:
    """A suggested single-axis translation to clear an overlap.

    ``axis`` is "x" / "y" / "z"; ``shift_mm`` is the signed magnitude of
    the translation applied to part A along that axis. ``overlap_dims`` is
    the per-axis penetration depth of the two AABBs (0 on an axis means
    the boxes already clear on that axis).
    """
    axis: str
    shift_mm: float
    clearance_mm: float
    overlap_dims: Tuple[float, float, float]
    overlaps: bool

    @property
    def vector(self) -> Tuple[float, float, float]:
        """The shift as an (dx, dy, dz) translation vector for part A."""
        idx = _AXES.index(self.axis)
        v = [0.0, 0.0, 0.0]
        v[idx] = self.shift_mm
        return (v[0], v[1], v[2])


def _overlap_depth(a_min: float, a_max: float,
                   b_min: float, b_max: float) -> float:
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def _axis_shift(a_min: float, a_max: float,
                b_min: float, b_max: float, clearance: float) -> float:
    """Signed minimal translation of interval A to clear interval B."""
    move_negative = b_min - clearance - a_max   # push A below B
    move_positive = b_max + clearance - a_min   # push A above B
    if abs(move_negative) < abs(move_positive):
        return move_negative
    if abs(move_positive) < abs(move_negative):
        return move_positive
    # tie: push A away from B's centre (stable, deterministic)
    center_a = (a_min + a_max) / 2.0
    center_b = (b_min + b_max) / 2.0
    return move_positive if center_a >= center_b else move_negative


def boxes_overlap(a: BBox, b: BBox, tol: float = 0.0) -> bool:
    """True if the two AABBs interpenetrate on all three axes.

    ``tol`` shrinks the test: a positive ``tol`` requires the overlap to
    exceed ``tol`` before it counts (bare face contact is not an overlap).
    """
    return (_overlap_depth(a[0], a[3], b[0], b[3]) > tol and
            _overlap_depth(a[1], a[4], b[1], b[4]) > tol and
            _overlap_depth(a[2], a[5], b[2], b[5]) > tol)


def suggest_clearance_shift(bbox_a: BBox, bbox_b: BBox,
                            clearance_mm: float = 1.0) -> ClearanceShift:
    """Cheapest single-axis shift of A to clear B with ``clearance_mm``.

    Always returns a ``ClearanceShift``. When the boxes do not overlap the
    result carries ``overlaps=False`` and a zero shift (nothing to do).
    """
    if clearance_mm < 0:
        raise ValueError("clearance_mm must be non-negative")
    ax0, ay0, az0, ax1, ay1, az1 = bbox_a
    bx0, by0, bz0, bx1, by1, bz1 = bbox_b

    ox = _overlap_depth(ax0, ax1, bx0, bx1)
    oy = _overlap_depth(ay0, ay1, by0, by1)
    oz = _overlap_depth(az0, az1, bz0, bz1)
    overlaps = ox > 0 and oy > 0 and oz > 0

    if not overlaps:
        return ClearanceShift(axis="x", shift_mm=0.0,
                              clearance_mm=clearance_mm,
                              overlap_dims=(ox, oy, oz), overlaps=False)

    candidates = [
        ("x", _axis_shift(ax0, ax1, bx0, bx1, clearance_mm)),
        ("y", _axis_shift(ay0, ay1, by0, by1, clearance_mm)),
        ("z", _axis_shift(az0, az1, bz0, bz1, clearance_mm)),
    ]
    # Cheapest axis first; stable ordering (x<y<z) breaks exact ties.
    candidates.sort(key=lambda c: abs(c[1]))
    axis, shift = candidates[0]
    return ClearanceShift(axis=axis, shift_mm=shift,
                          clearance_mm=clearance_mm,
                          overlap_dims=(ox, oy, oz), overlaps=True)
