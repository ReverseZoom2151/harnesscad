"""AABB interference detection with a minimum-clearance fix-vector suggestion.

Mined from CADCLAW's interference gate (``cadclaw/interference.py``) and reduced
to its deterministic, stdlib-only core. The original relies on OpenCascade
(``BRepAlgoAPI_Common``) for exact solid-solid overlap volume; that dependency is
dropped here. Instead we compute the *axis-aligned bounding-box* overlap, whose
intersection volume is exact for the box abstraction and needs only arithmetic.

Two ideas carry over verbatim because they are pure geometry:

*   **BBox pre-filter** -- two parts can only interfere if their AABBs overlap on
    all three axes (with a small negative tolerance so a shared face is not a clip).
*   **Cheapest-axis fix vector** -- when a clip is found, the smallest-overlap
    axis is the cheapest direction to push part A clear. The signed shift is the
    minimum interval translation that separates A from B with the requested
    clearance; for nested/contained overlaps it is larger than the raw overlap,
    because the contained interval must travel all the way past one side.

Everything is deterministic: same boxes in -> same clips out, in a stable order.

Usage::

    from harnesscad.domain.assembly.interference import AABB, check_interference
    boxes = {"plate": AABB(0, 0, 0, 10, 10, 4), "bar": AABB(8, 0, 0, 20, 3, 4)}
    result = check_interference(boxes, min_clearance=1.0)
    for clip in result.clips:
        print(clip.label_a, clip.label_b, clip.volume, clip.suggest_axis, clip.suggest_shift)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["AABB", "Clip", "InterferenceResult", "check_interference"]


@dataclass(frozen=True)
class AABB:
    """An axis-aligned bounding box. Mins must not exceed maxes."""

    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float

    def __post_init__(self) -> None:
        if self.xmin > self.xmax or self.ymin > self.ymax or self.zmin > self.zmax:
            raise ValueError(f"AABB has a min greater than its max: {self!r}")

    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.xmin + self.xmax) / 2.0,
            (self.ymin + self.ymax) / 2.0,
            (self.zmin + self.zmax) / 2.0,
        )

    def as_tuple(self) -> Tuple[float, float, float, float, float, float]:
        return (self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax)


@dataclass(frozen=True)
class Clip:
    """A detected AABB interference between two parts plus a suggested fix."""

    label_a: str
    label_b: str
    center_a: Tuple[float, float, float]
    center_b: Tuple[float, float, float]
    volume: float  # exact AABB overlap volume
    overlap_dims: Tuple[float, float, float]  # (dx, dy, dz)
    suggest_axis: str  # "x" | "y" | "z"
    suggest_shift: float  # signed; move A by this along suggest_axis to clear B
    clearance: float


@dataclass
class InterferenceResult:
    passed: bool
    checked_pairs: int
    clips: List[Clip] = field(default_factory=list)


def _overlap_on_axis(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def _bb_overlap(a: AABB, b: AABB, tol: float) -> bool:
    """True if boxes overlap on all three axes. ``tol`` < 0 ignores shared faces."""
    return (
        a.xmin < b.xmax + tol and b.xmin < a.xmax + tol
        and a.ymin < b.ymax + tol and b.ymin < a.ymax + tol
        and a.zmin < b.zmax + tol and b.zmin < a.zmax + tol
    )


def _axis_shift(a_min: float, a_max: float, b_min: float, b_max: float,
                clearance: float) -> float:
    """Smallest signed translation of A on one axis that clears B by ``clearance``."""
    move_negative = b_min - clearance - a_max
    move_positive = b_max + clearance - a_min
    if abs(move_negative) < abs(move_positive):
        return move_negative
    if abs(move_positive) < abs(move_negative):
        return move_positive
    # Tie: push in the direction A already leans relative to B.
    center_a = (a_min + a_max) / 2.0
    center_b = (b_min + b_max) / 2.0
    return move_positive if center_a >= center_b else move_negative


def _suggest_clear_shift(a: AABB, b: AABB,
                         clearance: float) -> Tuple[str, float, Tuple[float, float, float]]:
    ox = _overlap_on_axis(a.xmin, a.xmax, b.xmin, b.xmax)
    oy = _overlap_on_axis(a.ymin, a.ymax, b.ymin, b.ymax)
    oz = _overlap_on_axis(a.zmin, a.zmax, b.zmin, b.zmax)
    candidates = [
        ("x", _axis_shift(a.xmin, a.xmax, b.xmin, b.xmax, clearance)),
        ("y", _axis_shift(a.ymin, a.ymax, b.ymin, b.ymax, clearance)),
        ("z", _axis_shift(a.zmin, a.zmax, b.zmin, b.zmax, clearance)),
    ]
    # Cheapest fix first; ties broken by fixed axis order (x<y<z) for determinism.
    candidates.sort(key=lambda p: (abs(p[1]), p[0]))
    axis, shift = candidates[0]
    return axis, shift, (ox, oy, oz)


def check_interference(
    boxes: Dict[str, AABB],
    *,
    skip_labels: Optional[Set[str]] = None,
    min_volume: float = 1.0,
    min_clearance: float = 1.0,
    face_tolerance: float = -0.5,
) -> InterferenceResult:
    """Detect pairwise AABB interference among labelled parts.

    Args:
        boxes: mapping of part label -> its :class:`AABB`.
        skip_labels: labels excluded from checking (e.g. belts, wheels).
        min_volume: minimum overlap volume to report as a clip.
        min_clearance: clearance added to the suggested fix so the moved part
            lands clear rather than tangent.
        face_tolerance: negative tolerance so a shared face is not a clip.

    Returns a deterministic :class:`InterferenceResult`; clips are ordered by
    ``(label_a, label_b)``.
    """
    skip = skip_labels or set()
    items = sorted((lbl, bb) for lbl, bb in boxes.items() if lbl not in skip)
    clips: List[Clip] = []
    checked = 0
    for i in range(len(items)):
        la, a = items[i]
        for j in range(i + 1, len(items)):
            lb, b = items[j]
            if not _bb_overlap(a, b, face_tolerance):
                continue
            checked += 1
            axis, shift, overlap = _suggest_clear_shift(a, b, min_clearance)
            volume = overlap[0] * overlap[1] * overlap[2]
            if volume <= min_volume:
                continue
            clips.append(Clip(
                label_a=la, label_b=lb,
                center_a=a.center, center_b=b.center,
                volume=volume, overlap_dims=overlap,
                suggest_axis=axis, suggest_shift=shift,
                clearance=min_clearance,
            ))
    clips.sort(key=lambda c: (c.label_a, c.label_b))
    return InterferenceResult(passed=len(clips) == 0, checked_pairs=checked, clips=clips)
