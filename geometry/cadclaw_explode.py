"""Exploded-view offset vectors and outside-in removal ordering.

Given the axis-aligned bounding boxes of an assembly's parts, this module
computes the geometry needed to draw an exploded / disassembly view and
to order the parts for outside-in removal — all as pure, deterministic
arithmetic on bounding boxes, with no CAD kernel and no rendering
dependency.

Three deterministic pieces:

  * **Removal axis** — the natural pull-out direction for a part is the
    signed principal axis (+/- X/Y/Z) of the vector from the assembly
    centroid to the part's centre: the face of the assembly it is
    closest to. This is what an axis-aligned explode animation moves each
    part along.
  * **Radial offset** — for a symmetric "burst" view, each part is
    translated along the unit vector from the centroid to its centre by
    ``expansion * distance_from_centroid``, so parts already far out move
    further and the whole assembly scales open about its centre. A part
    sitting on the centroid is nudged along its removal axis by
    ``min_offset`` so it does not vanish inside its neighbours.
  * **Removal order** — parts are ordered by a per-label priority (lower
    = removed first) and, within a priority tier, by descending distance
    from the centroid (outermost first). The reverse of a removal order
    is a valid assembly order.

This is the visualization / sequencing geometry that complements the
collision-aware sequence planner elsewhere in the codebase: that planner
proves a collision-free escape exists; this produces the offset vectors
and the outside-in default order used to render the motion. It operates
purely on the ``(cx, cy, cz)`` centres derived from bounding boxes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
BBox = Tuple[float, float, float, float, float, float]
_AXES = ("X", "Y", "Z")


def bbox_center(bbox: BBox) -> Vec3:
    """Centre of an ``(xmin,ymin,zmin,xmax,ymax,zmax)`` bounding box."""
    return ((bbox[0] + bbox[3]) / 2.0,
            (bbox[1] + bbox[4]) / 2.0,
            (bbox[2] + bbox[5]) / 2.0)


def assembly_centroid(centers: Sequence[Vec3]) -> Vec3:
    """Mean of the part centres (unweighted centroid)."""
    if not centers:
        raise ValueError("cannot take the centroid of zero parts")
    n = len(centers)
    return (sum(c[0] for c in centers) / n,
            sum(c[1] for c in centers) / n,
            sum(c[2] for c in centers) / n)


def removal_axis(center: Vec3, centroid: Vec3) -> Tuple[str, float]:
    """Natural pull-out axis for a part, as ``(axis, direction)``.

    The axis is the dominant component of ``center - centroid``; the
    direction is +1 or -1. Ties resolve X > Y > Z (deterministic).
    """
    d = (center[0] - centroid[0],
         center[1] - centroid[1],
         center[2] - centroid[2])
    ad = (abs(d[0]), abs(d[1]), abs(d[2]))
    if ad[0] >= ad[1] and ad[0] >= ad[2]:
        i = 0
    elif ad[1] >= ad[2]:
        i = 1
    else:
        i = 2
    return (_AXES[i], 1.0 if d[i] >= 0 else -1.0)


def axis_offset(axis: str, direction: float, distance: float) -> Vec3:
    """Translation vector of ``distance`` along a signed principal axis."""
    idx = _AXES.index(axis.upper())
    v = [0.0, 0.0, 0.0]
    v[idx] = distance * direction
    return (v[0], v[1], v[2])


def radial_offset(center: Vec3, centroid: Vec3, expansion: float = 0.35,
                  min_offset: float = 0.0) -> Vec3:
    """Outward-from-centroid translation for a radial exploded view.

    Magnitude is ``expansion * distance_from_centroid`` along the unit
    vector centroid->centre. A part on the centroid is nudged by
    ``min_offset`` along its removal axis instead (or left in place when
    ``min_offset`` is 0).
    """
    if expansion < 0:
        raise ValueError("expansion must be non-negative")
    vx = center[0] - centroid[0]
    vy = center[1] - centroid[1]
    vz = center[2] - centroid[2]
    dist = math.sqrt(vx * vx + vy * vy + vz * vz)
    if dist > 1e-9:
        return (vx * expansion, vy * expansion, vz * expansion)
    if min_offset > 0:
        axis, direction = removal_axis(center, centroid)
        return axis_offset(axis, direction, min_offset)
    return (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class RemovalStep:
    """One part in an outside-in removal order."""
    part_index: int
    label: str
    center: Vec3
    distance_from_centroid: float
    removal_axis: str
    removal_direction: float

    def offset_at(self, distance: float) -> Vec3:
        """Axis-aligned translation for this part at a given explode distance."""
        return axis_offset(self.removal_axis, self.removal_direction, distance)


def removal_order(bboxes: Sequence[BBox], labels: Sequence[str],
                  priority: Optional[Dict[str, float]] = None,
                  default_priority: float = 5.0) -> List[RemovalStep]:
    """Outside-in removal order for the parts.

    Parts sort by ``priority[label]`` (lower first), then by descending
    distance from the centroid (outermost first), then by part index for
    a fully deterministic tiebreak. The reverse of this list is a valid
    assembly order.
    """
    if len(bboxes) != len(labels):
        raise ValueError("bboxes and labels must have equal length")
    if not bboxes:
        return []
    priority = priority or {}
    centers = [bbox_center(b) for b in bboxes]
    centroid = assembly_centroid(centers)

    entries = []
    for i, (c, lbl) in enumerate(zip(centers, labels)):
        dist = math.dist(c, centroid)
        axis, direction = removal_axis(c, centroid)
        pri = priority.get(lbl, default_priority)
        entries.append((pri, -dist, i, lbl, c, dist, axis, direction))

    entries.sort(key=lambda e: (e[0], e[1], e[2]))
    return [
        RemovalStep(part_index=i, label=lbl, center=c,
                    distance_from_centroid=dist,
                    removal_axis=axis, removal_direction=direction)
        for _pri, _negd, i, lbl, c, dist, axis, direction in entries
    ]
