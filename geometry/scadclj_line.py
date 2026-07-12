"""scadclj_line -- solid line / polyline geometry, after scad-clj's geometry.clj.

scad-clj ships a small ``geometry`` namespace with one genuinely geometric
routine: ``line``, which turns a pair of 3D points into a *solid* capsule (a
cylinder with a spherical cap at each end), and ``lines``, which chains that
over a polyline.  OpenSCAD has no line primitive -- to draw a strut between two
arbitrary points you must compute the rotation that carries the +Z axis onto the
segment direction, and that computation is the transferable deterministic core.

The direction-to-rotation math (``Math/acos`` of the normalised Z component for
the angle; the axis ``[-dy, dx, 0]`` perpendicular to both +Z and the segment)
is exposed on its own as :func:`direction_rotation`, because "rotate this part
to point from A to B" is a constantly-recurring need independent of drawing a
strut.  The degenerate cases scad-clj glosses over are handled explicitly:
coincident endpoints, and segments parallel to the Z axis (where ``[-dy, dx, 0]``
vanishes and a valid fallback axis is required).

The output is :mod:`programs.scadclj_data_ir` data, so a line/polyline can be
composed into a larger model and emitted with ``write_scad``.  Unlike the
original -- where the two end caps are left at the origin and [0,0,len] rather
than transformed onto the real endpoints -- both caps here are placed at the
true endpoints, so the capsule is geometrically correct.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from programs.scadclj_data_ir import (
    Node,
    cylinder,
    rotate,
    sphere,
    translate,
    union,
)

__all__ = [
    "direction_rotation",
    "segment_length",
    "line",
    "lines",
]

Vec3 = Sequence[float]


def segment_length(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((bi - ai) ** 2 for ai, bi in zip(a, b)))


def direction_rotation(a: Vec3, b: Vec3) -> Tuple[float, List[float], float]:
    """Rotation carrying +Z onto the direction ``a -> b``.

    Returns ``(angle_radians, axis, length)``.  ``angle`` is
    ``acos(dz / length)`` and ``axis`` is ``[-dy, dx, 0]`` (perpendicular to
    both +Z and the segment), matching scad-clj.  For a coincident pair the
    result is ``(0, [0, 0, 1], 0)``; for a segment along +/-Z (where the
    perpendicular axis degenerates) the X axis is substituted so the rotation
    is still well defined (angle 0 for +Z, pi for -Z)."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length == 0.0:
        return 0.0, [0.0, 0.0, 1.0], 0.0
    # clamp guards against tiny floating overshoot outside [-1, 1]
    cos_a = max(-1.0, min(1.0, dz / length))
    angle = math.acos(cos_a)
    axis = [-dy, dx, 0.0]
    if axis[0] == 0.0 and axis[1] == 0.0:
        # segment parallel to Z: any axis in the XY plane works.
        axis = [1.0, 0.0, 0.0]
    return angle, axis, length


def line(a: Vec3, b: Vec3, radius: float = 1.0) -> Node:
    """A solid capsule (cylinder + spherical end caps) from *a* to *b*."""
    angle, axis, length = direction_rotation(a, b)
    ax, ay, az = a[0], a[1], a[2]
    if length == 0.0:
        return translate([ax, ay, az], sphere(radius))
    shaft = translate(
        [ax, ay, az],
        rotate(
            angle, axis,
            translate([0.0, 0.0, length / 2.0], cylinder(radius, length)),
        ),
    )
    return union(
        translate([ax, ay, az], sphere(radius)),
        translate([b[0], b[1], b[2]], sphere(radius)),
        shaft,
    )


def lines(points: Sequence[Vec3], radius: float = 1.0) -> Node:
    """Union of solid segments through *points* (a polyline of capsules)."""
    pts = list(points)
    if len(pts) == 0:
        raise ValueError("lines needs at least one point")
    if len(pts) == 1:
        p = pts[0]
        return translate([p[0], p[1], p[2]], sphere(radius))
    segs: List[Node] = []
    for a, b in zip(pts, pts[1:]):
        segs.append(line(a, b, radius))
    return union(*segs)
