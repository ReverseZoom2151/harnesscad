"""Helical screw threads: sweep a tooth profile around a (possibly conical) axis.

Reimplementation of SolidPython's ``solid/screw_thread.py``, PyEuclid-free and
returning plain mesh data.

A 2D tooth cross-section (given in XY: X is the radial direction, Y the
elevation) is swept around the Z axis, climbing ``pitch`` per revolution for a
total of ``length``.  Each angular step places a copy of the profile at radius
``rad`` and elevation ``elev``; consecutive copies are stitched into a tube and
the two ends are closed with triangle fans.

The features that make the result printable rather than merely helical:

  * **neck-in / neck-out** -- over the first ``neck_in_degrees`` and the last
    ``neck_out_degrees`` the profile's radius is ramped from (or back to) the
    core radius, so the thread emerges from and sinks back into the shaft
    instead of ending in a sharp overhanging cliff.  ``map_segment`` is the
    linear remap that drives the ramps.
  * **conical threads** -- when ``rad_2 != inner_rad`` the profile is
    pre-rotated by ``-atan((rad_2 - inner_rad) / length)`` in the
    radial/elevation plane so the tooth stays perpendicular to the cone's side.
  * **internal threads** -- the profile is flipped 180 degrees so it cuts into a
    bore instead of standing out from a shaft.
  * **left-handed threads** -- ``inverse_thread_direction`` reverses the sweep
    and the face winding.

:func:`thread_scad` reproduces SolidPython's trimming step: the swept
polyhedron is intersected with a cylindrical tube (external) or a solid cylinder
(internal) so the thread ends flush.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from harnesscad.domain.programs.solidpy_scad_emit import (
    ScadNode,
    cylinder,
    difference,
    intersection,
    polyhedron,
)

__all__ = [
    "EPSILON",
    "map_segment",
    "default_thread_section",
    "thread",
    "thread_scad",
]

EPSILON = 1e-5

Vec3 = Tuple[float, float, float]
Face = Tuple[int, int, int]


def map_segment(x: float, domain_min: float, domain_max: float,
                range_min: float, range_max: float) -> float:
    """Linearly remap ``x`` from [domain_min, domain_max] to [range_min, range_max]."""
    if domain_min == domain_max or range_min == range_max:
        return range_min
    proportion = (x - domain_min) / (domain_max - domain_min)
    return (1 - proportion) * range_min + proportion * range_max


def default_thread_section(tooth_height: float,
                           tooth_depth: float) -> List[Tuple[float, float]]:
    """An isosceles triangle tooth: ``tooth_height`` tall, ``tooth_depth`` deep."""
    return [(0.0, -tooth_height / 2.0),
            (tooth_depth, 0.0),
            (0.0, tooth_height / 2.0)]


def _rotate_z(p: Sequence[float], theta: float) -> Vec3:
    c, s = math.cos(theta), math.sin(theta)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2])


def _bounds_2d(points: Sequence[Sequence[float]]):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys)), (max(xs), max(ys))


def thread(outline_pts: Sequence[Sequence[float]],
           inner_rad: float,
           pitch: float,
           length: float,
           external: bool = True,
           segments_per_rot: int = 32,
           neck_in_degrees: float = 0.0,
           neck_out_degrees: float = 0.0,
           rad_2: float = None,
           inverse_thread_direction: bool = False,
           ) -> Tuple[List[Vec3], List[Face]]:
    """Sweep the closed tooth profile ``outline_pts`` into a helix; return (points, faces)."""
    if len(outline_pts) < 3:
        raise ValueError("outline_pts must be a closed polygon of >= 3 points")
    if pitch <= 0 or length <= 0:
        raise ValueError("pitch and length must be positive")
    if segments_per_rot < 3:
        raise ValueError("segments_per_rot must be >= 3")
    if neck_in_degrees < 0 or neck_out_degrees < 0:
        raise ValueError("neck angles must be non-negative")

    rad_2 = inner_rad if rad_2 is None else rad_2
    rotations = length / pitch
    total_angle = 360.0 * rotations
    if neck_in_degrees + neck_out_degrees > total_angle:
        raise ValueError("neck_in_degrees + neck_out_degrees exceeds the total sweep")

    up_step = length / (rotations * segments_per_rot)
    total_steps = math.ceil(rotations * segments_per_rot) + 1
    step_angle = total_angle / (total_steps - 1)

    # Flip the profile inward for an internal thread
    profile = [(float(p[0]), float(p[1])) for p in outline_pts]
    if not external:
        profile = [(-x, -y) for x, y in profile]

    # Keep the tooth perpendicular to the side of a conical thread
    if inner_rad != rad_2:
        cone_angle = -math.atan((rad_2 - inner_rad) / length)
        c, s = math.cos(cone_angle), math.sin(cone_angle)
        profile = [(x * c - y * s, x * s + y * c) for x, y in profile]

    # The profile lives in XY but the sweep moves it through XZ around Z
    section = [(x, 0.0, y) for x, y in profile]
    poly_sides = len(section)

    (min_x, _), (max_x, _) = _bounds_2d(profile)
    outline_w = max_x - min_x

    neck_out_start = total_angle - neck_out_degrees
    neck_distance = (outline_w + EPSILON) * (1 if external else -1)
    section_rads = (
        max(0.0, inner_rad - neck_distance),                                 # start
        map_segment(neck_in_degrees, 0, total_angle, inner_rad, rad_2),      # neck-in end
        map_segment(neck_out_start, 0, total_angle, inner_rad, rad_2),       # neck-out start
        rad_2 - neck_distance,                                               # end
    )

    points: List[Vec3] = []
    faces: List[Face] = []

    for i in range(total_steps):
        angle = i * step_angle
        elevation = i * up_step
        if angle > total_angle:
            angle = total_angle
            elevation = length

        if angle < neck_in_degrees:
            rad = map_segment(angle, 0, neck_in_degrees,
                              section_rads[0], section_rads[1])
        elif angle < neck_out_start:
            rad = map_segment(angle, neck_in_degrees, neck_out_start,
                              section_rads[1], section_rads[2])
        else:
            rad = map_segment(angle, neck_out_start, total_angle,
                              section_rads[2], section_rads[3])

        theta = math.radians(angle) * (-1 if inverse_thread_direction else 1)
        for p in section:
            moved = (p[0] + rad, p[1], p[2] + elevation)
            points.append(_rotate_z(moved, theta))

        if i < total_steps - 1:
            base = i * poly_sides
            nxt = base + poly_sides
            for j in range(poly_sides):
                k = (j + 1) % poly_sides
                faces.append((base + j, base + k, nxt + j))
                faces.append((base + k, nxt + k, nxt + j))

    # Triangle fans closing the first and last profile
    last_loop = len(points) - poly_sides
    for i in range(poly_sides - 2):
        faces.append((0, i + 2, i + 1))
        faces.append((last_loop, last_loop + i + 1, last_loop + i + 2))

    if inverse_thread_direction:
        faces = [(f[2], f[1], f[0]) for f in faces]

    return points, faces


def thread_scad(outline_pts: Sequence[Sequence[float]],
                inner_rad: float,
                pitch: float,
                length: float,
                external: bool = True,
                segments_per_rot: int = 32,
                rad_2: float = None,
                convexity: int = 2,
                **kwargs) -> ScadNode:
    """:func:`thread` as a ``polyhedron()``, trimmed to the shaft/bore cylinder."""
    points, faces = thread(outline_pts, inner_rad, pitch, length,
                           external=external, segments_per_rot=segments_per_rot,
                           rad_2=rad_2, **kwargs)
    solid = polyhedron(points=points, faces=faces, convexity=convexity)

    rad_2 = inner_rad if rad_2 is None else rad_2
    (min_x, _), (max_x, _) = _bounds_2d([(p[0], p[1]) for p in outline_pts])
    outline_w = max_x - min_x

    if external:
        tube = difference()(
            cylinder(r1=inner_rad + outline_w + EPSILON,
                     r2=rad_2 + outline_w + EPSILON,
                     h=length, segments=segments_per_rot),
            cylinder(r1=inner_rad, r2=rad_2, h=length, segments=segments_per_rot),
        )
    else:
        tube = cylinder(r1=inner_rad, r2=rad_2, h=length,
                        segments=segments_per_rot)
    return intersection()(solid, tube)
