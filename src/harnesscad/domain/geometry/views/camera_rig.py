"""BlenderLLM object-framing camera rig (Du et al., 2024).

BlenderLLM renders every generated model from a fixed set of eight views to
score it. The deterministic geometry behind that rendering (its
``scripts/geometry_utils.py``) does three things:

* parse the vertex positions out of an exported Wavefront ``.obj`` file;
* compute the axis-aligned bounding box of those vertices, its centre and the
  largest side length ``delta_max``;
* place eight cameras, one per corner of a cube centred on the model, scaled so
  the model always fills the frame regardless of its size.

The corner placement is what makes the rig size-invariant: the horizontal
offset is ``delta_max * 2.5 / sqrt(2)`` and the vertical offset is
``delta_max * 2.5``, and (matching the original) the object's ``z`` axis is
mapped to the camera-space vertical while its ``y`` axis becomes the depth
axis. This module reimplements that pure-geometry core in stdlib Python; the
Blender render itself is out of scope.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

Vec3 = tuple[float, float, float]

# Horizontal cameras sit on a circle of radius delta_max*FRAME_SCALE/sqrt(2)
# so that a corner-on view of a delta_max cube just fills a square frame.
FRAME_SCALE = 2.5


def parse_obj_vertices(text: str) -> list[Vec3]:
    """Extract ``v x y z`` vertex positions from Wavefront OBJ text.

    Only geometric-vertex lines (``v``) are read; texture (``vt``), normal
    (``vn``) and face (``f``) lines are ignored. Extra coordinates (``w`` or
    vertex colours) beyond the first three are discarded.
    """
    vertices: list[Vec3] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "v":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return vertices


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box of a vertex cloud."""

    min_corner: Vec3
    max_corner: Vec3

    @property
    def center(self) -> Vec3:
        return tuple(
            (lo + hi) / 2.0 for lo, hi in zip(self.min_corner, self.max_corner)
        )  # type: ignore[return-value]

    @property
    def extents(self) -> Vec3:
        return tuple(
            hi - lo for lo, hi in zip(self.min_corner, self.max_corner)
        )  # type: ignore[return-value]

    @property
    def delta_max(self) -> float:
        return max(self.extents)


def bounding_box(vertices: Iterable[Vec3]) -> BoundingBox:
    """Axis-aligned bounding box of ``vertices``.

    Raises ``ValueError`` on an empty cloud (an unbounded box is undefined).
    """
    verts = list(vertices)
    if not verts:
        raise ValueError("cannot bound an empty vertex set")
    lo = [math.inf, math.inf, math.inf]
    hi = [-math.inf, -math.inf, -math.inf]
    for x, y, z in verts:
        coords = (x, y, z)
        for axis in range(3):
            lo[axis] = min(lo[axis], coords[axis])
            hi[axis] = max(hi[axis], coords[axis])
    return BoundingBox((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2]))


def camera_positions(box: BoundingBox, *, frame_scale: float = FRAME_SCALE) -> list[Vec3]:
    """Eight corner camera positions framing ``box``.

    Faithful to BlenderLLM's ``calculate_bounding_box``: the horizontal offset
    is ``delta_max * frame_scale / sqrt(2)`` and the vertical offset is
    ``delta_max * frame_scale``. The model's ``z`` axis (``center[1]`` in the
    returned tuples' vertical slot) is treated as up and its ``y`` axis as
    depth, so the returned coordinates are ``(x, z, y)`` in world terms.
    """
    cx, cy, cz = box.center
    delta = box.delta_max
    horizontal = delta * frame_scale / math.sqrt(2)
    vertical = delta * frame_scale
    positions: list[Vec3] = []
    for i, j, k in itertools.product((-1, 1), repeat=3):
        positions.append(
            (cx + i * horizontal, cz + j * horizontal, cy + k * vertical)
        )
    return positions


def camera_positions_from_obj(text: str, *, frame_scale: float = FRAME_SCALE) -> list[Vec3]:
    """Convenience: OBJ text straight to eight camera positions."""
    return camera_positions(bounding_box(parse_obj_vertices(text)), frame_scale=frame_scale)


def framing_radius(box: BoundingBox, *, frame_scale: float = FRAME_SCALE) -> float:
    """Distance from the box centre to each corner camera."""
    horizontal = box.delta_max * frame_scale / math.sqrt(2)
    vertical = box.delta_max * frame_scale
    return math.sqrt(2.0 * horizontal * horizontal + vertical * vertical)
