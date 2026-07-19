"""Architectural primitive geometry for sketch-to-CAD reconstruction.

Seven shapes are selected by architectural form analysis of typical
residential buildings::

    Cube, Cylinder, Pyramid, Shed, Hip, A-Frame, Mansard

The learned transformer only predicts a scene descriptor -- a shape *type* plus
position, rotation (yaw, pitch) and size. The actual 3D geometry is reconstructed
downstream by a visual-programming environment that reads those parameters. This
module is that deterministic *shape -> B-rep-like geometry* lifting, built in
pure stdlib so the descriptor can be turned into concrete vertices / edges
without a proprietary modelling application.

Each shape is generated in a canonical unit form centred on the origin footprint,
scaled by ``size = (sx, sy, sz)`` (sx, sy = footprint, sz = height), rotated by
``yaw`` (about +Z) then ``pitch`` (about +X) in degrees, and translated to
``position``. The result is a :class:`Mesh` of vertices + edges (the rendered
wire-frame) plus quad/tri faces.

The roof shapes are the distinguishing pieces:

  * **shed** -- a box with a single-slope (mono-pitch) roof: one ridge edge high,
    the opposite eave low.
  * **hip** -- a box with a four-sided hip roof: a short ridge, all four sides slope.
  * **aframe** -- a triangular prism ("A-frame"): gable ends, ridge along +y.
  * **mansard** -- a two-slope (steep lower, shallow upper) roof on a box.

This is distinct from :mod:`reconstruction.ppa_primitive` (2D line/circle/arc sketch
primitives): these are parametric 3D architectural solids. Pure ``math`` only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]

SHAPE_TYPES: tuple[str, ...] = (
    "cube",
    "cylinder",
    "pyramid",
    "shed",
    "hip",
    "aframe",
    "mansard",
)

# Fraction of total height taken by the walls (box) below the roof, for roofed shapes.
_WALL_FRACTION = 0.6
# Ridge half-length as a fraction of footprint depth for the hip roof.
_HIP_RIDGE = 0.25
# Mansard: fraction of footprint the upper (shallow) tier keeps.
_MANSARD_INSET = 0.4


@dataclass(frozen=True)
class Mesh:
    """A wire-frame + face mesh: vertices, undirected edges, and polygon faces."""

    vertices: tuple[Vec3, ...]
    edges: tuple[tuple[int, int], ...]
    faces: tuple[tuple[int, ...], ...]

    def bounding_box(self) -> tuple[Vec3, Vec3]:
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


# --- transform helpers -----------------------------------------------------
def _rot_z(p: Vec3, deg: float) -> Vec3:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    x, y, z = p
    return (c * x - s * y, s * x + c * y, z)


def _rot_x(p: Vec3, deg: float) -> Vec3:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    x, y, z = p
    return (x, c * y - s * z, s * y + c * z)


def _place(verts, size: Vec3, rotation, position: Vec3):
    """Scale by ``size``, rotate yaw(Z) then pitch(X), translate to ``position``."""
    sx, sy, sz = size
    yaw, pitch = rotation
    out = []
    for (x, y, z) in verts:
        p = (x * sx, y * sy, z * sz)
        p = _rot_z(p, yaw)
        p = _rot_x(p, pitch)
        out.append((p[0] + position[0], p[1] + position[1], p[2] + position[2]))
    return tuple(out)


def _box_edges() -> tuple[tuple[int, int], ...]:
    # 0..3 bottom (ccw), 4..7 top (ccw)
    bottom = ((0, 1), (1, 2), (2, 3), (3, 0))
    top = ((4, 5), (5, 6), (6, 7), (7, 4))
    verticals = ((0, 4), (1, 5), (2, 6), (3, 7))
    return bottom + top + verticals


# Canonical unit footprint corners (x,y in [-.5,.5]).
_FP = ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))


def _canonical_cube():
    verts = [(x, y, 0.0) for (x, y) in _FP] + [(x, y, 1.0) for (x, y) in _FP]
    faces = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    return verts, _box_edges(), faces


def _canonical_cylinder(segments: int = 16):
    verts = []
    for z in (0.0, 1.0):
        for i in range(segments):
            a = 2 * math.pi * i / segments
            verts.append((0.5 * math.cos(a), 0.5 * math.sin(a), z))
    edges = []
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        edges.append((i, j))  # bottom ring
        edges.append((segments + i, segments + j))  # top ring
        edges.append((i, segments + i))  # vertical
        faces.append((i, j, segments + j, segments + i))
    return verts, tuple(edges), tuple(faces)


def _canonical_pyramid():
    base = [(x, y, 0.0) for (x, y) in _FP]
    apex = (0.0, 0.0, 1.0)
    verts = base + [apex]
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4))
    faces = ((0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))
    return verts, edges, faces


def _canonical_shed():
    # Box walls to _WALL_FRACTION, then a single-slope roof: y=-.5 eave low, y=+.5 high.
    w = _WALL_FRACTION
    verts = [
        (-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0),   # 0-3 base
        (-0.5, -0.5, w), (0.5, -0.5, w),                                          # 4-5 low eave
        (0.5, 0.5, 1.0), (-0.5, 0.5, 1.0),                                        # 6-7 high eave
    ]
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (4, 5), (5, 6), (6, 7), (7, 4),
    )
    faces = (
        (0, 1, 2, 3),      # floor
        (4, 5, 6, 7),      # sloped roof
        (0, 1, 5, 4),      # low wall
        (2, 3, 7, 6),      # high wall
        (1, 2, 6, 5),      # right gable
        (3, 0, 4, 7),      # left gable
    )
    return verts, edges, faces


def _canonical_hip():
    w = _WALL_FRACTION
    r = _HIP_RIDGE
    base = [(x, y, 0.0) for (x, y) in _FP]
    top = [(x, y, w) for (x, y) in _FP]              # 4-7 wall top
    ridge = [(0.0, -r, 1.0), (0.0, r, 1.0)]          # 8-9 ridge ends
    verts = base + top + ridge
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (4, 8), (5, 8), (6, 9), (7, 9), (8, 9),
    )
    faces = (
        (0, 1, 2, 3),
        (4, 5, 8),          # front hip
        (6, 7, 9),          # back hip
        (5, 6, 9, 8),       # right slope
        (7, 4, 8, 9),       # left slope
        (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),  # walls
    )
    return verts, edges, faces


def _canonical_aframe():
    # Triangular prism: gable ends in x=+/-.5 planes, ridge along y at z=1.
    verts = [
        (-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0),  # 0-3 base
        (0.0, -0.5, 1.0), (0.0, 0.5, 1.0),                                       # 4-5 ridge
    ]
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (0, 4), (1, 4), (2, 5), (3, 5), (4, 5),
    )
    faces = (
        (0, 1, 2, 3),
        (0, 1, 4),          # front gable
        (2, 3, 5),          # back gable
        (0, 4, 5, 3),       # left slope
        (1, 2, 5, 4),       # right slope
    )
    return verts, edges, faces


def _canonical_mansard():
    w = _WALL_FRACTION
    k = _MANSARD_INSET * 0.5   # upper-tier half footprint
    base = [(x, y, 0.0) for (x, y) in _FP]
    mid = [(x, y, w) for (x, y) in _FP]                                  # 4-7 eave
    upper = [(-k, -k, 1.0), (k, -k, 1.0), (k, k, 1.0), (-k, k, 1.0)]     # 8-11 ridge deck
    verts = base + mid + upper
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (8, 9), (9, 10), (10, 11), (11, 8),
        (4, 8), (5, 9), (6, 10), (7, 11),
    )
    faces = (
        (0, 1, 2, 3),
        (8, 9, 10, 11),                 # flat deck
        (4, 5, 9, 8), (5, 6, 10, 9), (6, 7, 11, 10), (7, 4, 8, 11),  # steep lower roof
        (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),      # walls
    )
    return verts, edges, faces


_BUILDERS = {
    "cube": _canonical_cube,
    "cylinder": _canonical_cylinder,
    "pyramid": _canonical_pyramid,
    "shed": _canonical_shed,
    "hip": _canonical_hip,
    "aframe": _canonical_aframe,
    "mansard": _canonical_mansard,
}


def build_shape(
    shape: str,
    position: Vec3 = (0.0, 0.0, 0.0),
    rotation=(0.0, 0.0),
    size: Vec3 = (1.0, 1.0, 1.0),
) -> Mesh:
    """Reconstruct one architectural object's :class:`Mesh` from its parameters.

    ``rotation`` is ``(yaw, pitch)`` in degrees; ``size`` is ``(sx, sy, sz)`` with
    sx/sy the footprint and sz the height. Matches the scene-descriptor row order.
    """
    if shape not in _BUILDERS:
        raise ValueError(f"unknown shape {shape!r}")
    verts, edges, faces = _BUILDERS[shape]()
    placed = _place(verts, size, rotation, position)
    return Mesh(placed, tuple(edges), tuple(faces))


def build_from_object(obj) -> Mesh:
    """Build a mesh from a
    :class:`reconstruction.sketch2cad_scene_descriptor.SceneObject`."""
    return build_shape(obj.shape, obj.position, obj.rotation, obj.size)
