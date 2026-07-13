"""Deterministic interpreter for the global-coordinate sketch-and-extrude CAD DSL.

Implements the mini feature-based CAD kernel described in Makatura et al.,
"How Can Large Language Models Help Humans in Design and Manufacturing?"
(Appendix A.2 / Figure 84). A program is a sequence of ``createSketch`` and
``extrude`` operations whose primitive centres are given in *global* 3D
coordinates. Evaluating the program produces an ordered list of placed solids,
each with its global-space axis-aligned bounding box (AABB).

Only the global-coordinate variant is modelled here because it is fully
deterministic: every solid's placement is fixed by the program alone, with no
dependence on wall-clock time or randomness.

This module is deliberately independent from ``geometry/euclid_dsl.py`` (a
straightedge-and-compass construction DSL); the two share no code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Plane / face geometry tables
# ---------------------------------------------------------------------------

# Default sketch planes -> outward normal (unit vector) and the pair of
# in-plane axes (as axis indices 0=X, 1=Y, 2=Z).
_PLANE_NORMALS: Dict[str, Tuple[float, float, float]] = {
    "XY_PLANE": (0.0, 0.0, 1.0),
    "XZ_PLANE": (0.0, 1.0, 0.0),
    "ZY_PLANE": (1.0, 0.0, 0.0),
}

# In-plane axes per plane, following the paper's convention:
#   XY_PLANE in-plane axes = (X, Y)
#   XZ_PLANE in-plane axes = (X, Z)
#   ZY_PLANE in-plane axes = (Z, Y)
_PLANE_INPLANE_AXES: Dict[str, Tuple[int, int]] = {
    "XY_PLANE": (0, 1),
    "XZ_PLANE": (0, 2),
    "ZY_PLANE": (2, 1),
}

# cap() side -> outward normal.
_SIDE_NORMALS: Dict[str, Tuple[float, float, float]] = {
    "min_x": (-1.0, 0.0, 0.0),
    "max_x": (1.0, 0.0, 0.0),
    "min_y": (0.0, -1.0, 0.0),
    "max_y": (0.0, 1.0, 0.0),
    "min_z": (0.0, 0.0, -1.0),
    "max_z": (0.0, 0.0, 1.0),
}

# cap() side -> (axis index, is_max) telling which AABB coordinate the face
# sits at.
_SIDE_FACE: Dict[str, Tuple[int, bool]] = {
    "min_x": (0, False),
    "max_x": (0, True),
    "min_y": (1, False),
    "max_y": (1, True),
    "min_z": (2, False),
    "max_z": (2, True),
}

_AXIS_NAMES = ("x", "y", "z")


def _normal_axis(normal: Tuple[float, float, float]) -> int:
    """Return the axis index (0,1,2) of a purely axis-aligned normal."""
    for axis in range(3):
        if normal[axis] != 0.0:
            return axis
    raise ValueError("normal is the zero vector")


# ---------------------------------------------------------------------------
# Sketch primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Circle:
    """A circle primitive with a global-coordinate centre and radius."""

    center_x: float
    center_y: float
    center_z: float
    radius: float

    def __post_init__(self) -> None:
        if self.radius < 0.0:
            raise ValueError("circle radius must be non-negative")

    @property
    def center(self) -> Tuple[float, float, float]:
        return (float(self.center_x), float(self.center_y), float(self.center_z))


@dataclass(frozen=True)
class Rectangle:
    """A rectangle primitive.

    ``length`` is measured along the plane's first in-plane axis (u) and
    ``width`` along the second in-plane axis (v). The rectangle is centred on
    its global-coordinate centre.
    """

    center_x: float
    center_y: float
    center_z: float
    length: float
    width: float

    def __post_init__(self) -> None:
        if self.length < 0.0:
            raise ValueError("rectangle length must be non-negative")
        if self.width < 0.0:
            raise ValueError("rectangle width must be non-negative")

    @property
    def center(self) -> Tuple[float, float, float]:
        return (float(self.center_x), float(self.center_y), float(self.center_z))


# ---------------------------------------------------------------------------
# Plane specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DefaultPlane:
    """One of the three default sketch planes."""

    name: str


@dataclass(frozen=True)
class CapPlane:
    """A planar face of a previously-extruded solid, selected by ``cap()``."""

    solid_id: int
    side: str


# ---------------------------------------------------------------------------
# Sketch / solid records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sketch:
    """A resolved sketch: a primitive placed on a plane in global space.

    ``origin`` is the primitive centre after any cap() normal-axis override.
    ``normal`` is the plane's outward normal (extrusion direction).
    """

    id: int
    primitive: object
    normal: Tuple[float, float, float]
    inplane_axes: Tuple[int, int]
    origin: Tuple[float, float, float]


@dataclass(frozen=True)
class Solid:
    """An extruded solid with its global axis-aligned bounding box."""

    id: int
    kind: str  # "box" or "cylinder"
    normal: Tuple[float, float, float]
    aabb: Tuple[float, float, float, float, float, float]

    @property
    def xmin(self) -> float:
        return self.aabb[0]

    @property
    def ymin(self) -> float:
        return self.aabb[1]

    @property
    def zmin(self) -> float:
        return self.aabb[2]

    @property
    def xmax(self) -> float:
        return self.aabb[3]

    @property
    def ymax(self) -> float:
        return self.aabb[4]

    @property
    def zmax(self) -> float:
        return self.aabb[5]

    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.aabb[0] + self.aabb[3]) / 2.0,
            (self.aabb[1] + self.aabb[4]) / 2.0,
            (self.aabb[2] + self.aabb[5]) / 2.0,
        )


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------


class Interpreter:
    """Builder + evaluator for a sketch-and-extrude program.

    Programs are constructed without string ``eval``: call :meth:`create_sketch`
    and :meth:`extrude` to append operations; :meth:`cap` builds a face plane
    spec referencing an existing solid. The resulting solids (with global AABBs)
    are available via :attr:`solids`.
    """

    def __init__(self) -> None:
        self._sketches: Dict[int, Sketch] = {}
        self._solids: Dict[int, Solid] = {}
        self._order: List[int] = []  # solid ids in creation order
        self._next_sketch_id = 0
        self._next_solid_id = 0

    # -- plane / face helpers ------------------------------------------------

    def cap(self, solid_id: int, side: str) -> CapPlane:
        """Return a plane spec for a face of an existing solid."""
        if solid_id not in self._solids:
            raise ValueError("cap() references unknown solid id: %r" % (solid_id,))
        if side not in _SIDE_NORMALS:
            raise ValueError("unknown cap side: %r" % (side,))
        return CapPlane(solid_id=solid_id, side=side)

    # -- sketch --------------------------------------------------------------

    def create_sketch(self, primitive: object, plane: object) -> int:
        """Place ``primitive`` on ``plane`` and return a new sketch id.

        ``plane`` may be the string name of a default plane ("XY_PLANE",
        "XZ_PLANE", "ZY_PLANE"), a :class:`DefaultPlane`, or a :class:`CapPlane`
        produced by :meth:`cap`.
        """
        if not isinstance(primitive, (Circle, Rectangle)):
            raise ValueError("primitive must be a Circle or Rectangle")

        if isinstance(plane, str):
            plane = DefaultPlane(plane)

        if isinstance(plane, DefaultPlane):
            if plane.name not in _PLANE_NORMALS:
                raise ValueError("unknown plane: %r" % (plane.name,))
            normal = _PLANE_NORMALS[plane.name]
            inplane = _PLANE_INPLANE_AXES[plane.name]
            origin = primitive.center
        elif isinstance(plane, CapPlane):
            if plane.solid_id not in self._solids:
                raise ValueError(
                    "cap() references unknown solid id: %r" % (plane.solid_id,)
                )
            if plane.side not in _SIDE_NORMALS:
                raise ValueError("unknown cap side: %r" % (plane.side,))
            normal = _SIDE_NORMALS[plane.side]
            axis = _normal_axis(normal)
            # In-plane axes are the two axes other than the normal axis, kept in
            # ascending index order (matching the default-plane conventions:
            # +Z face -> (X,Y); +Y face -> (X,Z); +X face -> (Z,Y) uses (Z,Y)).
            inplane = _cap_inplane_axes(axis)
            # Override the normal-axis coordinate with the selected face's
            # extreme coordinate on the referenced solid.
            face_axis, is_max = _SIDE_FACE[plane.side]
            solid = self._solids[plane.solid_id]
            face_coord = solid.aabb[3 + face_axis] if is_max else solid.aabb[face_axis]
            base = list(primitive.center)
            base[face_axis] = face_coord
            origin = (base[0], base[1], base[2])
        else:
            raise ValueError("unknown plane specification: %r" % (plane,))

        sketch = Sketch(
            id=self._next_sketch_id,
            primitive=primitive,
            normal=normal,
            inplane_axes=inplane,
            origin=origin,
        )
        self._sketches[sketch.id] = sketch
        self._next_sketch_id += 1
        return sketch.id

    # -- extrude -------------------------------------------------------------

    def extrude(self, sketch_id: int, length: float) -> int:
        """Extrude a sketch along its plane normal by ``length`` (> 0)."""
        if sketch_id not in self._sketches:
            raise ValueError("extrude references unknown sketch id: %r" % (sketch_id,))
        if length <= 0.0:
            raise ValueError("extrude length must be positive")

        sketch = self._sketches[sketch_id]
        origin = sketch.origin
        normal = sketch.normal
        axis = _normal_axis(normal)
        u_axis, v_axis = sketch.inplane_axes

        # In-plane half extents.
        prim = sketch.primitive
        if isinstance(prim, Circle):
            kind = "cylinder"
            half_u = prim.radius
            half_v = prim.radius
        else:  # Rectangle
            kind = "box"
            half_u = prim.length / 2.0
            half_v = prim.width / 2.0

        lo = [0.0, 0.0, 0.0]
        hi = [0.0, 0.0, 0.0]

        # In-plane extents centred on origin.
        lo[u_axis] = origin[u_axis] - half_u
        hi[u_axis] = origin[u_axis] + half_u
        lo[v_axis] = origin[v_axis] - half_v
        hi[v_axis] = origin[v_axis] + half_v

        # Extrusion along the normal axis from the sketch plane.
        base = origin[axis]
        direction = normal[axis]  # +1 or -1
        far = base + direction * length
        lo[axis] = min(base, far)
        hi[axis] = max(base, far)

        aabb = (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])
        solid = Solid(id=self._next_solid_id, kind=kind, normal=normal, aabb=aabb)
        self._solids[solid.id] = solid
        self._order.append(solid.id)
        self._next_solid_id += 1
        return solid.id

    # -- results -------------------------------------------------------------

    @property
    def solids(self) -> List[Solid]:
        """Solids in creation order."""
        return [self._solids[i] for i in self._order]

    def solid(self, solid_id: int) -> Solid:
        if solid_id not in self._solids:
            raise ValueError("unknown solid id: %r" % (solid_id,))
        return self._solids[solid_id]

    def assembly_aabb(self) -> Tuple[float, float, float, float, float, float]:
        """Overall AABB enclosing every solid in the program."""
        if not self._order:
            raise ValueError("no solids in program")
        solids = self.solids
        xmin = min(s.aabb[0] for s in solids)
        ymin = min(s.aabb[1] for s in solids)
        zmin = min(s.aabb[2] for s in solids)
        xmax = max(s.aabb[3] for s in solids)
        ymax = max(s.aabb[4] for s in solids)
        zmax = max(s.aabb[5] for s in solids)
        return (xmin, ymin, zmin, xmax, ymax, zmax)


def _cap_inplane_axes(normal_axis: int) -> Tuple[int, int]:
    """In-plane axes for a cap face given its normal axis.

    Matches the default-plane conventions so that a cap face parallel to a
    default plane uses the same axis ordering:
      normal Z -> (X, Y)
      normal Y -> (X, Z)
      normal X -> (Z, Y)
    """
    if normal_axis == 2:
        return (0, 1)
    if normal_axis == 1:
        return (0, 2)
    return (2, 1)
