"""Object-layout scene-composition representation for authored 3D worlds.

Paper: *WorldCraft: Photo-Realistic 3D World Creation and Customization via LLM
Agents* (Liu, Tang, Tai).

WorldCraft's ArrangeIt stage (Sec. 3.3) composes a scene by *placing* a
collection of 3D assets, each with a 3D location ``p_i = (x, y, z)`` and an
orientation ``theta_i = (theta_x, theta_y, theta_z)`` in Euler angles, and by
recognising the **hierarchical dependencies between objects** (e.g. a bookshelf
and the books it holds) so the arrangement can be decomposed into an *object
tree* of subproblems (Fig. 5).

This module is the deterministic, stdlib-only **authored layout representation**
that stage produces and consumes. It is deliberately DISTINCT from
``reconstruction.scenegraph_model`` (paper 159), which *reads* typed spatial
relations off already-positioned geometry: here the poses are the authored
design variables, and the graph is a placement/containment *tree* of design
intent rather than a derived relation graph.

Provides:

* :class:`Pose` -- a rigid placement (position, Euler orientation, uniform or
  per-axis scale) with a value-semantics identity and translation/rotation
  helpers;
* :class:`ObjectPlacement` -- a placed asset: a stable id, a semantic category,
  a local (unscaled, unposed) half-extent box, a :class:`Pose`, an optional
  parent id (its host in the object tree) and free-form attributes;
* :class:`LayoutSpec` -- the ordered container of placements plus optional room
  bounds, with parent/child (object-tree) queries, topological ordering and a
  round-trippable ``to_dict`` / ``from_dict`` serialization.

Everything is pure and deterministic; there is no randomness, wall clock or I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Tuple

Vec3 = Tuple[float, float, float]

_TWO_PI = 2.0 * math.pi


def _as_vec3(value: object, name: str) -> Vec3:
    seq = tuple(value)  # type: ignore[arg-type]
    if len(seq) != 3:
        raise ValueError(f"{name} must have 3 components, got {len(seq)}")
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _wrap_angle(a: float) -> float:
    """Wrap an angle into the half-open interval ``[0, 2*pi)`` (paper's range)."""
    w = math.fmod(a, _TWO_PI)
    if w < 0.0:
        w += _TWO_PI
    # fmod of an exact multiple can produce -0.0; normalise.
    return 0.0 if w == 0.0 else w


# --------------------------------------------------------------------------- #
# Pose                                                                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pose:
    """A rigid placement: position, Euler orientation and per-axis scale.

    ``position`` is the world location ``p_i``. ``orientation`` holds the Euler
    angles ``theta_i`` (radians), each normalised into ``[0, 2*pi)`` to match the
    paper's ``[0, 2*pi]^3`` orientation domain. ``scale`` is a per-axis positive
    scale factor applied to the local geometry (uniform by default).
    """

    position: Vec3 = (0.0, 0.0, 0.0)
    orientation: Vec3 = (0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _as_vec3(self.position, "position"))
        orient = _as_vec3(self.orientation, "orientation")
        object.__setattr__(self, "orientation", tuple(_wrap_angle(a) for a in orient))
        scale = _as_vec3(self.scale, "scale")
        if any(s <= 0.0 for s in scale):
            raise ValueError("scale components must be strictly positive")
        object.__setattr__(self, "scale", scale)

    # -- constructors -------------------------------------------------------- #
    @staticmethod
    def at(x: float, y: float, z: float) -> "Pose":
        """A pose at ``(x, y, z)`` with identity orientation and unit scale."""
        return Pose(position=(x, y, z))

    @staticmethod
    def uniform_scale(s: float) -> "Pose":
        """Identity pose with a uniform scale ``s`` on all axes."""
        return Pose(scale=(s, s, s))

    # -- transforms ---------------------------------------------------------- #
    def translated(self, dx: float, dy: float, dz: float) -> "Pose":
        px, py, pz = self.position
        return replace(self, position=(px + dx, py + dy, pz + dz))

    def rotated_z(self, dtheta: float) -> "Pose":
        """Return a copy yawed by ``dtheta`` radians about the vertical axis."""
        ox, oy, oz = self.orientation
        return replace(self, orientation=(ox, oy, _wrap_angle(oz + dtheta)))

    def with_scale(self, sx: float, sy: float, sz: float) -> "Pose":
        return replace(self, scale=(sx, sy, sz))

    @property
    def yaw(self) -> float:
        """Rotation about the vertical (z) axis -- the primary furniture yaw."""
        return self.orientation[2]

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "position": list(self.position),
            "orientation": list(self.orientation),
            "scale": list(self.scale),
        }

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "Pose":
        return Pose(
            position=tuple(data.get("position", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
            orientation=tuple(data.get("orientation", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
            scale=tuple(data.get("scale", (1.0, 1.0, 1.0))),  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# Object placement                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class ObjectPlacement:
    """A placed 3D asset in an authored layout.

    ``half_extent`` is the *local* (unposed, unscaled) half-size of the asset's
    axis-aligned bounding box about its own origin. The world footprint follows
    from the :class:`Pose` (scale then translation); axis-aligned yaw of a
    multiple of 90 degrees is handled by :meth:`world_half_extent`.
    ``parent_id`` links this object to its host in the object tree (``None`` for
    a root/free object). ``attributes`` carries authored semantic annotations.
    """

    object_id: str
    category: str
    half_extent: Vec3
    pose: Pose = field(default_factory=Pose)
    parent_id: Optional[str] = None
    attributes: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.half_extent = _as_vec3(self.half_extent, "half_extent")
        if any(h < 0.0 for h in self.half_extent):
            raise ValueError("half_extent components must be non-negative")
        if not isinstance(self.pose, Pose):
            self.pose = Pose.from_dict(self.pose)  # type: ignore[arg-type]

    # -- derived world geometry --------------------------------------------- #
    def scaled_half_extent(self) -> Vec3:
        """Half-extent after applying the pose's per-axis scale (no rotation)."""
        hx, hy, hz = self.half_extent
        sx, sy, sz = self.pose.scale
        return (hx * sx, hy * sy, hz * sz)

    def world_half_extent(self) -> Vec3:
        """Axis-aligned world half-extent, accounting for a quarter-turn yaw.

        For the common furniture case the yaw is a multiple of 90 degrees; a
        90/270-degree yaw swaps the x and y footprint. Non-axis yaws fall back to
        the bounding half-extent of the rotated footprint about z.
        """
        sx, sy, sz = self.scaled_half_extent()
        yaw = self.pose.yaw
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        return (sx * c + sy * s, sx * s + sy * c, sz)

    def world_center(self) -> Vec3:
        return self.pose.position

    def world_bounds(self) -> Tuple[Vec3, Vec3]:
        """Axis-aligned ``(min, max)`` world corners of the placed footprint."""
        cx, cy, cz = self.world_center()
        hx, hy, hz = self.world_half_extent()
        return ((cx - hx, cy - hy, cz - hz), (cx + hx, cy + hy, cz + hz))

    def to_dict(self) -> Dict[str, object]:
        return {
            "object_id": self.object_id,
            "category": self.category,
            "half_extent": list(self.half_extent),
            "pose": self.pose.to_dict(),
            "parent_id": self.parent_id,
            "attributes": dict(self.attributes),
        }

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "ObjectPlacement":
        return ObjectPlacement(
            object_id=str(data["object_id"]),
            category=str(data["category"]),
            half_extent=tuple(data["half_extent"]),  # type: ignore[arg-type]
            pose=Pose.from_dict(data.get("pose", {})),  # type: ignore[arg-type]
            parent_id=data.get("parent_id"),  # type: ignore[assignment]
            attributes=dict(data.get("attributes", {})),  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# Layout spec                                                                  #
# --------------------------------------------------------------------------- #
class LayoutSpec:
    """Ordered collection of :class:`ObjectPlacement` forming a scene layout.

    Insertion order is preserved deterministically. ``room_bounds`` is an
    optional axis-aligned ``(min, max)`` extent of the enclosing sub-space (the
    room the coordinator carves out in Sec. 3.1). The parent links across
    placements form the *object tree* ArrangeIt decomposes into subproblems.
    """

    def __init__(self, room_bounds: Optional[Tuple[Vec3, Vec3]] = None) -> None:
        self._placements: Dict[str, ObjectPlacement] = {}
        self._order: List[str] = []
        self.room_bounds: Optional[Tuple[Vec3, Vec3]] = None
        if room_bounds is not None:
            lo = _as_vec3(room_bounds[0], "room_bounds.min")
            hi = _as_vec3(room_bounds[1], "room_bounds.max")
            if any(h < l for l, h in zip(lo, hi)):
                raise ValueError("room_bounds max component smaller than min")
            self.room_bounds = (lo, hi)

    # -- placements ---------------------------------------------------------- #
    def add(self, placement: ObjectPlacement) -> ObjectPlacement:
        if placement.object_id in self._placements:
            raise ValueError(f"duplicate object id: {placement.object_id!r}")
        if placement.parent_id is not None and placement.parent_id not in self._placements:
            raise KeyError(f"unknown parent id: {placement.parent_id!r}")
        self._placements[placement.object_id] = placement
        self._order.append(placement.object_id)
        return placement

    def get(self, object_id: str) -> ObjectPlacement:
        return self._placements[object_id]

    def has(self, object_id: str) -> bool:
        return object_id in self._placements

    @property
    def placements(self) -> List[ObjectPlacement]:
        return [self._placements[i] for i in self._order]

    @property
    def object_ids(self) -> List[str]:
        return list(self._order)

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, object_id: object) -> bool:
        return object_id in self._placements

    def __iter__(self):
        return iter(self.placements)

    # -- object tree --------------------------------------------------------- #
    def roots(self) -> List[ObjectPlacement]:
        """Placements with no parent (order-stable)."""
        return [self._placements[i] for i in self._order if self._placements[i].parent_id is None]

    def children(self, object_id: str) -> List[ObjectPlacement]:
        """Direct children of ``object_id`` (order-stable)."""
        return [
            self._placements[i]
            for i in self._order
            if self._placements[i].parent_id == object_id
        ]

    def descendants(self, object_id: str) -> List[ObjectPlacement]:
        """All transitive children of ``object_id`` in depth-first order."""
        out: List[ObjectPlacement] = []
        for child in self.children(object_id):
            out.append(child)
            out.extend(self.descendants(child.object_id))
        return out

    def ancestors(self, object_id: str) -> List[ObjectPlacement]:
        """Chain of hosts from the immediate parent up to a root."""
        out: List[ObjectPlacement] = []
        cur = self._placements[object_id].parent_id
        seen = {object_id}
        while cur is not None and cur not in seen:
            seen.add(cur)
            out.append(self._placements[cur])
            cur = self._placements[cur].parent_id
        return out

    def topological_order(self) -> List[ObjectPlacement]:
        """Placements ordered so every parent precedes its children.

        Deterministic: preserves insertion order among independent objects.
        """
        result: List[ObjectPlacement] = []
        emitted = set()

        def emit(pid: str) -> None:
            if pid in emitted:
                return
            emitted.add(pid)
            result.append(self._placements[pid])
            for child in self.children(pid):
                emit(child.object_id)

        for root in self.roots():
            emit(root.object_id)
        # Any placement not reachable from a root (should not happen given the
        # add-time parent check) is appended in insertion order for safety.
        for pid in self._order:
            if pid not in emitted:
                emit(pid)
        return result

    # -- serialization ------------------------------------------------------- #
    def to_dict(self) -> Dict[str, object]:
        return {
            "room_bounds": (
                None if self.room_bounds is None
                else [list(self.room_bounds[0]), list(self.room_bounds[1])]
            ),
            "placements": [self._placements[i].to_dict() for i in self._order],
        }

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "LayoutSpec":
        rb = data.get("room_bounds")
        bounds = None
        if rb is not None:
            bounds = (tuple(rb[0]), tuple(rb[1]))  # type: ignore[index]
        spec = LayoutSpec(room_bounds=bounds)
        for pd in data.get("placements", []):  # type: ignore[union-attr]
            spec.add(ObjectPlacement.from_dict(pd))  # type: ignore[arg-type]
        return spec
