"""Procedural scene instantiation from an authored layout spec.

Paper: *WorldCraft: Photo-Realistic 3D World Creation and Customization via LLM
Agents* (Liu, Tang, Tai), Sec. 3 (pipeline: object creation -> layout ->
instantiation into ``scene.blend``).

Once ForgeIt has produced an asset library and ArrangeIt has solved a layout,
WorldCraft *instantiates* the scene: each placement is realised as a concrete
object instance carrying a world transform (built from its pose), with the
object-tree parent transforms composed so a book placed relative to its shelf
follows the shelf. The renderer/Blender export is external; the deterministic,
locally buildable piece is the **instantiation of a layout spec against an asset
library into a flat list of transformed instances**.

This module is DISTINCT from the scene-graph and layout-solver modules: it does
not search poses or read relations -- it *evaluates* the authored
:class:`reconstruction.worldcraft_layout_spec.LayoutSpec` into concrete world
transforms, resolving parent chains and asset references.

Provides:

* :class:`AssetDef` -- an entry in the asset library (category -> a canonical
  local half-extent and default attributes ForgeIt produced);
* :class:`AssetLibrary` -- a registry of :class:`AssetDef` by name;
* :class:`Transform` -- an affine transform (scale, yaw about z, translation)
  with point application and composition, matching :class:`Pose` semantics;
* :class:`SceneInstance` -- a realised object: id, category, world transform,
  world AABB and merged attributes;
* :func:`instantiate_layout` -- realise every placement into a
  :class:`SceneInstance`, composing parent transforms in topological order;
* :func:`instances_bounds` -- overall axis-aligned bounds of an instantiated scene.

Deterministic and stdlib-only; no randomness, wall clock or I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from reconstruction.worldcraft_layout_spec import (
    LayoutSpec,
    ObjectPlacement,
    Pose,
)

Vec3 = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
# Asset library                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AssetDef:
    """A canonical asset produced by ForgeIt for a category.

    ``half_extent`` is the asset's local (unit-scale) half-size; ``attributes``
    are default annotations (material name, affordance, ...) merged into every
    instance of this asset unless overridden by the placement.
    """

    name: str
    half_extent: Vec3
    attributes: Dict[str, object] = field(default_factory=dict)


class AssetLibrary:
    """Registry of :class:`AssetDef` keyed by name (order-stable)."""

    def __init__(self) -> None:
        self._assets: Dict[str, AssetDef] = {}

    def register(self, asset: AssetDef) -> AssetDef:
        if asset.name in self._assets:
            raise ValueError(f"duplicate asset name: {asset.name!r}")
        self._assets[asset.name] = asset
        return asset

    def has(self, name: str) -> bool:
        return name in self._assets

    def get(self, name: str) -> AssetDef:
        return self._assets[name]

    @property
    def names(self) -> List[str]:
        return list(self._assets.keys())


# --------------------------------------------------------------------------- #
# Transform                                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Transform:
    """Affine transform: per-axis scale, yaw about z, then translation.

    Point mapping is ``world = R_z(yaw) . (scale * local) + translation``.
    Composition ``a.compose(b)`` yields the transform equivalent to applying
    ``b`` in ``a``'s frame (a is the parent, b is the child-local transform).
    """

    scale: Vec3 = (1.0, 1.0, 1.0)
    yaw: float = 0.0
    translation: Vec3 = (0.0, 0.0, 0.0)

    @staticmethod
    def from_pose(pose: Pose) -> "Transform":
        return Transform(scale=pose.scale, yaw=pose.yaw, translation=pose.position)

    def apply(self, point: Vec3) -> Vec3:
        sx, sy, sz = self.scale
        x, y, z = point
        x, y, z = x * sx, y * sy, z * sz
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        rx = c * x - s * y
        ry = s * x + c * y
        tx, ty, tz = self.translation
        return (rx + tx, ry + ty, z + tz)

    def compose(self, child: "Transform") -> "Transform":
        """Return parent(self) applied to a child-local transform."""
        # New scale multiplies component-wise (both axis-aligned scales).
        sx, sy, sz = self.scale
        cx, cy, cz = child.scale
        new_scale = (sx * cx, sy * cy, sz * cz)
        new_yaw = self.yaw + child.yaw
        # Child origin mapped through the parent gives the new translation.
        new_translation = self.apply(child.translation)
        return Transform(scale=new_scale, yaw=new_yaw, translation=new_translation)


# --------------------------------------------------------------------------- #
# Scene instance                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class SceneInstance:
    """A realised, world-placed object instance."""

    object_id: str
    category: str
    transform: Transform
    world_min: Vec3
    world_max: Vec3
    attributes: Dict[str, object] = field(default_factory=dict)

    @property
    def world_center(self) -> Vec3:
        return tuple((lo + hi) / 2.0 for lo, hi in zip(self.world_min, self.world_max))  # type: ignore[return-value]


def _axis_aligned_bounds(transform: Transform, half_extent: Vec3) -> Tuple[Vec3, Vec3]:
    """World AABB of a box of given local half-extent under ``transform``."""
    hx, hy, hz = half_extent
    corners = [
        (sx * hx, sy * hy, sz * hz)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]
    pts = [transform.apply(c) for c in corners]
    lo = tuple(min(p[i] for p in pts) for i in range(3))
    hi = tuple(max(p[i] for p in pts) for i in range(3))
    return lo, hi  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Instantiation                                                                #
# --------------------------------------------------------------------------- #
def instantiate_layout(
    spec: LayoutSpec,
    library: Optional[AssetLibrary] = None,
    *,
    compose_parents: bool = True,
) -> List[SceneInstance]:
    """Realise every placement in ``spec`` into a :class:`SceneInstance`.

    Placements are processed in the spec's topological order so a parent's world
    transform is available before its children. When ``compose_parents`` is true
    the child's local transform is expressed in the parent's frame (its pose is
    treated as relative to the parent); otherwise every pose is treated as global.

    The asset's canonical half-extent comes from ``library`` (looked up by the
    placement's ``category``) when available, else from the placement's own
    ``half_extent``. Attributes merge library defaults under placement overrides.

    Raises ``KeyError`` if ``compose_parents`` is off but this is unaffected; the
    topological order from the spec guarantees parents resolve first.
    """
    world_tf: Dict[str, Transform] = {}
    instances: List[SceneInstance] = []

    for placement in spec.topological_order():
        local_tf = Transform.from_pose(placement.pose)
        if compose_parents and placement.parent_id is not None:
            parent_tf = world_tf[placement.parent_id]
            tf = parent_tf.compose(local_tf)
        else:
            tf = local_tf
        world_tf[placement.object_id] = tf

        half_extent = placement.half_extent
        attrs: Dict[str, object] = {}
        if library is not None and library.has(placement.category):
            asset = library.get(placement.category)
            half_extent = asset.half_extent
            attrs.update(asset.attributes)
        attrs.update(placement.attributes)

        lo, hi = _axis_aligned_bounds(tf, half_extent)
        instances.append(SceneInstance(
            object_id=placement.object_id,
            category=placement.category,
            transform=tf,
            world_min=lo,
            world_max=hi,
            attributes=attrs,
        ))

    return instances


def instances_bounds(instances: List[SceneInstance]) -> Optional[Tuple[Vec3, Vec3]]:
    """Overall axis-aligned ``(min, max)`` of a scene, or ``None`` if empty."""
    if not instances:
        return None
    lo = tuple(min(i.world_min[a] for i in instances) for a in range(3))
    hi = tuple(max(i.world_max[a] for i in instances) for a in range(3))
    return lo, hi  # type: ignore[return-value]
