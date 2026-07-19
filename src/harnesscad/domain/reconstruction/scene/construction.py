"""Deterministic scene-graph construction from CAD geometric primitives.

Each mesh is embedded as a node carrying its centroid and 3D bounding box,
and an edge is added between two nodes when their meshes are within a fixed
proximity (``<= 1 cm``, the voxel grid size). This module reconstructs that
low-level *spatial* layer of the scene graph **purely from bounding-box
geometry** -- no learned model, no point cloud. Given a set of primitives
``(id, obj_type, AABB)`` it derives, deterministically:

* **touching / adjacency** -- when the axis-aligned gap between two boxes is at
  or below a proximity threshold (the ``1 cm`` rule);
* **containment** -- when one box fully encloses another (``CONTAINS`` /
  ``CONTAINED_BY``);
* **support / on-top-of** -- when one box rests on another: horizontally
  overlapping *footprints* with the upper box's base meeting the lower box's top;
* **directional relations** -- ``ABOVE`` / ``BELOW`` (z), ``LEFT_OF`` /
  ``RIGHT_OF`` (x), ``FRONT_OF`` / ``BEHIND`` (y) chosen by the axis of maximum
  centroid separation, so exactly one directional pair is emitted per near pair.

Each asymmetric relation is stored together with its inverse (inverse map
in :mod:`reconstruction.scenegraph_model`) so ``a ON_TOP_OF b`` implies
``b SUPPORTS a``. Construction is order-stable: nodes and the candidate pairs are
processed in input order, giving byte-reproducible graphs.

Only stdlib is used. Semantic enrichment (class / affordance / material) and the
functional-relation extraction are handled by sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from harnesscad.domain.reconstruction.scene.model import (
    AABB,
    RelationType,
    SceneGraph,
    SceneNode,
    Vec3,
)


@dataclass(frozen=True)
class Primitive:
    """A CAD primitive to be turned into a scene node."""

    prim_id: str
    obj_type: str
    aabb: AABB
    attributes: Optional[dict] = None


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConstructionConfig:
    """Thresholds for deterministic relation derivation.

    ``proximity`` mirrors the ``1 cm`` adjacency rule (default 0.01).
    ``support_overlap_frac`` is the minimum horizontal footprint-overlap
    fraction (of the smaller footprint) required to call one box supported by
    another. ``emit_directional`` toggles the ABOVE/BELOW/LEFT/RIGHT/FRONT/BEHIND
    layer.
    """

    proximity: float = 0.01
    support_overlap_frac: float = 0.05
    emit_directional: bool = True
    emit_adjacency: bool = True
    emit_containment: bool = True
    emit_support: bool = True


# --------------------------------------------------------------------------- #
# Pairwise relation predicates                                                 #
# --------------------------------------------------------------------------- #
def _footprint_overlap_frac(a: AABB, b: AABB) -> float:
    """Fraction of the smaller xy footprint covered by the xy intersection."""
    ax0, ay0, _ = a.min
    ax1, ay1, _ = a.max
    bx0, by0, _ = b.min
    bx1, by1, _ = b.max
    ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    oy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ox * oy
    if inter <= 0.0:
        return 0.0
    fa = (ax1 - ax0) * (ay1 - ay0)
    fb = (bx1 - bx0) * (by1 - by0)
    smaller = min(fa, fb)
    if smaller <= 0.0:
        return 0.0
    return inter / smaller


def is_on_top_of(upper: AABB, lower: AABB, cfg: ConstructionConfig) -> bool:
    """True if ``upper`` rests on ``lower`` (footprint overlap + z contact)."""
    if _footprint_overlap_frac(upper, lower) < cfg.support_overlap_frac:
        return False
    # upper's base must sit at/above lower's top within proximity, and upper
    # must be the higher box.
    upper_base = upper.min[2]
    lower_top = lower.max[2]
    if upper.centroid[2] <= lower.centroid[2]:
        return False
    return abs(upper_base - lower_top) <= cfg.proximity


def directional_relation(a: AABB, b: AABB) -> RelationType:
    """Directional relation of ``a`` relative to ``b`` by dominant centroid axis.

    Returns the relation ``a REL b`` (e.g. ABOVE means a is above b). The axis
    with the greatest absolute centroid separation is chosen; ties break in the
    order z, x, y for determinism.
    """
    ca = a.centroid
    cb = b.centroid
    dx = ca[0] - cb[0]
    dy = ca[1] - cb[1]
    dz = ca[2] - cb[2]
    # tie-break priority: z, x, y
    axis = max(((abs(dz), 0), (abs(dx), 1), (abs(dy), 2)))[1]
    if axis == 0:
        return RelationType.ABOVE if dz >= 0 else RelationType.BELOW
    if axis == 1:
        return RelationType.RIGHT_OF if dx >= 0 else RelationType.LEFT_OF
    return RelationType.BEHIND if dy >= 0 else RelationType.FRONT_OF


# --------------------------------------------------------------------------- #
# Construction                                                                 #
# --------------------------------------------------------------------------- #
def build_scene_graph(
    primitives: Sequence[Primitive],
    cfg: Optional[ConstructionConfig] = None,
) -> SceneGraph:
    """Build a spatial scene graph from primitives via deterministic AABB tests.

    Pairs are examined in input order (i < j). For each near pair the derived
    relations are added with their inverses. Directional relations are emitted
    for adjacent pairs only (so the graph stays sparse and matches the intended
    proximity edges).
    """
    cfg = cfg or ConstructionConfig()
    g = SceneGraph()
    for p in primitives:
        g.add_node(SceneNode(p.prim_id, p.obj_type, p.aabb, dict(p.attributes or {})))

    prims = list(primitives)
    for i in range(len(prims)):
        pi = prims[i]
        for j in range(i + 1, len(prims)):
            pj = prims[j]
            _relate_pair(g, pi, pj, cfg)
    return g


def _relate_pair(g: SceneGraph, pi: Primitive, pj: Primitive, cfg: ConstructionConfig) -> None:
    a, b = pi.aabb, pj.aabb

    # containment takes precedence (a strict hierarchy relation)
    if cfg.emit_containment:
        if a.contains(b) and a.volume > b.volume:
            g.add_edge(pi.prim_id, RelationType.CONTAINS, pj.prim_id, add_inverse=True)
            return
        if b.contains(a) and b.volume > a.volume:
            g.add_edge(pj.prim_id, RelationType.CONTAINS, pi.prim_id, add_inverse=True)
            return

    gap = a.gap(b)
    near = gap <= cfg.proximity
    if not near:
        return

    touching = a.overlaps(b, tol=cfg.proximity)

    # support relation (vertical, footprint overlap)
    if cfg.emit_support:
        if is_on_top_of(a, b, cfg):
            g.add_edge(pi.prim_id, RelationType.ON_TOP_OF, pj.prim_id, add_inverse=True)
        elif is_on_top_of(b, a, cfg):
            g.add_edge(pj.prim_id, RelationType.ON_TOP_OF, pi.prim_id, add_inverse=True)

    # adjacency / touching (symmetric proximity)
    if cfg.emit_adjacency:
        rel = RelationType.TOUCHING if touching else RelationType.ADJACENT_TO
        g.add_edge(pi.prim_id, rel, pj.prim_id)
        g.add_edge(pj.prim_id, rel, pi.prim_id)

    # directional layer
    if cfg.emit_directional:
        rel = directional_relation(a, b)
        g.add_edge(pi.prim_id, rel, pj.prim_id, add_inverse=True)


def connect_by_proximity(
    g: SceneGraph,
    primitives: Sequence[Primitive],
    relation: RelationType = RelationType.CONNECTED_TO,
    proximity: float = 0.01,
) -> int:
    """Add symmetric ``relation`` edges between every pair within ``proximity``.

    Returns the number of undirected connections added. Used to build the
    connectivity layer that the functional-relation extraction traverses.
    """
    prims = list(primitives)
    added = 0
    for i in range(len(prims)):
        for j in range(i + 1, len(prims)):
            if prims[i].aabb.gap(prims[j].aabb) <= proximity:
                e1 = g.add_edge(prims[i].prim_id, relation, prims[j].prim_id)
                e2 = g.add_edge(prims[j].prim_id, relation, prims[i].prim_id)
                if e1 is not None or e2 is not None:
                    added += 1
    return added
