"""Typed 3D scene-graph representation for CAD-based industrial environments.

An unstructured CAD environment is enriched into a
**multi-layered 3D scene graph** ``G = (V, E)`` whose nodes are meshes/objects
carrying geometric properties (centroid, 3D bounding box, usd path) and whose
edges encode *spatial adjacency* and, at a higher layer, *functional* relations.
The graph is the structured representation required for reasoning and dynamic
simulation.

This module is the deterministic, stdlib-only **data model** underneath that
pipeline. It provides:

* :class:`AABB` -- an axis-aligned bounding box with centroid / extent / volume
  and the containment / overlap / separation predicates used to derive relations;
* :class:`RelationType` -- the closed set of typed relation labels
  (``ON_TOP_OF``, ``ADJACENT_TO``, ``CONNECTED_TO``, ``CONTAINS``, ``SUPPORTS``
  plus directional ``ABOVE``/``BELOW``/``LEFT_OF``/``RIGHT_OF``/``FRONT_OF``/
  ``BEHIND`` and ``TOUCHING``) together with their inverse map, so an
  ``a ON_TOP_OF b`` fact implies ``b SUPPORTS a``;
* :class:`SceneNode` -- a typed object node with an :class:`AABB`, free-form
  semantic/geometric attributes and stable identity;
* :class:`RelationEdge` -- a typed directed edge ``(source, relation, target)``;
* :class:`SceneGraph` -- the container with deterministic insertion order,
  adjacency lookup, neighbour / relation queries and inverse-edge helpers.

Everything is pure, deterministic and network-free. The LVLM labelling and
DBSCAN clustering are handled by sibling modules; this file only
defines the representation they populate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

Vec3 = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
# Axis-aligned bounding box                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AABB:
    """Axis-aligned bounding box defined by inclusive ``min`` / ``max`` corners.

    Axis convention (matches the stored 3D bounding box / centroid):
    ``x`` = left/right, ``y`` = front/behind, ``z`` = up/down (vertical).
    """

    min: Vec3
    max: Vec3

    def __post_init__(self) -> None:
        for lo, hi in zip(self.min, self.max):
            if hi < lo:
                raise ValueError("AABB max component smaller than min component")

    # -- derived geometry ---------------------------------------------------- #
    @property
    def centroid(self) -> Vec3:
        return tuple((lo + hi) / 2.0 for lo, hi in zip(self.min, self.max))  # type: ignore[return-value]

    @property
    def extent(self) -> Vec3:
        return tuple(hi - lo for lo, hi in zip(self.min, self.max))  # type: ignore[return-value]

    @property
    def volume(self) -> float:
        ex, ey, ez = self.extent
        return ex * ey * ez

    # -- predicates ---------------------------------------------------------- #
    def contains_point(self, p: Vec3, tol: float = 0.0) -> bool:
        return all(lo - tol <= c <= hi + tol for lo, hi, c in zip(self.min, self.max, p))

    def contains(self, other: "AABB", tol: float = 0.0) -> bool:
        """True if ``self`` fully encloses ``other`` (with tolerance ``tol``)."""
        return all(
            slo - tol <= olo and ohi <= shi + tol
            for slo, shi, olo, ohi in zip(self.min, self.max, other.min, other.max)
        )

    def overlaps(self, other: "AABB", tol: float = 0.0) -> bool:
        """True if the two boxes intersect (touching counts when ``tol >= 0``)."""
        return all(
            slo - tol <= ohi and olo - tol <= shi
            for slo, shi, olo, ohi in zip(self.min, self.max, other.min, other.max)
        )

    def gap(self, other: "AABB") -> float:
        """Minimum axis-aligned separation distance between the two boxes.

        Zero when the boxes touch or overlap; otherwise the Euclidean length of
        the per-axis positive separations.
        """
        sep_sq = 0.0
        for slo, shi, olo, ohi in zip(self.min, self.max, other.min, other.max):
            if olo > shi:
                d = olo - shi
                sep_sq += d * d
            elif slo > ohi:
                d = slo - ohi
                sep_sq += d * d
        return sep_sq ** 0.5


# --------------------------------------------------------------------------- #
# Relation vocabulary                                                          #
# --------------------------------------------------------------------------- #
class RelationType(Enum):
    """Closed set of typed scene-graph relation labels."""

    # symmetric proximity
    ADJACENT_TO = "adjacent_to"
    CONNECTED_TO = "connected_to"
    TOUCHING = "touching"
    # vertical support pair
    ON_TOP_OF = "on_top_of"
    SUPPORTS = "supports"
    # containment pair
    CONTAINS = "contains"
    CONTAINED_BY = "contained_by"
    # directional pairs
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    FRONT_OF = "front_of"
    BEHIND = "behind"


# Inverse of each directed relation. Symmetric relations map to themselves.
_INVERSE: Dict[RelationType, RelationType] = {
    RelationType.ADJACENT_TO: RelationType.ADJACENT_TO,
    RelationType.CONNECTED_TO: RelationType.CONNECTED_TO,
    RelationType.TOUCHING: RelationType.TOUCHING,
    RelationType.ON_TOP_OF: RelationType.SUPPORTS,
    RelationType.SUPPORTS: RelationType.ON_TOP_OF,
    RelationType.CONTAINS: RelationType.CONTAINED_BY,
    RelationType.CONTAINED_BY: RelationType.CONTAINS,
    RelationType.ABOVE: RelationType.BELOW,
    RelationType.BELOW: RelationType.ABOVE,
    RelationType.LEFT_OF: RelationType.RIGHT_OF,
    RelationType.RIGHT_OF: RelationType.LEFT_OF,
    RelationType.FRONT_OF: RelationType.BEHIND,
    RelationType.BEHIND: RelationType.FRONT_OF,
}


def inverse_relation(rel: RelationType) -> RelationType:
    """Return the inverse of ``rel`` (a ON_TOP_OF b => b SUPPORTS a)."""
    return _INVERSE[rel]


def is_symmetric(rel: RelationType) -> bool:
    """True when the relation is its own inverse (adjacency / touching)."""
    return _INVERSE[rel] is rel


# --------------------------------------------------------------------------- #
# Nodes and edges                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class SceneNode:
    """A typed object node in the scene graph.

    ``node_id`` is a stable unique identity. ``obj_type`` is the coarse object
    class (the ``group`` label, e.g. ``"pipe"``, ``"valve"``).
    ``attributes`` holds free-form semantic / geometric annotations
    (``name`` label, ``material``, ``affordance``, ``usd_path`` etc.).
    """

    node_id: str
    obj_type: str
    aabb: AABB
    attributes: Dict[str, object] = field(default_factory=dict)

    @property
    def centroid(self) -> Vec3:
        return self.aabb.centroid


@dataclass(frozen=True)
class RelationEdge:
    """A typed directed edge ``source --relation--> target``."""

    source: str
    relation: RelationType
    target: str

    def inverse(self) -> "RelationEdge":
        return RelationEdge(self.target, inverse_relation(self.relation), self.source)

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.source, self.relation.value, self.target)


# --------------------------------------------------------------------------- #
# Scene graph                                                                  #
# --------------------------------------------------------------------------- #
class SceneGraph:
    """Container of typed nodes and typed directed relation edges.

    Insertion order is preserved deterministically. Duplicate edges (same
    source/relation/target) are ignored. All iteration is order-stable so that
    downstream serialization is reproducible.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, SceneNode] = {}
        self._edges: List[RelationEdge] = []
        self._edge_set: set = set()
        # adjacency: node_id -> list of outgoing edge indices (order-stable)
        self._out: Dict[str, List[int]] = {}
        self._in: Dict[str, List[int]] = {}

    # -- nodes --------------------------------------------------------------- #
    def add_node(self, node: SceneNode) -> SceneNode:
        if node.node_id in self._nodes:
            raise ValueError(f"duplicate node id: {node.node_id!r}")
        self._nodes[node.node_id] = node
        self._out.setdefault(node.node_id, [])
        self._in.setdefault(node.node_id, [])
        return node

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_node(self, node_id: str) -> SceneNode:
        return self._nodes[node_id]

    @property
    def nodes(self) -> List[SceneNode]:
        return list(self._nodes.values())

    @property
    def node_ids(self) -> List[str]:
        return list(self._nodes.keys())

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    # -- edges --------------------------------------------------------------- #
    def add_edge(
        self,
        source: str,
        relation: RelationType,
        target: str,
        *,
        add_inverse: bool = False,
    ) -> Optional[RelationEdge]:
        """Add a directed edge; optionally also add its inverse.

        Returns the created edge, or ``None`` if it already existed. Raises if
        either endpoint is unknown.
        """
        if source not in self._nodes:
            raise KeyError(f"unknown source node: {source!r}")
        if target not in self._nodes:
            raise KeyError(f"unknown target node: {target!r}")
        edge = self._insert(RelationEdge(source, relation, target))
        if add_inverse:
            self._insert(edge.inverse() if edge is not None else RelationEdge(source, relation, target).inverse())
        return edge

    def add_relation_edge(self, edge: RelationEdge, *, add_inverse: bool = False) -> Optional[RelationEdge]:
        return self.add_edge(edge.source, edge.relation, edge.target, add_inverse=add_inverse)

    def _insert(self, edge: RelationEdge) -> Optional[RelationEdge]:
        key = edge.as_tuple()
        if key in self._edge_set:
            return None
        idx = len(self._edges)
        self._edges.append(edge)
        self._edge_set.add(key)
        self._out[edge.source].append(idx)
        self._in[edge.target].append(idx)
        return edge

    @property
    def edges(self) -> List[RelationEdge]:
        return list(self._edges)

    def has_edge(self, source: str, relation: RelationType, target: str) -> bool:
        return (source, relation.value, target) in self._edge_set

    # -- adjacency queries --------------------------------------------------- #
    def out_edges(self, node_id: str, relation: Optional[RelationType] = None) -> List[RelationEdge]:
        edges = [self._edges[i] for i in self._out.get(node_id, [])]
        if relation is not None:
            edges = [e for e in edges if e.relation is relation]
        return edges

    def in_edges(self, node_id: str, relation: Optional[RelationType] = None) -> List[RelationEdge]:
        edges = [self._edges[i] for i in self._in.get(node_id, [])]
        if relation is not None:
            edges = [e for e in edges if e.relation is relation]
        return edges

    def neighbors(self, node_id: str, relation: Optional[RelationType] = None) -> List[str]:
        """Target node ids reachable via outgoing edges (optionally filtered)."""
        seen: Dict[str, None] = {}
        for e in self.out_edges(node_id, relation):
            seen.setdefault(e.target, None)
        return list(seen.keys())

    def undirected_neighbors(self, node_id: str) -> List[str]:
        """All nodes linked by an edge in either direction (order-stable)."""
        seen: Dict[str, None] = {}
        for e in self.out_edges(node_id):
            seen.setdefault(e.target, None)
        for e in self.in_edges(node_id):
            seen.setdefault(e.source, None)
        return list(seen.keys())

    def degree(self, node_id: str) -> int:
        return len(self._out.get(node_id, [])) + len(self._in.get(node_id, []))
