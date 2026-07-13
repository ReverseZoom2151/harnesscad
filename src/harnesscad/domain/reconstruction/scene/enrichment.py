"""Deterministic semantic enrichment of a CAD scene graph.

Paper: *Semantic Enrichment of CAD-Based Industrial Environments via Scene
Graphs for Simulation and Reasoning* (Walus et al.), Sec. III-A (Vocabulary),
III-B step 4 (Semantic Labelling).

The paper enriches every mesh with a coarse ``group`` label and a specific
``name`` label drawn from a **three-layer vocabulary tree** (root -> group ->
name), attaching them (plus the usd path) as node attributes. Labelling itself
is done by an LVLM (out of scope), but the paper stresses that a *predefined
vocabulary* is what keeps labels consistent, and that the **bounding-box
dimensions are provided precisely so thin gaskets can be told apart from thick
flanges** (Sec. V-B). That geometry-driven disambiguation is fully deterministic.

This module provides the offline, network-free enrichment scaffolding:

* :class:`Vocabulary` -- the three-layer group/name tree with validation, group
  membership lookup and (de)serialization to nested dicts;
* :func:`enrich_node` / :func:`enrich_graph` -- attach ``group``, ``name``,
  ``material``, ``affordance`` and ``usd_path`` attributes to nodes, validating
  labels against the vocabulary;
* :func:`classify_by_dimensions` -- the paper's bbox-based disambiguation rule
  (aspect-ratio thresholds that separate thin plate-like parts from thick
  block-like parts, e.g. gasket vs flange);
* :data:`DEFAULT_AFFORDANCES` and :func:`affordance_for` -- a deterministic
  ``group`` -> affordance mapping (valve -> ``turn``, gauge -> ``read`` ...)
  giving the actionable-element semantics the paper needs for simulation.

Everything is stdlib-only and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from harnesscad.domain.reconstruction.scene.model import AABB, SceneGraph, SceneNode


# --------------------------------------------------------------------------- #
# Three-layer vocabulary tree                                                  #
# --------------------------------------------------------------------------- #
class Vocabulary:
    """Three-layer label tree: root -> ``group`` -> ``name``.

    The root carries no information (per the paper). Each group maps to the set
    of allowed specific names. New groups / names can be proposed at runtime
    (the paper permits the LVLM to extend the vocabulary).
    """

    def __init__(self, groups: Optional[Dict[str, List[str]]] = None) -> None:
        self._groups: Dict[str, List[str]] = {}
        self._name_index: Dict[str, str] = {}  # name -> group
        for group, names in (groups or {}).items():
            self.add_group(group)
            for name in names:
                self.add_name(group, name)

    def add_group(self, group: str) -> None:
        self._groups.setdefault(group, [])

    def add_name(self, group: str, name: str) -> None:
        if group not in self._groups:
            self.add_group(group)
        if name in self._name_index and self._name_index[name] != group:
            raise ValueError(f"name {name!r} already registered under group "
                             f"{self._name_index[name]!r}")
        if name not in self._groups[group]:
            self._groups[group].append(name)
        self._name_index[name] = group

    def has_group(self, group: str) -> bool:
        return group in self._groups

    def has_name(self, name: str) -> bool:
        return name in self._name_index

    def group_of(self, name: str) -> Optional[str]:
        return self._name_index.get(name)

    def names(self, group: str) -> List[str]:
        return list(self._groups.get(group, []))

    @property
    def groups(self) -> List[str]:
        return list(self._groups.keys())

    def validate(self, group: str, name: Optional[str]) -> bool:
        """True if ``group`` exists and ``name`` (if given) belongs to it."""
        if group not in self._groups:
            return False
        if name is None:
            return True
        return name in self._groups[group]

    def to_dict(self) -> Dict[str, List[str]]:
        return {g: list(ns) for g, ns in self._groups.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, List[str]]) -> "Vocabulary":
        return cls(data)


# --------------------------------------------------------------------------- #
# Affordances                                                                  #
# --------------------------------------------------------------------------- #
# Deterministic group -> affordance mapping for actionable industrial elements.
DEFAULT_AFFORDANCES: Dict[str, str] = {
    "valve": "turn",
    "wheel_valve": "turn",
    "gauge": "read",
    "display": "read",
    "pump": "actuate",
    "pump_unit": "actuate",
    "pipe": "convey",
    "pipe_assembly": "convey",
    "tank": "store",
    "flange": "fasten",
    "gasket": "seal",
    "bolt": "fasten",
    "connection_assembly": "connect",
}


def affordance_for(group: str, affordances: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Return the affordance for a ``group`` label, or ``None`` if unknown."""
    table = affordances if affordances is not None else DEFAULT_AFFORDANCES
    return table.get(group)


# --------------------------------------------------------------------------- #
# Geometry-driven disambiguation                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DimensionRule:
    """Threshold rule mapping a bbox shape signature to a label.

    ``flatness`` = smallest-extent / largest-extent. A small value means a thin,
    plate-like part; a large value means a chunky, block-like part.
    """

    label: str
    max_flatness: float = 1.0
    min_flatness: float = 0.0


# Default rule set separating thin plate-like parts (gasket) from thick
# block-like parts (flange), exactly the distinction the paper highlights.
DEFAULT_DIMENSION_RULES: Tuple[DimensionRule, ...] = (
    DimensionRule("gasket", max_flatness=0.15, min_flatness=0.0),
    DimensionRule("flange", max_flatness=1.0, min_flatness=0.15),
)


def flatness(aabb: AABB) -> float:
    """Smallest / largest bounding-box extent (0 = infinitely thin, 1 = cube)."""
    ex = sorted(aabb.extent)
    largest = ex[2]
    if largest <= 0.0:
        return 0.0
    return ex[0] / largest


def classify_by_dimensions(
    aabb: AABB,
    rules: Tuple[DimensionRule, ...] = DEFAULT_DIMENSION_RULES,
) -> Optional[str]:
    """Deterministically pick a label from bbox shape via the rule set.

    Rules are checked in order; the first whose flatness band contains the box
    wins. Returns ``None`` if no rule matches.
    """
    f = flatness(aabb)
    for rule in rules:
        if rule.min_flatness <= f <= rule.max_flatness:
            return rule.label
    return None


# --------------------------------------------------------------------------- #
# Node / graph enrichment                                                      #
# --------------------------------------------------------------------------- #
def enrich_node(
    node: SceneNode,
    group: str,
    name: Optional[str] = None,
    *,
    material: Optional[str] = None,
    usd_path: Optional[str] = None,
    vocabulary: Optional[Vocabulary] = None,
    affordances: Optional[Dict[str, str]] = None,
) -> SceneNode:
    """Attach semantic attributes to a node in place and return it.

    Sets ``node.obj_type = group`` (the coarse ``group`` label) and records
    ``group``, ``name``, ``material``, ``affordance`` and ``usd_path`` in
    ``node.attributes``. If a ``vocabulary`` is supplied the (group, name) pair
    is validated against it (raising on mismatch).
    """
    if vocabulary is not None and not vocabulary.validate(group, name):
        raise ValueError(f"label ({group!r}, {name!r}) not in vocabulary")
    node.obj_type = group
    node.attributes["group"] = group
    if name is not None:
        node.attributes["name"] = name
    if material is not None:
        node.attributes["material"] = material
    if usd_path is not None:
        node.attributes["usd_path"] = usd_path
    aff = affordance_for(group, affordances)
    if aff is not None:
        node.attributes["affordance"] = aff
    return node


def enrich_graph(
    graph: SceneGraph,
    labels: Dict[str, Dict[str, object]],
    *,
    vocabulary: Optional[Vocabulary] = None,
    affordances: Optional[Dict[str, str]] = None,
) -> int:
    """Enrich many nodes from a ``node_id -> {group,name,material,usd_path}`` map.

    Returns the number of nodes enriched. Unknown node ids raise ``KeyError``.
    """
    count = 0
    for node_id, spec in labels.items():
        node = graph.get_node(node_id)
        group = str(spec["group"])
        enrich_node(
            node,
            group,
            name=spec.get("name"),  # type: ignore[arg-type]
            material=spec.get("material"),  # type: ignore[arg-type]
            usd_path=spec.get("usd_path"),  # type: ignore[arg-type]
            vocabulary=vocabulary,
            affordances=affordances,
        )
        count += 1
    return count


def coverage(graph: SceneGraph, attribute: str = "group") -> float:
    """Fraction of nodes carrying ``attribute`` (semantic-labelling coverage)."""
    if len(graph) == 0:
        return 0.0
    have = sum(1 for n in graph.nodes if attribute in n.attributes)
    return have / len(graph)
