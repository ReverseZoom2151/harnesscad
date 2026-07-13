"""Query and reasoning engine over a CAD scene graph.

Paper: *Semantic Enrichment of CAD-Based Industrial Environments via Scene
Graphs for Simulation and Reasoning* (Walus et al.), Sec. I / V-D / VI.

Once the scene graph is built and semantically enriched, the paper's motivation
is *reasoning*: giving an LLM agent a structured representation it can query to
answer high-level questions ("which components influence what other components
and in what order"). This module provides the deterministic query primitives an
agent would call:

* :func:`objects_of_type` -- all nodes of a given ``group`` (object class);
* :func:`objects_with_attribute` / :func:`objects_by_affordance` -- attribute
  filters (material, affordance, arbitrary key/value);
* :func:`related` -- the neighbours of a node under a specific relation type
  (relation queries such as "what does X support?");
* :func:`relation_between` -- the relation label(s) directly linking two nodes;
* :func:`shortest_path` -- BFS shortest undirected path between two nodes with
  order-stable tie-breaking, plus :func:`path_exists`;
* :func:`connected_component` / :func:`connected_components` -- undirected
  reachability, used to isolate freestanding structures (the paper's clustered
  ``structures``);
* :func:`count_by_type` -- a type histogram (the paper's label-distribution
  analysis).

Traversals are deterministic (nodes visited in insertion order) and stdlib-only.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from harnesscad.domain.reconstruction.scenegraph_model import RelationType, SceneGraph, SceneNode


# --------------------------------------------------------------------------- #
# Type / attribute queries                                                     #
# --------------------------------------------------------------------------- #
def objects_of_type(graph: SceneGraph, obj_type: str) -> List[SceneNode]:
    """All nodes whose ``obj_type`` (group label) equals ``obj_type``."""
    return [n for n in graph.nodes if n.obj_type == obj_type]


def objects_with_attribute(
    graph: SceneGraph, key: str, value: Optional[object] = None
) -> List[SceneNode]:
    """Nodes carrying ``key``; if ``value`` is given also match on equality."""
    out = []
    for n in graph.nodes:
        if key in n.attributes and (value is None or n.attributes[key] == value):
            out.append(n)
    return out


def objects_by_affordance(graph: SceneGraph, affordance: str) -> List[SceneNode]:
    """Nodes whose ``affordance`` attribute equals ``affordance``."""
    return objects_with_attribute(graph, "affordance", affordance)


def count_by_type(graph: SceneGraph) -> Dict[str, int]:
    """Order-stable histogram of ``obj_type`` over all nodes."""
    hist: Dict[str, int] = {}
    for n in graph.nodes:
        hist[n.obj_type] = hist.get(n.obj_type, 0) + 1
    return hist


# --------------------------------------------------------------------------- #
# Relation queries                                                             #
# --------------------------------------------------------------------------- #
def related(graph: SceneGraph, node_id: str, relation: RelationType) -> List[str]:
    """Targets reachable from ``node_id`` via ``relation`` (order-stable)."""
    return graph.neighbors(node_id, relation)


def relation_between(graph: SceneGraph, source: str, target: str) -> List[RelationType]:
    """All direct relation labels on edges ``source -> target`` (order-stable)."""
    return [e.relation for e in graph.out_edges(source) if e.target == target]


# --------------------------------------------------------------------------- #
# Path / connectivity                                                          #
# --------------------------------------------------------------------------- #
def shortest_path(
    graph: SceneGraph,
    source: str,
    target: str,
    relation: Optional[RelationType] = None,
) -> Optional[List[str]]:
    """BFS shortest undirected path ``source .. target`` or ``None``.

    Edges are traversed in both directions. If ``relation`` is given only edges
    of that type are used. Neighbours are expanded in insertion order so the
    returned path is deterministic.
    """
    if not graph.has_node(source) or not graph.has_node(target):
        return None
    if source == target:
        return [source]
    prev: Dict[str, str] = {}
    visited = {source}
    q: deque = deque([source])
    while q:
        cur = q.popleft()
        for nb in _undirected_neighbors(graph, cur, relation):
            if nb in visited:
                continue
            visited.add(nb)
            prev[nb] = cur
            if nb == target:
                return _reconstruct(prev, source, target)
            q.append(nb)
    return None


def _undirected_neighbors(
    graph: SceneGraph, node_id: str, relation: Optional[RelationType]
) -> List[str]:
    seen: Dict[str, None] = {}
    for e in graph.out_edges(node_id, relation):
        seen.setdefault(e.target, None)
    for e in graph.in_edges(node_id, relation):
        seen.setdefault(e.source, None)
    return list(seen.keys())


def _reconstruct(prev: Dict[str, str], source: str, target: str) -> List[str]:
    path = [target]
    cur = target
    while cur != source:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path


def path_exists(
    graph: SceneGraph,
    source: str,
    target: str,
    relation: Optional[RelationType] = None,
) -> bool:
    return shortest_path(graph, source, target, relation) is not None


def connected_component(
    graph: SceneGraph, node_id: str, relation: Optional[RelationType] = None
) -> List[str]:
    """Undirected reachable set from ``node_id`` (inclusive), insertion-ordered."""
    if not graph.has_node(node_id):
        return []
    visited = {node_id}
    order = [node_id]
    q: deque = deque([node_id])
    while q:
        cur = q.popleft()
        for nb in _undirected_neighbors(graph, cur, relation):
            if nb not in visited:
                visited.add(nb)
                order.append(nb)
                q.append(nb)
    # return in graph insertion order for stability
    return [nid for nid in graph.node_ids if nid in visited]


def connected_components(
    graph: SceneGraph, relation: Optional[RelationType] = None
) -> List[List[str]]:
    """Partition nodes into undirected connected components (order-stable)."""
    seen = set()
    comps: List[List[str]] = []
    for nid in graph.node_ids:
        if nid in seen:
            continue
        comp = connected_component(graph, nid, relation)
        seen.update(comp)
        comps.append(comp)
    return comps
