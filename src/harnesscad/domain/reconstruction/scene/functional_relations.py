"""Functional-relation extraction over a CAD scene graph (Algorithm 1).

Given the enriched scene graph and its ``group`` semantics, Algorithm 1 infers
the direct functional relationships between the functional units inside a pipe
system. It needs:

* a set of **connector** group labels ``S_pipe`` (here ``{"pipe_assembly"}``)
  which may be traversed while non-relevant meshes (structural supports) are
  ignored;
* a list of **functional units** ``F = {F_1 .. F_k}``, each a set of nodes --
  obtained by finding interconnected node clusters that share the same ``group``
  label (e.g. all nodes of a wheel valve).

The algorithm grows every functional unit outward through connector nodes,
claiming each connector for at most one unit (a global marked set ``M`` prevents
double-claiming), iterating until fixpoint. Two functional units are declared
functionally related when a scene-graph edge links a node claimed by one unit to
a node claimed by another. It returns the compact **functional graph**
``G_func = (V_func, E_func)`` over unit indices.

This module implements exactly that, deterministically and stdlib-only:

* :func:`find_functional_units` -- group same-``group`` connected nodes into
  units ("interconnected node clusters of the same group label");
* :func:`extract_functional_relations` -- the full Algorithm 1
  (marked-set growth to fixpoint + edge-induced unit adjacency);
* :class:`FunctionalGraph` -- ``V_func`` / ``E_func`` with per-unit membership
  and a neighbour query.

The learned LVLM labelling that produces the ``group`` semantics is upstream and
out of scope; here the semantics are taken as given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from harnesscad.domain.reconstruction.scene.model import RelationType, SceneGraph


# --------------------------------------------------------------------------- #
# Functional graph result                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class FunctionalGraph:
    """Compact functional graph ``G_func = (V_func, E_func)`` over unit indices.

    ``units[i]`` is the (grown) set of node ids claimed by functional unit ``i``.
    ``edges`` is the set of undirected ``(i, j)`` unit-pairs (``i < j``) that are
    functionally related. ``labels[i]`` is the shared ``group`` label of unit i.
    """

    units: List[Set[str]] = field(default_factory=list)
    edges: Set[Tuple[int, int]] = field(default_factory=set)
    labels: List[str] = field(default_factory=list)

    @property
    def vertices(self) -> List[int]:
        return list(range(len(self.units)))

    def unit_of(self, node_id: str) -> Optional[int]:
        for i, members in enumerate(self.units):
            if node_id in members:
                return i
        return None

    def neighbors(self, unit_index: int) -> List[int]:
        out = []
        for (i, j) in sorted(self.edges):
            if i == unit_index:
                out.append(j)
            elif j == unit_index:
                out.append(i)
        return out


# --------------------------------------------------------------------------- #
# Functional-unit discovery                                                    #
# --------------------------------------------------------------------------- #
def find_functional_units(
    graph: SceneGraph,
    unit_groups: Iterable[str],
    *,
    relation: Optional[RelationType] = None,
) -> List[Set[str]]:
    """Group same-``group`` interconnected nodes into functional units.

    A functional unit is a maximal set of nodes that (a) all carry the same
    ``group`` label in ``unit_groups`` and (b) are connected to each other
    through edges that only pass through nodes of that same group. Units are
    returned in graph insertion order (of their first-seen member).
    """
    wanted = set(unit_groups)
    units: List[Set[str]] = []
    seen: Set[str] = set()
    for nid in graph.node_ids:
        if nid in seen:
            continue
        grp = graph.get_node(nid).obj_type
        if grp not in wanted:
            continue
        # BFS restricted to same-group nodes
        comp: Set[str] = set()
        stack = [nid]
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            for nb in _neighbors(graph, cur, relation):
                if nb not in comp and graph.get_node(nb).obj_type == grp:
                    stack.append(nb)
        seen.update(comp)
        units.append(comp)
    return units


def _neighbors(graph: SceneGraph, node_id: str, relation: Optional[RelationType]) -> List[str]:
    seen: Dict[str, None] = {}
    for e in graph.out_edges(node_id, relation):
        seen.setdefault(e.target, None)
    for e in graph.in_edges(node_id, relation):
        seen.setdefault(e.source, None)
    return list(seen.keys())


# --------------------------------------------------------------------------- #
# Algorithm 1                                                                  #
# --------------------------------------------------------------------------- #
def extract_functional_relations(
    graph: SceneGraph,
    functional_units: Sequence[Set[str]],
    pipe_groups: Iterable[str],
    *,
    labels: Optional[Sequence[str]] = None,
    relation: Optional[RelationType] = None,
) -> FunctionalGraph:
    """Run Algorithm 1: grow units through connectors and induce unit adjacency.

    Parameters: ``functional_units`` = ``F``; ``pipe_groups`` =
    the connector label set ``S_pipe``. Growth claims each connector node for at
    most one unit (global marked set ``M``), iterating to fixpoint. Then every
    scene-graph edge whose endpoints fall in two *different* units contributes an
    undirected functional edge.
    """
    pipe = set(pipe_groups)
    # copy units so we don't mutate the caller's sets
    units: List[Set[str]] = [set(u) for u in functional_units]

    # M = union of all functional-unit nodes (Algorithm 1, line 1)
    marked: Set[str] = set()
    for u in units:
        marked |= u

    # Grow to fixpoint (lines 2-15)
    changed = True
    while changed:
        changed = False
        for i in range(len(units)):
            new_nodes: List[str] = []
            for v in list(units[i]):
                for u in _neighbors(graph, v, relation):
                    if u in marked:
                        continue
                    if graph.get_node(u).obj_type in pipe:
                        new_nodes.append(u)
                        marked.add(u)  # claim u for unit i
            if new_nodes:
                units[i].update(new_nodes)
                changed = True

    # Induce functional edges from scene-graph edges (lines 16-24)
    func_edges: Set[Tuple[int, int]] = set()
    member_of: Dict[str, int] = {}
    for i, u in enumerate(units):
        for nid in u:
            member_of[nid] = i
    for e in graph.edges:
        i = member_of.get(e.source)
        j = member_of.get(e.target)
        if i is None or j is None or i == j:
            continue
        func_edges.add((min(i, j), max(i, j)))

    if labels is None:
        derived: List[str] = []
        for u in functional_units:
            if u:
                first = sorted(u)[0]
                derived.append(graph.get_node(first).obj_type)
            else:
                derived.append("")
        labels = derived
    return FunctionalGraph(units=units, edges=func_edges, labels=list(labels))
