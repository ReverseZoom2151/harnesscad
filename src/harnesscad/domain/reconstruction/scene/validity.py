"""Consistency and validity checking for a CAD scene graph.

A scene graph is only useful for reasoning if it is *well formed*: a **correct
scene graph** is essential for successful functional identification, and typical
structural errors are concrete -- a continuous pipe split into two clusters, and
two unrelated structures joined by a single spurious edge. This module supplies
the deterministic validity
checks that guard against such malformed graphs before reasoning runs.

Checks (each yields typed :class:`Issue` records; nothing is mutated):

* **dangling edges** -- an edge referencing a missing endpoint;
* **self loops** -- an edge from a node to itself;
* **inverse consistency** -- every asymmetric relation ``a REL b`` must have its
  inverse ``b INV(REL) a`` present, and every symmetric relation must be
  mirrored (a ADJACENT_TO b implies b ADJACENT_TO a);
* **containment acyclicity** -- the ``CONTAINS`` relation must form a DAG (no
  node transitively contains itself);
* **support acyclicity** -- likewise for ``SUPPORTS``/``ON_TOP_OF`` stacking;
* **relation-type validity** -- edge labels must be members of the closed
  :class:`RelationType` vocabulary (guards hand-built graphs);
* **isolated nodes** -- reported as informational (a lone node may be a
  freestanding structure, or a fragment of a wrongly split cluster).

:func:`check_scene_graph` runs every check and returns a :class:`ValidationReport`
whose ``ok`` flag is true iff no error-severity issue was found. Stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from harnesscad.domain.reconstruction.scene.model import (
    RelationType,
    SceneGraph,
    inverse_relation,
    is_symmetric,
)

ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass(frozen=True)
class Issue:
    """A single validity finding."""

    code: str
    severity: str
    message: str
    nodes: Tuple[str, ...] = ()


@dataclass
class ValidationReport:
    """Aggregate result of :func:`check_scene_graph`."""

    issues: List[Issue] = field(default_factory=list)

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self) -> List[str]:
        return [i.code for i in self.issues]


# --------------------------------------------------------------------------- #
# Individual checks                                                            #
# --------------------------------------------------------------------------- #
def check_dangling_and_self_loops(graph: SceneGraph) -> List[Issue]:
    issues: List[Issue] = []
    for e in graph.edges:
        if not graph.has_node(e.source) or not graph.has_node(e.target):
            issues.append(Issue("dangling_edge", ERROR,
                                 f"edge references missing endpoint: {e.as_tuple()}",
                                 (e.source, e.target)))
        elif e.source == e.target:
            issues.append(Issue("self_loop", ERROR,
                                 f"self-loop on {e.source!r} via {e.relation.value}",
                                 (e.source,)))
    return issues


def check_relation_types(graph: SceneGraph) -> List[Issue]:
    issues: List[Issue] = []
    for e in graph.edges:
        if not isinstance(e.relation, RelationType):
            issues.append(Issue("bad_relation_type", ERROR,
                                 f"unknown relation on {e.as_tuple()}",
                                 (e.source, e.target)))
    return issues


def check_inverse_consistency(graph: SceneGraph) -> List[Issue]:
    """Every asymmetric/symmetric edge must have its required counterpart."""
    issues: List[Issue] = []
    for e in graph.edges:
        if not isinstance(e.relation, RelationType):
            continue
        inv = inverse_relation(e.relation)
        if not graph.has_edge(e.target, inv, e.source):
            kind = "symmetric" if is_symmetric(e.relation) else "inverse"
            issues.append(Issue("missing_inverse", WARNING,
                                 f"{kind} counterpart missing for {e.as_tuple()} "
                                 f"(expected {e.target} {inv.value} {e.source})",
                                 (e.source, e.target)))
    return issues


def _has_cycle(adj: Dict[str, List[str]]) -> Optional[List[str]]:
    """Return a cycle path if the directed adjacency has one, else ``None``."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {n: WHITE for n in adj}
    parent: Dict[str, Optional[str]] = {n: None for n in adj}

    def visit(start: str) -> Optional[List[str]]:
        stack = [(start, iter(adj[start]))]
        color[start] = GRAY
        while stack:
            node, it = stack[-1]
            advanced = False
            for nb in it:
                if color.get(nb, WHITE) == GRAY:
                    # reconstruct cycle
                    cyc = [nb, node]
                    p = parent[node]
                    while p is not None and p != nb:
                        cyc.append(p)
                        p = parent[p]
                    cyc.reverse()
                    return cyc
                if color.get(nb, WHITE) == WHITE:
                    parent[nb] = node
                    color[nb] = GRAY
                    stack.append((nb, iter(adj[nb])))
                    advanced = True
                    break
            if not advanced:
                color[node] = BLACK
                stack.pop()
        return None

    for n in adj:
        if color[n] == WHITE:
            cyc = visit(n)
            if cyc:
                return cyc
    return None


def _relation_adjacency(graph: SceneGraph, relation: RelationType) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {nid: [] for nid in graph.node_ids}
    for e in graph.edges:
        if e.relation is relation and graph.has_node(e.source) and graph.has_node(e.target):
            adj[e.source].append(e.target)
    return adj


def check_acyclic(graph: SceneGraph, relation: RelationType, code: str) -> List[Issue]:
    adj = _relation_adjacency(graph, relation)
    cyc = _has_cycle(adj)
    if cyc:
        return [Issue(code, ERROR,
                      f"cycle in {relation.value} relation: {' -> '.join(cyc)}",
                      tuple(cyc))]
    return []


def check_isolated_nodes(graph: SceneGraph) -> List[Issue]:
    issues: List[Issue] = []
    for nid in graph.node_ids:
        if graph.degree(nid) == 0:
            issues.append(Issue("isolated_node", INFO,
                                 f"node {nid!r} has no relations", (nid,)))
    return issues


# --------------------------------------------------------------------------- #
# Aggregate                                                                    #
# --------------------------------------------------------------------------- #
def check_scene_graph(
    graph: SceneGraph,
    *,
    require_inverses: bool = True,
    report_isolated: bool = True,
) -> ValidationReport:
    """Run all validity checks and aggregate into a :class:`ValidationReport`."""
    issues: List[Issue] = []
    issues += check_dangling_and_self_loops(graph)
    issues += check_relation_types(graph)
    if require_inverses:
        issues += check_inverse_consistency(graph)
    issues += check_acyclic(graph, RelationType.CONTAINS, "containment_cycle")
    issues += check_acyclic(graph, RelationType.SUPPORTS, "support_cycle")
    if report_isolated:
        issues += check_isolated_nodes(graph)
    return ValidationReport(issues=issues)
