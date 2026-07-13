"""Parametric feature-tree DAG: rebuild, stale propagation, suppression.

OpenCAD's ``opencad_tree`` service is the piece that turns a bag of operations
into a *parametric* model: nodes form a DAG (``parent_id`` -- the shape being
modified -- plus ``tool_refs`` -- shapes consumed as tools, e.g. the cutter of a
boolean), and every mutation propagates through it:

* editing a node's parameters marks it and **all its descendants** ``stale`` and
  drops their cached shape IDs;
* suppressing a node marks it ``suppressed``, and a descendant becomes
  transitively suppressed **only when every one of its parents is suppressed** --
  otherwise it is merely ``stale`` (blocked by a suppressed input).  That rule
  (walked in topological order so parents settle before children) is the subtle
  bit most reimplementations get wrong;
* deleting a node with dependents is refused unless ``cascade``;
* rebuilding walks the DAG in topological order, skips suppressed nodes, and
  either aborts at the first failure or continues and marks the blocked
  descendants (``continue_on_error``).

The harness already event-sources the op stream (:mod:`state.opdag` -- content-hashed
history, branch/merge/bisect) and models SolidWorks-style feature trees for
*reconstruction* (:mod:`reconstruction.sldprtnet_feature_tree`), but it had no
rebuild/invalidation engine: the status lattice
(``pending -> built | failed | stale | suppressed``) and its propagation rules.
This module supplies exactly that, kernel-agnostically -- the caller passes a
``builder`` callable that turns a node into a shape ID, so it works with any
backend or with a stub in tests.

Deterministic: Kahn topological order with alphabetical tie-breaks; no clock, no
randomness.

Public API
----------
``FeatureNode``, ``FeatureTree``, ``RebuildReport``
``topological_order``, ``descendants``, ``direct_dependents``
``add_feature``, ``edit_feature``, ``suppress_feature``, ``delete_feature``, ``rebuild``
``CircularDependencyError``, ``MissingDependencyError``
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

__all__ = [
    "CircularDependencyError",
    "MissingDependencyError",
    "FeatureNode",
    "FeatureTree",
    "RebuildReport",
    "topological_order",
    "descendants",
    "direct_dependents",
    "add_feature",
    "edit_feature",
    "suppress_feature",
    "delete_feature",
    "rebuild",
]

PENDING = "pending"
BUILT = "built"
FAILED = "failed"
STALE = "stale"
SUPPRESSED = "suppressed"
BLOCKED = "blocked"

STATUSES = (PENDING, BUILT, FAILED, STALE, SUPPRESSED, BLOCKED)


class CircularDependencyError(ValueError):
    """The feature graph contains a cycle."""


class MissingDependencyError(ValueError):
    """A node references a parent that does not exist."""


@dataclass
class FeatureNode:
    id: str
    operation: str
    name: str = ""
    parameters: Dict[str, object] = field(default_factory=dict)
    parent_id: Optional[str] = None
    tool_refs: List[str] = field(default_factory=list)
    shape_id: Optional[str] = None
    status: str = PENDING
    suppressed: bool = False

    @property
    def depends_on(self) -> List[str]:
        """Inputs of this feature: its parent shape plus any tool shapes."""
        if self.parent_id is None:
            return list(self.tool_refs)
        return [self.parent_id, *self.tool_refs]


@dataclass
class FeatureTree:
    nodes: Dict[str, FeatureNode] = field(default_factory=dict)
    revision: int = 0

    def copy(self) -> "FeatureTree":
        return deepcopy(self)

    def statuses(self) -> Dict[str, str]:
        return {nid: node.status for nid, node in sorted(self.nodes.items())}


# ── graph ───────────────────────────────────────────────────────────


def _validate(nodes: Dict[str, FeatureNode]) -> None:
    for node_id, node in nodes.items():
        for parent in node.depends_on:
            if parent not in nodes:
                raise MissingDependencyError(
                    "Feature node '%s' depends on missing parent '%s'."
                    % (node_id, parent)
                )
            if parent == node_id:
                raise CircularDependencyError(
                    "Feature node '%s' cannot depend on itself." % node_id
                )


def topological_order(nodes: Dict[str, FeatureNode]) -> List[str]:
    """Kahn topological order; ties broken alphabetically for determinism."""
    _validate(nodes)
    indegree = {nid: 0 for nid in nodes}
    children: Dict[str, Set[str]] = {nid: set() for nid in nodes}
    for node_id, node in nodes.items():
        for parent in node.depends_on:
            children[parent].add(node_id)
            indegree[node_id] += 1

    ready = sorted(nid for nid, deg in indegree.items() if deg == 0)
    ordered: List[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for child in sorted(children[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort()

    if len(ordered) != len(nodes):
        raise CircularDependencyError("Circular dependency detected.")
    return ordered


def descendants(nodes: Dict[str, FeatureNode], source: str) -> Set[str]:
    """Every node transitively downstream of *source*."""
    children: Dict[str, Set[str]] = {nid: set() for nid in nodes}
    for node_id, node in nodes.items():
        for parent in node.depends_on:
            if parent in children:
                children[parent].add(node_id)
    out: Set[str] = set()
    stack = [source]
    while stack:
        current = stack.pop()
        for child in sorted(children.get(current, set())):
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


def direct_dependents(nodes: Dict[str, FeatureNode], source: str) -> List[str]:
    return sorted(
        nid for nid, node in nodes.items() if source in node.depends_on
    )


# ── mutations ───────────────────────────────────────────────────────


def _commit(tree: FeatureTree) -> FeatureTree:
    tree.revision += 1
    return tree


def _invalidate_descendants(tree: FeatureTree, node_id: str) -> None:
    for child_id in descendants(tree.nodes, node_id):
        child = tree.nodes[child_id]
        child.status = STALE
        child.shape_id = None


def add_feature(tree: FeatureTree, node: FeatureNode) -> FeatureTree:
    """Insert *node*, validating that its inputs exist and the graph stays acyclic."""
    updated = tree.copy()
    if node.id in updated.nodes:
        raise ValueError("Feature node '%s' already exists." % node.id)
    for parent in node.depends_on:
        if parent not in updated.nodes:
            raise MissingDependencyError("Dependency '%s' does not exist." % parent)
    updated.nodes[node.id] = deepcopy(node)
    topological_order(updated.nodes)  # raises on cycle
    return _commit(updated)


def edit_feature(
    tree: FeatureTree, node_id: str, parameters: Dict[str, object]
) -> FeatureTree:
    """Merge *parameters* into a node and stale it plus everything downstream."""
    updated = tree.copy()
    node = updated.nodes.get(node_id)
    if node is None:
        raise ValueError("Feature node '%s' does not exist." % node_id)
    node.parameters = {**node.parameters, **parameters}
    node.status = STALE
    node.shape_id = None
    _invalidate_descendants(updated, node_id)
    return _commit(updated)


def suppress_feature(
    tree: FeatureTree, node_id: str, suppressed: bool = True
) -> FeatureTree:
    """Suppress/unsuppress a node, propagating the OpenCAD transitive rule.

    A descendant becomes suppressed only when *all* of its parents are suppressed;
    otherwise it is stale (its input chain is blocked but it still has live inputs).
    """
    updated = tree.copy()
    node = updated.nodes.get(node_id)
    if node is None:
        raise ValueError("Feature node '%s' does not exist." % node_id)

    node.suppressed = suppressed
    node.status = SUPPRESSED if suppressed else STALE
    node.shape_id = None

    affected = descendants(updated.nodes, node_id)
    for child_id in topological_order(updated.nodes):
        if child_id not in affected:
            continue
        child = updated.nodes[child_id]
        child.shape_id = None
        if suppressed:
            all_parents_suppressed = all(
                updated.nodes[p].suppressed for p in child.depends_on
            )
            child.suppressed = all_parents_suppressed
            child.status = SUPPRESSED if all_parents_suppressed else STALE
        else:
            child.suppressed = False
            child.status = STALE
    return _commit(updated)


def delete_feature(
    tree: FeatureTree, node_id: str, cascade: bool = False
) -> FeatureTree:
    """Delete a node; refuse when dependents exist unless *cascade*."""
    updated = tree.copy()
    if node_id not in updated.nodes:
        raise ValueError("Feature node '%s' does not exist." % node_id)
    deps = direct_dependents(updated.nodes, node_id)
    if deps and not cascade:
        raise ValueError(
            "Cannot delete node '%s' because dependents exist: %s"
            % (node_id, ", ".join(deps))
        )
    doomed = {node_id}
    if cascade:
        doomed |= descendants(updated.nodes, node_id)
    for dead in doomed:
        updated.nodes.pop(dead, None)
    return _commit(updated)


# ── rebuild ─────────────────────────────────────────────────────────


@dataclass
class RebuildReport:
    tree: FeatureTree
    order: List[str] = field(default_factory=list)
    built: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)   # suppressed
    failed: List[str] = field(default_factory=list)
    blocked: List[str] = field(default_factory=list)   # downstream of a failure
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed and not self.blocked


Builder = Callable[[FeatureNode, FeatureTree], str]


def rebuild(
    tree: FeatureTree,
    builder: Builder,
    *,
    continue_on_error: bool = False,
    force: bool = False,
) -> RebuildReport:
    """Rebuild the tree in topological order.

    ``builder(node, tree)`` returns the node's new shape ID and may raise to signal
    a build failure.  Nodes already ``built`` are reused unless *force*.  Suppressed
    nodes are skipped; nodes whose inputs failed or were skipped are ``blocked``.
    """
    updated = tree.copy()
    order = topological_order(updated.nodes)
    report = RebuildReport(tree=updated, order=list(order))
    unavailable: Set[str] = set()

    for node_id in order:
        node = updated.nodes[node_id]

        if node.suppressed:
            node.status = SUPPRESSED
            node.shape_id = None
            report.skipped.append(node_id)
            unavailable.add(node_id)
            continue

        blocked_by = [p for p in node.depends_on if p in unavailable]
        if blocked_by:
            node.status = BLOCKED
            node.shape_id = None
            report.blocked.append(node_id)
            unavailable.add(node_id)
            report.errors[node_id] = "Blocked by upstream node(s): %s" % ", ".join(
                sorted(blocked_by)
            )
            continue

        if node.status == BUILT and node.shape_id is not None and not force:
            report.built.append(node_id)
            continue

        try:
            shape_id = builder(node, updated)
        except Exception as exc:  # deliberate: any builder failure is a node failure
            node.status = FAILED
            node.shape_id = None
            report.failed.append(node_id)
            report.errors[node_id] = str(exc)
            unavailable.add(node_id)
            if not continue_on_error:
                for remaining in order[order.index(node_id) + 1 :]:
                    remaining_node = updated.nodes[remaining]
                    remaining_node.status = (
                        SUPPRESSED if remaining_node.suppressed else STALE
                    )
                    remaining_node.shape_id = None
                break
            continue

        node.shape_id = shape_id
        node.status = BUILT
        report.built.append(node_id)

    return report
