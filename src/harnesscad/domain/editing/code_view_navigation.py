"""Traceability between the CSG code and the 3D view (reverse & forward search).

Two navigation features are built on top of the AST<->CSG reference:

  * **Reverse search** -- from a selected element in the 3D view, find the code.
    The target elements are the nodes in the branch of the CSG tree from the root
    to the selected node, inclusive. The impacted elements are the other nodes of
    the CSG tree created by the same code statement as the selected part. The
    code editor adds a number in the margin of the targeted nodes indicating the
    call order of the instruction in the call stack.
  * **Forward search** -- from a selected code statement, find the view elements.
    All nodes created by the selected statement are collected: if there is only
    one it is marked as targeted; if there is more than one, all are marked as
    impacted.
  * **Ghosts** -- intersection/difference operations subtract volume, so their
    operands are not visible. When the selected element is one of these
    operations, the elements used in its creation are drawn as ghosts, cloned
    from the CSG children and classified as target/impacted.

This module is deterministic, view-agnostic (no rendering), and works purely on
the traced geometry tree from :mod:`programs.bidircsg_forward`. Pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from harnesscad.domain.programs.ast.bidirectional_csg import Node, Path, node_at
from harnesscad.domain.programs.runtime.csg_forward_eval import GeomNode, get, iter_geom


# --------------------------------------------------------------------------
# Reverse search: view element -> code.
# --------------------------------------------------------------------------

@dataclass
class Reverse:
    """Result of a reverse search from a selected output node."""

    selected: GeomNode
    target_source_paths: List[Path]   # root -> selected AST branch (target set)
    impacted: List[GeomNode]          # other outputs of the same AST node
    call_order: List[int]             # per targeted branch node: call-stack order


def _branch_to(tree: GeomNode, target: GeomNode) -> List[GeomNode]:
    """The chain of geometry nodes root..target (inclusive), or [] if absent."""
    def dfs(node: GeomNode, acc: List[GeomNode]) -> List[GeomNode]:
        acc = acc + [node]
        if node is target:
            return acc
        for ch in node.children:
            found = dfs(ch, acc)
            if found:
                return found
        return []
    return dfs(tree, [])


def reverse_search(tree: GeomNode, selected: GeomNode) -> Reverse:
    """Reverse navigation from a selected element.

    *Target* = the AST branch from the root down to the selected node. *Impacted*
    = every other output node produced by the same AST statement (e.g. the other
    instances of a repeated module / loop).
    """
    branch = _branch_to(tree, selected)
    if not branch:
        raise ValueError("selected node is not in the tree")
    target_paths = [g.source_path for g in branch]
    impacted = [
        g for g in iter_geom(tree)
        if g.source_path == selected.source_path and g is not selected
    ]
    # Call order = position of each branch node within the group of outputs that
    # share its AST node (the call order within the call stack).
    call_order: List[int] = []
    for g in branch:
        siblings = [x for x in iter_geom(tree) if x.source_path == g.source_path]
        siblings.sort(key=lambda x: x.call_stack)
        call_order.append(siblings.index(g))
    return Reverse(selected, target_paths, impacted, call_order)


# --------------------------------------------------------------------------
# Forward search: code statement -> view elements.
# --------------------------------------------------------------------------

@dataclass
class Forward:
    source_path: Path
    target: List[GeomNode]     # the single created node, if unique
    impacted: List[GeomNode]   # all created nodes, if more than one


def forward_search(tree: GeomNode, source_path: Path) -> Forward:
    """Forward navigation from a code statement.

    One created node -> *targeted*; several -> all *impacted*.
    """
    created = [g for g in iter_geom(tree) if g.source_path == source_path]
    if len(created) == 1:
        return Forward(source_path, list(created), [])
    return Forward(source_path, [], list(created))


# --------------------------------------------------------------------------
# Ghosts for removed (subtracted) elements.
# --------------------------------------------------------------------------

@dataclass
class Ghost:
    """A cloned representation of an operand removed by an intersect/difference."""

    element: GeomNode
    role: str  # "target" or "impacted"


def ghosts(tree: GeomNode, operation: GeomNode) -> List[Ghost]:
    """Ghost clones of the operands of an intersection/difference.

    The first operand (kept) is classified ``target``; the subtracted operands
    ``impacted`` -- driving the distinct target/impacted colouring of the cloned
    children.
    """
    if operation.kind not in ("Intersection", "Difference"):
        raise ValueError("ghosts only apply to intersection/difference nodes")
    out: List[Ghost] = []
    for i, child in enumerate(operation.children):
        role = "target" if i == 0 else "impacted"
        out.append(Ghost(child, role))
    return out


def removed_operands(tree: GeomNode) -> List[GeomNode]:
    """Every operand that a difference/intersection removes from the view.

    These are exactly the outputs that have no visual representation of their own
    and thus need ghosts to be navigable.
    """
    out: List[GeomNode] = []
    for g in iter_geom(tree):
        if g.kind == "Difference":
            out.extend(g.children[1:])
        elif g.kind == "Intersection":
            out.extend(g.children)
    return out


# --------------------------------------------------------------------------
# Traceability helpers (output feature <-> source node).
# --------------------------------------------------------------------------

def locate_source(program: Node, output_node: GeomNode) -> Node:
    """The AST node that produced ``output_node`` (output feature -> source)."""
    return node_at(program, output_node.source_path)


def consistency(program: Node) -> bool:
    """Every output node's ``source_path`` resolves to a real AST node.

    A basic navigation-consistency invariant: the AST<->CSG reference is total.
    """
    tree = get(program)
    for g in iter_geom(tree):
        try:
            node_at(program, g.source_path)
        except (IndexError, TypeError):
            return False
    return True
