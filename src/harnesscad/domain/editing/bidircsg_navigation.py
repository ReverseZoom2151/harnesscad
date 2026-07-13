"""Traceability between the CSG code and the 3D view (reverse & forward search).

From *Introducing Bidirectional Programming in Constructive Solid Geometry-Based
CAD* (Gonzalez et al., SUI '23), Sec. 4.1-4.2. The paper adds two navigation
features on top of the AST<->CSG reference:

  * **Reverse search** (Sec. 4.1) -- from a selected element in the 3D view, find
    the code. "The target element refers to the nodes in the branch of the CSG
    tree from the root to the selected node, included. The impacted elements refer
    to the other nodes of the CSG tree that were created with the same code
    statement of the selected part" (F2). "The code editor adds a number in the
    margin of the targeted nodes indicating the call order of the instruction in
    the call stack."
  * **Forward search** (Sec. 4.2) -- from a selected code statement, find the view
    elements. "The system searches all the nodes created by the selected
    statement. If there is only one, the program marks it as targeted; if there is
    more than one, it marks all nodes as impacted" (F4).
  * **Ghosts** (Sec. 4.1, F3) -- intersection/difference operations subtract
    volume, so their operands are not visible. "When the selected element is one
    of these operations ... we draw the elements used in its creation as ghosts",
    cloned from the CSG children and classified as target/impacted.

This module is deterministic, view-agnostic (no rendering), and works purely on
the traced geometry tree from :mod:`programs.bidircsg_forward`. Pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from harnesscad.domain.programs.ast.bidircsg_ast import Node, Path, node_at
from harnesscad.domain.programs.runtime.bidircsg_forward import GeomNode, get, iter_geom


# --------------------------------------------------------------------------
# Reverse search: view element -> code.
# --------------------------------------------------------------------------

@dataclass
class Reverse:
    """Result of a reverse search from a selected output node."""

    selected: GeomNode
    target_source_paths: List[Path]   # root -> selected AST branch (F2 green)
    impacted: List[GeomNode]          # other outputs of the same AST node (pink)
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
    """Reverse navigation from a selected element (Sec. 4.1, F1-F3).

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
    # share its AST node (the "call order in the call stack", F2).
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
    """Forward navigation from a code statement (Sec. 4.2, F4).

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
    """Ghost clones of the operands of an intersection/difference (Sec. 4.1, F3).

    The first operand (kept) is classified ``target``; the subtracted operands
    ``impacted`` -- mirroring the paper's green (target) / pink (impacted)
    colouring of the cloned children.
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

    These are exactly the outputs that "do not have a visual representation"
    (Sec. 3.4.1) and thus need ghosts to be navigable.
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
