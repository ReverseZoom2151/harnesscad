"""Derive parametric handle positions by walking a CSG tree (paper Section 4.1-4.2).

This implements the two retrieval features the paper builds on top of the handle
grids:

* **Position** (Section 4.1, Figure 2). "Given a determined handle of a selected
  node, the application can define the position of the handle in terms of the
  variables used in the code. The application iterates on the CSG tree to locate
  the selected node. Then, the selected node provides the definition of the
  position of the handle relative to the node's centre. Later, the application
  iterates on translate nodes in the branch of the selected node, adding their
  definitions to the position of the handle." That is: walk from the root down to
  the target node, summing every ``translate`` vector on the path, then add the
  handle's offset — all as :class:`~programs.paramgeom_linform.LinearForm`
  arithmetic, giving a parametric position relative to the CSG root ``[0,0,0]``.

* **Delta vector** (Section 4.2, Figure 4). The vector between two handles,
  ``destination - origin``, "allowing users to determine the necessary
  transformation to align the origin and destination points". Worked example
  from the paper: aligning a sphere handle to a cylinder-top handle yields
  ``[r_top - r_sphere, 0, thickness + h_stem + h_top]``.

The tree model here is deliberately small and deterministic: :class:`TransformNode`
carries a translate offset and children; :class:`PrimitiveNode` is a leaf that
owns a named handle grid. Non-translate transforms (rotate/scale) are supported
as opaque pass-throughs that break linear derivation, matching the paper's stated
limitation that only ``translate`` accumulates cleanly. The paper's SymPy
simplification step is subsumed by :class:`LinearForm`'s canonical form (trivial
``translate 0`` contributions simply vanish).

Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from programs.paramgeom_handles import Offset
from programs.paramgeom_linform import LinearForm

Vec3 = Tuple[LinearForm, LinearForm, LinearForm]

_ZERO_VEC: Vec3 = (LinearForm.const(0), LinearForm.const(0), LinearForm.const(0))


class DerivationError(ValueError):
    """Raised when a parametric position cannot be derived (e.g. rotate on path)."""


@dataclass
class Node:
    """Base CSG node."""

    node_id: str


@dataclass
class PrimitiveNode(Node):
    """A leaf primitive owning a handle grid (name -> offset relative to centre)."""

    handles: Dict[str, Offset] = field(default_factory=dict)


@dataclass
class TransformNode(Node):
    """An internal node applying a translate offset to its children.

    ``offset`` is the parametric translate vector. ``linear`` is True for
    ``translate`` (the paper's accumulable case) and False for opaque transforms
    (rotate/scale) that must not be linearly accumulated.
    """

    offset: Vec3 = field(default_factory=lambda: _ZERO_VEC)
    children: List[Node] = field(default_factory=list)
    linear: bool = True


def translate(node_id: str, offset: Vec3, *children: Node) -> TransformNode:
    """Convenience constructor for a translate node."""
    return TransformNode(node_id, offset=offset, children=list(children), linear=True)


def opaque_transform(node_id: str, *children: Node) -> TransformNode:
    """A non-translate transform (rotate/scale) that blocks linear derivation."""
    return TransformNode(node_id, offset=_ZERO_VEC, children=list(children), linear=False)


def _add_vec(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub_vec(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _find_path(root: Node, target_id: str) -> Optional[List[Node]]:
    """Return the node path root..target (inclusive), or None if absent."""
    if root.node_id == target_id:
        return [root]
    if isinstance(root, TransformNode):
        for child in root.children:
            sub = _find_path(child, target_id)
            if sub is not None:
                return [root] + sub
    return None


def accumulate_translations(path: Sequence[Node]) -> Vec3:
    """Sum the translate offsets along ``path`` (root..node), as a Vec3.

    Raises :class:`DerivationError` if the path crosses a non-linear transform,
    mirroring the paper's limitation that only ``translate`` accumulates.
    """
    total = _ZERO_VEC
    for node in path:
        if isinstance(node, TransformNode):
            if not node.linear:
                raise DerivationError(
                    f"non-translate transform {node.node_id!r} on path blocks derivation"
                )
            total = _add_vec(total, node.offset)
    return total


def derive_position(root: Node, node_id: str, handle_name: str) -> Vec3:
    """Parametric position of ``handle_name`` on node ``node_id`` relative to the root.

    Implements the Position feature: accumulated translations along the branch
    plus the handle's offset relative to the node centre. The result is a Vec3 of
    canonical linear forms (already simplified).
    """
    path = _find_path(root, node_id)
    if path is None:
        raise DerivationError(f"node {node_id!r} not found in tree")
    node = path[-1]
    if not isinstance(node, PrimitiveNode):
        # Non-primitive nodes carry a single handle at the node's position
        # (paper Section 5): the accumulated translation is the position.
        if handle_name not in ("center", "node"):
            raise DerivationError(
                f"non-primitive node {node_id!r} only exposes a 'center' handle"
            )
        return accumulate_translations(path)
    if handle_name not in node.handles:
        raise DerivationError(
            f"handle {handle_name!r} not defined on node {node_id!r}"
        )
    branch = accumulate_translations(path)
    offset = node.handles[handle_name]
    return _add_vec(branch, offset)


def delta_vector(
    root: Node,
    origin: Tuple[str, str],
    destination: Tuple[str, str],
) -> Vec3:
    """Vector ``destination - origin`` between two handles (Delta-vector feature).

    ``origin`` and ``destination`` are ``(node_id, handle_name)`` pairs. The
    returned Vec3 is the transformation that moves the origin handle onto the
    destination handle — e.g. paste it into a ``translate`` to snap the objects.
    """
    o_pos = derive_position(root, origin[0], origin[1])
    d_pos = derive_position(root, destination[0], destination[1])
    return _sub_vec(d_pos, o_pos)


def vec_to_code(vec: Vec3, var_order: Optional[List[str]] = None) -> str:
    """Render a Vec3 as an OpenSCAD-style vector literal ``[x, y, z]``."""
    return "[" + ", ".join(c.to_code(var_order) for c in vec) + "]"


def translate_statement(vec: Vec3, var_order: Optional[List[str]] = None) -> str:
    """Render a Vec3 as a full ``translate([...])`` call (ready to paste)."""
    return f"translate({vec_to_code(vec, var_order)})"
