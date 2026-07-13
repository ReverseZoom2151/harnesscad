"""Forward evaluation ``get``: CSG program AST -> traced geometry tree.

From *Introducing Bidirectional Programming in Constructive Solid Geometry-Based
CAD* (Gonzalez et al., SUI '23), Sec. 4:

    "OpenSCAD parses the code to create an Abstract Syntax Tree (AST) ... Then it
    processes the AST ... to create an Abstract CSG Tree. Each node in this tree
    represents an element that contributes to the creation of the model ...
    OpenSCAD uses the CSG to compute a mesh hierarchy that contains the 3D points,
    normal vectors, and colors of all nodes ... and stores it in a Geometric Tree."

This module is the forward half of the bidirectional transformation -- the paper's
``get``. It evaluates a :mod:`programs.bidircsg_ast` program into a **traced
geometry tree** in which every output node carries:

  * ``source_path`` -- the AST node reference (Sec. 4.1, F2: "the reference of the
    AST node in each CSG node"). This is what makes navigation possible: one AST
    node may produce *several* output nodes (a loop / repeated module), which is
    exactly the *impacted* relationship.
  * ``call_stack`` -- the sequence of loop-instance indices from the root to this
    node ("the call order of the instruction in the call stack", F2).
  * ``world_transform`` / ``parent_transform`` -- the accumulated affine from the
    root to this node (and to its parent frame). The gizmo in the paper is placed
    "applying previous translation and rotation from the root to the selected
    object" (F6); the backward ``put`` needs the parent frame to convert a
    world-space edit into a local parameter change.
  * ``anchor`` -- the world position of the node's local origin (a cheap stand-in
    for the mesh, enough to verify the lens laws).

Boolean semantics are not meshed; the tree mirrors the CSG structure and records
transforms. Pure stdlib; deterministic (rotations use ``math``, no wall clock).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

from harnesscad.domain.programs.bidircsg_ast import (
    Difference,
    Intersection,
    Node,
    Path,
    Primitive,
    Repeat,
    Rotate,
    Scale,
    Translate,
    Union,
    Vec3,
    children,
)


# --------------------------------------------------------------------------
# Minimal affine transform (3x3 linear + translation).
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Affine:
    """point' = lin @ point + t.  ``lin`` is a row-major 3x3 tuple."""

    lin: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    t: Vec3 = (0.0, 0.0, 0.0)

    def apply(self, p: Vec3) -> Vec3:
        m = self.lin
        return (
            m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + self.t[0],
            m[3] * p[0] + m[4] * p[1] + m[5] * p[2] + self.t[1],
            m[6] * p[0] + m[7] * p[1] + m[8] * p[2] + self.t[2],
        )

    def apply_linear(self, v: Vec3) -> Vec3:
        m = self.lin
        return (
            m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
        )

    def compose(self, other: "Affine") -> "Affine":
        """self ∘ other: apply ``other`` first, then ``self``."""
        a, b = self.lin, other.lin
        lin = tuple(
            sum(a[3 * r + k] * b[3 * k + c] for k in range(3))
            for r in range(3)
            for c in range(3)
        )
        t = self.apply(other.t)
        return Affine(lin, t)

    def inverse_linear(self) -> Tuple[float, ...]:
        """Inverse of the 3x3 linear part (raises if singular)."""
        m = self.lin
        det = (
            m[0] * (m[4] * m[8] - m[5] * m[7])
            - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6])
        )
        if abs(det) < 1e-12:
            raise ValueError("singular linear transform")
        inv = (
            (m[4] * m[8] - m[5] * m[7]) / det,
            (m[2] * m[7] - m[1] * m[8]) / det,
            (m[1] * m[5] - m[2] * m[4]) / det,
            (m[5] * m[6] - m[3] * m[8]) / det,
            (m[0] * m[8] - m[2] * m[6]) / det,
            (m[2] * m[3] - m[0] * m[5]) / det,
            (m[3] * m[7] - m[4] * m[6]) / det,
            (m[1] * m[6] - m[0] * m[7]) / det,
            (m[0] * m[4] - m[1] * m[3]) / det,
        )
        return inv

    def apply_inverse_linear(self, v: Vec3) -> Vec3:
        inv = self.inverse_linear()
        return (
            inv[0] * v[0] + inv[1] * v[1] + inv[2] * v[2],
            inv[3] * v[0] + inv[4] * v[1] + inv[5] * v[2],
            inv[6] * v[0] + inv[7] * v[1] + inv[8] * v[2],
        )


IDENTITY = Affine()


def translation(offset: Vec3) -> Affine:
    return Affine(IDENTITY.lin, tuple(float(c) for c in offset))


def scaling(factors: Vec3) -> Affine:
    fx, fy, fz = factors
    return Affine((fx, 0.0, 0.0, 0.0, fy, 0.0, 0.0, 0.0, fz), (0.0, 0.0, 0.0))


def rotation(angles_deg: Vec3) -> Affine:
    """Rotation about x, then y, then z (degrees), matching Rotate semantics."""
    rx, ry, rz = (math.radians(a) for a in angles_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    ax = Affine((1, 0, 0, 0, cx, -sx, 0, sx, cx))
    ay = Affine((cy, 0, sy, 0, 1, 0, -sy, 0, cy))
    az = Affine((cz, -sz, 0, sz, cz, 0, 0, 0, 1))
    return az.compose(ay).compose(ax)


# --------------------------------------------------------------------------
# Traced geometry tree.
# --------------------------------------------------------------------------

@dataclass
class GeomNode:
    kind: str                       # AST node type name, or primitive kind
    source_path: Path               # reference back to the AST node
    call_stack: Tuple[int, ...]     # loop-instance indices root -> here
    world_transform: Affine         # root -> this node (incl. own transform)
    parent_transform: Affine        # root -> parent frame (excl. own transform)
    anchor: Vec3                    # world position of the local origin
    params: Tuple[float, ...] = ()  # primitive params (empty otherwise)
    children: List["GeomNode"] = field(default_factory=list)

    def is_primitive(self) -> bool:
        return not self.children and self.kind not in (
            "Union", "Intersection", "Difference",
            "Translate", "Rotate", "Scale", "Repeat",
        )


def _eval(node: Node, path: Path, frame: Affine, stack: Tuple[int, ...]) -> GeomNode:
    if isinstance(node, Primitive):
        return GeomNode(
            kind=node.kind, source_path=path, call_stack=stack,
            world_transform=frame, parent_transform=frame,
            anchor=frame.apply((0.0, 0.0, 0.0)), params=tuple(node.params),
        )

    if isinstance(node, (Translate, Rotate, Scale)):
        if isinstance(node, Translate):
            local = translation(node.offset)
        elif isinstance(node, Rotate):
            local = rotation(node.angles)
        else:
            local = scaling(node.factors)
        world = frame.compose(local)
        child = _eval(node.child, path + (0,), world, stack)
        return GeomNode(
            kind=type(node).__name__, source_path=path, call_stack=stack,
            world_transform=world, parent_transform=frame,
            anchor=world.apply((0.0, 0.0, 0.0)), children=[child],
        )

    if isinstance(node, Repeat):
        kids: List[GeomNode] = []
        for i in range(node.count):
            step = translation(tuple(c * i for c in node.step))
            world = frame.compose(step)
            kids.append(_eval(node.child, path + (0,), world, stack + (i,)))
        return GeomNode(
            kind="Repeat", source_path=path, call_stack=stack,
            world_transform=frame, parent_transform=frame,
            anchor=frame.apply((0.0, 0.0, 0.0)), children=kids,
        )

    if isinstance(node, (Union, Intersection, Difference)):
        kids = [
            _eval(ch, path + (i,), frame, stack)
            for i, ch in enumerate(children(node))
        ]
        return GeomNode(
            kind=type(node).__name__, source_path=path, call_stack=stack,
            world_transform=frame, parent_transform=frame,
            anchor=frame.apply((0.0, 0.0, 0.0)), children=kids,
        )

    raise TypeError("unknown CSG node: %r" % (node,))


def get(program: Node) -> GeomNode:
    """Forward evaluate a program into a traced geometry tree (the lens ``get``)."""
    return _eval(program, (), IDENTITY, ())


def iter_geom(tree: GeomNode) -> Iterator[GeomNode]:
    """Pre-order walk over the geometry tree."""
    yield tree
    for ch in tree.children:
        yield from iter_geom(ch)


def leaves(tree: GeomNode) -> List[GeomNode]:
    """All primitive (leaf) output nodes."""
    return [g for g in iter_geom(tree) if g.is_primitive()]


def find_instances(tree: GeomNode, source_path: Path) -> List[GeomNode]:
    """Every output node produced by the AST node at ``source_path``."""
    return [g for g in iter_geom(tree) if g.source_path == source_path]


def find_instance(
    tree: GeomNode, source_path: Path, call_stack: Tuple[int, ...] = ()
) -> Optional[GeomNode]:
    """The unique output node for a source path + call stack (or ``None``)."""
    for g in iter_geom(tree):
        if g.source_path == source_path and g.call_stack == call_stack:
            return g
    return None
