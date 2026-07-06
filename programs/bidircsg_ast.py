"""Bidirectional CSG program AST (source representation).

From *Introducing Bidirectional Programming in Constructive Solid Geometry-Based
CAD* (Gonzalez, Kieken, Pietrzak, Girouard, Casiez, SUI '23).

The paper (Sec. 4) describes the OpenSCAD architecture: the code is parsed into an
Abstract Syntax Tree (AST); each node "represents an element that contributes to
the creation of the model and is a module instance. The tree leaves are always
primitives (e.g. spheres or cylinders). Intermediate nodes can be boolean
operations (e.g. union), transformations (e.g. translate), or groups such as
control structures (e.g. conditionals or loops)." Evaluating the AST yields an
Abstract CSG tree whose nodes carry a *reference* back to the AST node -- this
reference is what makes bidirectional navigation possible (Sec. 4.1, F2).

This module is the SOURCE side of the bidirectional transformation: a small,
deterministic, stdlib-only CSG program AST plus the structural machinery the
navigation/lens layers need:

  * frozen node types -- ``Primitive`` (leaf), the transforms ``Translate`` /
    ``Rotate`` / ``Scale``, the booleans ``Union`` / ``Intersection`` /
    ``Difference``, and the loop ``Repeat`` (a group that replicates its child --
    the paper's "module called multiple times inside a loop", the source of the
    *impacted* relationship);
  * a **path** = tuple of child indices from the root, used as the stable AST
    node identity (one AST node -> possibly many CSG output nodes);
  * generic traversal (:func:`iter_nodes`, :func:`node_at`), functional update
    (:func:`replace_at`, :func:`wrap_at`) so the backward ``put`` never mutates
    the original program, and (:func:`parent_path`);
  * deterministic serialisation (:func:`serialize` to OpenSCAD-like text and
    :func:`to_dict` / :func:`from_dict` for an exact round-trip) and a
    structural signature.

The learned/UI parts of the paper (interview coding, gizmo rendering, variable
inference) are out of scope. Pure stdlib; deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Path = Tuple[int, ...]


# --------------------------------------------------------------------------
# Node types.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Primitive:
    """A leaf primitive, e.g. ``sphere(r)`` or ``cube([x, y, z])``."""

    kind: str
    params: Tuple[float, ...]


@dataclass(frozen=True)
class Translate:
    offset: Vec3
    child: "Node"


@dataclass(frozen=True)
class Rotate:
    """Euler rotation in degrees about x, then y, then z (OpenSCAD order)."""

    angles: Vec3
    child: "Node"


@dataclass(frozen=True)
class Scale:
    factors: Vec3
    child: "Node"


@dataclass(frozen=True)
class Union:
    children: Tuple["Node", ...]


@dataclass(frozen=True)
class Intersection:
    children: Tuple["Node", ...]


@dataclass(frozen=True)
class Difference:
    """First child minus the union of the rest (OpenSCAD ``difference``)."""

    children: Tuple["Node", ...]


@dataclass(frozen=True)
class Repeat:
    """A loop that instantiates ``child`` ``count`` times.

    Instance ``i`` (0-based) is translated by ``step * i``. This models an
    OpenSCAD ``for`` loop: all instances share the *same* AST subtree, so a
    single source edit propagates to every instance (the paper's *impacted*
    elements, Sec. 4.1).
    """

    count: int
    step: Vec3
    child: "Node"


Node = object  # any of the above; kept loose to avoid a heavy Union type.

_TRANSFORMS = (Translate, Rotate, Scale)
_BOOLEANS = (Union, Intersection, Difference)


# --------------------------------------------------------------------------
# Generic child access / functional rebuild.
# --------------------------------------------------------------------------

def children(node: Node) -> Tuple[Node, ...]:
    if isinstance(node, Primitive):
        return ()
    if isinstance(node, _TRANSFORMS) or isinstance(node, Repeat):
        return (node.child,)
    if isinstance(node, _BOOLEANS):
        return tuple(node.children)
    raise TypeError("unknown CSG node: %r" % (node,))


def with_children(node: Node, new_children: Sequence[Node]) -> Node:
    """Rebuild ``node`` with replaced children (structure preserved)."""
    nc = tuple(new_children)
    if isinstance(node, Primitive):
        if nc:
            raise ValueError("primitive takes no children")
        return node
    if isinstance(node, Translate):
        return Translate(node.offset, nc[0])
    if isinstance(node, Rotate):
        return Rotate(node.angles, nc[0])
    if isinstance(node, Scale):
        return Scale(node.factors, nc[0])
    if isinstance(node, Repeat):
        return Repeat(node.count, node.step, nc[0])
    if isinstance(node, Union):
        return Union(nc)
    if isinstance(node, Intersection):
        return Intersection(nc)
    if isinstance(node, Difference):
        return Difference(nc)
    raise TypeError("unknown CSG node: %r" % (node,))


# --------------------------------------------------------------------------
# Path-based traversal and update.
# --------------------------------------------------------------------------

def iter_nodes(program: Node, prefix: Path = ()) -> Iterator[Tuple[Path, Node]]:
    """Pre-order walk yielding ``(path, node)`` for every AST node."""
    yield prefix, program
    for i, ch in enumerate(children(program)):
        yield from iter_nodes(ch, prefix + (i,))


def node_at(program: Node, path: Path) -> Node:
    node = program
    for idx in path:
        node = children(node)[idx]
    return node


def parent_path(path: Path) -> Path:
    if not path:
        raise ValueError("root has no parent")
    return path[:-1]


def replace_at(program: Node, path: Path, new_node: Node) -> Node:
    """Return a copy of ``program`` with the node at ``path`` replaced."""
    if not path:
        return new_node
    idx = path[0]
    kids = list(children(program))
    kids[idx] = replace_at(kids[idx], path[1:], new_node)
    return with_children(program, kids)


def wrap_at(program: Node, path: Path, wrapper) -> Node:
    """Replace the node at ``path`` with ``wrapper(old_node)``.

    ``wrapper`` receives the existing node and returns a new node (typically a
    transform wrapping it). Used by the backward ``put`` to insert a transform.
    """
    old = node_at(program, path)
    return replace_at(program, path, wrapper(old))


# --------------------------------------------------------------------------
# Serialisation.
# --------------------------------------------------------------------------

def _fmt(x: float) -> str:
    if x == 0:
        x = 0.0
    if float(x).is_integer():
        return str(int(x))
    return repr(round(float(x), 6))


def _vec(v: Vec3) -> str:
    return "[%s]" % ", ".join(_fmt(c) for c in v)


def serialize(program: Node, indent: int = 0) -> str:
    """Render ``program`` as OpenSCAD-like source (deterministic, one-way)."""
    pad = "  " * indent

    def block(head: str, kids: Sequence[Node]) -> str:
        lines = [pad + head + " {"]
        for ch in kids:
            lines.append(serialize(ch, indent + 1))
        lines.append(pad + "}")
        return "\n".join(lines)

    if isinstance(program, Primitive):
        return "%s%s(%s);" % (
            pad, program.kind, ", ".join(_fmt(p) for p in program.params)
        )
    if isinstance(program, Translate):
        return block("translate(%s)" % _vec(program.offset), (program.child,))
    if isinstance(program, Rotate):
        return block("rotate(%s)" % _vec(program.angles), (program.child,))
    if isinstance(program, Scale):
        return block("scale(%s)" % _vec(program.factors), (program.child,))
    if isinstance(program, Repeat):
        return block(
            "for(i=[0:%d]) translate(i*%s)" % (program.count - 1, _vec(program.step)),
            (program.child,),
        )
    if isinstance(program, Union):
        return block("union()", program.children)
    if isinstance(program, Intersection):
        return block("intersection()", program.children)
    if isinstance(program, Difference):
        return block("difference()", program.children)
    raise TypeError("unknown CSG node: %r" % (program,))


def to_dict(program: Node) -> dict:
    """Exact, JSON-friendly serialisation (round-trips via :func:`from_dict`)."""
    t = type(program).__name__
    if isinstance(program, Primitive):
        return {"t": t, "kind": program.kind, "params": list(program.params)}
    if isinstance(program, Translate):
        return {"t": t, "offset": list(program.offset), "child": to_dict(program.child)}
    if isinstance(program, Rotate):
        return {"t": t, "angles": list(program.angles), "child": to_dict(program.child)}
    if isinstance(program, Scale):
        return {"t": t, "factors": list(program.factors), "child": to_dict(program.child)}
    if isinstance(program, Repeat):
        return {
            "t": t, "count": program.count, "step": list(program.step),
            "child": to_dict(program.child),
        }
    if isinstance(program, _BOOLEANS):
        return {"t": t, "children": [to_dict(c) for c in program.children]}
    raise TypeError("unknown CSG node: %r" % (program,))


def from_dict(d: dict) -> Node:
    t = d["t"]
    if t == "Primitive":
        return Primitive(d["kind"], tuple(d["params"]))
    if t == "Translate":
        return Translate(tuple(d["offset"]), from_dict(d["child"]))
    if t == "Rotate":
        return Rotate(tuple(d["angles"]), from_dict(d["child"]))
    if t == "Scale":
        return Scale(tuple(d["factors"]), from_dict(d["child"]))
    if t == "Repeat":
        return Repeat(d["count"], tuple(d["step"]), from_dict(d["child"]))
    if t == "Union":
        return Union(tuple(from_dict(c) for c in d["children"]))
    if t == "Intersection":
        return Intersection(tuple(from_dict(c) for c in d["children"]))
    if t == "Difference":
        return Difference(tuple(from_dict(c) for c in d["children"]))
    raise ValueError("unknown node tag: %r" % (t,))


def structural_signature(program: Node) -> tuple:
    """Structure ignoring numeric parameters (kinds + shape of the tree)."""
    if isinstance(program, Primitive):
        return ("Primitive", program.kind)
    return (type(program).__name__,) + tuple(
        structural_signature(c) for c in children(program)
    )


def node_count(program: Node) -> int:
    return sum(1 for _ in iter_nodes(program))
