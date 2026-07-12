"""Kernel-free TopoDS topology explorer (pythonocc ``OCC/Extend/TopologyUtils.py``).

pythonocc-core is a SWIG binding to the compiled OpenCASCADE (OCCT) kernel, so
almost every helper it ships is a thin wrapper over a live C++ call.  Its
``TopologyExplorer`` class is the one exception whose *value* is pure-Python graph
logic layered on top of the kernel primitives -- and that logic is fully
kernel-independent once the shape is modelled as data.  This module lifts exactly
that algorithm out of the binding.

Two ideas carry across from ``TopologyExplorer``:

* **Orientation-aware sub-shape deduplication.**  In OCCT a ``TopoDS_Shape`` is a
  triple ``(TShape, orientation, location)``; the same underlying geometry (the
  same ``TShape``) is reused for every incident parent, each time with its own
  orientation.  A raw traversal therefore reports a cube's edge 24 times (4 per
  face x 6 faces).  ``TopologyExplorer`` collapses those to the 12 *unique*
  geometric edges by bucketing on ``hash`` and confirming identity with
  ``IsSame`` (which ignores orientation).  We reproduce this with a stable
  first-seen dedup keyed on a ``tshape`` identity.

* **The shapes<->ancestors inverse map** (OCCT's ``TopExp::MapShapesAndAncestors``).
  Given the containment DAG solid -> shell -> face -> wire -> edge -> vertex, it
  answers upward queries -- *which faces bound this edge?*, *which edges meet this
  vertex?* -- by inverting the downward containment.  This is the classic B-rep
  adjacency query and is a deterministic graph walk, not a kernel operation.

This is distinct from the harness's other topology modules.
``reconstruction.cadparser_brep_graph`` builds a face/edge/*coedge* graph with
mate/next/prev half-edge relations from explicit loop definitions;
``geometry.manifold_halfedge`` is a triangle-mesh half-edge runtime; and
``geometry.opencad_synthetic_topology`` synthesises faces/edges for analytic
primitives.  None of them model the full eight-level TopAbs containment hierarchy,
orientation-aware unique-sub-shape counting, or the ancestor-map inversion this
module provides.

Pure stdlib, deterministic (first-seen ordering, no clock, no randomness).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = [
    "SHAPE_TYPES",
    "FORWARD",
    "REVERSED",
    "Shape",
    "sub_shapes",
    "count",
    "topology_summary",
    "map_shapes_and_ancestors",
    "ancestors_of",
    "make_edge",
    "make_box",
]

# TopAbs_ShapeEnum ordering: outermost container first, atom last.
SHAPE_TYPES: Tuple[str, ...] = (
    "COMPOUND",
    "COMPSOLID",
    "SOLID",
    "SHELL",
    "FACE",
    "WIRE",
    "EDGE",
    "VERTEX",
)

FORWARD = "FORWARD"
REVERSED = "REVERSED"


@dataclass(frozen=True)
class Shape:
    """A minimal, kernel-free stand-in for ``TopoDS_Shape``.

    :param shape_type: one of :data:`SHAPE_TYPES`.
    :param tshape: identity of the underlying geometry.  Two shapes are
        ``IsSame`` iff their ``tshape`` values are equal, regardless of
        orientation -- so a shared edge is represented by reusing the *same*
        :class:`Shape` instance (or one carrying the same ``tshape``) under
        every parent.
    :param orientation: :data:`FORWARD` or :data:`REVERSED`.
    :param children: directly contained sub-shapes, in traversal order.
    """

    shape_type: str
    tshape: str
    orientation: str = FORWARD
    children: Tuple["Shape", ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.shape_type not in SHAPE_TYPES:
            raise ValueError(f"unknown shape type: {self.shape_type!r}")
        if self.orientation not in (FORWARD, REVERSED):
            raise ValueError(f"unknown orientation: {self.orientation!r}")

    def is_same(self, other: "Shape") -> bool:
        """Same geometry ignoring orientation (mirrors ``TopoDS_Shape::IsSame``)."""
        return self.shape_type == other.shape_type and self.tshape == other.tshape

    def reversed(self) -> "Shape":
        """A view of this shape with the opposite orientation (same ``tshape``)."""
        flip = REVERSED if self.orientation == FORWARD else FORWARD
        return Shape(self.shape_type, self.tshape, flip, self.children)


def _walk(shape: Shape) -> Iterable[Shape]:
    """Depth-first traversal yielding *every* occurrence of every sub-shape.

    Shared sub-shapes are yielded once per incident parent, exactly as
    ``TopExp_Explorer`` reports them; deduplication is a separate step.
    """
    stack: List[Shape] = [shape]
    while stack:
        node = stack.pop()
        yield node
        # push in reverse so children are visited in declared order
        for child in reversed(node.children):
            stack.append(child)


def _dedup(shapes: Iterable[Shape]) -> List[Shape]:
    """First-seen dedup by ``(shape_type, tshape)`` -- the ignore-orientation filter."""
    seen: set = set()
    unique: List[Shape] = []
    for shp in shapes:
        key = (shp.shape_type, shp.tshape)
        if key not in seen:
            seen.add(key)
            unique.append(shp)
    return unique


def sub_shapes(shape: Shape, shape_type: str, *, unique: bool = True) -> List[Shape]:
    """All sub-shapes of ``shape_type`` contained in ``shape``.

    With ``unique`` (the default, matching ``TopologyExplorer(ignore_orientation=True)``)
    each distinct geometry is returned once; otherwise every oriented occurrence is
    returned.  If ``shape`` itself is of ``shape_type`` it is included, as
    ``TopExp_Explorer`` does when the root matches the searched type.
    """
    if shape_type not in SHAPE_TYPES:
        raise ValueError(f"unknown shape type: {shape_type!r}")
    found = [node for node in _walk(shape) if node.shape_type == shape_type]
    return _dedup(found) if unique else found


def count(shape: Shape, shape_type: str, *, unique: bool = True) -> int:
    """Number of ``shape_type`` sub-shapes (unique geometries by default)."""
    return len(sub_shapes(shape, shape_type, unique=unique))


def topology_summary(shape: Shape) -> Dict[str, int]:
    """Unique sub-shape counts per type (mirrors ``get_topology_summary``)."""
    return {stype.lower(): count(shape, stype) for stype in SHAPE_TYPES}


def map_shapes_and_ancestors(
    shape: Shape, sub_type: str, ancestor_type: str
) -> Dict[str, Tuple[Shape, ...]]:
    """Inverse containment map: ``tshape`` of a ``sub_type`` -> its ``ancestor_type`` shapes.

    Reproduces ``TopExp::MapShapesAndAncestors``: an ancestor is any
    ``ancestor_type`` shape whose sub-tree contains the sub-shape.  Both keys and
    the returned ancestor lists are deduplicated by geometry and ordered by first
    appearance, so the result is deterministic.
    """
    if sub_type not in SHAPE_TYPES:
        raise ValueError(f"unknown shape type: {sub_type!r}")
    if ancestor_type not in SHAPE_TYPES:
        raise ValueError(f"unknown shape type: {ancestor_type!r}")
    if SHAPE_TYPES.index(ancestor_type) >= SHAPE_TYPES.index(sub_type):
        raise ValueError(
            f"{ancestor_type} cannot be an ancestor of {sub_type} "
            "(ancestor must be an outer container type)"
        )

    result: Dict[str, List[Shape]] = {}
    order: List[str] = []
    for ancestor in sub_shapes(shape, ancestor_type, unique=True):
        for sub in sub_shapes(ancestor, sub_type, unique=True):
            bucket = result.setdefault(sub.tshape, [])
            if sub.tshape not in order:
                order.append(sub.tshape)
            if not any(a.is_same(ancestor) for a in bucket):
                bucket.append(ancestor)
    return {tshape: tuple(result[tshape]) for tshape in order}


def ancestors_of(shape: Shape, entity: Shape, ancestor_type: str) -> List[Shape]:
    """The ``ancestor_type`` shapes that contain ``entity`` (unique, ordered)."""
    amap = map_shapes_and_ancestors(shape, entity.shape_type, ancestor_type)
    return list(amap.get(entity.tshape, ()))


# --- convenience constructors ------------------------------------------------
def make_edge(tag: str, v0: Shape, v1: Shape, orientation: str = FORWARD) -> Shape:
    """An EDGE bounded by two shared vertices (reuse vertex instances to share them)."""
    return Shape("EDGE", tag, orientation, (v0, v1))


def make_box() -> Shape:
    """A topologically closed unit cube: 1 solid, 1 shell, 6 faces, 12 edges, 8 vertices.

    Edges and vertices are *shared* between incident faces, so orientation-aware
    dedup recovers 12 unique edges and 8 unique vertices from 24 oriented edge and
    24 oriented vertex occurrences -- the canonical ``TopologyExplorer`` example.
    """
    verts = [Shape("VERTEX", f"v{i}") for i in range(8)]

    # 12 edges as ordered vertex pairs (bottom ring, top ring, verticals).
    edge_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # e0..e3  bottom
        (4, 5), (5, 6), (6, 7), (7, 4),   # e4..e7  top
        (0, 4), (1, 5), (2, 6), (3, 7),   # e8..e11 verticals
    ]
    edges = [make_edge(f"e{i}", verts[a], verts[b]) for i, (a, b) in enumerate(edge_pairs)]

    # 6 faces; each is a wire of 4 shared edges. Every edge is used by 2 faces.
    face_edges = [
        (0, 1, 2, 3),     # bottom
        (4, 5, 6, 7),     # top
        (0, 9, 4, 8),     # front  y=0
        (1, 10, 5, 9),    # right  x=1
        (2, 11, 6, 10),   # back   y=1
        (3, 8, 7, 11),    # left   x=0
    ]
    faces: List[Shape] = []
    for fi, eids in enumerate(face_edges):
        wire = Shape("WIRE", f"w{fi}", FORWARD, tuple(edges[e] for e in eids))
        faces.append(Shape("FACE", f"f{fi}", FORWARD, (wire,)))

    shell = Shape("SHELL", "sh0", FORWARD, tuple(faces))
    return Shape("SOLID", "so0", FORWARD, (shell,))
