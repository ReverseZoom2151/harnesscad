"""Type-pair degrees-of-freedom table and the autoconstrain validity mask.

``reconstruction/sketchgraphs_taxonomy.py`` records a *nominal* DOF figure per
constraint type -- "coincident removes 2".  That number is only true for one
pairing.  Coincident between two points removes 2, between a line and a point
removes 1, and between two circles removes 3, because what a constraint actually
pins down depends on the types of the primitives it joins.  This module carries
the finer accounting as a two-level table
(``EDGE_DOF_REMOVED[constraint][(t1, t2)]``) and uses it for two distinct jobs.

1. **DOF accounting over a construction sequence.**  Each node op *adds* its
   primitive's DOF; each edge op *removes* the tabulated amount for its type
   pair.  The running total is the sketch's under-constrainedness at every
   prefix, which is the signal a sequence model can be conditioned on.
   The accounting rules are exact and non-obvious:

   * subnode labels (a line's ``.start``, a circle's ``.center``) collapse to
     ``Point`` -- they *are* points for DOF purposes;
   * a self-loop (a one-reference edge, e.g. ``horizontal`` on a single line) is
     scored against the pair ``(t, t)``;
   * a hyperedge (3+ references, e.g. ``mirror``) removes 0 -- the heuristic
     declines to model it rather than guessing;
   * any edge touching the ``External`` node (the origin) removes 0;
   * the table is symmetric: ``(t1, t2)`` is tried, then ``(t2, t1)``;
   * an unlisted pairing removes 0, so an inapplicable constraint is free
     rather than an error.

2. **The autoconstrain validity mask.**  A model that proposes constraints must
   not propose a *radius* between two lines.  The set of type pairs a constraint
   appears under in the table *is* its applicability domain, so inverting the
   table by type pair yields, for free, the legal constraint set for any pair of
   primitives (:func:`valid_constraints`).  This is packed into a
   lower-triangular mask over ``(node i, node j<=i)`` in a flat layout with row
   offsets ``i * (i + 1) // 2`` (:func:`mask_offset`); node 0 is the external
   origin node and is left fully permissive.  :func:`constraint_mask` builds that
   mask as booleans (``True`` = allowed), so it can drive a logit mask
   (allowed -> 0, disallowed -> -inf) or a plain rejection filter.

Public API
----------
``NODE_DOF`` / ``EDGE_DOF_REMOVED``    -- the tables.
``dof_for_node`` / ``dof_removed_for_edge``.
``sequence_dof(ops)`` / ``cumulative_dof(ops)``.
``valid_constraints(t1, t2)``          -- legal constraint types for a pair.
``mask_offset(i)`` / ``mask_size(n)``  -- the triangular flat layout.
``constraint_mask(labels)``            -- the full validity mask.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Sequence, Tuple, Union

from harnesscad.io.formats.onshape_json import EntityType, SubnodeType

__all__ = [
    "NODE_DOF",
    "EDGE_DOF_REMOVED",
    "NodeOp",
    "EdgeOp",
    "node_label_for_dof",
    "dof_for_node",
    "dof_removed_for_edge",
    "sequence_dof",
    "cumulative_dof",
    "valid_constraints",
    "constraint_types",
    "mask_offset",
    "mask_size",
    "constraint_mask",
]


#: DOF each primitive type contributes.  Types absent here (spline, ellipse,
#: external, ...) contribute 0 under the heuristic.
NODE_DOF: Dict[EntityType, int] = {
    EntityType.Point: 2,
    EntityType.Line: 4,
    EntityType.Circle: 3,
    EntityType.Arc: 5,
}


def _pairs(**spec: int) -> Dict[Tuple[EntityType, EntityType], int]:
    """Helper: ``arc_line=1`` -> ``{(Arc, Line): 1}``."""
    out: Dict[Tuple[EntityType, EntityType], int] = {}
    for key, value in spec.items():
        a, b = key.split("_")
        out[(EntityType[a.capitalize()], EntityType[b.capitalize()])] = value
    return out


#: ``constraint -> {(entity type, entity type): dof removed}``.
#: The keys of the inner dict double as the constraint's applicability domain.
EDGE_DOF_REMOVED: Dict[str, Dict[Tuple[EntityType, EntityType], int]] = {
    "angle": _pairs(line_line=1),
    "centerline_dimension": _pairs(arc_line=0, circle_line=0, line_line=0, line_point=0),
    "coincident": _pairs(
        arc_arc=3, arc_circle=3, arc_point=1, circle_circle=3, circle_point=1,
        line_line=2, line_point=1, point_point=2,
    ),
    "concentric": _pairs(
        arc_arc=2, arc_circle=2, arc_point=2, circle_circle=2, circle_point=2,
        point_point=2,
    ),
    "diameter": _pairs(arc_arc=1, circle_circle=1),
    "distance": _pairs(
        arc_arc=1, arc_circle=1, arc_line=1, arc_point=1, circle_circle=1,
        circle_line=1, circle_point=1, line_line=1, line_point=1, point_point=1,
    ),
    "equal": _pairs(arc_arc=1, arc_circle=1, circle_circle=1, line_line=1),
    "fix": _pairs(arc_arc=3, circle_circle=3, line_line=2, point_point=2),
    "horizontal": _pairs(line_line=1, point_point=1),
    "intersected": {},
    "length": _pairs(arc_arc=1, line_line=1),
    "midpoint": _pairs(arc_point=2, line_point=2),
    "normal": _pairs(arc_line=1, circle_line=1),
    "offset": _pairs(arc_arc=2, arc_circle=2, circle_circle=2, line_line=1),
    "parallel": _pairs(line_line=1),
    "perpendicular": _pairs(line_line=1),
    "radius": _pairs(arc_arc=1, circle_circle=1),
    "subnode": _pairs(arc_point=0, circle_point=0, line_point=0),
    "tangent": _pairs(
        arc_arc=1, arc_circle=1, arc_line=1, circle_circle=1, circle_line=1,
    ),
    "vertical": _pairs(line_line=1, point_point=1),
}

#: The ``subnode`` pseudo-constraint records structural containment (a curve owns
#: its endpoints); it is not a user constraint and is excluded from prediction.
_STRUCTURAL = frozenset({"subnode"})

Label = Union[EntityType, SubnodeType]


class NodeOp:
    """A construction op that introduces a primitive."""

    __slots__ = ("label",)

    def __init__(self, label: Label) -> None:
        self.label = label

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"NodeOp({self.label!r})"


class EdgeOp:
    """A construction op that applies ``label`` to the referenced node indices."""

    __slots__ = ("label", "references")

    def __init__(self, label: str, references: Sequence[int]) -> None:
        self.label = label
        self.references = tuple(references)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"EdgeOp({self.label!r}, {self.references!r})"


# ---------------------------------------------------------------------------
# DOF accounting
# ---------------------------------------------------------------------------
def node_label_for_dof(label: Label) -> EntityType:
    """Collapse a subnode label to ``Point`` -- subnodes are points."""
    if isinstance(label, SubnodeType):
        return EntityType.Point
    return label


def dof_for_node(label: Label) -> int:
    """DOF contributed by a node op.  Unmodelled types contribute 0."""
    return NODE_DOF.get(node_label_for_dof(label), 0)


def dof_removed_for_edge(edge: EdgeOp, nodes: Sequence[NodeOp]) -> int:
    """DOF removed by an edge op, given the nodes constructed so far.

    Returns 0 -- rather than raising -- for a hyperedge, an edge touching the
    external origin node, an unknown constraint, or a type pairing the table does
    not list.  See the module docstring for why each case is free.
    """
    ref_types = [node_label_for_dof(nodes[r].label) for r in edge.references]

    if len(ref_types) == 1:
        ref_types = ref_types * 2
    elif len(ref_types) > 2:
        return 0

    if EntityType.External in ref_types:
        return 0

    table = EDGE_DOF_REMOVED.get(edge.label)
    if table is None:
        return 0

    t1, t2 = ref_types
    if (t1, t2) in table:
        return table[(t1, t2)]
    if (t2, t1) in table:
        return table[(t2, t1)]
    return 0


def sequence_dof(ops: Sequence[Union[NodeOp, EdgeOp]]) -> List[int]:
    """Per-op DOF delta: ``+dof`` for a node op, ``-dof_removed`` for an edge op."""
    out: List[int] = []
    nodes: List[NodeOp] = []
    for op in ops:
        if isinstance(op, NodeOp):
            nodes.append(op)
            out.append(dof_for_node(op.label))
        else:
            out.append(-dof_removed_for_edge(op, nodes))
    return out


def cumulative_dof(ops: Sequence[Union[NodeOp, EdgeOp]]) -> List[int]:
    """Running DOF total after each op -- the sketch's residual freedom.

    A well-constrained sketch (origin pinned) ends at 0; a positive tail means
    under-constrained, negative means over-constrained under the heuristic.
    """
    total = 0
    out: List[int] = []
    for delta in sequence_dof(ops):
        total += delta
        out.append(total)
    return out


# ---------------------------------------------------------------------------
# Validity mask
# ---------------------------------------------------------------------------
def constraint_types(include_structural: bool = False) -> Tuple[str, ...]:
    """The predictable constraint types, in a stable (sorted) order."""
    names = sorted(EDGE_DOF_REMOVED)
    if not include_structural:
        names = [n for n in names if n not in _STRUCTURAL]
    return tuple(names)


def _build_valid_by_pair() -> Dict[Tuple[EntityType, EntityType], FrozenSet[str]]:
    out: Dict[Tuple[EntityType, EntityType], set] = {}
    for name, table in EDGE_DOF_REMOVED.items():
        if name in _STRUCTURAL:
            continue
        for pair in table:
            out.setdefault(tuple(sorted(pair)), set()).add(name)
    return {k: frozenset(v) for k, v in out.items()}


_VALID_BY_PAIR = _build_valid_by_pair()


def valid_constraints(label_a: Label, label_b: Label) -> FrozenSet[str]:
    """The constraint types applicable between two primitive types.

    Order-insensitive; subnode labels collapse to ``Point``.  A pairing that
    appears in no constraint's table (e.g. two splines) yields the empty set.
    """
    key = tuple(sorted((node_label_for_dof(label_a), node_label_for_dof(label_b))))
    return _VALID_BY_PAIR.get(key, frozenset())


def mask_offset(node_index: int) -> int:
    """Flat start offset of node ``i``'s row in the triangular mask."""
    if node_index < 0:
        raise ValueError("node index must be non-negative")
    return node_index * (node_index + 1) // 2


def mask_size(num_nodes: int) -> int:
    """Number of ``(i, j<=i)`` slots for ``num_nodes`` nodes."""
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    return num_nodes * (num_nodes + 1) // 2


def constraint_mask(
    labels: Sequence[Label], types: Sequence[str] | None = None
) -> List[List[bool]]:
    """Validity mask over ``(node i, partner j<=i)`` slots and constraint types.

    Row ``mask_offset(i) + j`` holds, for each type in ``types``, whether that
    constraint may legally join node ``j`` to node ``i``.  Node 0 is the external
    origin node: its own slot and every ``j == 0`` slot are fully permissive,
    because an origin constraint is a projection whose type the DOF table does
    not restrict.

    ``types`` defaults to :func:`constraint_types` (structural ``subnode``
    excluded).
    """
    names = tuple(types) if types is not None else constraint_types()
    num_nodes = len(labels)
    mask = [[False] * len(names) for _ in range(mask_size(num_nodes))]

    if num_nodes == 0:
        return mask

    # Node 0 is the external origin: fully permissive.
    mask[0] = [True] * len(names)

    for i in range(1, num_nodes):
        offset = mask_offset(i)
        mask[offset] = [True] * len(names)  # partner is the origin node
        for j in range(1, i + 1):
            allowed = valid_constraints(labels[i], labels[j])
            mask[offset + j] = [name in allowed for name in names]

    return mask
