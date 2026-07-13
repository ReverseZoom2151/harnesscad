"""SketchGraphs geometric constraint graph (Seff et al., NeurIPS 2020, Sec. 3.2).

A CAD sketch is represented as a *geometric constraint graph* ``G = (V, E)``: nodes
``V`` are primitives, edges ``E`` are the designer-imposed constraints between them.
The paper stresses several structural properties that make this a *multi-hypergraph*
rather than a plain graph, all of which this module models faithfully:

  * **Loops** -- a constraint acting on a single primitive (e.g. a radius/scale
    constraint) is an edge connecting a node to itself (member arity 1).
  * **Hyperedges** -- a constraint acting on three or more primitives (e.g. a
    ``mirror``, which names a third primitive as the axis of symmetry) joins more
    than two nodes.
  * **Multi-edges** -- multiple constraints may share the same member nodes.
  * **Sub-primitive nodes** -- constraints are often applied to a *specific point*
    on a primitive (an endpoint or centre). To represent these unambiguously the
    paper adds the sub-primitive as its own node, joined to its parent by a
    dedicated sub-primitive edge. This module represents that with
    :meth:`SketchGraph.add_subprimitive`, which links a child point node to its
    parent and marks the connecting edge as structural (not a designer constraint).

Beyond construction, the module offers the relational analyses the paper reports:
node degree, adjacency, multi-edge / loop / hyperedge detection, and an
approximate DOF budget (``total primitive DOF`` vs. ``DOF removed by constraints``,
Sec. 3.2 / Fig. 4). The DOF accounting draws primitive and constraint weights from
:mod:`reconstruction.sketchgraphs_taxonomy`; sub-primitive point nodes contribute
their own 2 DOF and each sub-primitive edge removes 2 (pinning the point onto its
parent), consistent with the paper's endpoint modelling.

This is deliberately distinct from the root-level ``constraints.ConstraintGraph``:
that class is a rank-style DOF *solver* over the ``cisp.ops`` abstract model (4
primitive / 8 constraint types, no arcs, no sub-primitive nodes, no hyperedges).
Here we model the *relational structure* SketchGraphs actually stores -- the full
primitive/constraint taxonomy, loops, hyperedges and sub-primitive nodes -- and
expose it for graph-level analysis and sequence extraction.

Pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.reconstruction.sketch import constraint_taxonomy as tax

# DOF contributed by a sub-primitive point node, and removed by the structural
# edge that pins it onto its parent primitive.
_SUBPRIM_DOF = 2


@dataclass(frozen=True)
class PrimitiveNode:
    """A node in the constraint graph.

    ``ptype`` is a SketchGraphs primitive type (or ``'point'`` for a sub-primitive
    endpoint/centre node). ``is_construction`` mirrors the paper's ``isConstruction``
    boolean (True -> reference-only geometry, not physically realized). ``parent``
    is set for sub-primitive nodes and names the primitive they belong to.
    ``dof`` overrides the taxonomy DOF (used for splines with variable DOF).
    """

    node_id: str
    ptype: str
    is_construction: bool = False
    is_subprimitive: bool = False
    parent: Optional[str] = None
    dof: Optional[int] = None


@dataclass(frozen=True)
class ConstraintEdge:
    """A constraint edge over one or more member nodes.

    ``members`` is the ordered tuple of node ids the constraint acts on. ``ctype``
    is a SketchGraphs constraint type. ``is_structural`` marks a sub-primitive edge
    (a graph-internal link, not a designer constraint). ``value`` carries the
    numeric quantity of a dimensional constraint (length/angle).
    """

    ctype: str
    members: Tuple[str, ...]
    is_structural: bool = False
    value: Optional[float] = None

    @property
    def kind(self) -> str:
        """``'loop'`` / ``'edge'`` / ``'hyperedge'`` by member count."""
        return tax.classify_edge(len(self.members))


@dataclass
class DofBudget:
    """Approximate DOF accounting for a sketch (Sec. 3.2 / Fig. 4).

    ``primitive_dof`` is the total DOF of all primitive nodes; ``removed_dof`` is
    the DOF removed by constraints that carry a fixed weight (``mirror`` /
    ``projected`` are skipped and counted in ``variable_constraints``).
    ``remaining_dof`` is the signed difference (may go negative if over-determined).
    """

    primitive_dof: int
    removed_dof: int
    remaining_dof: int
    variable_constraints: int


class SketchGraph:
    """A geometric constraint multi-hypergraph for one CAD sketch."""

    def __init__(self) -> None:
        self._nodes: Dict[str, PrimitiveNode] = {}
        self._order: List[str] = []              # node insertion order
        self._edges: List[ConstraintEdge] = []

    # -- construction -------------------------------------------------------
    def add_primitive(
        self,
        node_id: str,
        ptype: str,
        is_construction: bool = False,
        dof: Optional[int] = None,
    ) -> str:
        """Add a primitive node. ``dof`` is required for splines (variable DOF)."""
        if ptype not in tax.PRIMITIVE_SPECS:
            raise ValueError(f"unknown primitive type '{ptype}'")
        if node_id in self._nodes:
            raise ValueError(f"duplicate node '{node_id}'")
        if dof is None and tax.PRIMITIVE_SPECS[ptype].dof is None:
            raise ValueError(f"primitive '{ptype}' needs an explicit dof")
        self._nodes[node_id] = PrimitiveNode(
            node_id=node_id,
            ptype=ptype,
            is_construction=is_construction,
            dof=dof,
        )
        self._order.append(node_id)
        return node_id

    def add_subprimitive(self, node_id: str, parent: str) -> str:
        """Add a sub-primitive point node (an endpoint/centre) linked to ``parent``.

        Adds the node *and* the structural sub-primitive edge joining it to its
        parent primitive, exactly as the paper includes sub-primitives as separate
        nodes to disambiguate point-level constraints.
        """
        if parent not in self._nodes:
            raise KeyError(f"unknown parent primitive '{parent}'")
        if node_id in self._nodes:
            raise ValueError(f"duplicate node '{node_id}'")
        self._nodes[node_id] = PrimitiveNode(
            node_id=node_id,
            ptype="point",
            is_subprimitive=True,
            parent=parent,
        )
        self._order.append(node_id)
        self._edges.append(
            ConstraintEdge("coincident", (parent, node_id), is_structural=True)
        )
        return node_id

    def add_constraint(
        self,
        ctype: str,
        members: Sequence[str],
        value: Optional[float] = None,
    ) -> int:
        """Add a constraint edge over ``members``; returns its edge index.

        The member count is validated against the constraint's Appendix-B schemata
        (member arities), so a 2-node ``mirror`` or a 3-node ``coincident`` is
        rejected.
        """
        if ctype not in tax.CONSTRAINT_SPECS:
            raise ValueError(f"unknown constraint type '{ctype}'")
        members = tuple(members)
        if not members:
            raise ValueError("a constraint needs at least one member")
        for m in members:
            if m not in self._nodes:
                raise KeyError(f"unknown member node '{m}'")
        spec = tax.CONSTRAINT_SPECS[ctype]
        if len(members) not in spec.member_arities:
            raise ValueError(
                f"constraint '{ctype}' takes {spec.member_arities} members, "
                f"got {len(members)}"
            )
        self._edges.append(ConstraintEdge(ctype, members, value=value))
        return len(self._edges) - 1

    # -- accessors ----------------------------------------------------------
    @property
    def nodes(self) -> Tuple[PrimitiveNode, ...]:
        return tuple(self._nodes[n] for n in self._order)

    @property
    def edges(self) -> Tuple[ConstraintEdge, ...]:
        return tuple(self._edges)

    @property
    def node_order(self) -> Tuple[str, ...]:
        return tuple(self._order)

    def node(self, node_id: str) -> PrimitiveNode:
        return self._nodes[node_id]

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def constraint_edges(self) -> Tuple[ConstraintEdge, ...]:
        """Designer constraint edges (excludes structural sub-primitive edges)."""
        return tuple(e for e in self._edges if not e.is_structural)

    # -- relational analysis ------------------------------------------------
    def degree(self, node_id: str) -> int:
        """Number of edge-incidences on a node (a loop counts once)."""
        if node_id not in self._nodes:
            raise KeyError(f"unknown node '{node_id}'")
        return sum(1 for e in self._edges if node_id in e.members)

    def degrees(self) -> Dict[str, int]:
        return {n: self.degree(n) for n in self._order}

    def neighbors(self, node_id: str) -> Tuple[str, ...]:
        """Distinct other nodes sharing an edge with ``node_id`` (sorted)."""
        if node_id not in self._nodes:
            raise KeyError(f"unknown node '{node_id}'")
        out = set()
        for e in self._edges:
            if node_id in e.members:
                out.update(m for m in e.members if m != node_id)
        return tuple(sorted(out))

    def loops(self) -> Tuple[ConstraintEdge, ...]:
        return tuple(e for e in self._edges if e.kind == "loop")

    def hyperedges(self) -> Tuple[ConstraintEdge, ...]:
        return tuple(e for e in self._edges if e.kind == "hyperedge")

    def has_multi_edges(self) -> bool:
        """True if two edges share the exact same member set (order-independent)."""
        seen = set()
        for e in self._edges:
            key = frozenset(e.members)
            if key in seen:
                return True
            seen.add(key)
        return False

    def is_multigraph(self) -> bool:
        """True if the graph has loops, hyperedges or multi-edges."""
        return bool(self.loops()) or bool(self.hyperedges()) or self.has_multi_edges()

    # -- DOF budget ---------------------------------------------------------
    def _node_dof(self, node: PrimitiveNode) -> int:
        if node.is_subprimitive:
            return _SUBPRIM_DOF
        if node.dof is not None:
            return node.dof
        return tax.primitive_dof(node.ptype)

    def dof_budget(self) -> DofBudget:
        """Approximate DOF accounting for the sketch (Sec. 3.2 / Fig. 4).

        Sums primitive DOF and subtracts the DOF removed by fixed-weight
        constraints (sub-primitive structural edges remove ``2`` each; ``mirror``
        and ``projected`` are variable/external and counted separately).
        """
        primitive_dof = sum(self._node_dof(n) for n in self._nodes.values())
        removed = 0
        variable = 0
        for e in self._edges:
            if e.is_structural:
                removed += _SUBPRIM_DOF
                continue
            w = tax.CONSTRAINT_SPECS[e.ctype].dof_removed
            if w is None:
                variable += 1
            else:
                removed += w
        return DofBudget(
            primitive_dof=primitive_dof,
            removed_dof=removed,
            remaining_dof=primitive_dof - removed,
            variable_constraints=variable,
        )


def build_from_sketch(
    primitives: Sequence[Tuple[str, str]],
    constraints: Sequence[Tuple[str, Sequence[str]]],
) -> SketchGraph:
    """Build a :class:`SketchGraph` from a flat primitive/constraint description.

    ``primitives`` is a sequence of ``(node_id, ptype)`` and ``constraints`` a
    sequence of ``(ctype, members)``. A convenience for the common case with no
    sub-primitive nodes or construction flags.
    """
    g = SketchGraph()
    for node_id, ptype in primitives:
        g.add_primitive(node_id, ptype)
    for ctype, members in constraints:
        g.add_constraint(ctype, members)
    return g
