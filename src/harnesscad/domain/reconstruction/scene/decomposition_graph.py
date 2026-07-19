"""Hierarchical geometry-aware decomposition graph.

A hierarchical, geometry-aware graph representation for text-to-CAD. The model
is trained, but its **intermediate representation** is a deterministic
structure: a top-down assembly decomposition whose multi-level nodes are
parts/components and whose edges are explicit geometric constraints, serialised
to a layered structured-text form::

    # Layer 0
    || microwave_oven | Composite of Door + Body ||
    # Layer 1
    || Door | Composite of Door_window + Control_panel | Align(XYZ) Door.back_face to Body.front_face ||

This module implements the representation, its serialisation/parse round-trip,
and graph-fidelity metrics: node-level accuracy, hierarchy-level accuracy, and
Geometric Constraint Satisfaction (GCS). Deterministic and stdlib-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Constraint",
    "Node",
    "DecompositionGraph",
    "serialize",
    "parse",
    "node_accuracy",
    "hierarchy_accuracy",
    "geometric_constraint_satisfaction",
]


@dataclass(frozen=True)
class Constraint:
    """A geometric constraint edge, e.g. ``Align(XYZ) Door.back_face to Body.front_face``."""

    kind: str            # e.g. "Align(XYZ)"
    src_ref: str         # e.g. "Door.back_face"
    dst_ref: str         # e.g. "Body.front_face"

    def render(self) -> str:
        return f"{self.kind} {self.src_ref} to {self.dst_ref}"


@dataclass(frozen=True)
class Node:
    """One decomposition node: name, layer depth, descriptor, and constraints."""

    name: str
    layer: int
    descriptor: str
    constraints: Tuple[Constraint, ...] = ()

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be >= 0")


@dataclass
class DecompositionGraph:
    """A multi-level decomposition graph keyed by node name."""

    nodes: List[Node] = field(default_factory=list)

    def by_name(self) -> Dict[str, Node]:
        return {n.name: n for n in self.nodes}

    def max_layer(self) -> int:
        return max((n.layer for n in self.nodes), default=0)

    def all_constraints(self) -> List[Tuple[str, Constraint]]:
        return [(n.name, c) for n in self.nodes for c in n.constraints]


def serialize(graph: DecompositionGraph) -> str:
    """Render the graph to the layered structured-text form."""
    lines: List[str] = []
    for layer in range(graph.max_layer() + 1):
        layer_nodes = [n for n in graph.nodes if n.layer == layer]
        if not layer_nodes:
            continue
        lines.append(f"# Layer {layer}")
        for n in layer_nodes:
            parts = [n.name, n.descriptor]
            if n.constraints:
                parts.append("; ".join(c.render() for c in n.constraints))
            lines.append("|| " + " | ".join(parts) + " ||")
    return "\n".join(lines)


_LAYER_RE = re.compile(r"^#\s*Layer\s+(\d+)\s*$")
_NODE_RE = re.compile(r"^\|\|\s*(.*?)\s*\|\|$")
_CONS_RE = re.compile(r"^(\S+)\s+(\S+)\s+to\s+(\S+)$")


def _parse_constraint(text: str) -> Constraint:
    m = _CONS_RE.match(text.strip())
    if not m:
        raise ValueError(f"cannot parse constraint {text!r}")
    return Constraint(kind=m.group(1), src_ref=m.group(2), dst_ref=m.group(3))


def parse(text: str) -> DecompositionGraph:
    """Parse the structured-text form back into a :class:`DecompositionGraph`."""
    layer = 0
    nodes: List[Node] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lm = _LAYER_RE.match(line)
        if lm:
            layer = int(lm.group(1))
            continue
        nm = _NODE_RE.match(line)
        if not nm:
            continue
        fields = [f.strip() for f in nm.group(1).split("|")]
        name = fields[0]
        descriptor = fields[1] if len(fields) > 1 else ""
        constraints: List[Constraint] = []
        if len(fields) > 2 and fields[2]:
            for chunk in fields[2].split(";"):
                chunk = chunk.strip()
                if chunk:
                    constraints.append(_parse_constraint(chunk))
        nodes.append(Node(name=name, layer=layer, descriptor=descriptor,
                          constraints=tuple(constraints)))
    return DecompositionGraph(nodes=nodes)


def node_accuracy(pred: DecompositionGraph, ref: DecompositionGraph) -> float:
    """Fraction of reference nodes whose name AND descriptor are reproduced."""
    if not ref.nodes:
        raise ValueError("reference graph has no nodes")
    p = pred.by_name()
    hits = sum(
        1 for n in ref.nodes
        if n.name in p and p[n.name].descriptor == n.descriptor
    )
    return hits / len(ref.nodes)


def hierarchy_accuracy(pred: DecompositionGraph, ref: DecompositionGraph) -> float:
    """Fraction of reference nodes placed at the correct layer depth."""
    if not ref.nodes:
        raise ValueError("reference graph has no nodes")
    p = pred.by_name()
    hits = sum(1 for n in ref.nodes if n.name in p and p[n.name].layer == n.layer)
    return hits / len(ref.nodes)


def geometric_constraint_satisfaction(
    pred: DecompositionGraph, ref: DecompositionGraph
) -> float:
    """GCS: fraction of reference constraints present (exactly) in the prediction.

    Returns 1.0 when the reference declares no constraints.
    """
    ref_cons = ref.all_constraints()
    if not ref_cons:
        return 1.0
    pred_cons = set(pred.all_constraints())
    hits = sum(1 for c in ref_cons if c in pred_cons)
    return hits / len(ref_cons)
