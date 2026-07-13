"""STEP entity-reference graph and validity checks.

The paper stresses that a STEP file is intrinsically *graph-structured*: entity
instances form a directed acyclic graph (DAG) via cross-references, and "small
errors in entity ordering or identifier usage can render the entire file
invalid". Because STEP-LLM generates the file text *directly* (no CAD kernel
enforcing validity), a deterministic structural check is essential to decide
whether a generated file is even well-formed before any geometric evaluation.

This module builds the reference graph over a :class:`~formats.stepllm_parser.
StepFile` and exposes the checks a generated file must pass:

  * **no dangling references** - every ``#M`` resolves to a defined instance;
  * **acyclicity** - the reference relation is a DAG (topological order exists);
  * **required roots present** - at least one top-level solid/shell root entity;
  * **reachability** - which instances are reachable from the roots (dead
    instances are reported but not fatal);
  * **schema arity/kind** conformance for the known-entity subset
    (:mod:`formats.stepllm_schema`).

It complements :mod:`reconstruction.cadparser_brep_graph` (which builds a
face/edge/coedge adjacency graph for a learned convolution): here the graph is
the raw part-21 *instance*-reference graph used for file validity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harnesscad.io.formats.stepllm_parser import StepFile, entity_refs
from harnesscad.io.formats.stepllm_schema import check_attributes

# Entity types that act as top-level B-rep roots (nothing references them).
ROOT_TYPES: frozenset = frozenset({
    "MANIFOLD_SOLID_BREP", "CLOSED_SHELL", "OPEN_SHELL",
    "BREP_WITH_VOIDS", "SHELL_BASED_SURFACE_MODEL",
    "GEOMETRIC_CURVE_SET", "FACETED_BREP",
})


@dataclass
class ReferenceGraph:
    """Adjacency of a STEP file's instance-reference DAG."""

    out_edges: dict = field(default_factory=dict)   # id -> [referenced ids]
    in_degree: dict = field(default_factory=dict)   # id -> count of referrers

    def successors(self, ent_id: int) -> list:
        return self.out_edges.get(ent_id, [])


def build_graph(step: StepFile) -> ReferenceGraph:
    """Build the reference graph; unresolved targets are simply omitted here."""

    ids = set(step.entities)
    out_edges: dict = {}
    in_degree: dict = {i: 0 for i in step.order}
    for ent_id in step.order:
        refs = entity_refs(step.entities[ent_id])
        out_edges[ent_id] = refs
        for target in refs:
            if target in in_degree:
                in_degree[target] += 1
    return ReferenceGraph(out_edges=out_edges, in_degree=in_degree)


def dangling_references(step: StepFile) -> list:
    """Return ``(from_id, missing_id)`` pairs for every unresolved reference."""

    ids = set(step.entities)
    out: list = []
    for ent_id in step.order:
        for target in entity_refs(step.entities[ent_id]):
            if target not in ids:
                out.append((ent_id, target))
    return out


def roots(step: StepFile) -> list:
    """Instances of a root B-rep type that nothing references."""

    graph = build_graph(step)
    return [i for i in step.order
            if step.entities[i].keyword in ROOT_TYPES
            and graph.in_degree.get(i, 0) == 0]


def topological_order(step: StepFile) -> list:
    """Kahn topological order (referrers before referents).

    Raises :class:`ValueError` if the reference relation contains a cycle.
    Assumes references resolve; check :func:`dangling_references` first.
    """

    graph = build_graph(step)
    # Referrer -> referent edges; order so that a node precedes its targets.
    remaining = dict(graph.in_degree)
    queue = [i for i in step.order if remaining.get(i, 0) == 0]
    order: list = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for target in graph.successors(node):
            if target not in remaining:
                continue
            remaining[target] -= 1
            if remaining[target] == 0:
                queue.append(target)
    if len(order) != len(step.order):
        raise ValueError("reference graph contains a cycle")
    return order


def is_acyclic(step: StepFile) -> bool:
    try:
        topological_order(step)
        return True
    except ValueError:
        return False


def reachable(step: StepFile, start=None) -> set:
    """Set of ids reachable from ``start`` (defaults to the detected roots)."""

    graph = build_graph(step)
    if start is None:
        start = roots(step)
    seen: set = set()
    stack = list(start)
    while stack:
        node = stack.pop()
        if node in seen or node not in step.entities:
            continue
        seen.add(node)
        stack.extend(graph.successors(node))
    return seen


def unreachable(step: StepFile, start=None) -> list:
    seen = reachable(step, start)
    return [i for i in step.order if i not in seen]


@dataclass
class ValidityReport:
    dangling: list = field(default_factory=list)
    acyclic: bool = True
    roots: list = field(default_factory=list)
    unreachable: list = field(default_factory=list)
    schema_problems: list = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """A file is structurally valid iff references resolve, the graph is a
        DAG, at least one root exists, and no schema arity/kind violations."""

        return (not self.dangling and self.acyclic and bool(self.roots)
                and not self.schema_problems)

    def summary(self) -> str:
        parts = [
            f"valid={self.valid}",
            f"dangling={len(self.dangling)}",
            f"acyclic={self.acyclic}",
            f"roots={len(self.roots)}",
            f"unreachable={len(self.unreachable)}",
            f"schema_problems={len(self.schema_problems)}",
        ]
        return ", ".join(parts)


def validate(step: StepFile) -> ValidityReport:
    """Run the full structural validity check over a parsed STEP file."""

    dangling = dangling_references(step)
    acyclic = is_acyclic(step)
    rts = roots(step)
    unreach = unreachable(step) if rts else list(step.order)
    schema_problems: list = []
    for ent_id in step.order:
        schema_problems.extend(check_attributes(step.entities[ent_id]))
    return ValidityReport(
        dangling=dangling,
        acyclic=acyclic,
        roots=rts,
        unreachable=unreach,
        schema_problems=schema_problems,
    )
