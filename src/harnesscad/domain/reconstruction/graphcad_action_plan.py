"""Graph-CAD stage 2: decomposition graph -> ordered CAD action sequence.

Graph-CAD's second stage turns the hierarchical decomposition graph into an
ordered CAD action sequence that "respects assembly structure and local
dependencies" before a third stage emits executable ``bpy`` code. The *model*
does that with a fine-tuned LLM, but the ordering it must respect is fully
determined by the graph: children exist before their parent consumes them,
boolean operands exist before the boolean, ``after`` / ``depends_on`` edges are
honoured, and a parent's ``assembly_order`` groups run one after another.

This module builds that ground-truth linearisation deterministically. It walks
the build waves of :mod:`reconstruction.graphcad_knowledge_graph`, expands
``pattern=`` template nodes into their instances via
:mod:`geometry.graphcad_pattern`, and emits one typed action per step:

* ``create``   -- instantiate a primitive / sketch extrusion leaf
* ``place``    -- apply a node's align / pos / orientation / rotation fields
* ``boolean``  -- subtract, union or intersect a tool with a target
* ``bevel``    -- fillet or chamfer a target
* ``assemble`` -- combine a composite node's children, group by group

``plan_actions`` guarantees every action's operands are already defined, and
``validate_plan`` re-checks that property independently so a *predicted* action
sequence can be scored against the graph it claims to realise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.graphcad_pattern import expand_pattern, parse_pattern
from harnesscad.domain.reconstruction.graphcad_knowledge_graph import (
    BOOLEAN_METHODS,
    GraphNode,
    KnowledgeGraph,
    build_waves,
)

__all__ = [
    "Action",
    "plan_actions",
    "render_plan",
    "validate_plan",
    "action_histogram",
]

_BEVEL_METHODS = ("bevel", "fillet", "chamfer")
_ASSEMBLY_METHODS = ("composite", "union", "assembly", "auto_connect")


@dataclass(frozen=True)
class Action:
    """One step of the CAD action sequence."""

    op: str
    node_id: str
    method: Optional[str] = None
    operands: Tuple[str, ...] = ()
    detail: Optional[str] = None

    def render(self) -> str:
        parts = [f"{self.op}({self.node_id}"]
        if self.method:
            parts.append(f", method={self.method}")
        if self.operands:
            parts.append(", operands=[" + ", ".join(self.operands) + "]")
        if self.detail:
            parts.append(f", {self.detail}")
        return "".join(parts) + ")"


def _placement_detail(node: GraphNode) -> Optional[str]:
    fields = []
    for key in ("align", "anchor", "pos", "connect", "orientation", "rotation"):
        value = getattr(node, key)
        if value is not None:
            fields.append(f"{key}={value}")
    return "; ".join(fields) if fields else None


def _instance_ids(node: GraphNode) -> Tuple[str, ...]:
    if node.pattern is None:
        return ()
    pattern = parse_pattern(
        node.pattern if "pattern" in node.pattern else f"pattern={node.pattern}"
    )
    return tuple(instance.instance_id for instance in expand_pattern(node.node_id, pattern))


def _ordered_children(graph: KnowledgeGraph, node: GraphNode) -> Tuple[str, ...]:
    """Children in ``assembly_order`` group order, declaration order otherwise."""
    children = [child.node_id for child in graph.children_of(node.node_id)]
    groups = node.assembly_groups()
    if not groups:
        return tuple(children)
    ordered: List[str] = []
    for group in groups:
        ordered.extend(item for item in group if item in children and item not in ordered)
    ordered.extend(item for item in children if item not in ordered)
    return tuple(ordered)


def plan_actions(graph: KnowledgeGraph) -> Tuple[Action, ...]:
    """Linearise the graph into a dependency-respecting CAD action sequence.

    Nodes are visited in build-wave order (children before parents). Each node
    contributes a ``create`` / ``boolean`` / ``bevel`` / ``assemble`` action
    plus, if it carries placement fields, a following ``place`` action. Pattern
    template nodes create every instance first, then place them.
    """
    index = graph.by_id()
    actions: List[Action] = []

    for wave in build_waves(graph):
        for node_id in wave:
            node = index[node_id]
            method = node.create_method
            children = _ordered_children(graph, node)

            if node.pattern is not None:
                instances = _instance_ids(node)
                for instance_id in instances:
                    actions.append(
                        Action("create", instance_id, method or "primitive",
                               detail=f"size={node.size}" if node.size else None)
                    )
                actions.append(
                    Action("assemble", node.node_id, "pattern", instances,
                           detail=f"pattern={node.pattern}")
                )
            elif method in BOOLEAN_METHODS:
                operands = tuple(
                    value for value in (node.target_id, node.tool_id) if value is not None
                )
                actions.append(Action("boolean", node.node_id, method, operands))
            elif method in _BEVEL_METHODS:
                operands = (node.tool_id,) if node.tool_id else ()
                actions.append(
                    Action("bevel", node.node_id, method, operands, node.constraint)
                )
            elif children:
                actions.append(
                    Action("assemble", node.node_id, method or "composite", children)
                )
            else:
                actions.append(
                    Action("create", node.node_id, method or "primitive",
                           detail=f"size={node.size}" if node.size else None)
                )

            detail = _placement_detail(node)
            if detail is not None:
                actions.append(Action("place", node.node_id, detail=detail))

    return tuple(actions)


def render_plan(actions: Sequence[Action]) -> str:
    """Render the action sequence as one numbered line per action."""
    return "\n".join(
        f"{index + 1}. {action.render()}" for index, action in enumerate(actions)
    )


def validate_plan(actions: Sequence[Action]) -> Tuple[str, ...]:
    """Check that every action's operands were defined by an earlier action.

    Also flags an id defined twice and a ``place`` / ``bevel`` acting on an id
    that was never created -- the two failure modes an out-of-order predicted
    sequence produces.
    """
    errors: List[str] = []
    defined: Dict[str, int] = {}

    for step, action in enumerate(actions, 1):
        for operand in action.operands:
            if operand not in defined:
                errors.append(
                    f"step {step}: {action.op} {action.node_id!r} uses undefined {operand!r}"
                )
        if action.op == "place":
            if action.node_id not in defined:
                errors.append(f"step {step}: place on undefined {action.node_id!r}")
            continue
        if action.node_id in defined:
            errors.append(
                f"step {step}: {action.node_id!r} was already defined at step "
                f"{defined[action.node_id]}"
            )
        else:
            defined[action.node_id] = step
    return tuple(errors)


def action_histogram(actions: Sequence[Action]) -> Dict[str, int]:
    """Count actions per op -- a cheap structural signature of a plan."""
    histogram: Dict[str, int] = {}
    for action in actions:
        histogram[action.op] = histogram.get(action.op, 0) + 1
    return histogram
