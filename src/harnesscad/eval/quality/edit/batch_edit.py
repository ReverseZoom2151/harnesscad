"""batch_edit — semantic, query-driven bulk edits over a HarnessCAD part.

The "find-and-replace" of parametric CAD: instead of editing one feature at a
time, *query the semantic feature graph by a predicate* ("every hole under
5 mm", "all fillets", "chamfers on this body") and emit exactly one CISP op per
match — returned as a single, atomic, reviewable edit list.

This module deliberately **does not apply** anything. It is pure planning: it
selects the matching nodes and turns each into an :class:`cisp.ops.Op` via a
caller-supplied ``op_template`` (matched node -> Op). The caller reviews the
:class:`BatchEdit` (``matches`` / ``ops`` / ``summary``) and decides whether to
push those ops through the backend — keeping edit and review separate, and
leaving provenance (the op-DAG) the single place mutations happen.

Common predicates are provided (:func:`by_type`, :func:`by_param_threshold`,
:func:`all_of`, :func:`any_of`, :func:`negate`) but a predicate is just any
callable ``FeatureNode -> bool``, so arbitrary selection logic is welcome.

Pure, stdlib-only, deterministic: matches follow feature-graph node order.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from harnesscad.core.cisp.ops import Op
from harnesscad.eval.quality.graph.feature_graph import (
    FeatureGraph, FeatureNode, build_feature_graph,
)

Predicate = Callable[[FeatureNode], bool]
OpTemplate = Callable[[FeatureNode], Optional[Op]]

_COMPARATORS: Dict[str, Callable[[Any, Any], bool]] = {
    "<": operator.lt, "<=": operator.le, "lt": operator.lt, "le": operator.le,
    ">": operator.gt, ">=": operator.ge, "gt": operator.gt, "ge": operator.ge,
    "==": operator.eq, "eq": operator.eq, "=": operator.eq,
    "!=": operator.ne, "ne": operator.ne,
}


# --------------------------------------------------------------------------- #
# Predicate factories
# --------------------------------------------------------------------------- #
def by_type(node_type: str) -> Predicate:
    """Match nodes of a given feature-graph type (e.g. ``by_type('hole')``)."""
    def _pred(node: FeatureNode) -> bool:
        return node.type == node_type
    return _pred


def by_param_threshold(param: str, comparator: str, value: Any) -> Predicate:
    """Match nodes whose ``params[param]`` compares to ``value``.

    ``comparator`` is one of ``< <= > >= == !=`` (or their word forms). Nodes
    lacking the parameter (or carrying a non-comparable value) never match.
    """
    cmp = _COMPARATORS.get(comparator)
    if cmp is None:
        raise ValueError(f"unknown comparator {comparator!r}")

    def _pred(node: FeatureNode) -> bool:
        if param not in node.params:
            return False
        actual = node.params.get(param)
        if actual is None:
            return False
        try:
            return bool(cmp(actual, value))
        except TypeError:
            return False
    return _pred


def all_of(*predicates: Predicate) -> Predicate:
    """Match nodes satisfying *every* predicate (logical AND)."""
    preds = list(predicates)

    def _pred(node: FeatureNode) -> bool:
        return all(p(node) for p in preds)
    return _pred


def any_of(*predicates: Predicate) -> Predicate:
    """Match nodes satisfying *any* predicate (logical OR)."""
    preds = list(predicates)

    def _pred(node: FeatureNode) -> bool:
        return any(p(node) for p in preds)
    return _pred


def negate(predicate: Predicate) -> Predicate:
    """Match nodes the wrapped predicate does *not* match (logical NOT)."""
    def _pred(node: FeatureNode) -> bool:
        return not predicate(node)
    return _pred


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #
@dataclass
class BatchEdit:
    """One atomic, reviewable batch edit: the matches and the ops they yield.

    ``matches`` are the selected :class:`FeatureNode` (graph order); ``ops`` are
    the emitted :class:`cisp.ops.Op` (one per match whose template returned an
    op — a template may return ``None`` to skip). Nothing has been applied.
    """

    matches: List[FeatureNode] = field(default_factory=list)
    ops: List[Op] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ops)

    @property
    def is_empty(self) -> bool:
        return not self.ops

    def to_dict(self) -> dict:
        return {
            "matches": [n.to_dict() for n in self.matches],
            "ops": [o.to_dict() for o in self.ops],
            "summary": dict(self.summary),
        }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def batch_edit(featuregraph_or_backend: Any,
               predicate: Predicate,
               op_template: OpTemplate) -> BatchEdit:
    """Select graph nodes by ``predicate`` and emit one op per match.

    ``featuregraph_or_backend`` is a :class:`quality.featuregraph.FeatureGraph`,
    or a backend / op-DAG from which one is built. ``predicate`` is any
    ``FeatureNode -> bool``; ``op_template`` maps a matched node to an
    :class:`cisp.ops.Op` (or ``None`` to skip that match). The result is a single
    reviewable :class:`BatchEdit`; nothing is applied.
    """
    graph = _as_graph(featuregraph_or_backend)

    matches: List[FeatureNode] = [n for n in graph.nodes if predicate(n)]

    ops: List[Op] = []
    matched_with_op: List[FeatureNode] = []
    for node in matches:
        op = op_template(node)
        if op is None:
            continue
        if not isinstance(op, Op):
            raise TypeError(
                f"op_template must return a cisp.ops.Op or None, got "
                f"{type(op).__name__}")
        ops.append(op)
        matched_with_op.append(node)

    type_counts: Dict[str, int] = {}
    for node in matches:
        type_counts[node.type] = type_counts.get(node.type, 0) + 1

    summary = {
        "match_count": len(matches),
        "op_count": len(ops),
        "skipped": len(matches) - len(ops),
        "types": dict(sorted(type_counts.items())),
        "match_ids": [n.id for n in matches],
    }
    return BatchEdit(matches=matches, ops=ops, summary=summary)


def _as_graph(obj: Any) -> FeatureGraph:
    if isinstance(obj, FeatureGraph):
        return obj
    # Duck-type: something already exposing graph nodes/find is graph-like.
    if hasattr(obj, "nodes") and hasattr(obj, "find") and not hasattr(obj, "ops"):
        return obj  # type: ignore[return-value]
    return build_feature_graph(obj)
