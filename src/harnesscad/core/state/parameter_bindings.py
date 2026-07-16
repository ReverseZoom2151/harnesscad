"""Parameter bindings: drive feature parameters from solver results and sibling nodes.

OpenCAD's feature tree (``opencad_tree/models.py`` ``ParameterBinding`` plus the
two-pass resolution in ``opencad_tree/service.py``) closes the loop between the
2D constraint solver and the parametric feature DAG: a feature parameter can be
*driven* rather than typed, either

* from another node's payload (``source="node"``, a dotted ``source_path`` into
  ``{"shape_id": ..., "parameters": {...}}``),
* from a cached solver result (``source="solver"``, a dotted path into the solved
  sketch payload -- e.g. ``entities.l1.x2``), or
* through a safe arithmetic ``expression`` whose namespace is built from the
  node's literal parameters plus the leaf values of the path bindings.

Resolution is two-pass, and the order is the point: pass one resolves every
path binding; a path binding that also carries an expression does NOT set the
parameter directly -- it contributes its resolved value to the expression
namespace under the *leaf name* of its path. Pass two evaluates the expression
bindings against that namespace in declaration order, and each result becomes
visible to the expressions after it, so ``hole_offset = width / 2`` can feed
``slot_end = hole_offset + 8``.

The other half of the idea is *invalidation*: when a new solver result lands
(``apply_solver_result``), every node holding a matching solver binding gets its
parameter rewritten; if the value actually changed the node and all its
descendants go stale, and -- the subtle rule -- any node whose expression
mentions a parameter name that just changed goes stale too, found by extracting
the free symbols of each expression (OpenCAD
``FeatureTreeService._invalidate_expression_dependents``).

The harness already has the two halves this composes: the safe expression
evaluator and dependency-ordered parameter table
(:mod:`harnesscad.domain.numeric.parameter_expressions`) and the stale-propagating
feature DAG (:mod:`harnesscad.core.state.feature_tree`). What it lacked was the
binding layer between them -- the record that says "this op parameter is not a
literal, it is driven by that solver variable through this expression". This
module supplies exactly that, without touching either neighbour: bindings live
in a :class:`BindingSet` keyed by node id, so ``feature_tree.FeatureNode`` stays
unchanged.

Deterministic: dict iteration is never relied on for ordering (bindings resolve
in declaration order; staled sets come back sorted); no clock, no randomness.

Public API
----------
``ParameterBinding``, ``BindingSet``, ``SolverCache``
``resolve_path``, ``cast_value``, ``runtime_parameters``
``apply_solver_result``, ``expression_dependents``, ``SolverApplyReport``
``BindingError``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from harnesscad.core.state.feature_tree import (
    STALE,
    SUPPRESSED,
    FeatureTree,
    descendants,
)
from harnesscad.domain.numeric.parameter_expressions import (
    ExpressionError,
    evaluate,
    extract_symbols,
)

__all__ = [
    "BindingError",
    "ParameterBinding",
    "BindingSet",
    "SolverCache",
    "SolverApplyReport",
    "resolve_path",
    "cast_value",
    "runtime_parameters",
    "apply_solver_result",
    "expression_dependents",
]

_SOURCES = ("node", "solver", "")
_CASTS = ("int", "float", "bool", "string")


class BindingError(ValueError):
    """A binding is malformed or cannot be resolved."""


@dataclass(frozen=True)
class ParameterBinding:
    """One driven parameter on one feature node.

    ``parameter``   -- the node parameter this binding writes.
    ``source``      -- ``"node"`` | ``"solver"`` | ``""`` (pure expression).
    ``source_key``  -- the source node id or the solver-cache sketch id.
    ``source_path`` -- dotted path into the source payload (``entities.l1.x2``).
    ``cast_as``     -- optional cast (``int``/``float``/``bool``/``string``).
    ``expression``  -- optional arithmetic expression; when present the resolved
                       path value feeds the expression namespace under the leaf
                       name of ``source_path`` instead of writing the parameter.
    """

    parameter: str
    source: str = ""
    source_key: str = ""
    source_path: str = ""
    cast_as: Optional[str] = None
    expression: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.parameter:
            raise BindingError("Binding requires a target parameter name.")
        if self.source not in _SOURCES:
            raise BindingError(
                "Unknown binding source %r (expected 'node', 'solver' or '')."
                % (self.source,)
            )
        if self.cast_as is not None and self.cast_as not in _CASTS:
            raise BindingError("Unknown cast %r." % (self.cast_as,))
        if self.source and not self.source_key:
            raise BindingError(
                "Binding on '%s' names source %r but no source_key."
                % (self.parameter, self.source)
            )
        if not self.source and self.expression is None:
            raise BindingError(
                "Binding on '%s' has neither a source nor an expression."
                % (self.parameter,)
            )

    @property
    def leaf(self) -> str:
        """Namespace name a path-fed expression binding contributes under."""
        if self.source_path:
            return self.source_path.rsplit(".", 1)[-1]
        return self.parameter


class SolverCache:
    """Cached solver payloads keyed by sketch id (OpenCAD ``solver_cache``)."""

    def __init__(self) -> None:
        self._payloads: Dict[str, Dict[str, object]] = {}

    def put(self, sketch_id: str, payload: Mapping[str, object]) -> None:
        self._payloads[sketch_id] = dict(payload)

    def get(self, sketch_id: str) -> Optional[Dict[str, object]]:
        return self._payloads.get(sketch_id)

    def ids(self) -> List[str]:
        return sorted(self._payloads)


class BindingSet:
    """Bindings for a whole tree, keyed by node id.

    Kept outside :class:`~harnesscad.core.state.feature_tree.FeatureNode` so the
    DAG module stays binding-agnostic; a caller that persists trees serialises
    this alongside them.
    """

    def __init__(self) -> None:
        self._by_node: Dict[str, List[ParameterBinding]] = {}

    def add(self, node_id: str, binding: ParameterBinding) -> None:
        self._by_node.setdefault(node_id, []).append(binding)

    def for_node(self, node_id: str) -> List[ParameterBinding]:
        return list(self._by_node.get(node_id, []))

    def node_ids(self) -> List[str]:
        return sorted(self._by_node)

    def items(self) -> List[Tuple[str, List[ParameterBinding]]]:
        return [(nid, list(bs)) for nid, bs in sorted(self._by_node.items())]


def resolve_path(payload: Mapping[str, object], path: str) -> object:
    """Walk a dotted *path* through nested mappings; raise on a miss."""
    current: object = payload
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            raise BindingError("Unable to resolve path '%s'." % path)
    return current


def cast_value(value: object, cast_as: Optional[str]) -> object:
    """Apply an optional cast; unknown casts were rejected at construction."""
    if cast_as == "int":
        return int(value)  # type: ignore[arg-type]
    if cast_as == "float":
        return float(value)  # type: ignore[arg-type]
    if cast_as == "bool":
        return bool(value)
    if cast_as == "string":
        return str(value)
    return value


def _resolve_source(
    binding: ParameterBinding,
    tree: FeatureTree,
    solver_cache: Optional[SolverCache],
) -> Tuple[bool, object]:
    """Resolve a path binding's raw value. Returns ``(found, value)``."""
    if binding.source == "node":
        source_node = tree.nodes.get(binding.source_key)
        if source_node is None:
            return False, None
        payload: Dict[str, object] = {
            "shape_id": source_node.shape_id,
            "parameters": source_node.parameters,
        }
        try:
            return True, resolve_path(payload, binding.source_path)
        except BindingError:
            return False, None
    if binding.source == "solver":
        if solver_cache is None:
            return False, None
        cached = solver_cache.get(binding.source_key)
        if cached is None:
            return False, None
        try:
            return True, resolve_path(cached, binding.source_path)
        except BindingError:
            return False, None
    return False, None


def runtime_parameters(
    tree: FeatureTree,
    node_id: str,
    bindings: BindingSet,
    solver_cache: Optional[SolverCache] = None,
) -> Dict[str, object]:
    """Resolve a node's effective parameters through its bindings (two passes).

    Pass one resolves path bindings: a binding without an expression writes the
    parameter directly (cast applied); a binding *with* an expression feeds the
    resolved value into the expression namespace under its path's leaf name.
    Pass two evaluates expression bindings in declaration order against a
    namespace of every numeric parameter plus the pass-one leaf values; each
    result becomes available to the expressions after it.

    Unresolvable path bindings are skipped silently (the OpenCAD behaviour: a
    missing solver result must not block a rebuild that does not need it); an
    invalid *expression* raises, because a wrong formula is a modelling error,
    not a missing input.
    """
    node = tree.nodes.get(node_id)
    if node is None:
        raise BindingError("Unknown feature node '%s'." % node_id)

    params: Dict[str, object] = dict(node.parameters)
    expr_extras: Dict[str, float] = {}

    node_bindings = bindings.for_node(node_id)

    for binding in node_bindings:
        if not binding.source:
            continue
        found, raw = _resolve_source(binding, tree, solver_cache)
        if not found:
            continue
        if binding.expression is not None:
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                expr_extras[binding.leaf] = float(raw)
        else:
            params[binding.parameter] = cast_value(raw, binding.cast_as)

    namespace: Dict[str, float] = {
        key: float(value)
        for key, value in params.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    namespace.update(expr_extras)

    for binding in node_bindings:
        if binding.expression is None:
            continue
        try:
            value = evaluate(binding.expression, namespace)
        except ExpressionError as exc:
            raise BindingError(
                "Expression binding on '%s.%s' failed: %s"
                % (node_id, binding.parameter, exc)
            ) from exc
        params[binding.parameter] = cast_value(value, binding.cast_as)
        namespace[binding.parameter] = float(value)

    return params


def expression_dependents(
    bindings: BindingSet, changed_parameters: Set[str]
) -> List[str]:
    """Node ids whose expression bindings reference any changed parameter name.

    The OpenCAD ``_invalidate_expression_dependents`` rule: a parameter change
    invalidates not only the node it lives on but every node whose *formula*
    mentions that name -- found statically from the expression's free symbols,
    so no evaluation happens here. Expressions that fail to parse are ignored
    (they will raise at resolution time instead).
    """
    if not changed_parameters:
        return []
    hit: Set[str] = set()
    for node_id, node_bindings in bindings.items():
        for binding in node_bindings:
            if binding.expression is None:
                continue
            try:
                symbols = extract_symbols(binding.expression)
            except ExpressionError:
                continue
            if symbols & changed_parameters:
                hit.add(node_id)
                break
    return sorted(hit)


@dataclass
class SolverApplyReport:
    """What one solver result did to the tree."""

    tree: FeatureTree
    changed_nodes: List[str] = field(default_factory=list)
    changed_parameters: List[str] = field(default_factory=list)
    staled: List[str] = field(default_factory=list)


def _stale(tree: FeatureTree, node_id: str) -> None:
    node = tree.nodes[node_id]
    if node.status != SUPPRESSED:
        node.status = STALE
        node.shape_id = None


def apply_solver_result(
    tree: FeatureTree,
    bindings: BindingSet,
    solver_cache: SolverCache,
    sketch_id: str,
    solved_payload: Mapping[str, object],
) -> SolverApplyReport:
    """Land a solver result: cache it, rewrite driven parameters, stale movers.

    Non-mutating (works on a copy of *tree*; the cache IS updated -- it is the
    landing site). For every node holding a non-expression binding with
    ``source="solver"`` and a matching ``source_key``, the bound parameter is
    rewritten from the payload. Where the value actually changed, the node and
    all its descendants go stale, and so does every expression-dependent node
    (with its descendants). Expression bindings are left alone here -- they
    resolve at rebuild time via :func:`runtime_parameters`.
    """
    solver_cache.put(sketch_id, solved_payload)
    updated = tree.copy()

    changed_nodes: Set[str] = set()
    changed_parameters: Set[str] = set()

    for node_id in sorted(updated.nodes):
        node = updated.nodes[node_id]
        node_changed = False
        for binding in bindings.for_node(node_id):
            if binding.source != "solver" or binding.source_key != sketch_id:
                continue
            if binding.expression is not None:
                continue
            try:
                raw = resolve_path(solved_payload, binding.source_path)
            except BindingError:
                continue
            coerced = cast_value(raw, binding.cast_as)
            old = node.parameters.get(binding.parameter)
            node.parameters[binding.parameter] = coerced
            if old != coerced:
                node_changed = True
                changed_parameters.add(binding.parameter)
        if node_changed:
            changed_nodes.add(node_id)

    staled: Set[str] = set()
    for root in sorted(changed_nodes):
        _stale(updated, root)
        staled.add(root)
        for child_id in descendants(updated.nodes, root):
            _stale(updated, child_id)
            staled.add(child_id)

    for extra_root in expression_dependents(bindings, changed_parameters):
        if extra_root in staled or extra_root not in updated.nodes:
            continue
        _stale(updated, extra_root)
        staled.add(extra_root)
        for child_id in descendants(updated.nodes, extra_root):
            _stale(updated, child_id)
            staled.add(child_id)

    updated.revision += 1
    return SolverApplyReport(
        tree=updated,
        changed_nodes=sorted(changed_nodes),
        changed_parameters=sorted(changed_parameters),
        staled=sorted(staled),
    )


# ── selfcheck ───────────────────────────────────────────────────────


def selfcheck(verbose: bool = False) -> bool:
    """Exercise two-pass resolution and solver-result invalidation.

    Synthetic tree, no kernel, no solver: the solved payload is written by
    hand, so every expected value is known by construction.
    """
    from harnesscad.core.state.feature_tree import BUILT, FeatureNode

    checks: List[Tuple[str, bool]] = []

    tree = FeatureTree()
    tree.nodes["sk1"] = FeatureNode(
        id="sk1", operation="sketch", status=BUILT, shape_id="s-sk1",
        parameters={"width": 30.0},
    )
    tree.nodes["f1"] = FeatureNode(
        id="f1", operation="extrude", status=BUILT, shape_id="s-f1",
        parent_id="sk1", parameters={"depth": 8.0},
    )
    tree.nodes["f2"] = FeatureNode(
        id="f2", operation="fillet", status=BUILT, shape_id="s-f2",
        parent_id="f1", parameters={"radius": 1.0},
    )

    bindings = BindingSet()
    cache = SolverCache()
    cache.put("sk1", {"entities": {"l1": {"x2": 30.0}}})

    # Direct solver binding: f1.depth driven by the solved line end.
    bindings.add("f1", ParameterBinding(
        parameter="depth", source="solver", source_key="sk1",
        source_path="entities.l1.x2", cast_as="float",
    ))
    params = runtime_parameters(tree, "f1", bindings, cache)
    checks.append(("solver path binding", params["depth"] == 30.0))

    # Expression binding fed by a path leaf: radius = x2 / 10.
    bindings.add("f2", ParameterBinding(
        parameter="radius", source="solver", source_key="sk1",
        source_path="entities.l1.x2", expression="x2 / 10",
    ))
    params = runtime_parameters(tree, "f2", bindings, cache)
    checks.append(("expression from path leaf", params["radius"] == 3.0))

    # Chained expressions: later expression sees the earlier result.
    bindings.add("f2", ParameterBinding(
        parameter="setback", expression="radius * 2",
    ))
    params = runtime_parameters(tree, "f2", bindings, cache)
    checks.append(("chained expression", params["setback"] == 6.0))

    # Node-source binding.
    bindings.add("f1", ParameterBinding(
        parameter="stock_width", source="node", source_key="sk1",
        source_path="parameters.width",
    ))
    params = runtime_parameters(tree, "f1", bindings, cache)
    checks.append(("node path binding", params["stock_width"] == 30.0))

    # A missing solver payload skips silently.
    bindings.add("f1", ParameterBinding(
        parameter="ghost", source="solver", source_key="sk-missing",
        source_path="entities.q.x",
    ))
    params = runtime_parameters(tree, "f1", bindings, cache)
    checks.append(("missing payload skipped", "ghost" not in params))

    # New solver result: value changed -> f1 and descendant f2 stale.
    report = apply_solver_result(
        tree, bindings, cache, "sk1", {"entities": {"l1": {"x2": 42.0}}}
    )
    checks.append(("changed node recorded", report.changed_nodes == ["f1"]))
    checks.append(("descendants staled", set(report.staled) >= {"f1", "f2"}))
    checks.append((
        "stale status set",
        report.tree.nodes["f1"].status == STALE
        and report.tree.nodes["f2"].status == STALE,
    ))
    checks.append(("original untouched", tree.nodes["f1"].status == BUILT))
    checks.append((
        "parameter rewritten", report.tree.nodes["f1"].parameters["depth"] == 42.0,
    ))

    # Same result again: nothing changes, nothing stales.
    tree2 = report.tree
    report2 = apply_solver_result(
        tree2, bindings, cache, "sk1", {"entities": {"l1": {"x2": 42.0}}}
    )
    checks.append(("idempotent re-apply", report2.changed_nodes == []))

    # Expression-dependency invalidation is found statically.
    deps = expression_dependents(bindings, {"radius"})
    checks.append(("expression dependents", deps == ["f2"]))
    checks.append(("no false dependents", expression_dependents(bindings, {"zzz"}) == []))

    # Malformed bindings are refused at construction.
    try:
        ParameterBinding(parameter="p", source="wormhole", source_key="x")
        checks.append(("bad source refused", False))
    except BindingError:
        checks.append(("bad source refused", True))
    try:
        ParameterBinding(parameter="p")
        checks.append(("empty binding refused", False))
    except BindingError:
        checks.append(("empty binding refused", True))

    ok = all(passed for _, passed in checks)
    if verbose:
        for name, passed in checks:
            print("  %-28s %s" % (name, "ok" if passed else "FAIL"))
        print("parameter_bindings selfcheck: %s" % ("ok" if ok else "FAILED"))
    return ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.core.state.parameter_bindings",
        description="Solver/node-driven parameter bindings for the feature DAG.",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the synthetic binding self-check (no real data)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.selfcheck:
        return 0 if selfcheck(verbose=True) else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
