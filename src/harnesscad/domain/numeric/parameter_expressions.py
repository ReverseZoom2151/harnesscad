"""Safe parametric-expression evaluator and parameter table.

A feature tree drives every feature parameter from a *named parameter table*
whose entries may be either literal numbers or math expressions referencing
other parameters (``"thickness * 2 + clearance"``).  A safe evaluator walks a
Python ``ast`` and whitelists a small operator/function set, so no attribute
access, comprehension, subscript, lambda or arbitrary call can ever be
executed; bindings are then resolved in two passes so that an expression can
consume values produced by earlier bindings.

This module implements that idea as a self-contained, deterministic engine and
adds the piece such designs usually leave implicit: a *parameter table* that

* extracts the free symbols of every expression (``extract_symbols``),
* builds the parameter dependency graph and evaluates the table in topological
  order (so ``a = "b + 1"`` works regardless of insertion order),
* detects reference cycles and unknown symbols with precise error messages, and
* reports, for any parameter, the transitive set of parameters it depends on and
  the set that would become stale if it changed (the driver of feature-tree
  invalidation).

Deterministic: pure arithmetic and sorted graph traversal; no clock, no
randomness, no I/O.

Public API
----------
``evaluate(expr, namespace) -> float``
``extract_symbols(expr) -> set[str]``
``ParameterTable`` -- ``set``/``set_expr``/``evaluate_all``/``dependencies``/
``dependents``/``evaluation_order``
``ExpressionError``, ``CyclicParameterError``
"""

from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set

__all__ = [
    "ExpressionError",
    "CyclicParameterError",
    "SAFE_FUNCTIONS",
    "SAFE_CONSTANTS",
    "evaluate",
    "extract_symbols",
    "Parameter",
    "ParameterTable",
]


class ExpressionError(ValueError):
    """Raised when an expression cannot be parsed or evaluated safely."""


class CyclicParameterError(ExpressionError):
    """Raised when parameter expressions reference each other cyclically."""


_BINARY_OPS: Dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_UNARY_OPS: Dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

SAFE_FUNCTIONS: Dict[str, Callable[..., float]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "hypot": math.hypot,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "ceil": math.ceil,
    "floor": math.floor,
    "radians": math.radians,
    "degrees": math.degrees,
}

SAFE_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "tau": math.tau,
    "e": math.e,
}


def _eval_node(node: ast.AST, namespace: Mapping[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, namespace)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ExpressionError("Boolean constants are not supported")
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ExpressionError(
            "Unsupported constant type: %s" % type(node.value).__name__
        )

    if isinstance(node, ast.Name):
        name = node.id
        if name in namespace:
            return float(namespace[name])
        if name in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[name]
        raise ExpressionError("Unknown symbol '%s'" % name)

    if isinstance(node, ast.UnaryOp):
        fn = _UNARY_OPS.get(type(node.op))
        if fn is None:
            raise ExpressionError(
                "Unsupported unary operator: %s" % type(node.op).__name__
            )
        return float(fn(_eval_node(node.operand, namespace)))

    if isinstance(node, ast.BinOp):
        fn2 = _BINARY_OPS.get(type(node.op))
        if fn2 is None:
            raise ExpressionError(
                "Unsupported binary operator: %s" % type(node.op).__name__
            )
        left = _eval_node(node.left, namespace)
        right = _eval_node(node.right, namespace)
        try:
            return float(fn2(left, right))
        except ZeroDivisionError as exc:
            raise ExpressionError("Division by zero") from exc
        except (OverflowError, ValueError) as exc:
            raise ExpressionError("Arithmetic error: %s" % exc) from exc

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("Only simple function calls are allowed")
        fn3 = SAFE_FUNCTIONS.get(node.func.id)
        if fn3 is None:
            raise ExpressionError("Function '%s' is not allowed" % node.func.id)
        if node.keywords:
            raise ExpressionError("Keyword arguments are not allowed")
        args = [_eval_node(a, namespace) for a in node.args]
        try:
            return float(fn3(*args))
        except (TypeError, ValueError) as exc:
            raise ExpressionError(
                "Bad call to '%s': %s" % (node.func.id, exc)
            ) from exc

    raise ExpressionError("Unsupported expression node: %s" % type(node).__name__)


def _parse(expression: str) -> ast.Expression:
    try:
        return ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError("Invalid expression syntax: %s" % exc) from exc


def evaluate(expression: str, namespace: Optional[Mapping[str, float]] = None) -> float:
    """Evaluate *expression* against *namespace*, whitelisting all node kinds."""
    return _eval_node(_parse(expression), namespace or {})


def extract_symbols(expression: str) -> Set[str]:
    """Return the free variable names of *expression* (constants/functions excluded)."""
    tree = _parse(expression)
    names: Set[str] = set()
    called: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in SAFE_CONSTANTS or node.id in called:
                continue
            names.add(node.id)
    return names


@dataclass
class Parameter:
    """A named table entry: either a literal ``value`` or an ``expression``."""

    name: str
    value: Optional[float] = None
    expression: Optional[str] = None
    unit: Optional[str] = None

    @property
    def is_driven(self) -> bool:
        return self.expression is not None


@dataclass
class ParameterTable:
    """Named parameters with expression bindings, resolved in dependency order."""

    parameters: Dict[str, Parameter] = field(default_factory=dict)

    # -- mutation ---------------------------------------------------------
    def set(self, name: str, value: float, unit: Optional[str] = None) -> None:
        """Define (or redefine) *name* as a literal value."""
        self.parameters[name] = Parameter(name=name, value=float(value), unit=unit)

    def set_expr(self, name: str, expression: str, unit: Optional[str] = None) -> None:
        """Define (or redefine) *name* as a driven expression."""
        extract_symbols(expression)  # syntax check up front
        self.parameters[name] = Parameter(name=name, expression=expression, unit=unit)

    def remove(self, name: str) -> None:
        self.parameters.pop(name, None)

    def __contains__(self, name: object) -> bool:
        return name in self.parameters

    def names(self) -> List[str]:
        return sorted(self.parameters)

    # -- dependency graph -------------------------------------------------
    def direct_dependencies(self, name: str) -> Set[str]:
        """Symbols referenced directly by *name*'s expression."""
        param = self._require(name)
        if param.expression is None:
            return set()
        return extract_symbols(param.expression)

    def evaluation_order(self, external: Optional[Iterable[str]] = None) -> List[str]:
        """Topological order over the parameter dependency graph (ties: alphabetical).

        Symbols listed in *external* are treated as already-known inputs (they are
        supplied by the caller's namespace) rather than unknown references.
        """
        known_external = set(external or ())
        deps: Dict[str, Set[str]] = {}
        for name in sorted(self.parameters):
            raw = self.direct_dependencies(name) - known_external
            unknown = sorted(s for s in raw if s not in self.parameters)
            if unknown:
                raise ExpressionError(
                    "Parameter '%s' references unknown symbol(s): %s"
                    % (name, ", ".join(unknown))
                )
            if name in raw:
                raise CyclicParameterError(
                    "Parameter '%s' references itself" % name
                )
            deps[name] = raw

        ordered: List[str] = []
        done: Set[str] = set()
        # Kahn with a sorted ready-set for a stable, deterministic order.
        while len(ordered) < len(deps):
            ready = sorted(n for n in deps if n not in done and deps[n] <= done)
            if not ready:
                stuck = sorted(n for n in deps if n not in done)
                raise CyclicParameterError(
                    "Cyclic parameter references among: %s" % ", ".join(stuck)
                )
            for n in ready:
                ordered.append(n)
                done.add(n)
        return ordered

    def dependencies(self, name: str) -> Set[str]:
        """Transitive set of parameters *name* depends on."""
        self._require(name)
        out: Set[str] = set()
        stack = sorted(self.direct_dependencies(name))
        while stack:
            current = stack.pop()
            if current in out or current not in self.parameters:
                continue
            out.add(current)
            stack.extend(sorted(self.direct_dependencies(current)))
        out.discard(name)
        return out

    def dependents(self, name: str) -> Set[str]:
        """Parameters that would go stale if *name* changed (transitive)."""
        self._require(name)
        out: Set[str] = set()
        changed = True
        while changed:
            changed = False
            for other in sorted(self.parameters):
                if other == name or other in out:
                    continue
                direct = self.direct_dependencies(other)
                if name in direct or direct & out:
                    out.add(other)
                    changed = True
        return out

    # -- evaluation -------------------------------------------------------
    def evaluate_all(self, extra: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
        """Resolve every parameter to a float, in dependency order."""
        namespace: Dict[str, float] = {k: float(v) for k, v in (extra or {}).items()}
        for name in self.evaluation_order(external=namespace.keys()):
            param = self.parameters[name]
            if param.expression is not None:
                namespace[name] = evaluate(param.expression, namespace)
            else:
                namespace[name] = float(param.value if param.value is not None else 0.0)
        return namespace

    def evaluate_one(self, name: str, extra: Optional[Mapping[str, float]] = None) -> float:
        return self.evaluate_all(extra)[self._require(name).name]

    def _require(self, name: str) -> Parameter:
        param = self.parameters.get(name)
        if param is None:
            raise ExpressionError("Unknown parameter '%s'" % name)
        return param


def build_table(entries: Iterable[Any]) -> ParameterTable:
    """Build a table from ``(name, literal_or_expression)`` pairs or mappings."""
    table = ParameterTable()
    for entry in entries:
        if isinstance(entry, Mapping):
            name = str(entry["name"])
            raw = entry.get("value", entry.get("expression"))
            unit = entry.get("unit")
        else:
            name, raw = entry  # type: ignore[misc]
            unit = None
        if isinstance(raw, str):
            table.set_expr(str(name), raw, unit=unit)
        else:
            table.set(str(name), float(raw), unit=unit)
    return table
