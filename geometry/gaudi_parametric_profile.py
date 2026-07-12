"""Parametric-curve profile sampler (Gaudi-backend style), without ``eval``.

Gaudi's ``parametric`` plates define a closed 2D outline as a pair of formulas
``x(t)`` and ``y(t)`` sampled over a parameter ``range`` in ``steps`` steps
(``template.py:evaluate_parametric_curve``).  The upstream code evaluates each
formula string with Python ``eval`` inside a ``{"math": math, "t": t}``
namespace -- convenient but unsafe (any callable reachable from ``math`` or a
crafted string runs) and non-introspectable.

This module reimplements the sampler deterministically and safely:

  * a tiny arithmetic evaluator built on the Python ``ast`` module that
    whitelists ``+ - * / // % **`` and unary +/-, the single free variable ``t``,
    the constants ``pi``/``e``/``tau``, and the common ``math.*`` functions used
    for curve authoring (``sin``, ``cos``, ``tan``, ``sqrt``, ``exp``, ``log``,
    ``atan2``, ``hypot`` ...) -- with or without the ``math.`` prefix, so both
    ``"4*cos(t)"`` and ``"4*math.cos(t)"`` parse.  No attribute access outside
    the ``math`` module, no subscript, no comprehension, no lambda, no arbitrary
    call is ever executable;
  * :func:`sample_curve` evaluates ``x(t)``/``y(t)`` at ``steps`` values of ``t``
    equally spaced across ``[start, end)`` (matching Gaudi's half-open sampling,
    which leaves the loop open so the first and last points can be stitched);
  * profile hygiene the upstream code omits: :func:`polygon_signed_area`,
    :func:`ensure_ccw` (a mesh cap needs a consistent winding),
    :func:`dedupe_points` (drop a near-repeated closing point) and
    :func:`is_degenerate` (reject a zero-area / collinear outline before it
    reaches a mesher).

Deterministic: pure arithmetic over a fixed sample grid; no clock, no
randomness, no I/O.

Public API
----------
``evaluate(expr, t) -> float``
``compile_expr(expr) -> callable``
``sample_curve(fx, fy, start, end, steps) -> list[(x, y)]``
``polygon_signed_area(points) -> float``
``ensure_ccw(points) -> list``
``dedupe_points(points, tol=1e-9) -> list``
``is_degenerate(points, tol=1e-12) -> bool``
``ParametricExprError``
"""

from __future__ import annotations

import ast
import math
from typing import Callable, List, Sequence, Tuple

Point = Tuple[float, float]


class ParametricExprError(ValueError):
    """Raised when a parametric formula uses an unsupported construct."""


_ALLOWED_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "pow": math.pow,
    "fabs": math.fabs,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "hypot": math.hypot,
    "copysign": math.copysign,
    "degrees": math.degrees,
    "radians": math.radians,
}

_ALLOWED_CONSTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}


def _eval_node(node: ast.AST, t: float) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, t)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ParametricExprError("only numeric constants are allowed")
    if isinstance(node, ast.Name):
        if node.id == "t":
            return float(t)
        if node.id in _ALLOWED_CONSTS:
            return _ALLOWED_CONSTS[node.id]
        raise ParametricExprError("unknown name '{0}'".format(node.id))
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ParametricExprError("operator not allowed")
        return op(_eval_node(node.left, t), _eval_node(node.right, t))
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, t)
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        raise ParametricExprError("unary operator not allowed")
    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        if name not in _ALLOWED_FUNCS:
            raise ParametricExprError("call to '{0}' not allowed".format(name))
        if node.keywords:
            raise ParametricExprError("keyword arguments not allowed")
        args = [_eval_node(a, t) for a in node.args]
        return float(_ALLOWED_FUNCS[name](*args))
    raise ParametricExprError("unsupported expression: {0}".format(type(node).__name__))


def _call_name(func: ast.AST) -> str:
    # Accept bare ``cos`` and ``math.cos`` (attribute access limited to ``math``).
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id == "math":
            return func.attr
        raise ParametricExprError(
            "attribute access on '{0}' not allowed".format(func.value.id)
        )
    raise ParametricExprError("unsupported call target")


def evaluate(expr: str, t: float) -> float:
    """Evaluate a single formula string at parameter value ``t``."""
    if not isinstance(expr, str):
        raise ParametricExprError("expression must be a string")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ParametricExprError("syntax error: {0}".format(exc.msg))
    return _eval_node(tree, t)


def compile_expr(expr: str) -> Callable[[float], float]:
    """Return a reusable ``f(t) -> float`` for a formula, validating it once."""
    if not isinstance(expr, str):
        raise ParametricExprError("expression must be a string")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ParametricExprError("syntax error: {0}".format(exc.msg))
    _eval_node(tree, 0.0)  # validate node set once

    def _fn(t: float) -> float:
        return _eval_node(tree, t)

    return _fn


def sample_curve(
    fx: str, fy: str, start: float, end: float, steps: int
) -> List[Point]:
    """Sample ``(x(t), y(t))`` at ``steps`` equally spaced ``t`` in ``[start, end)``.

    Mirrors Gaudi's half-open sampling: ``t_i = start + i * (end - start) / steps``
    for ``i`` in ``0 .. steps-1``, leaving the loop open.
    """
    if not (isinstance(steps, int) and not isinstance(steps, bool)) or steps < 1:
        raise ParametricExprError("steps must be a positive integer")
    cx = compile_expr(fx)
    cy = compile_expr(fy)
    step_size = (end - start) / steps
    pts: List[Point] = []
    for i in range(steps):
        t = start + i * step_size
        pts.append((cx(t), cy(t)))
    return pts


def polygon_signed_area(points: Sequence[Point]) -> float:
    """Shoelace signed area; positive when the outline winds counter-clockwise."""
    n = len(points)
    if n < 3:
        return 0.0
    acc = 0.0
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        acc += x0 * y1 - x1 * y0
    return acc / 2.0


def ensure_ccw(points: Sequence[Point]) -> List[Point]:
    """Return the outline wound counter-clockwise (reversed if it was CW)."""
    if polygon_signed_area(points) < 0.0:
        return list(reversed(list(points)))
    return list(points)


def dedupe_points(points: Sequence[Point], tol: float = 1e-9) -> List[Point]:
    """Drop consecutive (and wrap-around closing) near-duplicate points."""
    out: List[Point] = []
    for p in points:
        if out and _close(out[-1], p, tol):
            continue
        out.append((p[0], p[1]))
    while len(out) >= 2 and _close(out[0], out[-1], tol):
        out.pop()
    return out


def is_degenerate(points: Sequence[Point], tol: float = 1e-12) -> bool:
    """True if the outline has < 3 distinct points or ~zero area (collinear)."""
    cleaned = dedupe_points(points)
    if len(cleaned) < 3:
        return True
    return abs(polygon_signed_area(cleaned)) <= tol


def _close(a: Point, b: Point, tol: float) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol
