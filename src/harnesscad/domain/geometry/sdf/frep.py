"""Functional-representation (f-rep) expression-graph IR.

An f-rep solid is a directed acyclic graph (DAG) of *opcodes* over the implicit
coordinate functions ``x``, ``y``, ``z``. A shape is the level set
``f(x, y, z) = 0`` of the expression the graph evaluates, with ``f < 0``
denoting the interior.

This module reifies that graph as an *introspectable, optimisable* data
structure -- distinct from a fixed library of Python SDF callables.  The graph
is what makes interval pruning, exact automatic differentiation and structural
optimisation possible: they are all traversals of this same tree.

Core properties:

* the opcode set (``VAR_X``, ``OP_ADD``, ``OP_MIN``, ``OP_SQRT``, ...), each
  with an arity and a printable symbol (``opcode.cpp``);
* **common-subexpression sharing**: identical subexpressions are deduplicated to
  a single node by a structural hash, so ``f + f`` stores ``f`` once
  (libfive builds a hash-consed DAG);
* **constant folding** and commutative-operand canonicalisation as cheap graph
  optimisations;
* a deterministic evaluator over a single point;
* an infix pretty-printer and an S-expression serialiser (libfive speaks a
  Scheme-symbol dialect, e.g. ``(max (- x 1) (sqrt y))``).

Pure stdlib, deterministic, no floating wall-clock behaviour.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Opcodes
# ---------------------------------------------------------------------------

# name -> (arity, infix-or-scheme symbol).  Nullary leaves carry their own text.
_UNARY = {
    "neg": lambda a: -a,
    "square": lambda a: a * a,
    "sqrt": math.sqrt,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "exp": math.exp,
    "log": math.log,
    "recip": lambda a: 1.0 / a,
}

_BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "min": min,
    "max": max,
    "pow": lambda a, b: math.pow(a, b),
    "atan2": math.atan2,
    "mod": lambda a, b: a - b * math.floor(a / b),
}

# Opcodes that are commutative: operands may be reordered for canonical dedup.
_COMMUTATIVE = frozenset({"add", "mul", "min", "max"})

# Infix operator glyphs for the pretty-printer (others print as function calls).
_INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/"}

_LEAVES = frozenset({"var-x", "var-y", "var-z", "const"})


def arity(op: str) -> int:
    """Number of operands an opcode takes (0, 1 or 2)."""
    if op in _LEAVES:
        return 0
    if op in _UNARY:
        return 1
    if op in _BINARY:
        return 2
    raise KeyError("unknown opcode: %r" % (op,))


def is_commutative(op: str) -> bool:
    return op in _COMMUTATIVE


# ---------------------------------------------------------------------------
# Nodes and the hash-consing graph
# ---------------------------------------------------------------------------


class Node:
    """A node in an f-rep graph.

    Never construct directly -- use a :class:`Graph` so that structural sharing
    (common-subexpression elimination) is preserved.  Two structurally identical
    expressions built from the same graph are the *same* ``Node`` object, so
    ``a is b`` iff ``a`` and ``b`` denote the same subexpression.
    """

    __slots__ = ("op", "value", "a", "b", "id", "_graph")

    def __init__(self, graph: "Graph", op: str, value: Optional[float],
                 a: Optional["Node"], b: Optional["Node"], nid: int):
        self.op = op
        self.value = value
        self.a = a
        self.b = b
        self.id = nid
        self._graph = graph

    # -- ergonomic operator overloading (all routed through the graph) --------
    def _lift(self, other) -> "Node":
        if isinstance(other, Node):
            return other
        return self._graph.constant(float(other))

    def __add__(self, o): return self._graph.op("add", self, self._lift(o))
    def __radd__(self, o): return self._graph.op("add", self._lift(o), self)
    def __sub__(self, o): return self._graph.op("sub", self, self._lift(o))
    def __rsub__(self, o): return self._graph.op("sub", self._lift(o), self)
    def __mul__(self, o): return self._graph.op("mul", self, self._lift(o))
    def __rmul__(self, o): return self._graph.op("mul", self._lift(o), self)
    def __truediv__(self, o): return self._graph.op("div", self, self._lift(o))
    def __rtruediv__(self, o): return self._graph.op("div", self._lift(o), self)
    def __neg__(self): return self._graph.op("neg", self, None)

    def __repr__(self) -> str:
        return "Node(#%d %s)" % (self.id, to_infix(self))


class Graph:
    """A hash-consed DAG builder.

    Every distinct subexpression is stored once, keyed by a structural hash of
    ``(op, value, child-ids)``.  Rebuilding an identical expression returns the
    cached node, so the graph is a DAG (shared subtrees), not a tree.
    """

    def __init__(self, *, fold: bool = True):
        self._cache: Dict[Tuple, Node] = {}
        self._nodes: List[Node] = []
        self._fold = fold

    # -- interning -----------------------------------------------------------
    def _intern(self, key: Tuple, op: str, value: Optional[float],
                a: Optional[Node], b: Optional[Node]) -> Node:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        node = Node(self, op, value, a, b, len(self._nodes))
        self._nodes.append(node)
        self._cache[key] = node
        return node

    # -- leaves --------------------------------------------------------------
    def x(self) -> Node:
        return self._intern(("var-x",), "var-x", None, None, None)

    def y(self) -> Node:
        return self._intern(("var-y",), "var-y", None, None, None)

    def z(self) -> Node:
        return self._intern(("var-z",), "var-z", None, None, None)

    def constant(self, value: float) -> Node:
        v = float(value)
        # Normalise -0.0 to 0.0 so the cache key is stable.
        if v == 0.0:
            v = 0.0
        return self._intern(("const", v), "const", v, None, None)

    # -- generic op with folding + canonicalisation --------------------------
    def op(self, op: str, a: Node, b: Optional[Node] = None) -> Node:
        n = arity(op)
        if n == 1:
            assert b is None
            # constant folding
            if self._fold and a.op == "const":
                try:
                    return self.constant(_UNARY[op](a.value))
                except (ValueError, ZeroDivisionError, OverflowError):
                    pass
            return self._intern((op, a.id), op, None, a, None)
        if n == 2:
            assert b is not None
            if self._fold and a.op == "const" and b.op == "const":
                try:
                    return self.constant(_BINARY[op](a.value, b.value))
                except (ValueError, ZeroDivisionError, OverflowError):
                    pass
            # canonical operand order for commutative ops (smaller id first)
            if is_commutative(op) and a.id > b.id:
                a, b = b, a
            return self._intern((op, a.id, b.id), op, None, a, b)
        raise ValueError("op %r is nullary" % (op,))

    # -- named builders for unary/binary math --------------------------------
    def sqrt(self, a): return self.op("sqrt", a)
    def square(self, a): return self.op("square", a)
    def abs(self, a): return self.op("abs", a)
    def sin(self, a): return self.op("sin", a)
    def cos(self, a): return self.op("cos", a)
    def exp(self, a): return self.op("exp", a)
    def log(self, a): return self.op("log", a)
    def min(self, a, b): return self.op("min", a, b)
    def max(self, a, b): return self.op("max", a, b)

    # -- introspection -------------------------------------------------------
    def num_nodes(self) -> int:
        """Total distinct nodes interned (the DAG size)."""
        return len(self._nodes)


# ---------------------------------------------------------------------------
# Traversals: evaluation, sizing, printing, serialisation
# ---------------------------------------------------------------------------


def _post_order(root: Node) -> List[Node]:
    """Unique nodes reachable from ``root`` in dependency (child-first) order."""
    seen = set()
    order: List[Node] = []

    def visit(n: Node) -> None:
        if n.id in seen:
            return
        seen.add(n.id)
        if n.a is not None:
            visit(n.a)
        if n.b is not None:
            visit(n.b)
        order.append(n)

    visit(root)
    return order


def eval_point(root: Node, x: float, y: float, z: float = 0.0) -> float:
    """Evaluate ``root`` at ``(x, y, z)``.

    Uses a memo over the DAG so each shared subexpression is computed once.
    """
    memo: Dict[int, float] = {}
    for n in _post_order(root):
        op = n.op
        if op == "var-x":
            memo[n.id] = x
        elif op == "var-y":
            memo[n.id] = y
        elif op == "var-z":
            memo[n.id] = z
        elif op == "const":
            memo[n.id] = n.value
        elif n.b is None:
            memo[n.id] = _UNARY[op](memo[n.a.id])
        else:
            memo[n.id] = _BINARY[op](memo[n.a.id], memo[n.b.id])
    return memo[root.id]


def make_callable(root: Node) -> Callable[[float, float, float], float]:
    """Return ``f(x, y, z)`` evaluating this graph (pre-flattened traversal)."""
    order = _post_order(root)
    rid = root.id

    def f(x: float, y: float, z: float = 0.0) -> float:
        memo: Dict[int, float] = {}
        for n in order:
            op = n.op
            if op == "var-x":
                memo[n.id] = x
            elif op == "var-y":
                memo[n.id] = y
            elif op == "var-z":
                memo[n.id] = z
            elif op == "const":
                memo[n.id] = n.value
            elif n.b is None:
                memo[n.id] = _UNARY[op](memo[n.a.id])
            else:
                memo[n.id] = _BINARY[op](memo[n.a.id], memo[n.b.id])
        return memo[rid]

    return f


def _fmt_const(v: float) -> str:
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def to_infix(root: Node) -> str:
    """Human-readable infix rendering, e.g. ``max((x - 1), sqrt(y))``."""
    def render(n: Node) -> str:
        op = n.op
        if op == "var-x":
            return "x"
        if op == "var-y":
            return "y"
        if op == "var-z":
            return "z"
        if op == "const":
            return _fmt_const(n.value)
        if op == "neg":
            return "-%s" % render(n.a)
        if n.b is None:
            return "%s(%s)" % (op, render(n.a))
        if op in _INFIX:
            return "(%s %s %s)" % (render(n.a), _INFIX[op], render(n.b))
        return "%s(%s, %s)" % (op, render(n.a), render(n.b))

    return render(root)


def to_sexpr(root: Node) -> str:
    """Serialise to a libfive-style Scheme S-expression, e.g.
    ``(max (- x 1) (sqrt y))``."""
    def render(n: Node) -> str:
        op = n.op
        if op == "var-x":
            return "x"
        if op == "var-y":
            return "y"
        if op == "var-z":
            return "z"
        if op == "const":
            return _fmt_const(n.value)
        sym = _INFIX.get(op, op)
        if n.b is None:
            return "(%s %s)" % (sym, render(n.a))
        return "(%s %s %s)" % (sym, render(n.a), render(n.b))

    return render(root)


# ---------------------------------------------------------------------------
# A small stdlib of shapes, so downstream modules have graphs to chew on
# ---------------------------------------------------------------------------


def circle(g: Graph, cx: float, cy: float, r: float) -> Node:
    """f-rep circle: ``sqrt((x-cx)^2 + (y-cy)^2) - r`` (an exact 2D SDF)."""
    dx = g.x() - cx
    dy = g.y() - cy
    return g.sqrt(g.square(dx) + g.square(dy)) - r


def rectangle(g: Graph, x0: float, y0: float, x1: float, y1: float) -> Node:
    """f-rep axis-aligned rectangle as an intersection of four half-planes.

    ``max`` of the four signed half-plane distances -- negative inside.
    """
    x, y = g.x(), g.y()
    left = g.constant(x0) - x
    right = x - x1
    bottom = g.constant(y0) - y
    top = y - y1
    return g.max(g.max(left, right), g.max(bottom, top))


def sphere(g: Graph, cx: float, cy: float, cz: float, r: float) -> Node:
    """f-rep sphere: distance to centre minus radius."""
    dx = g.x() - cx
    dy = g.y() - cy
    dz = g.z() - cz
    return g.sqrt(g.square(dx) + g.square(dy) + g.square(dz)) - r


def union(g: Graph, a: Node, b: Node) -> Node:
    """CSG union of two f-rep solids (``min`` of the fields)."""
    return g.min(a, b)


def intersection(g: Graph, a: Node, b: Node) -> Node:
    """CSG intersection (``max`` of the fields)."""
    return g.max(a, b)


def difference(g: Graph, a: Node, b: Node) -> Node:
    """CSG difference ``a \\ b`` (``max(a, -b)``)."""
    return g.max(a, -b)
