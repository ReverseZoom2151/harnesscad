"""Forward-mode automatic differentiation over the f-rep IR, after libfive.

libfive's derivative evaluator carries, alongside each intermediate value, its
partial derivatives with respect to ``x``, ``y`` and ``z``.  Propagating those
through the chain rule as the graph is evaluated yields the *exact* gradient
``(df/dx, df/dy, df/dz)`` at a point in a single pass -- no finite differences,
no truncation error.  For an implicit surface that gradient is (proportional to)
the surface normal, which is exactly the Hermite data dual contouring needs.

This is genuinely distinct from finite-difference derivative estimation: it is
symbolic differentiation performed numerically via *dual numbers*.  A dual
number ``(v; g)`` pairs a value ``v`` with a gradient vector ``g``; the
arithmetic rules

    (v;g) + (w;h)   = (v+w;  g+h)
    (v;g) * (w;h)   = (v*w;  v*h + w*g)      # product rule
    sqrt(v;g)       = (sqrt v; g / (2 sqrt v))   # chain rule
    ...

are the derivative rules of calculus, and applying them node-by-node over the
DAG gives the derivative of the whole expression.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from harnesscad.domain.geometry import libfive_frep_ir as ir

Vec3 = Tuple[float, float, float]


class Dual:
    """A value paired with its gradient ``(d/dx, d/dy, d/dz)``."""

    __slots__ = ("v", "g")

    def __init__(self, v: float, g: Vec3):
        self.v = float(v)
        self.g = (float(g[0]), float(g[1]), float(g[2]))

    # -- gradient helpers ----------------------------------------------------
    @staticmethod
    def _add_g(a: Vec3, b: Vec3) -> Vec3:
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    @staticmethod
    def _scale_g(s: float, a: Vec3) -> Vec3:
        return (s * a[0], s * a[1], s * a[2])

    # -- arithmetic (each rule is a line of calculus) ------------------------
    def __add__(self, o: "Dual") -> "Dual":
        return Dual(self.v + o.v, self._add_g(self.g, o.g))

    def __sub__(self, o: "Dual") -> "Dual":
        return Dual(self.v - o.v,
                    (self.g[0] - o.g[0], self.g[1] - o.g[1], self.g[2] - o.g[2]))

    def __neg__(self) -> "Dual":
        return Dual(-self.v, self._scale_g(-1.0, self.g))

    def __mul__(self, o: "Dual") -> "Dual":
        # product rule: (uv)' = u'v + uv'
        return Dual(self.v * o.v,
                    self._add_g(self._scale_g(o.v, self.g),
                                self._scale_g(self.v, o.g)))

    def __truediv__(self, o: "Dual") -> "Dual":
        # quotient rule: (u/v)' = (u'v - uv') / v^2
        inv = 1.0 / o.v
        num_g = self._add_g(self._scale_g(o.v, self.g),
                            self._scale_g(-self.v, o.g))
        return Dual(self.v * inv, self._scale_g(inv * inv, num_g))

    def square(self) -> "Dual":
        return Dual(self.v * self.v, self._scale_g(2.0 * self.v, self.g))

    def sqrt(self) -> "Dual":
        s = math.sqrt(self.v)
        d = 0.0 if s == 0.0 else 0.5 / s
        return Dual(s, self._scale_g(d, self.g))

    def abs(self) -> "Dual":
        sign = 0.0 if self.v == 0.0 else math.copysign(1.0, self.v)
        return Dual(abs(self.v), self._scale_g(sign, self.g))

    def recip(self) -> "Dual":
        inv = 1.0 / self.v
        return Dual(inv, self._scale_g(-inv * inv, self.g))

    def sin(self) -> "Dual":
        return Dual(math.sin(self.v), self._scale_g(math.cos(self.v), self.g))

    def cos(self) -> "Dual":
        return Dual(math.cos(self.v), self._scale_g(-math.sin(self.v), self.g))

    def tan(self) -> "Dual":
        c = math.cos(self.v)
        return Dual(math.tan(self.v), self._scale_g(1.0 / (c * c), self.g))

    def exp(self) -> "Dual":
        e = math.exp(self.v)
        return Dual(e, self._scale_g(e, self.g))

    def log(self) -> "Dual":
        return Dual(math.log(self.v), self._scale_g(1.0 / self.v, self.g))

    def atan(self) -> "Dual":
        return Dual(math.atan(self.v),
                    self._scale_g(1.0 / (1.0 + self.v * self.v), self.g))

    @staticmethod
    def _select(a: "Dual", b: "Dual", take_a: bool) -> "Dual":
        # min/max are piecewise: the derivative follows the selected branch.
        return a if take_a else b

    def __repr__(self) -> str:
        return "Dual(v=%g, g=%r)" % (self.v, self.g)


_UNARY_AD = {
    "neg": lambda a: -a,
    "square": Dual.square,
    "sqrt": Dual.sqrt,
    "abs": Dual.abs,
    "recip": Dual.recip,
    "sin": Dual.sin,
    "cos": Dual.cos,
    "tan": Dual.tan,
    "exp": Dual.exp,
    "log": Dual.log,
    "atan": Dual.atan,
}


def eval_dual(root: ir.Node, x: float, y: float, z: float = 0.0) -> Dual:
    """Evaluate value + exact gradient of the graph at ``(x, y, z)``."""
    memo: Dict[int, Dual] = {}
    for n in ir._post_order(root):
        op = n.op
        if op == "var-x":
            memo[n.id] = Dual(x, (1.0, 0.0, 0.0))
        elif op == "var-y":
            memo[n.id] = Dual(y, (0.0, 1.0, 0.0))
        elif op == "var-z":
            memo[n.id] = Dual(z, (0.0, 0.0, 1.0))
        elif op == "const":
            memo[n.id] = Dual(n.value, (0.0, 0.0, 0.0))
        elif op == "add":
            memo[n.id] = memo[n.a.id] + memo[n.b.id]
        elif op == "sub":
            memo[n.id] = memo[n.a.id] - memo[n.b.id]
        elif op == "mul":
            memo[n.id] = memo[n.a.id] * memo[n.b.id]
        elif op == "div":
            memo[n.id] = memo[n.a.id] / memo[n.b.id]
        elif op == "min":
            a, b = memo[n.a.id], memo[n.b.id]
            memo[n.id] = Dual._select(a, b, a.v <= b.v)
        elif op == "max":
            a, b = memo[n.a.id], memo[n.b.id]
            memo[n.id] = Dual._select(a, b, a.v >= b.v)
        elif op in _UNARY_AD:
            memo[n.id] = _UNARY_AD[op](memo[n.a.id])
        else:
            raise NotImplementedError("no AD rule for opcode %r" % (op,))
    return memo[root.id]


def gradient(root: ir.Node, x: float, y: float, z: float = 0.0) -> Vec3:
    """Exact ``(df/dx, df/dy, df/dz)`` at a point."""
    return eval_dual(root, x, y, z).g


def normal(root: ir.Node, x: float, y: float, z: float = 0.0) -> Vec3:
    """Unit surface normal ``grad f / |grad f|`` (zero vector if degenerate)."""
    g = gradient(root, x, y, z)
    mag = math.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
    if mag == 0.0:
        return (0.0, 0.0, 0.0)
    return (g[0] / mag, g[1] / mag, g[2] / mag)
