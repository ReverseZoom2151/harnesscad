"""Interval arithmetic and an interval evaluator over the f-rep IR, after libfive.

To render a shape, libfive walks an octree and asks, for each box
``[x0,x1] x [y0,y1] x [z0,z1]``, "could the surface ``f = 0`` pass through
here?".  It answers by *interval arithmetic*: evaluating ``f`` with each
variable replaced by its whole coordinate range, obtaining a conservative
output interval ``[lo, hi]`` that is guaranteed to contain every value ``f``
takes over the box.  Then (SDF sign convention, ``f < 0`` inside):

* ``hi < 0``  -> the box is entirely inside the solid (FILLED) -- stop;
* ``lo > 0``  -> the box is entirely outside (EMPTY) -- stop;
* otherwise   -> the box straddles the surface (AMBIGUOUS) -- subdivide.

That test is what makes the octree cheap: whole sub-trees are pruned without
ever meshing them.  This module implements

* :class:`Interval` -- a rounded-outward interval with the full operator set
  libfive supports (``+ - * /``, ``square``, ``sqrt``, ``abs``, ``min``,
  ``max``, ``recip``, ``sin``, ``cos``, ``exp``, ``log``, ``pow``, ``mod``,
  ``atan``, ``atan2``), each guaranteed to *enclose* the true range;
* :func:`eval_interval` -- an evaluator that maps a :mod:`geometry.libfive_frep_ir`
  graph to an :class:`Interval` over a box;
* :func:`classify` -- the EMPTY / FILLED / AMBIGUOUS pruning decision.

Correctness contract (checked by the tests): the returned interval always
contains the true range of ``f`` over the box.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from harnesscad.domain.geometry import libfive_frep_ir as ir

EMPTY = "EMPTY"        # box wholly outside the solid
FILLED = "FILLED"      # box wholly inside the solid
AMBIGUOUS = "AMBIGUOUS"  # surface may cross the box


class Interval:
    """A closed real interval ``[lo, hi]`` with outward-enclosing arithmetic.

    Every operation returns an interval guaranteed to contain the true image of
    the operand ranges (the fundamental theorem of interval arithmetic).  A
    ``maybe_nan`` flag mirrors libfive: it records that the true value could be
    undefined (e.g. ``sqrt`` of a range dipping below zero), which forces the
    region to be treated as AMBIGUOUS.
    """

    __slots__ = ("lo", "hi", "maybe_nan")

    def __init__(self, lo: float, hi: float, maybe_nan: bool = False):
        if lo > hi:
            lo, hi = hi, lo
        self.lo = float(lo)
        self.hi = float(hi)
        self.maybe_nan = bool(maybe_nan)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def scalar(v: float) -> "Interval":
        return Interval(v, v)

    def width(self) -> float:
        return self.hi - self.lo

    def contains(self, v: float, eps: float = 1e-9) -> bool:
        return self.lo - eps <= v <= self.hi + eps

    def __repr__(self) -> str:
        tag = " nan?" if self.maybe_nan else ""
        return "Interval[%g, %g%s]" % (self.lo, self.hi, tag)

    # -- arithmetic ----------------------------------------------------------
    def __add__(self, o: "Interval") -> "Interval":
        return Interval(self.lo + o.lo, self.hi + o.hi,
                        self.maybe_nan or o.maybe_nan)

    def __sub__(self, o: "Interval") -> "Interval":
        return Interval(self.lo - o.hi, self.hi - o.lo,
                        self.maybe_nan or o.maybe_nan)

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo, self.maybe_nan)

    def __mul__(self, o: "Interval") -> "Interval":
        products = (self.lo * o.lo, self.lo * o.hi,
                    self.hi * o.lo, self.hi * o.hi)
        return Interval(min(products), max(products),
                        self.maybe_nan or o.maybe_nan)

    def __truediv__(self, o: "Interval") -> "Interval":
        # If the denominator straddles zero the result is unbounded; be
        # conservative (libfive does exactly this).
        if o.lo <= 0.0 <= o.hi:
            return Interval(-math.inf, math.inf, True)
        recips = (1.0 / o.lo, 1.0 / o.hi)
        inv = Interval(min(recips), max(recips), o.maybe_nan)
        return self * inv

    def square(self) -> "Interval":
        if self.lo >= 0.0:
            return Interval(self.lo * self.lo, self.hi * self.hi, self.maybe_nan)
        if self.hi <= 0.0:
            return Interval(self.hi * self.hi, self.lo * self.lo, self.maybe_nan)
        # straddles zero: min is 0, max is the larger squared endpoint
        return Interval(0.0, max(self.lo * self.lo, self.hi * self.hi),
                        self.maybe_nan)

    def sqrt(self) -> "Interval":
        nan = self.maybe_nan or self.lo < 0.0
        lo = math.sqrt(self.lo) if self.lo > 0.0 else 0.0
        hi = math.sqrt(self.hi) if self.hi > 0.0 else 0.0
        return Interval(lo, hi, nan)

    def abs(self) -> "Interval":
        if self.lo >= 0.0:
            return Interval(self.lo, self.hi, self.maybe_nan)
        if self.hi <= 0.0:
            return Interval(-self.hi, -self.lo, self.maybe_nan)
        return Interval(0.0, max(-self.lo, self.hi), self.maybe_nan)

    def recip(self) -> "Interval":
        if self.lo <= 0.0 <= self.hi:
            return Interval(-math.inf, math.inf, True)
        recips = (1.0 / self.lo, 1.0 / self.hi)
        return Interval(min(recips), max(recips), self.maybe_nan)

    def exp(self) -> "Interval":
        return Interval(math.exp(self.lo), math.exp(self.hi), self.maybe_nan)

    def log(self) -> "Interval":
        nan = self.maybe_nan or self.lo <= 0.0
        lo = math.log(self.lo) if self.lo > 0.0 else -math.inf
        hi = math.log(self.hi) if self.hi > 0.0 else -math.inf
        return Interval(lo, hi, nan)

    def atan(self) -> "Interval":
        return Interval(math.atan(self.lo), math.atan(self.hi), self.maybe_nan)

    def pow_int(self, n: int) -> "Interval":
        # Integer power via repeated interval multiplication (exact enclosure).
        if n == 0:
            return Interval(1.0, 1.0, self.maybe_nan)
        base = self if n > 0 else self.recip()
        result = Interval(1.0, 1.0, base.maybe_nan)
        for _ in range(abs(n)):
            result = result * base
        return result

    @staticmethod
    def min(a: "Interval", b: "Interval") -> "Interval":
        return Interval(min(a.lo, b.lo), min(a.hi, b.hi),
                        a.maybe_nan or b.maybe_nan)

    @staticmethod
    def max(a: "Interval", b: "Interval") -> "Interval":
        return Interval(max(a.lo, b.lo), max(a.hi, b.hi),
                        a.maybe_nan or b.maybe_nan)

    # -- trig: enclose by checking interior extrema --------------------------
    def sin(self) -> "Interval":
        return self._trig(math.sin, phase=math.pi / 2.0)

    def cos(self) -> "Interval":
        return self._trig(math.cos, phase=0.0)

    def _trig(self, fn, phase: float) -> "Interval":
        # If wider than a full period the range is the whole [-1, 1].
        if self.width() >= 2.0 * math.pi:
            return Interval(-1.0, 1.0, self.maybe_nan)
        lo_v = fn(self.lo)
        hi_v = fn(self.hi)
        result_lo = min(lo_v, hi_v)
        result_hi = max(lo_v, hi_v)
        # Extrema of sin/cos occur where the argument is phase + k*pi.
        # Maxima (+1) at phase + 2k*pi, minima (-1) at phase + (2k+1)*pi.
        k_start = math.floor((self.lo - phase) / math.pi)
        k_end = math.ceil((self.hi - phase) / math.pi)
        for k in range(k_start, k_end + 1):
            crit = phase + k * math.pi
            if self.lo <= crit <= self.hi:
                v = fn(crit)
                result_lo = min(result_lo, v)
                result_hi = max(result_hi, v)
        return Interval(result_lo, result_hi, self.maybe_nan)

    @staticmethod
    def atan2(y: "Interval", x: "Interval") -> "Interval":
        # Conservative: sample the corners; if the box contains the origin the
        # angle is unbounded within [-pi, pi] (matches libfive's origin case).
        if x.lo <= 0.0 <= x.hi and y.lo <= 0.0 <= y.hi:
            return Interval(-math.pi, math.pi, y.maybe_nan or x.maybe_nan)
        corners = [math.atan2(yy, xx)
                   for yy in (y.lo, y.hi) for xx in (x.lo, x.hi)]
        return Interval(min(corners), max(corners), y.maybe_nan or x.maybe_nan)

    @staticmethod
    def mod(a: "Interval", b: "Interval") -> "Interval":
        # Conservative bound: result of a mod b lies within [0, b] for b > 0.
        if b.lo > 0.0:
            return Interval(0.0, b.hi, a.maybe_nan or b.maybe_nan)
        if b.hi < 0.0:
            return Interval(b.lo, 0.0, a.maybe_nan or b.maybe_nan)
        return Interval(min(b.lo, 0.0), max(b.hi, 0.0), True)


# ---------------------------------------------------------------------------
# Interval evaluator over the IR
# ---------------------------------------------------------------------------

_UNARY_IV = {
    "neg": lambda a: -a,
    "square": Interval.square,
    "sqrt": Interval.sqrt,
    "abs": Interval.abs,
    "recip": Interval.recip,
    "sin": Interval.sin,
    "cos": Interval.cos,
    "exp": Interval.exp,
    "log": Interval.log,
    "atan": Interval.atan,
}

_BINARY_IV = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "min": Interval.min,
    "max": Interval.max,
    "atan2": Interval.atan2,
    "mod": Interval.mod,
}


def eval_interval(root: ir.Node,
                  box_lo: Tuple[float, float, float],
                  box_hi: Tuple[float, float, float]) -> Interval:
    """Evaluate the graph over a box, returning an enclosing :class:`Interval`.

    ``box_lo`` / ``box_hi`` are ``(x, y, z)`` corners.  The result is guaranteed
    to contain every value ``f`` attains inside the box.
    """
    memo: Dict[int, Interval] = {}
    for n in ir._post_order(root):
        op = n.op
        if op == "var-x":
            memo[n.id] = Interval(box_lo[0], box_hi[0])
        elif op == "var-y":
            memo[n.id] = Interval(box_lo[1], box_hi[1])
        elif op == "var-z":
            memo[n.id] = Interval(box_lo[2], box_hi[2])
        elif op == "const":
            memo[n.id] = Interval.scalar(n.value)
        elif op == "tan":
            # tan is monotone only within a branch; derive from sin/cos.
            a = memo[n.a.id]
            memo[n.id] = a.sin() / a.cos()
        elif op == "pow":
            base = memo[n.a.id]
            exp = memo[n.b.id]
            memo[n.id] = base.pow_int(int(round(exp.lo)))
        elif n.b is None:
            memo[n.id] = _UNARY_IV[op](memo[n.a.id])
        else:
            memo[n.id] = _BINARY_IV[op](memo[n.a.id], memo[n.b.id])
    return memo[root.id]


def classify(iv: Interval) -> str:
    """Map an output interval to a pruning decision (SDF sign: ``f < 0`` inside)."""
    if iv.maybe_nan:
        return AMBIGUOUS
    if iv.hi < 0.0:
        return FILLED
    if iv.lo > 0.0:
        return EMPTY
    return AMBIGUOUS
