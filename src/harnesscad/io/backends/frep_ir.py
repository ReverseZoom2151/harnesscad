"""Compile an :class:`~harnesscad.io.backends.frep.Node` CSG tree into the f-rep IR.

:mod:`harnesscad.io.backends.frep` evaluates its CSG tree with a Python closure
(:func:`~harnesscad.io.backends.frep.eval_node`). That is fine for sampling, but
a closure is opaque: you cannot differentiate it exactly and you cannot bound it
over a box. The repo already carries the machinery that *can* do both --
:mod:`harnesscad.domain.geometry.sdf.frep` (a hash-consed expression DAG),
:mod:`harnesscad.domain.numeric.forward_ad` (dual numbers -> exact gradients) and
:mod:`harnesscad.domain.numeric.interval_arithmetic` (conservative range bounds
-> octree pruning) -- but nothing ever handed them a real model.

This module is the missing bridge. :func:`compile_node` walks the backend's CSG
tree and re-emits it as an IR graph whose evaluation is *the same function* as
``eval_node``: every combinator is transcribed literally (``rect_exact``,
``_slab``, ``_combine_prism``, ``smooth_min_poly``, ``chamfer_min``,
``round_field``, ``shell``, the revolve coordinate remap and its angular wedge).
With the IR in hand:

*   :func:`exact_normal` gives the analytic surface normal (forward-mode AD) --
    no finite-difference truncation error, one pass, no epsilon to tune;
*   :func:`classify_box` gives the libfive EMPTY / FILLED / AMBIGUOUS verdict for
    a whole box, which lets the sampler skip regions the surface cannot enter.

Not every tree compiles. A sketch profile made of line segments is a polygon
whose sign comes from a winding number -- a *branch*, not an arithmetic
expression -- and the IR opcode set is pure arithmetic. Such a tree raises
:class:`CompileError` and the callers fall back to their existing behaviour.
That is a real limit, stated rather than papered over.

stdlib-only, deterministic.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from harnesscad.domain.geometry.sdf import frep as ir
from harnesscad.domain.numeric import forward_ad
from harnesscad.domain.numeric import interval_arithmetic as iv

__all__ = [
    "CompileError",
    "CompiledField",
    "compile_node",
    "exact_normal",
    "classify_box",
    "AMBIGUOUS",
    "EMPTY",
    "FILLED",
]

AMBIGUOUS = iv.AMBIGUOUS
EMPTY = iv.EMPTY
FILLED = iv.FILLED

Vec3 = Tuple[float, float, float]
_TWO_PI = 2.0 * math.pi


class CompileError(Exception):
    """The CSG tree contains a node the arithmetic IR cannot express."""


class CompiledField:
    """An IR graph plus its root -- the model as one arithmetic expression."""

    __slots__ = ("graph", "root")

    def __init__(self, graph: "ir.Graph", root: "ir.Node") -> None:
        self.graph = graph
        self.root = root

    def value(self, p: Sequence[float]) -> float:
        return ir.eval_point(self.root, float(p[0]), float(p[1]), float(p[2]))

    def normal(self, p: Sequence[float]) -> Vec3:
        return forward_ad.normal(self.root, float(p[0]), float(p[1]), float(p[2]))

    def gradient(self, p: Sequence[float]) -> Vec3:
        return forward_ad.gradient(self.root, float(p[0]), float(p[1]), float(p[2]))

    def node_count(self) -> int:
        return len(ir._post_order(self.root))  # noqa: SLF001 - the IR's own walker


# ---------------------------------------------------------------------------
# small IR helpers (each one mirrors a float helper in frep.py / the sdf package)
# ---------------------------------------------------------------------------

class _Builder:
    """A graph plus the encoding to use for ``min`` / ``max``.

    Two algebraically identical encodings, chosen for what the graph is FOR:

    *   ``smooth=False`` (default) emits the ``min`` / ``max`` opcodes. This is
        what interval arithmetic wants: ``min``/``max`` of two intervals is
        tight, so the pruning verdicts stay sharp.
    *   ``smooth=True`` emits the identity ``min(a,b) = (a + b - |a-b|)/2``.
        Same function, but its dual-number derivative at a TIE is the average of
        the two branches instead of an arbitrary pick. That matters: the exact
        rectangle/prism fields contain ``min(u, 0)``, and on the model's surface
        ``u`` is exactly zero -- precisely the tie -- where an arbitrary pick
        returns a zero gradient and hence no normal at all. The averaged
        subgradient points the right way, which after normalisation is the true
        normal. Used only for gradients, never for sampled values.
    """

    __slots__ = ("graph", "smooth")

    def __init__(self, graph: "ir.Graph", smooth: bool = False) -> None:
        self.graph = graph
        self.smooth = bool(smooth)

    def op(self, op, a, b=None):
        return self.graph.op(op, a, b)

    def constant(self, v):
        return self.graph.constant(v)

    def x(self):
        return self.graph.x()

    def y(self):
        return self.graph.y()

    def z(self):
        return self.graph.z()


def _c(g, v: float):
    return g.constant(float(v))


def _min(g, a, b):
    if getattr(g, "smooth", False):
        return _c(g, 0.5) * ((a + b) - _abs(g, a - b))
    return g.op("min", a, b)


def _max(g, a, b):
    if getattr(g, "smooth", False):
        return _c(g, 0.5) * ((a + b) + _abs(g, a - b))
    return g.op("max", a, b)


def _abs(g, a):
    return g.op("abs", a)


def _sq(g, a):
    return g.op("square", a)


def _sqrt(g, a):
    return g.op("sqrt", a)


def _hypot(g, a, b):
    """``math.hypot(a, b)`` as ``sqrt(a^2 + b^2)`` (identical for finite inputs)."""
    return _sqrt(g, _sq(g, a) + _sq(g, b))


def _clamp01(g, a):
    return _min(g, _max(g, a, _c(g, 0.0)), _c(g, 1.0))


def _smooth_min_poly(g, a, b, k: float):
    """combinators.smooth_min_poly, transcribed (k > 0)."""
    h = _clamp01(g, _c(g, 0.5) + (b - a) * _c(g, 0.5 / k))
    # lerp(b, a, h) - k*h*(1-h)  ==  b + (a-b)*h - k*h*(1-h)
    return (b + (a - b) * h) - _c(g, k) * h * (_c(g, 1.0) - h)


def _smooth_max_poly(g, a, b, k: float):
    return -_smooth_min_poly(g, -a, -b, k)


def _chamfer_min(g, a, b, r: float):
    """combinators.chamfer_min: ``min(a,b) - 0.5*max(r - |a-b|, 0)``."""
    e = _max(g, _c(g, r) - _abs(g, a - b), _c(g, 0.0))
    return _min(g, a, b) - _c(g, 0.5) * e


def _chamfer_max(g, a, b, r: float):
    return -_chamfer_min(g, -a, -b, r)


# ---------------------------------------------------------------------------
# the compiler
# ---------------------------------------------------------------------------

def _profile_ir(g, prof, u, v):
    """``frep._Profile.sdf`` as an expression in the in-plane coordinates."""
    if prof.polys:
        raise CompileError(
            "a polygon profile (sketch made of line segments) has a winding-number "
            "sign test, which the arithmetic f-rep IR cannot express")
    vals = []
    for (x, y, w, h) in prof.rects:
        cx, cy = x + w / 2.0, y + h / 2.0
        # primitives.rect_exact
        dx = _abs(g, u - _c(g, cx)) - _c(g, w / 2.0)
        dy = _abs(g, v - _c(g, cy)) - _c(g, h / 2.0)
        inside = _min(g, _max(g, dx, dy), _c(g, 0.0))
        outside = _hypot(g, _max(g, dx, _c(g, 0.0)), _max(g, dy, _c(g, 0.0)))
        vals.append(inside + outside)
    for (cx, cy, r) in prof.circles:
        # primitives.circle(p, d=2r)  ==  |p| - r
        vals.append(_hypot(g, u - _c(g, cx), v - _c(g, cy)) - _c(g, r))
    if not vals:
        raise CompileError("empty profile has no field (union_all identity is +inf)")
    out = vals[0]
    for nxt in vals[1:]:
        out = _min(g, out, nxt)
    return out


def _slab_ir(g, w, w0: float, w1: float):
    lo, hi = (w0, w1) if w0 <= w1 else (w1, w0)
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return _abs(g, w - _c(g, mid)) - _c(g, half)


def _combine_prism_ir(g, d2, dw, r_round: float, r_cham: float):
    """``frep._combine_prism``, transcribed."""
    if r_round > 0.0:
        a = d2 + _c(g, r_round)
        b = dw + _c(g, r_round)
        inside = _min(g, _max(g, a, b), _c(g, 0.0))
        outside = _hypot(g, _max(g, a, _c(g, 0.0)), _max(g, b, _c(g, 0.0)))
        return (inside + outside) - _c(g, r_round)      # xf.round_field
    if r_cham > 0.0:
        return _chamfer_max(g, d2, dw, r_cham)          # comb.chamfer_intersection
    inside = _min(g, _max(g, d2, dw), _c(g, 0.0))
    outside = _hypot(g, _max(g, d2, _c(g, 0.0)), _max(g, dw, _c(g, 0.0)))
    return inside + outside


def _boolean_ir(g, node, a, b):
    op = node.d["op"]
    blend = node.d.get("blend", "hard")
    k = float(node.d.get("k", 0.0))
    if blend == "smooth" and k > 0.0:
        if op == "union":
            return _smooth_min_poly(g, a, b, k)
        if op == "intersect":
            return _smooth_max_poly(g, a, b, k)
        return _smooth_max_poly(g, a, -b, k)            # smooth_difference
    if blend == "chamfer" and k > 0.0:
        if op == "union":
            return _chamfer_min(g, a, b, k)
        if op == "intersect":
            return _chamfer_max(g, a, b, k)
        return _chamfer_max(g, a, -b, k)
    if op == "union":
        return _min(g, a, b)
    if op == "intersect":
        return _max(g, a, b)
    return _max(g, a, -b)                                # difference


def _reflect_ir(p, plane: str):
    pl = str(plane).upper()
    x, y, z = p
    if pl == "XY":
        return (x, y, -z)
    if pl == "YZ":
        return (-x, y, z)
    return (x, -y, z)                                    # XZ


def _untransform_ir(g, p, tr):
    """Inverse of a pattern instance transform (``frep._untransform``)."""
    dx, dy, dz, ang = tr
    x = p[0] - _c(g, dx)
    y = p[1] - _c(g, dy)
    z = p[2] - _c(g, dz)
    if ang:
        a = math.radians(-float(ang))
        ca, sa = math.cos(a), math.sin(a)
        x, y = (_c(g, ca) * x - _c(g, sa) * y, _c(g, sa) * x + _c(g, ca) * y)
    return (x, y, z)


def _compile(g, node, p):
    """The IR expression of ``node``'s field at the point expression ``p``."""
    from harnesscad.io.backends import frep as backend

    t = node.t
    if t == "extrude":
        iu, ivx, iw = backend._plane_axes(node.d["plane"])  # noqa: SLF001
        d2 = _profile_ir(g, node.d["profile"], p[iu], p[ivx])
        dw = _slab_ir(g, p[iw], node.d["w0"], node.d["w1"])
        return _combine_prism_ir(g, d2, dw, float(node.d.get("round", 0.0)),
                                 float(node.d.get("cham", 0.0)))
    if t == "cyl":
        iu, ivx, iw = backend._plane_axes(node.d["plane"])  # noqa: SLF001
        d2 = _hypot(g, p[iu] - _c(g, node.d["cu"]),
                    p[ivx] - _c(g, node.d["cv"])) - _c(g, node.d["r"])
        dw = _slab_ir(g, p[iw], node.d["w0"], node.d["w1"])
        return _combine_prism_ir(g, d2, dw, 0.0, 0.0)
    if t == "revolve":
        return _compile_revolve(g, node, p)
    if t == "bool":
        a = _compile(g, node.d["a"], p)
        b = _compile(g, node.d["b"], p)
        return _boolean_ir(g, node, a, b)
    if t == "shell":
        d = _compile(g, node.d["child"], p)
        return _abs(g, d) - _c(g, float(node.d["thickness"]) / 2.0)
    if t == "mirror":
        a = _compile(g, node.d["child"], p)
        b = _compile(g, node.d["child"], _reflect_ir(p, node.d["plane"]))
        return _min(g, a, b)
    if t == "pattern":
        child = node.d["child"]
        vals = [_compile(g, child, _untransform_ir(g, p, tr))
                for tr in node.d["transforms"]]
        out = vals[0]
        for nxt in vals[1:]:
            out = _min(g, out, nxt)
        return out
    raise CompileError("unknown F-rep node kind '%s'" % (t,))


def _compile_revolve(g, node, p):
    from harnesscad.io.backends import frep as backend

    iu, ivx, iw = backend._plane_axes(node.d["plane"])  # noqa: SLF001
    au, av, du, dv, nu, nv = node.d["axis"]
    pu, pv, pw = p[iu], p[ivx], p[iw]
    s = (pu - _c(g, au)) * _c(g, du) + (pv - _c(g, av)) * _c(g, dv)
    perp = (pu - _c(g, au)) * _c(g, nu) + (pv - _c(g, av)) * _c(g, nv)
    rad = _hypot(g, perp, pw)
    qu = _c(g, au) + s * _c(g, du) + rad * _c(g, nu)
    qv = _c(g, av) + s * _c(g, dv) + rad * _c(g, nv)
    d = _profile_ir(g, node.d["profile"], qu, qv)
    angle = float(node.d.get("angle", 360.0))
    if abs(angle) >= 360.0:
        return d
    # the angular wedge, exactly as frep._eval_revolve computes it
    theta = g.op("atan2", pw, perp)
    half = math.radians(abs(angle)) / 2.0
    # _wrap_angle(a) for a in [-2pi, pi] is mod(a + pi, 2pi) - pi
    a = theta - _c(g, half)
    wrapped = g.op("mod", a + _c(g, math.pi), _c(g, _TWO_PI)) - _c(g, math.pi)
    dt = _abs(g, wrapped)
    wedge = (dt - _c(g, half)) * _max(g, rad, _c(g, 1e-9))
    return _max(g, d, wedge)


def compile_node(node, smooth: bool = False) -> CompiledField:
    """Compile a backend CSG ``Node`` into an arithmetic f-rep IR graph.

    ``smooth=True`` selects the tie-averaging encoding of ``min``/``max`` (see
    :class:`_Builder`) -- the graph to differentiate. The default encoding is the
    graph to bound.

    Raises :class:`CompileError` when the tree contains something the opcode set
    cannot express (today: polygon sketch profiles).
    """
    if node is None:
        raise CompileError("there is no solid to compile")
    graph = ir.Graph()
    g = _Builder(graph, smooth=smooth)
    root = _compile(g, node, (g.x(), g.y(), g.z()))
    return CompiledField(graph, root)


# ---------------------------------------------------------------------------
# the two things the IR buys us
# ---------------------------------------------------------------------------

def exact_normal(field: CompiledField, p: Sequence[float]) -> Vec3:
    """Unit surface normal by forward-mode AD (exact, no finite differences)."""
    return field.normal(p)


def classify_box(field: CompiledField, lo: Sequence[float], hi: Sequence[float],
                 margin: float = 0.0) -> str:
    """libfive's EMPTY / FILLED / AMBIGUOUS verdict for the box ``[lo, hi]``.

    ``margin`` widens the computed interval before the verdict is read, so that
    floating-point disagreement between the IR evaluation and the backend's own
    Python evaluation of the same function can never turn a straddling box into
    a confidently-pruned one. Widening can only ever make a box *more*
    ambiguous, so pruning stays conservative.
    """
    box = iv.eval_interval(field.root, (float(lo[0]), float(lo[1]), float(lo[2])),
                           (float(hi[0]), float(hi[1]), float(hi[2])))
    if margin:
        box = iv.Interval(box.lo - abs(margin), box.hi + abs(margin), box.maybe_nan)
    return iv.classify(box)


def try_compile(node, smooth: bool = False) -> Optional[CompiledField]:
    """:func:`compile_node`, or ``None`` when the tree is not IR-expressible."""
    try:
        return compile_node(node, smooth=smooth)
    except CompileError:
        return None
