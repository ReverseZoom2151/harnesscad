"""Data-first keyword-tagged S-expression CSG IR and emitter.

A shape is not an object with
methods; it is an inert nested tuple whose head is a keyword tag::

    (":cube", {"x": 1, "y": 2, "z": 3, "center": True})
    (":translate", [10, 0, 0], (":sphere", {"r": 4}))
    (":union", <node>, <node>, ...)

This is a *different representation* from the harness's existing
``programs.solidpy_scad_emit`` (an object tree of ``ScadNode`` instances) and
from ``geometry.libfive_frep_ir`` (an SDF opcode DAG over implicit x/y/z).  The
value a data-first form adds:

* **Canonical inert data.**  A model is tuples and dicts -- directly
serialisable, hashable-after-freezing, diffable, and walkable without any
class. ``postwalk`` is a generic bottom-up transform for structural rewrites.
* **Radian rotation convention.** ``rotate`` stores its angle in
  *radians*; the emitter converts to degrees on the way out (OpenSCAD wants
  degrees).  SolidPython passes the angle through untouched -- a genuinely
  different transform handling.
* **Special-variable dynamic binding.**  ``with_fn`` / ``with_fa`` / ``with_fs``
  / ``with_center`` are context managers; ``circle`` / ``sphere`` / ``cylinder``
  resolve ``$fn`` / ``$fa`` / ``$fs`` and ``center`` from the active binding at
  construction time, mirroring scad-clj's dynamic ``*fn*`` vars.  The object-tree
  emitter has no such ambient resolution.
* **module / library emission.**  ``include`` / ``use`` / ``import_`` / ``call``
  / ``call_module`` / ``define_module`` render OpenSCAD's library-call surface.
* ``excise`` -- scad-clj's "difference from the *last* node" operator.

The emitter (:func:`write_scad`) reproduces scad-clj's ``write-expr``
formatting: two-space indentation per depth, ``name (...) {\\n ... }\\n`` block
layout, the ``$fa=.., $fn=.., $fs=..`` special-variable prefix on curved
primitives, and the polygon / polyhedron point layout.  Floats are formatted
deterministically (fixed precision, trailing zeros trimmed) so a given tree
always yields byte-identical source.

Pure stdlib, deterministic, no wall-clock behaviour.
"""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "Node",
    "write_scad",
    "postwalk",
    "rad_to_deg",
    "deg_to_rad",
    "format_number",
    # special-variable bindings
    "with_fn",
    "with_fa",
    "with_fs",
    "with_center",
    # modifiers
    "background",
    "debug",
    "root",
    "disable",
    # libraries / modules
    "include",
    "use",
    "import_",
    "call",
    "call_module",
    "define_module",
    # 2D
    "square",
    "circle",
    "polygon",
    "text",
    # 3D
    "cube",
    "sphere",
    "cylinder",
    "polyhedron",
    # transforms
    "translate",
    "rotate",
    "rotatev",
    "rotatec",
    "scale",
    "mirror",
    "color",
    "resize",
    "multmatrix",
    "hull",
    "minkowski",
    "offset",
    # booleans
    "union",
    "intersection",
    "difference",
    "excise",
    # extrusion / other
    "extrude_linear",
    "extrude_rotate",
    "projection",
    "project",
    "cut",
    "render",
    "surface",
    # special vars (literal)
    "fa",
    "fn",
    "fs",
]

# A node is a tuple whose first element is a str keyword tag (":cube", ...).
Node = Tuple[Any, ...]

PI = math.pi
TAU = 2.0 * math.pi


def rad_to_deg(radians: float) -> float:
    return radians * 180.0 / PI


def deg_to_rad(degrees: float) -> float:
    return degrees * PI / 180.0


# =========================================================================
# = Dynamic special-variable bindings ($fn / $fa / $fs / center)          =
# =========================================================================
# scad-clj uses Clojure dynamic vars (*fn* *fa* *fs* *center*).  In eager
# Python the faithful idiom is a thread-local binding stack that constructors
# read while a `with_*` block is active.
class _Dyn(threading.local):
    def __init__(self) -> None:
        self.fn: Any = None
        self.fa: Any = None
        self.fs: Any = None
        self.center: bool = True  # scad-clj default *center* is true


_DYN = _Dyn()


@contextmanager
def _bind(attr: str, value: Any) -> Iterator[None]:
    prev = getattr(_DYN, attr)
    setattr(_DYN, attr, value)
    try:
        yield
    finally:
        setattr(_DYN, attr, prev)


def with_fn(value: Any) -> Any:
    """Context manager: resolve ``$fn`` to *value* for shapes built inside."""
    return _bind("fn", value)


def with_fa(value: Any) -> Any:
    return _bind("fa", value)


def with_fs(value: Any) -> Any:
    return _bind("fs", value)


def with_center(value: bool) -> Any:
    return _bind("center", value)


def _fargs() -> Dict[str, Any]:
    """Current $fa/$fn/$fs special variables, if any are bound."""
    out: Dict[str, Any] = {}
    if _DYN.fa is not None:
        out["fa"] = _DYN.fa
    if _DYN.fn is not None:
        out["fn"] = _DYN.fn
    if _DYN.fs is not None:
        out["fs"] = _DYN.fs
    return out


# =========================================================================
# = Constructors (return inert data)                                      =
# =========================================================================
def _flatten(block: Sequence[Any]) -> List[Any]:
    """Flatten a block argument list the way scad-clj's :list handler does.

    A child may be a single node, or a list/tuple *of* nodes (e.g. from
    ``union(*shapes)`` where ``shapes`` is itself a list)."""
    out: List[Any] = []
    for item in block:
        if _is_node(item):
            out.append(item)
        elif isinstance(item, (list, tuple)):
            out.extend(_flatten(item))
        elif item is not None:
            out.append(item)
    return out


def _is_node(x: Any) -> bool:
    return isinstance(x, tuple) and len(x) >= 1 and isinstance(x[0], str) \
        and x[0].startswith(":")


# -- 2D -------------------------------------------------------------------
def square(x: float, y: float, center: Optional[bool] = None) -> Node:
    if center is None:
        center = _DYN.center
    return (":square", {"x": x, "y": y, "center": center})


def circle(r: float) -> Node:
    args = {"r": r}
    args.update(_fargs())
    return (":circle", args)


def polygon(points: Sequence[Sequence[float]],
            paths: Optional[Sequence[Sequence[int]]] = None,
            convexity: Optional[int] = None) -> Node:
    d: Dict[str, Any] = {"points": [list(p) for p in points]}
    if paths is not None:
        d["paths"] = [list(p) for p in paths]
    if convexity is not None:
        d["convexity"] = convexity
    return (":polygon", d)


def text(s: str, **kwargs: Any) -> Node:
    args: Dict[str, Any] = {"text": s}
    if _DYN.fn is not None:
        args["fn"] = _DYN.fn
    args.update(kwargs)
    return (":text", args)


# -- 3D -------------------------------------------------------------------
def sphere(r: float) -> Node:
    args = {"r": r}
    args.update(_fargs())
    return (":sphere", args)


def cube(x: float, y: float, z: float, center: Optional[bool] = None) -> Node:
    if center is None:
        center = _DYN.center
    return (":cube", {"x": x, "y": y, "z": z, "center": center})


def cylinder(rs: Any, h: float, center: Optional[bool] = None) -> Node:
    if center is None:
        center = _DYN.center
    args = dict(_fargs())
    if isinstance(rs, (list, tuple)):
        r1, r2 = rs
        args.update({"h": h, "r1": r1, "r2": r2, "center": center})
    else:
        args.update({"h": h, "r": rs, "center": center})
    return (":cylinder", args)


def polyhedron(points: Sequence[Sequence[float]],
               faces: Sequence[Sequence[int]],
               convexity: Optional[int] = None) -> Node:
    d: Dict[str, Any] = {"points": [list(p) for p in points],
                         "faces": [list(f) for f in faces]}
    if convexity is not None:
        d["convexity"] = convexity
    return (":polyhedron", d)


# -- transforms -----------------------------------------------------------
def translate(v: Sequence[float], *block: Any) -> Node:
    x, y, z = v
    return (":translate", [x, y, z], *_flatten(block))


def rotatev(a: float, v: Sequence[float], *block: Any) -> Node:
    x, y, z = v
    return (":rotatev", [a, [x, y, z]], *_flatten(block))


def rotatec(v: Sequence[float], *block: Any) -> Node:
    x, y, z = v
    return (":rotatec", [x, y, z], *_flatten(block))


def rotate(*args: Any) -> Node:
    """scad-clj rotate: number head -> (angle, axis) form, else vector form.

    The angle is in RADIANS and converted to degrees by the emitter."""
    if args and isinstance(args[0], (int, float)):
        a = args[0]
        v = args[1]
        return rotatev(a, v, *args[2:])
    return rotatec(args[0], *args[1:])


def scale(v: Sequence[float], *block: Any) -> Node:
    x, y, z = v
    return (":scale", [x, y, z], *_flatten(block))


def mirror(v: Sequence[float], *block: Any) -> Node:
    x, y, z = v
    return (":mirror", [x, y, z], *_flatten(block))


def color(rgba: Sequence[float], *block: Any) -> Node:
    r, g, b, a = rgba
    return (":color", [r, g, b, a], *_flatten(block))


def resize(v: Sequence[float], *block: Any, auto: Any = None) -> Node:
    x, y, z = v
    return (":resize", {"x": x, "y": y, "z": z, "auto": auto}, *_flatten(block))


def multmatrix(m: Sequence[Sequence[float]], *block: Any) -> Node:
    return (":multmatrix", [list(row) for row in m], *_flatten(block))


def hull(*block: Any) -> Node:
    return (":hull", *_flatten(block))


def minkowski(*block: Any) -> Node:
    return (":minkowski", *_flatten(block))


def offset(r: Optional[float] = None, *block: Any,
           delta: Optional[float] = None, chamfer: bool = False) -> Node:
    return (":offset", {"r": r, "delta": delta, "chamfer": chamfer},
            *_flatten(block))


# -- booleans -------------------------------------------------------------
def union(*block: Any) -> Node:
    return (":union", *_flatten(block))


def intersection(*block: Any) -> Node:
    return (":intersection", *_flatten(block))


def difference(*block: Any) -> Node:
    return (":difference", *_flatten(block))


def excise(*nodes: Any) -> Node:
    """Like difference, but the subtraction is from the LAST node.

    ``excise(a, b, target)`` == ``target - (a + b)`` in scad-clj terms."""
    flat = _flatten(nodes)
    return difference(flat[-1], *flat[:-1])


# -- extrusion / other ----------------------------------------------------
def extrude_linear(options: Dict[str, Any], *block: Any) -> Node:
    o = dict(options)
    o.setdefault("center", _DYN.center)
    return (":extrude-linear", o, *_flatten(block))


def extrude_rotate(*args: Any) -> Node:
    if args and isinstance(args[0], dict):
        options = dict(args[0])
        block = args[1:]
    else:
        options = {}
        block = args
    if _DYN.fn is not None:
        options.setdefault("fn", _DYN.fn)
    return (":extrude-rotate", options, *_flatten(block))


def projection(cut_flag: bool, *block: Any) -> Node:
    return (":projection", {"cut": cut_flag}, *_flatten(block))


def project(*block: Any) -> Node:
    return projection(False, *block)


def cut(*block: Any) -> Node:
    return projection(True, *block)


def render(*args: Any) -> Node:
    if args and isinstance(args[0], (int, float)):
        return (":render", {"convexity": args[0]}, *_flatten(args[1:]))
    return (":render", {"convexity": 1}, *_flatten(args))


def surface(filepath: str, convexity: Optional[int] = None,
            center: Optional[bool] = None, invert: Optional[bool] = None) -> Node:
    if center is None:
        center = _DYN.center
    return (":surface", {"filepath": filepath, "convexity": convexity,
                         "center": center, "invert": invert})


# -- modifiers ------------------------------------------------------------
def _modifier(char: str, *block: Any) -> Node:
    return (":modifier", char, *_flatten(block))


def debug(*block: Any) -> Node:
    return _modifier("#", *block)


def background(*block: Any) -> Node:
    return _modifier("%", *block)


def disable(*block: Any) -> Node:
    return _modifier("*", *block)


def root(*block: Any) -> Node:
    return _modifier("!", *block)


# -- libraries / modules --------------------------------------------------
def include(library: str) -> Node:
    return (":include", {"library": library})


def use(library: str) -> Node:
    return (":use", {"library": library})


def import_(file: str) -> Node:
    return (":import", file)


def call(function: str, *args: Any) -> Node:
    return (":call", {"function": function}, list(args))


def call_module(module: str, *args: Any) -> Node:
    return (":call-module-no-block", {"module": module}, list(args))


def define_module(module: str, *body: Any) -> Node:
    # body = (arg1, arg2, ..., block-node); scad-clj puts params then the block.
    return (":define-module", {"module": module}, list(body))


# -- special vars (literal statements) ------------------------------------
def fa(x: float) -> Node:
    return (":fa", x)


def fn(x: float) -> Node:
    return (":fn", x)


def fs(x: float) -> Node:
    return (":fs", x)


# =========================================================================
# = Generic tree walk                                                     =
# =========================================================================
def postwalk(f: Any, node: Any) -> Any:
    """Bottom-up structural transform, after clojure.walk/postwalk.

    Rebuilds *node*, replacing every subnode (and finally the node itself)
    with ``f(subnode)``.  Dicts and coordinate lists are walked too."""
    if _is_node(node):
        walked = tuple(postwalk(f, part) for part in node)
        return f(walked)
    if isinstance(node, tuple):
        return f(tuple(postwalk(f, part) for part in node))
    if isinstance(node, list):
        return f([postwalk(f, part) for part in node])
    if isinstance(node, dict):
        return f({k: postwalk(f, v) for k, v in node.items()})
    return f(node)


# =========================================================================
# = Number formatting                                                     =
# =========================================================================
_PRECISION = 10


def format_number(value: Any) -> str:
    """Deterministic OpenSCAD literal for a scalar (trailing zeros trimmed)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            return "0"
        s = "%.*f" % (_PRECISION, value)
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        if s in ("", "-", "-0"):
            s = "0"
        return s
    return str(value)


def _n(value: Any) -> str:
    return format_number(value)


# =========================================================================
# = Emitter (write-expr dispatch)                                         =
# =========================================================================
def _indent(depth: int) -> str:
    return "  " * depth


def _fargs_prefix(args: Dict[str, Any]) -> str:
    out = ""
    if args.get("fa") is not None:
        out += "$fa=" + _n(args["fa"]) + ", "
    if args.get("fn") is not None:
        out += "$fn=" + _n(args["fn"]) + ", "
    if args.get("fs") is not None:
        out += "$fs=" + _n(args["fs"]) + ", "
    return out


def _emit_block(depth: int, block: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for child in block:
        out.extend(_write_expr(depth, child))
    return out


def _points_2d(points: Sequence[Sequence[float]]) -> str:
    return "], [".join(", ".join(_n(v) for v in p) for p in points)


def _write_expr(depth: int, node: Any) -> List[str]:
    # A plain sequence of nodes (scad-clj :list) -- flatten.
    if isinstance(node, (list, tuple)) and node and _is_node(node[0]):
        out: List[str] = []
        for sub in node:
            out.extend(_write_expr(depth, sub))
        return out
    if not _is_node(node):
        return []

    tag = node[0]
    handler = _EMITTERS.get(tag)
    if handler is None:
        return ["//(" + str(tag) + " " + str(node[1:]) + ")"]
    return handler(depth, node)


# -- individual emitters --------------------------------------------------
def _e_square(depth: int, node: Node) -> List[str]:
    a = node[1]
    s = _indent(depth) + "square ([" + _n(a["x"]) + ", " + _n(a["y"]) + "]"
    if a.get("center"):
        s += ", center=true"
    return [s + ");\n"]


def _e_circle(depth: int, node: Node) -> List[str]:
    a = node[1]
    return [_indent(depth) + "circle (" + _fargs_prefix(a) + "r=" + _n(a["r"]) + ");\n"]


def _e_polygon(depth: int, node: Node) -> List[str]:
    a = node[1]
    s = _indent(depth) + "polygon (points=[[" + _points_2d(a["points"]) + "]]"
    if a.get("paths") is not None:
        s += ", paths=[[" + "], [".join(",".join(_n(v) for v in p)
                                         for p in a["paths"]) + "]]"
    if a.get("convexity") is not None:
        s += ", convexity=" + _n(a["convexity"])
    return [s + ");\n"]


def _e_text(depth: int, node: Node) -> List[str]:
    a = node[1]
    s = _indent(depth) + 'text ("' + str(a["text"]) + '"'
    order = ["fn", "size", "font", "halign", "valign", "spacing", "direction",
             "language", "script"]
    labels = {"fn": "$fn"}
    quoted = {"font", "halign", "valign", "direction", "language", "script"}
    for k in order:
        if a.get(k) is not None:
            label = labels.get(k, k)
            if k in quoted:
                s += ', %s="%s"' % (label, a[k])
            else:
                s += ", %s=%s" % (label, _n(a[k]))
    return [s + ");\n"]


def _e_sphere(depth: int, node: Node) -> List[str]:
    a = node[1]
    return [_indent(depth) + "sphere (" + _fargs_prefix(a) + "r=" + _n(a["r"]) + ");\n"]


def _e_cube(depth: int, node: Node) -> List[str]:
    a = node[1]
    s = _indent(depth) + "cube ([" + _n(a["x"]) + ", " + _n(a["y"]) + ", " \
        + _n(a["z"]) + "]"
    if a.get("center"):
        s += ", center=true"
    return [s + ");\n"]


def _e_cylinder(depth: int, node: Node) -> List[str]:
    a = node[1]
    s = _indent(depth) + "cylinder (" + _fargs_prefix(a) + "h=" + _n(a["h"])
    if "r" in a:
        s += ", r=" + _n(a["r"])
    else:
        s += ", r1=" + _n(a["r1"]) + ", r2=" + _n(a["r2"])
    if a.get("center"):
        s += ", center=true"
    return [s + ");\n"]


def _e_polyhedron(depth: int, node: Node) -> List[str]:
    a = node[1]
    faces = "], [".join(", ".join(_n(v) for v in f) for f in a["faces"])
    s = _indent(depth) + "polyhedron (points=[[" + _points_2d(a["points"]) \
        + "]], faces=[[" + faces + "]]"
    if a.get("convexity") is not None:
        s += ", convexity=" + _n(a["convexity"])
    return [s + ");\n"]


def _block_wrap(depth: int, head: str, block: Sequence[Any]) -> List[str]:
    return [_indent(depth) + head + " {\n"] + _emit_block(depth + 1, block) \
        + [_indent(depth) + "}\n"]


def _e_translate(depth: int, node: Node) -> List[str]:
    x, y, z = node[1]
    return _block_wrap(depth, "translate ([" + _n(x) + ", " + _n(y) + ", "
                       + _n(z) + "])", node[2:])


def _e_rotatev(depth: int, node: Node) -> List[str]:
    a, (x, y, z) = node[1]
    head = "rotate (a=" + _n(rad_to_deg(a)) + ", v=[" + _n(x) + ", " + _n(y) \
        + ", " + _n(z) + "])"
    return _block_wrap(depth, head, node[2:])


def _e_rotatec(depth: int, node: Node) -> List[str]:
    x, y, z = node[1]
    # scad-clj uses no spaces after commas in the vector rotate form.
    head = "rotate ([" + _n(rad_to_deg(x)) + "," + _n(rad_to_deg(y)) + "," \
        + _n(rad_to_deg(z)) + "])"
    return _block_wrap(depth, head, node[2:])


def _e_scale(depth: int, node: Node) -> List[str]:
    x, y, z = node[1]
    return _block_wrap(depth, "scale ([" + _n(x) + ", " + _n(y) + ", " + _n(z)
                       + "])", node[2:])


def _e_mirror(depth: int, node: Node) -> List[str]:
    x, y, z = node[1]
    return _block_wrap(depth, "mirror ([" + _n(x) + ", " + _n(y) + ", " + _n(z)
                       + "])", node[2:])


def _e_color(depth: int, node: Node) -> List[str]:
    r, g, b, a = node[1]
    return _block_wrap(depth, "color ([" + _n(r) + ", " + _n(g) + ", " + _n(b)
                       + ", " + _n(a) + "])", node[2:])


def _e_resize(depth: int, node: Node) -> List[str]:
    a = node[1]
    head = "resize ([" + _n(a["x"]) + ", " + _n(a["y"]) + ", " + _n(a["z"]) + "]"
    auto = a.get("auto")
    if auto is not None:
        if isinstance(auto, (list, tuple)):
            head += " auto=[" + ", ".join("true" if bool(v) else "false"
                                          for v in auto) + "]"
        else:
            head += " auto=" + ("true" if bool(auto) else "false")
    head += ")"
    return _block_wrap(depth, head, node[2:])


def _e_multmatrix(depth: int, node: Node) -> List[str]:
    m = node[1]
    body = "[" + ",".join("[" + ",".join(_n(v) for v in row) + "]"
                          for row in m) + "]"
    return _block_wrap(depth, "multmatrix(" + body + ")", node[2:])


def _e_hull(depth: int, node: Node) -> List[str]:
    return _block_wrap(depth, "hull ()", node[1:])


def _e_minkowski(depth: int, node: Node) -> List[str]:
    return _block_wrap(depth, "minkowski ()", node[1:])


def _e_offset(depth: int, node: Node) -> List[str]:
    a = node[1]
    if a.get("r") is not None:
        head = "offset (r = " + _n(a["r"])
    else:
        head = "offset (delta = " + _n(a["delta"])
    if a.get("chamfer"):
        head += ", chamfer=true"
    head += ")"
    return _block_wrap(depth, head, node[2:])


def _e_union(depth: int, node: Node) -> List[str]:
    return _block_wrap(depth, "union ()", node[1:])


def _e_difference(depth: int, node: Node) -> List[str]:
    return _block_wrap(depth, "difference ()", node[1:])


def _e_intersection(depth: int, node: Node) -> List[str]:
    return _block_wrap(depth, "intersection ()", node[1:])


def _e_extrude_linear(depth: int, node: Node) -> List[str]:
    o = node[1]
    head = "linear_extrude (height=" + _n(o["height"])
    if o.get("twist") is not None:
        head += ", twist=" + _n(rad_to_deg(o["twist"]))
    if o.get("convexity") is not None:
        head += ", convexity=" + _n(o["convexity"])
    if o.get("slices") is not None:
        head += ", slices=" + _n(o["slices"])
    sc = o.get("scale")
    if sc is not None:
        if isinstance(sc, (list, tuple)):
            head += ", scale=[" + _n(sc[0]) + ", " + _n(sc[1]) + "]"
        else:
            head += ", scale=" + _n(sc)
    if o.get("center"):
        head += ", center=true"
    head += ")"
    return _block_wrap(depth, head, node[2:])


def _e_extrude_rotate(depth: int, node: Node) -> List[str]:
    o = node[1]
    parts = []
    if o.get("convexity") is not None:
        parts.append("convexity=" + _n(o["convexity"]))
    if o.get("angle") is not None:
        parts.append("angle=" + _n(o["angle"]))
    if o.get("fn") is not None:
        parts.append("$fn=" + _n(o["fn"]))
    return _block_wrap(depth, "rotate_extrude (" + ", ".join(parts) + ")",
                       node[2:])


def _e_projection(depth: int, node: Node) -> List[str]:
    cut_flag = node[1]["cut"]
    return _block_wrap(depth, "projection (cut = "
                       + ("true" if cut_flag else "false") + ")", node[2:])


def _e_render(depth: int, node: Node) -> List[str]:
    return _block_wrap(depth, "render (convexity=" + _n(node[1]["convexity"])
                       + ")", node[2:])


def _e_surface(depth: int, node: Node) -> List[str]:
    a = node[1]
    s = _indent(depth) + 'surface (file = "' + a["filepath"] + '"'
    if a.get("convexity") is not None:
        s += ", convexity=" + _n(a["convexity"])
    if a.get("center"):
        s += ", center=true"
    if a.get("invert"):
        s += ", invert=true"
    return [s + ");\n"]


def _e_modifier(depth: int, node: Node) -> List[str]:
    char = node[1]
    return [_indent(depth) + char + "union () {\n"] \
        + _emit_block(depth, node[2:]) + [_indent(depth) + "}\n"]


def _e_include(depth: int, node: Node) -> List[str]:
    return [_indent(depth) + "include <" + node[1]["library"] + ">\n"]


def _e_use(depth: int, node: Node) -> List[str]:
    return [_indent(depth) + "use <" + node[1]["library"] + ">\n"]


def _e_import(depth: int, node: Node) -> List[str]:
    return [_indent(depth) + 'import ("' + node[1] + '");\n']


def _make_args(args: Sequence[Any]) -> str:
    pieces = []
    for a in args:
        if isinstance(a, dict):
            pieces.append(", ".join(str(k) + "=" + _make_args([v])
                                    for k, v in a.items()))
        elif isinstance(a, (list, tuple)):
            pieces.append("[" + _make_args(a) + "]")
        else:
            pieces.append(_n(a))
    return ", ".join(pieces)


def _e_call(depth: int, node: Node) -> List[str]:
    return [_indent(depth) + node[1]["function"] + "("
            + _make_args(node[2]) + ");\n"]


def _e_call_module_no_block(depth: int, node: Node) -> List[str]:
    return [_indent(depth) + node[1]["module"] + " ("
            + _make_args(node[2]) + ");\n"]


def _e_define_module(depth: int, node: Node) -> List[str]:
    body = node[2]
    params, block = body[:-1], body[-1]
    return [_indent(depth) + "module " + node[1]["module"] + "("
            + _make_args(params) + ") {\n"] \
        + _emit_block(depth + 1, [block]) + [_indent(depth) + "};\n"]


def _e_fa(depth: int, node: Node) -> List[str]:
    return [_indent(depth) + "$fa = " + _n(node[1]) + ";\n"]


def _e_fn(depth: int, node: Node) -> List[str]:
    return [_indent(depth) + "$fn = " + _n(node[1]) + ";\n"]


def _e_fs(depth: int, node: Node) -> List[str]:
    return [_indent(depth) + "$fs = " + _n(node[1]) + ";\n"]


_EMITTERS = {
    ":square": _e_square,
    ":circle": _e_circle,
    ":polygon": _e_polygon,
    ":text": _e_text,
    ":sphere": _e_sphere,
    ":cube": _e_cube,
    ":cylinder": _e_cylinder,
    ":polyhedron": _e_polyhedron,
    ":translate": _e_translate,
    ":rotatev": _e_rotatev,
    ":rotatec": _e_rotatec,
    ":scale": _e_scale,
    ":mirror": _e_mirror,
    ":color": _e_color,
    ":resize": _e_resize,
    ":multmatrix": _e_multmatrix,
    ":hull": _e_hull,
    ":minkowski": _e_minkowski,
    ":offset": _e_offset,
    ":union": _e_union,
    ":difference": _e_difference,
    ":intersection": _e_intersection,
    ":extrude-linear": _e_extrude_linear,
    ":extrude-rotate": _e_extrude_rotate,
    ":projection": _e_projection,
    ":render": _e_render,
    ":surface": _e_surface,
    ":modifier": _e_modifier,
    ":include": _e_include,
    ":use": _e_use,
    ":import": _e_import,
    ":call": _e_call,
    ":call-module-no-block": _e_call_module_no_block,
    ":define-module": _e_define_module,
    ":fa": _e_fa,
    ":fn": _e_fn,
    ":fs": _e_fs,
}


def write_scad(*block: Any) -> str:
    """Render one or more nodes to OpenSCAD source text (scad-clj write-scad)."""
    return "".join(_emit_block(0, _flatten(block)))
