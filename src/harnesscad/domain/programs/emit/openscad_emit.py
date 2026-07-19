"""Python object model -> OpenSCAD source emitter.

A tree of Python objects (``cube``, ``translate``, ``difference``, ...) is
rendered to OpenSCAD source text. No geometry kernel is involved -- rendering
is a deterministic tree walk plus value formatting.

The harness already has the *inverse* direction (``programs.scadlm_ast`` parses
OpenSCAD source into an AST; ``geometry.scadlm_csg_eval`` evaluates it into a
CSG tree).  What it lacked is the forward direction: a first-class, composable
Python object model that *emits* OpenSCAD.  That closes the loop -- a generator
can build a model programmatically, emit it, and immediately verify it by
parsing and evaluating the emitted source with the existing modules.

The object model provides:

  * :class:`ScadNode` -- name + params + children + modifier, with the operator
    sugar ``+`` (union), ``-`` (difference), ``*`` (intersection) and
    ``obj(child, ...)`` to add children;
  * the whole builtin vocabulary: 2D/3D primitives, transforms, booleans,
    extrusions, ``hull``/``minkowski``/``offset``/``projection``/``render``,
    ``text``, ``import_``, ``surface``;
  * the ``* ! # %`` modifier characters (:func:`disable`, :func:`root`,
    :func:`debug`, :func:`background`);
  * SolidPython's ``segments`` -> ``$fn`` parameter aliasing and its
    Python-reserved-word escaping (``or_`` -> ``or``);
  * SolidPython's **hole** / **part** mechanism, which OpenSCAD has no
    equivalent of: a subtree marked as a hole is *lifted* out of its position
    and subtracted at the root (or at the nearest enclosing ``part``), so that
    later unions can never fill it back in.  Booleans on the path to a hole are
    rewritten to ``union`` when the hole branch is re-emitted -- an
    intersection/difference must not shrink the void.  Here the rewrite is
    structural (SolidPython does it with a string ``replace``, which corrupts
    identifiers that merely contain "union"/"difference").

Deterministic: parameters are emitted in a stable order (positional indices
first, then named keys sorted), floats are formatted with a fixed precision and
trailing zeros trimmed, so the same tree always yields byte-identical source.
Pure stdlib.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

__all__ = [
    "ScadNode",
    "scad_render",
    "format_value",
    # primitives
    "cube",
    "sphere",
    "cylinder",
    "polyhedron",
    "circle",
    "square",
    "polygon",
    "text",
    "surface",
    "import_",
    # booleans
    "union",
    "difference",
    "intersection",
    # transforms
    "translate",
    "rotate",
    "scale",
    "mirror",
    "resize",
    "multmatrix",
    "color",
    "offset",
    "hull",
    "minkowski",
    "projection",
    "render",
    "linear_extrude",
    "rotate_extrude",
    # solidpython extensions
    "hole",
    "part",
    "debug",
    "background",
    "root",
    "disable",
    # direction helpers
    "up",
    "down",
    "left",
    "right",
    "forward",
    "back",
]

# Names that exist only in SolidPython, never in OpenSCAD output.
NON_RENDERED = ("hole", "part")

MODIFIERS = {
    "disable": "*",
    "debug": "#",
    "background": "%",
    "root": "!",
    "*": "*",
    "#": "#",
    "%": "%",
    "!": "!",
}

# Python keywords that are legal OpenSCAD identifiers: users write ``or_``,
# OpenSCAD wants ``or``.
_RESERVED = (
    "and",
    "or",
    "not",
    "import",
    "for",
    "if",
    "else",
    "in",
    "is",
    "lambda",
    "class",
    "def",
    "return",
    "from",
    "global",
    "pass",
    "assert",
)

FLOAT_PRECISION = 10


def _unsub_keyword(word: str) -> str:
    if word.endswith("_") and word[:-1] in _RESERVED:
        return word[:-1]
    return word


def format_value(value: Any) -> str:
    """Format a Python value as an OpenSCAD literal (SolidPython ``py2openscad``)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = "%.*f" % (FLOAT_PRECISION, value)
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        if text in ("", "-0", "-"):
            text = "0"
        return text
    if isinstance(value, str):
        return '"%s"' % value.replace("\\", "\\\\").replace('"', '\\"')
    if isinstance(value, ScadNode):
        raise TypeError("ScadNode is not a valid parameter value")
    if isinstance(value, Iterable):
        return "[" + ", ".join(format_value(v) for v in value) + "]"
    return str(value)


class ScadNode:
    """One OpenSCAD callable: ``name(params) { children }``."""

    def __init__(self, name: str, params: Optional[Dict[Any, Any]] = None) -> None:
        self.name = name
        self.params: Dict[Any, Any] = dict(params or {})
        self.children: List["ScadNode"] = []
        self.modifier = ""
        self.is_hole = False
        self.is_part_root = False
        self.traits: Dict[str, Dict[str, Any]] = {}

    # -- tree building ----------------------------------------------------
    def add(self, child: Union["ScadNode", Sequence["ScadNode"]]) -> "ScadNode":
        if isinstance(child, ScadNode):
            self.children.append(child)
        elif isinstance(child, (list, tuple)):
            for c in child:
                self.add(c)
        else:
            raise TypeError("cannot add %r to a ScadNode" % type(child).__name__)
        return self

    def __call__(self, *args: Union["ScadNode", Sequence["ScadNode"]]) -> "ScadNode":
        for a in args:
            self.add(a)
        return self

    def add_param(self, key: str, value: Any) -> "ScadNode":
        if key == "$fn":
            key = "segments"
        self.params[key] = value
        return self

    def set_modifier(self, m: str) -> "ScadNode":
        self.modifier = MODIFIERS.get(m.lower(), "")
        return self

    def set_hole(self, is_hole: bool = True) -> "ScadNode":
        self.is_hole = is_hole
        return self

    def set_part_root(self, is_root: bool = True) -> "ScadNode":
        self.is_part_root = is_root
        return self

    def add_trait(self, name: str, data: Dict[str, Any]) -> "ScadNode":
        self.traits[name] = data
        return self

    def get_trait(self, name: str) -> Optional[Dict[str, Any]]:
        return self.traits.get(name)

    def copy(self) -> "ScadNode":
        other = ScadNode(self.name, self.params)
        other.modifier = self.modifier
        other.is_hole = self.is_hole
        other.is_part_root = self.is_part_root
        other.traits = {k: dict(v) for k, v in self.traits.items()}
        other.children = [c.copy() for c in self.children]
        return other

    # -- operator sugar ---------------------------------------------------
    def __add__(self, other: "ScadNode") -> "ScadNode":
        return ScadNode("union")(self, other)

    def __radd__(self, other: Any) -> "ScadNode":
        if other == 0:  # allows sum([...])
            return self
        return ScadNode("union")(other, self)

    def __sub__(self, other: "ScadNode") -> "ScadNode":
        return ScadNode("difference")(self, other)

    def __mul__(self, other: "ScadNode") -> "ScadNode":
        return ScadNode("intersection")(self, other)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "ScadNode(%r, %d children)" % (self.name, len(self.children))

    # -- rendering --------------------------------------------------------
    def render(self, header: str = "") -> str:
        return scad_render(self, header)


# =========================================================================
# = Rendering                                                             =
# =========================================================================
def _param_text(node: ScadNode) -> str:
    params = {_unsub_keyword(str(k)) if isinstance(k, str) else k: v
              for k, v in node.params.items()}
    if "segments" in params:
        params["$fn"] = params.pop("segments")

    positional = sorted(k for k in params if isinstance(k, int))
    named = sorted(k for k in params if not isinstance(k, int))

    parts: List[str] = []
    for k in positional:
        v = params[k]
        if v is None:
            continue
        parts.append(format_value(v))
    for k in named:
        v = params[k]
        if v is None:
            continue
        parts.append("%s = %s" % (k, format_value(v)))
    return ", ".join(parts)


def _emit(node: ScadNode, level: int, indent: str) -> List[str]:
    pad = indent * level
    if node.name in NON_RENDERED:
        # hole()/part() are transparent groups: emit only their children
        lines: List[str] = []
        for c in node.children:
            lines.extend(_emit(c, level, indent))
        return lines

    head = "%s%s%s(%s)" % (pad, node.modifier, node.name, _param_text(node))
    if not node.children:
        return [head + ";"]

    lines = [head + " {"]
    for c in node.children:
        lines.extend(_emit(c, level + 1, indent))
    lines.append(pad + "}")
    return lines


def _has_hole(node: ScadNode) -> bool:
    if node.is_hole:
        return True
    for c in node.children:
        if c.is_part_root:
            continue
        if _has_hole(c):
            return True
    return False


def _strip_holes(node: ScadNode, is_scope_root: bool = False) -> ScadNode:
    """Positive geometry: drop hole subtrees, resolve nested parts locally."""
    if node.is_part_root and not is_scope_root:
        return _resolve(node)
    out = ScadNode(node.name, node.params)
    out.modifier = node.modifier
    for c in node.children:
        if c.is_hole:
            continue
        out.add(_strip_holes(c))
    return out


def _hole_branch(node: ScadNode) -> Optional[ScadNode]:
    """Re-emit only the paths leading to holes; booleans become unions."""
    if node.is_hole:
        out = node.copy()
        out.is_hole = False
        out.name = "union" if out.name == "hole" else out.name
        return out
    kids: List[ScadNode] = []
    for c in node.children:
        if c.is_part_root and not c.is_hole:
            continue  # a part resolves its own holes
        branch = _hole_branch(c)
        if branch is not None:
            kids.append(branch)
    if not kids:
        return None
    name = node.name
    if name in ("difference", "intersection"):
        # An intersection/difference must never shrink a void.
        name = "union"
    if name in NON_RENDERED:
        name = "union"
    out = ScadNode(name, node.params)
    for k in kids:
        out.add(k)
    return out


def _resolve(node: ScadNode) -> ScadNode:
    """Materialise SolidPython holes into a real ``difference()``."""
    positive = _strip_holes(node, is_scope_root=True)
    hole_kids: List[ScadNode] = []
    for c in node.children:
        if c.is_part_root and not c.is_hole:
            continue
        branch = _hole_branch(c)
        if branch is not None:
            hole_kids.append(branch)
    if not hole_kids:
        return positive
    return ScadNode("difference")(positive, *hole_kids)


def scad_render(node: ScadNode, header: str = "", indent: str = "    ") -> str:
    """Render ``node`` (and its subtree) to OpenSCAD source text."""
    if not isinstance(node, ScadNode):
        raise TypeError("scad_render() expects a ScadNode")
    resolved = _resolve(node)
    body = "\n".join(_emit(resolved, 0, indent))
    if header and not header.endswith("\n"):
        header += "\n"
    return header + body + "\n"


# =========================================================================
# = Builtin vocabulary                                                    =
# =========================================================================
def _node(name: str, **params: Any) -> ScadNode:
    return ScadNode(name, {k: v for k, v in params.items() if v is not None})


# -- 3D primitives --------------------------------------------------------
def cube(size: Any = 1, center: Optional[bool] = None) -> ScadNode:
    return _node("cube", size=size, center=center)


def sphere(r: Optional[float] = None, d: Optional[float] = None,
           segments: Optional[int] = None) -> ScadNode:
    return _node("sphere", r=r, d=d, segments=segments)


def cylinder(r: Optional[float] = None, h: Optional[float] = None,
             r1: Optional[float] = None, r2: Optional[float] = None,
             d: Optional[float] = None, d1: Optional[float] = None,
             d2: Optional[float] = None, center: Optional[bool] = None,
             segments: Optional[int] = None) -> ScadNode:
    return _node("cylinder", r=r, h=h, r1=r1, r2=r2, d=d, d1=d1, d2=d2,
                 center=center, segments=segments)


def polyhedron(points: Sequence[Sequence[float]], faces: Sequence[Sequence[int]],
               convexity: Optional[int] = None) -> ScadNode:
    return _node("polyhedron", points=[list(p) for p in points],
                 faces=[list(f) for f in faces], convexity=convexity)


# -- 2D primitives --------------------------------------------------------
def circle(r: Optional[float] = None, d: Optional[float] = None,
           segments: Optional[int] = None) -> ScadNode:
    return _node("circle", r=r, d=d, segments=segments)


def square(size: Any = 1, center: Optional[bool] = None) -> ScadNode:
    return _node("square", size=size, center=center)


def polygon(points: Sequence[Sequence[float]],
            paths: Optional[Sequence[Sequence[int]]] = None) -> ScadNode:
    node = _node("polygon", points=[list(p) for p in points])
    if paths is not None:
        node.params["paths"] = [list(p) for p in paths]
    return node


def text(text: str, size: Optional[float] = None, font: Optional[str] = None,
         halign: Optional[str] = None, valign: Optional[str] = None,
         spacing: Optional[float] = None, direction: Optional[str] = None,
         language: Optional[str] = None, script: Optional[str] = None,
         segments: Optional[int] = None) -> ScadNode:
    return _node("text", text=text, size=size, font=font, halign=halign,
                 valign=valign, spacing=spacing, direction=direction,
                 language=language, script=script, segments=segments)


def surface(file: str, center: Optional[bool] = None,
            invert: Optional[bool] = None, convexity: Optional[int] = None) -> ScadNode:
    return _node("surface", file=file, center=center, invert=invert,
                 convexity=convexity)


def import_(file: str, layer: Optional[str] = None,
            convexity: Optional[int] = None) -> ScadNode:
    return _node("import", file=file, layer=layer, convexity=convexity)


# -- booleans -------------------------------------------------------------
def union() -> ScadNode:
    return ScadNode("union")


def difference() -> ScadNode:
    return ScadNode("difference")


def intersection() -> ScadNode:
    return ScadNode("intersection")


# -- transforms -----------------------------------------------------------
def translate(v: Sequence[float]) -> ScadNode:
    return _node("translate", v=list(v))


def rotate(a: Any = None, v: Optional[Sequence[float]] = None) -> ScadNode:
    node = ScadNode("rotate")
    if a is not None:
        node.params["a"] = list(a) if isinstance(a, (list, tuple)) else a
    if v is not None:
        node.params["v"] = list(v)
    return node


def scale(v: Any) -> ScadNode:
    return _node("scale", v=list(v) if isinstance(v, (list, tuple)) else v)


def mirror(v: Sequence[float]) -> ScadNode:
    return _node("mirror", v=list(v))


def resize(newsize: Sequence[float], auto: Any = None) -> ScadNode:
    return _node("resize", newsize=list(newsize), auto=auto)


def multmatrix(m: Sequence[Sequence[float]]) -> ScadNode:
    return _node("multmatrix", m=[list(row) for row in m])


def color(c: Any, alpha: Optional[float] = None) -> ScadNode:
    return _node("color", c=list(c) if isinstance(c, (list, tuple)) else c,
                 alpha=alpha)


def offset(r: Optional[float] = None, delta: Optional[float] = None,
           chamfer: Optional[bool] = None, segments: Optional[int] = None) -> ScadNode:
    return _node("offset", r=r, delta=delta, chamfer=chamfer, segments=segments)


def hull() -> ScadNode:
    return ScadNode("hull")


def minkowski() -> ScadNode:
    return ScadNode("minkowski")


def projection(cut: Optional[bool] = None) -> ScadNode:
    return _node("projection", cut=cut)


def render(convexity: Optional[int] = None) -> ScadNode:
    return _node("render", convexity=convexity)


def linear_extrude(height: Optional[float] = None, center: Optional[bool] = None,
                   convexity: Optional[int] = None, twist: Optional[float] = None,
                   slices: Optional[int] = None, scale: Any = None,
                   segments: Optional[int] = None) -> ScadNode:
    return _node("linear_extrude", height=height, center=center,
                 convexity=convexity, twist=twist, slices=slices,
                 scale=list(scale) if isinstance(scale, (list, tuple)) else scale,
                 segments=segments)


def rotate_extrude(angle: Optional[float] = None, convexity: Optional[int] = None,
                   segments: Optional[int] = None) -> ScadNode:
    return _node("rotate_extrude", angle=angle, convexity=convexity,
                 segments=segments)


# -- SolidPython extensions ----------------------------------------------
def hole() -> ScadNode:
    """A subtree that is subtracted at the root, not where it appears."""
    return ScadNode("hole").set_hole(True)


def part() -> ScadNode:
    """A scope whose holes are resolved locally instead of at the root."""
    return ScadNode("part").set_part_root(True)


def debug(node: ScadNode) -> ScadNode:
    return node.set_modifier("#")


def background(node: ScadNode) -> ScadNode:
    return node.set_modifier("%")


def root(node: ScadNode) -> ScadNode:
    return node.set_modifier("!")


def disable(node: ScadNode) -> ScadNode:
    return node.set_modifier("*")


# -- direction helpers ----------------------------------------------------
def up(z: float) -> ScadNode:
    return translate((0, 0, z))


def down(z: float) -> ScadNode:
    return translate((0, 0, -z))


def right(x: float) -> ScadNode:
    return translate((x, 0, 0))


def left(x: float) -> ScadNode:
    return translate((-x, 0, 0))


def forward(y: float) -> ScadNode:
    return translate((0, y, 0))


def back(y: float) -> ScadNode:
    return translate((0, -y, 0))
