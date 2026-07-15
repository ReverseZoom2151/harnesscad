"""Cross-family CSG vocabulary superset -- RapCAD / AngelCAD / OpenJSCAD / replicad.

The harness already has a full OpenSCAD front end (:mod:`...ast.openscad`).  But
OpenSCAD is only one dialect of the same underlying idea: a Constructive Solid
Geometry program built from booleans, transforms, primitives and the two set
operators that are *not* plain booleans -- convex ``hull`` and ``minkowski`` sum.
Four sibling projects speak the same idea in different words:

*   **RapCAD** -- an OpenSCAD-compatible language (Giles Bathgate); same keywords
    plus its own additions (``writeln``, ``assign``, ``pull``).
*   **AngelCAD** -- a C++/AngelScript CAD language; solids are objects and booleans
    are the ``+ - *`` operators over ``solid`` values, primitives are
    ``box``/``cone``/``sphere``.
*   **OpenJSCAD / JSCAD** -- a JavaScript API: ``union``/``subtract``/``intersect``,
    ``hull``/``hullChain``, primitives ``cube``/``cuboid``/``sphere``.
*   **replicad** -- a JavaScript B-rep API over OpenCascade: booleans are the
    methods ``fuse``/``cut``/``intersect`` on a shape.

This module is the **union of their vocabularies**, canonicalised: one
:class:`CsgOp` enum names each concept once, and :data:`DIALECTS` records how every
family spells it (a family that lacks a concept simply has no entry).  That gives:

*   :func:`canonicalise` -- map a family-specific name to the canonical op;
*   :func:`spelling` / :func:`families_supporting` -- the inverse and the coverage;
*   :func:`parse_call` -- recognise a ``name(args)`` call in any family and report
    the canonical op it denotes -- a tiny cross-dialect CSG parser.

Where an operation is portable and geometric rather than merely a keyword, it is
implemented for real, deterministically, in stdlib: :func:`convex_hull_2d`
(Andrew's monotone chain) and :func:`minkowski_sum_2d` (the convex Minkowski sum,
the hull of the pairwise vertex sums).  These are the two operators every family in
this list exposes and that a boolean-only kernel cannot express.

Pure stdlib, deterministic, no execution of any external CAD tool.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "CsgOp",
    "FAMILIES",
    "DIALECTS",
    "canonicalise",
    "spelling",
    "families_supporting",
    "coverage",
    "parse_call",
    "convex_hull_2d",
    "minkowski_sum_2d",
    "polygon_area",
]


class CsgOp(str, Enum):
    """The canonical, family-agnostic CSG concept set (the superset)."""

    # booleans
    UNION = "union"
    DIFFERENCE = "difference"
    INTERSECTION = "intersection"
    # non-boolean set operators
    HULL = "hull"
    MINKOWSKI = "minkowski"
    # transforms
    TRANSLATE = "translate"
    ROTATE = "rotate"
    SCALE = "scale"
    MIRROR = "mirror"
    RESIZE = "resize"
    OFFSET = "offset"
    # 3D primitives
    CUBE = "cube"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CONE = "cone"
    POLYHEDRON = "polyhedron"
    # 2D primitives
    SQUARE = "square"
    CIRCLE = "circle"
    POLYGON = "polygon"
    # 2D -> 3D
    LINEAR_EXTRUDE = "linear_extrude"
    ROTATE_EXTRUDE = "rotate_extrude"


FAMILIES: Tuple[str, ...] = ("openscad", "rapcad", "angelcad", "openjscad", "replicad")


# canonical op -> {family: (spelling, ...)}.  A family may spell an op several ways
# (aliases); absence means the family has no direct equivalent.
DIALECTS: Dict[CsgOp, Dict[str, Tuple[str, ...]]] = {
    CsgOp.UNION: {
        "openscad": ("union",), "rapcad": ("union",),
        "angelcad": ("union", "+"),
        "openjscad": ("union",), "replicad": ("fuse",),
    },
    CsgOp.DIFFERENCE: {
        "openscad": ("difference",), "rapcad": ("difference",),
        "angelcad": ("difference", "-"),
        "openjscad": ("subtract",), "replicad": ("cut",),
    },
    CsgOp.INTERSECTION: {
        "openscad": ("intersection",), "rapcad": ("intersection",),
        "angelcad": ("intersection", "*"),
        "openjscad": ("intersect",), "replicad": ("intersect",),
    },
    CsgOp.HULL: {
        "openscad": ("hull",), "rapcad": ("hull",),
        "angelcad": ("hull",),
        "openjscad": ("hull", "hullChain"), "replicad": ("hull",),
    },
    CsgOp.MINKOWSKI: {
        "openscad": ("minkowski",), "rapcad": ("minkowski",),
        "openjscad": ("expand",),  # JSCAD's expand is a Minkowski-with-a-disc
    },
    CsgOp.TRANSLATE: {
        "openscad": ("translate",), "rapcad": ("translate",),
        "angelcad": ("translate",),
        "openjscad": ("translate",), "replicad": ("translate",),
    },
    CsgOp.ROTATE: {
        "openscad": ("rotate",), "rapcad": ("rotate",),
        "angelcad": ("rotate",),
        "openjscad": ("rotate",), "replicad": ("rotate",),
    },
    CsgOp.SCALE: {
        "openscad": ("scale",), "rapcad": ("scale",),
        "angelcad": ("scale",),
        "openjscad": ("scale",), "replicad": ("scale",),
    },
    CsgOp.MIRROR: {
        "openscad": ("mirror",), "rapcad": ("mirror",),
        "angelcad": ("mirror",),
        "openjscad": ("mirror",), "replicad": ("mirror",),
    },
    CsgOp.RESIZE: {
        "openscad": ("resize",), "rapcad": ("resize",),
        "openjscad": ("resize",),
    },
    CsgOp.OFFSET: {
        "openscad": ("offset",), "rapcad": ("offset",),
        "openjscad": ("offset",), "replicad": ("offset",),
    },
    CsgOp.CUBE: {
        "openscad": ("cube",), "rapcad": ("cube",),
        "angelcad": ("box",),
        "openjscad": ("cube", "cuboid"), "replicad": ("makeBox",),
    },
    CsgOp.SPHERE: {
        "openscad": ("sphere",), "rapcad": ("sphere",),
        "angelcad": ("sphere",),
        "openjscad": ("sphere",), "replicad": ("makeSphere",),
    },
    CsgOp.CYLINDER: {
        "openscad": ("cylinder",), "rapcad": ("cylinder",),
        "angelcad": ("cylinder",),
        "openjscad": ("cylinder",), "replicad": ("makeCylinder",),
    },
    CsgOp.CONE: {
        "angelcad": ("cone",),
        "openjscad": ("cylinderElliptic",),
    },
    CsgOp.POLYHEDRON: {
        "openscad": ("polyhedron",), "rapcad": ("polyhedron",),
        "openjscad": ("polyhedron",),
    },
    CsgOp.SQUARE: {
        "openscad": ("square",), "rapcad": ("square",),
        "openjscad": ("square", "rectangle"), "replicad": ("drawRectangle",),
    },
    CsgOp.CIRCLE: {
        "openscad": ("circle",), "rapcad": ("circle",),
        "openjscad": ("circle",), "replicad": ("drawCircle",),
    },
    CsgOp.POLYGON: {
        "openscad": ("polygon",), "rapcad": ("polygon",),
        "openjscad": ("polygon",), "replicad": ("drawPolysides",),
    },
    CsgOp.LINEAR_EXTRUDE: {
        "openscad": ("linear_extrude",), "rapcad": ("linear_extrude",),
        "angelcad": ("linear_extrude",),
        "openjscad": ("extrudeLinear",), "replicad": ("sketchOnPlane",),
    },
    CsgOp.ROTATE_EXTRUDE: {
        "openscad": ("rotate_extrude",), "rapcad": ("rotate_extrude",),
        "angelcad": ("rotate_extrude",),
        "openjscad": ("extrudeRotate",), "replicad": ("revolution",),
    },
}


# reverse index: (family, spelling) -> CsgOp, built once, deterministically.
def _build_reverse() -> Dict[Tuple[str, str], CsgOp]:
    rev: Dict[Tuple[str, str], CsgOp] = {}
    for op, per_family in DIALECTS.items():
        for family, spellings in per_family.items():
            for name in spellings:
                rev[(family, name)] = op
    return rev


_REVERSE = _build_reverse()


# --------------------------------------------------------------------------- #
# vocabulary queries                                                          #
# --------------------------------------------------------------------------- #
def canonicalise(name: str, family: str) -> Optional[CsgOp]:
    """Map a family-specific spelling to its canonical :class:`CsgOp` (or ``None``)."""
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}")
    return _REVERSE.get((family, name))


def spelling(op: CsgOp, family: str) -> Tuple[str, ...]:
    """Every spelling ``family`` uses for ``op`` (empty if it has no equivalent)."""
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}")
    return DIALECTS.get(op, {}).get(family, ())


def families_supporting(op: CsgOp) -> Tuple[str, ...]:
    """The families that can express ``op`` (sorted, deterministic)."""
    return tuple(sorted(DIALECTS.get(op, {})))


def coverage() -> Dict[str, int]:
    """How many canonical ops each family covers -- the vocabulary-overlap report."""
    counts = {f: 0 for f in FAMILIES}
    for per_family in DIALECTS.values():
        for family in per_family:
            counts[family] += 1
    return counts


def parse_call(text: str, family: str) -> Optional[Tuple[CsgOp, List[str]]]:
    """Recognise a ``name(arg, arg, ...)`` CSG call in ``family``.

    Returns ``(canonical_op, [raw_arg, ...])`` if the callee names a known op in
    that family, else ``None``.  Whitespace-tolerant; a bare ``name()`` yields no
    args.  This is a minimal cross-dialect parser -- it identifies *which* CSG
    concept a call denotes without needing the family's full grammar.
    """
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}")
    s = text.strip().rstrip(";").strip()
    if "(" not in s or not s.endswith(")"):
        return None
    head, _, rest = s.partition("(")
    name = head.strip()
    op = _REVERSE.get((family, name))
    if op is None:
        return None
    inner = rest[:-1].strip()
    if not inner:
        return op, []
    args = _split_top_level(inner)
    return op, args


def _split_top_level(inner: str) -> List[str]:
    """Split on commas that are not nested inside brackets/parens/quotes."""
    args: List[str] = []
    depth = 0
    quote: Optional[str] = None
    buf: List[str] = []
    for ch in inner:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        args.append("".join(buf).strip())
    return args


# --------------------------------------------------------------------------- #
# real geometry: convex hull + Minkowski sum (2D)                             #
# --------------------------------------------------------------------------- #
Point = Tuple[float, float]


def _cross(o: Point, a: Point, b: Point) -> float:
    """Z-component of (a-o) x (b-o): >0 left turn, <0 right turn, 0 collinear."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull_2d(points: Sequence[Point]) -> List[Point]:
    """Convex hull of 2D points (Andrew's monotone chain), CCW, no repeated last.

    Deterministic: points are sorted; collinear points on the hull edges are
    dropped.  Returns the hull vertices in counter-clockwise order.  Fewer than 3
    unique points return the sorted unique set unchanged.  This is exactly the
    ``hull`` operator every family in :data:`FAMILIES` exposes.
    """
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 2:
        return pts

    lower: List[Point] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # concatenate, dropping each list's last point (shared with the other's start).
    return lower[:-1] + upper[:-1]


def polygon_area(poly: Sequence[Point]) -> float:
    """Signed area (shoelace); positive for a CCW polygon."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def minkowski_sum_2d(a: Sequence[Point], b: Sequence[Point]) -> List[Point]:
    """Minkowski sum of two 2D point sets, returned as a convex polygon (CCW).

    For convex operands the exact Minkowski sum is the convex hull of the pairwise
    vertex sums ``{p + q : p in A, q in B}`` -- and taking the hull makes the result
    correct for *any* input point sets (it yields ``hull(A) (+) hull(B)``).  This
    is the genuine ``minkowski`` operator, computed deterministically, not a
    keyword stub.
    """
    if not a or not b:
        return []
    sums: List[Point] = [
        (float(ax) + float(bx), float(ay) + float(by))
        for (ax, ay) in a
        for (bx, by) in b
    ]
    return convex_hull_2d(sums)
