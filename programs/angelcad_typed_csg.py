"""AngelCAD-style TYPED CSG language: a dimension-checked CSG AST.

AngelCAD (arnholm/angelcad, the ``as_csg`` script compiler) is the interesting
counter-example to OpenSCAD: its script language is *statically typed*.  Where
OpenSCAD has one untyped "geometry" value and happily lets a program write
``union() { circle(3); cube(2); }`` -- a 2D shape unioned with a 3D solid, which
the kernel then quietly mangles -- AngelCAD's AngelScript API declares a real
class hierarchy::

    shape                       (abstract)
      shape2d                   (abstract)  circle square rectangle polygon
                                            union2d difference2d intersection2d
                                            hull2d fill2d offset2d minkowski2d
                                            projection2d
      solid                     (abstract)  sphere cube cuboid cylinder cone
                                            polyhedron
                                            union3d difference3d intersection3d
                                            hull3d minkowski3d
                                            linear_extrude rotate_extrude
                                            transform_extrude sweep
    tmatrix                     (abstract)  translate rotate_x/y/z scale mirror
                                            hmatrix
    pos2 pos3 vec2 vec3 pface spline_path   (value types)

and the registered operators are typed too::

    solid@   opAdd(solid@ b)      // solid + solid  -> union3d
    shape2d@ opAdd(shape2d@ b)    // shape2d + shape2d -> union2d
    solid@   opMul(tmatrix@ m)    // matrix * solid -> solid
    tmatrix@ opMul(tmatrix@ B)    // matrix composition

so ``circle(3) + cube(2)`` is a *compile error*, not a silent modelling bug.
Only two operators cross the dimension boundary: the extrudes (2D -> 3D) and
``projection2d`` (3D -> 2D).

This module reimplements that idea as a small typed CSG AST for the harness:

* :class:`Node` -- an operation with named parameters and children;
* :data:`OPS` -- the operator table (result type, required child type, arity,
  parameter schema) transcribed from the AngelCAD ``InstallType`` registrations;
* :func:`check` -- a deterministic type/dimension checker returning a sorted
  list of :class:`Diagnostic` records (2D/3D mixing, wrong arity, missing or
  non-positive parameters, out-of-range polyhedron indices, ...);
* :class:`TMatrix` -- the ``tmatrix`` value type: composable 4x4 homogeneous
  transforms with ``xdir()/ydir()/zdir()/origin()`` accessors and the
  ``hmatrix(xvec, yvec[, zvec], pos)`` constructor that orthonormalises a frame.

Difference from what the harness already has: ``programs.scadlm_ast`` +
``geometry.scadlm_csg_eval`` are an *untyped* OpenSCAD front end (parse and
evaluate), and ``programs.solidpy_scad_emit`` emits untyped OpenSCAD.  Nothing
in the harness type-checks a CSG program, and nothing distinguishes 2D profiles
from 3D solids at the type level.  That is exactly the class of error an LLM
makes when writing CAD code, and it is catchable offline with zero geometry.

Pure stdlib, deterministic, no kernel.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "SOLID",
    "SHAPE2D",
    "TMatrix",
    "Node",
    "Diagnostic",
    "CsgTypeError",
    "OPS",
    "OpSpec",
    "result_type",
    "check",
    "type_check",
    "identity",
    "translate",
    "rotate_x",
    "rotate_y",
    "rotate_z",
    "scale",
    "mirror",
    "hmatrix",
    "circle",
    "square",
    "rectangle",
    "polygon",
    "sphere",
    "cube",
    "cuboid",
    "cylinder",
    "cone",
    "polyhedron",
    "union2d",
    "difference2d",
    "intersection2d",
    "hull2d",
    "fill2d",
    "offset2d",
    "minkowski2d",
    "projection2d",
    "union3d",
    "difference3d",
    "intersection3d",
    "hull3d",
    "minkowski3d",
    "linear_extrude",
    "rotate_extrude",
    "transform_extrude",
    "sweep",
    "transform",
]

SOLID = "solid"
SHAPE2D = "shape2d"

_DEG = math.pi / 180.0


# --------------------------------------------------------------------------
# tmatrix
# --------------------------------------------------------------------------


class TMatrix:
    """A 4x4 homogeneous transform (AngelCAD ``tmatrix``).

    Row-major, column-vector convention: ``p' = M * p``.  ``A * B`` is the
    matrix product, i.e. *B is applied first*, matching AngelCAD's
    ``m_transform = matrix->matrix() * m_transform``.
    """

    __slots__ = ("rows", "kind")

    def __init__(self, rows: Sequence[Sequence[float]], kind: str = "hmatrix") -> None:
        r = tuple(tuple(float(v) for v in row) for row in rows)
        if len(r) != 4 or any(len(row) != 4 for row in r):
            raise ValueError("TMatrix requires 4x4 rows")
        self.rows = r
        self.kind = kind

    # -- algebra ----------------------------------------------------------
    def __mul__(self, other: Any) -> Any:
        if isinstance(other, TMatrix):
            a, b = self.rows, other.rows
            return TMatrix(
                [
                    [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
                    for i in range(4)
                ],
                kind="hmatrix",
            )
        if isinstance(other, Node):
            return transform(self, other)
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TMatrix) and other.rows == self.rows

    def __hash__(self) -> int:
        return hash(self.rows)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "TMatrix(%s, kind=%r)" % (self.rows, self.kind)

    # -- accessors --------------------------------------------------------
    def xdir(self) -> Tuple[float, float, float]:
        return (self.rows[0][0], self.rows[1][0], self.rows[2][0])

    def ydir(self) -> Tuple[float, float, float]:
        return (self.rows[0][1], self.rows[1][1], self.rows[2][1])

    def zdir(self) -> Tuple[float, float, float]:
        return (self.rows[0][2], self.rows[1][2], self.rows[2][2])

    def origin(self) -> Tuple[float, float, float]:
        return (self.rows[0][3], self.rows[1][3], self.rows[2][3])

    def is_identity(self, tol: float = 0.0) -> bool:
        ident = identity().rows
        total = 0.0
        for i in range(4):
            for j in range(4):
                total += abs(self.rows[i][j] - ident[i][j])
        return total <= tol

    # -- application ------------------------------------------------------
    def apply_pos(self, p: Sequence[float]) -> Tuple[float, float, float]:
        x, y, z = float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0
        m = self.rows
        w = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3]
        if w == 0.0:
            raise ZeroDivisionError("degenerate homogeneous transform")
        return (
            (m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3]) / w,
            (m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3]) / w,
            (m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]) / w,
        )

    def apply_vec(self, v: Sequence[float]) -> Tuple[float, float, float]:
        x, y, z = float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 0.0
        m = self.rows
        return (
            m[0][0] * x + m[0][1] * y + m[0][2] * z,
            m[1][0] * x + m[1][1] * y + m[1][2] * z,
            m[2][0] * x + m[2][1] * y + m[2][2] * z,
        )


def identity() -> TMatrix:
    return TMatrix(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], kind="identity"
    )


def translate(dx: float, dy: float = 0.0, dz: float = 0.0) -> TMatrix:
    return TMatrix(
        [[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]], kind="translate"
    )


def _angle(deg: Optional[float], rad: Optional[float]) -> float:
    """AngelCAD's ``to_radian(deg, rad)``: exactly one of the two must be given."""
    if (deg is None) == (rad is None):
        raise ValueError("specify exactly one of deg=, rad=")
    return float(rad) if rad is not None else float(deg) * _DEG


def rotate_x(deg: Optional[float] = None, rad: Optional[float] = None) -> TMatrix:
    a = _angle(deg, rad)
    c, s = math.cos(a), math.sin(a)
    return TMatrix(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], kind="rotate_x"
    )


def rotate_y(deg: Optional[float] = None, rad: Optional[float] = None) -> TMatrix:
    a = _angle(deg, rad)
    c, s = math.cos(a), math.sin(a)
    return TMatrix(
        [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], kind="rotate_y"
    )


def rotate_z(deg: Optional[float] = None, rad: Optional[float] = None) -> TMatrix:
    a = _angle(deg, rad)
    c, s = math.cos(a), math.sin(a)
    return TMatrix(
        [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], kind="rotate_z"
    )


def scale(sx: float, sy: Optional[float] = None, sz: Optional[float] = None) -> TMatrix:
    if sy is None and sz is None:
        sy = sz = sx
    if sy is None or sz is None:
        raise ValueError("scale needs 1 or 3 factors")
    return TMatrix(
        [[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]], kind="scale"
    )


def mirror(nx: float, ny: float, nz: float) -> TMatrix:
    """Householder reflection in the plane through origin with normal (nx,ny,nz)."""
    n = math.sqrt(nx * nx + ny * ny + nz * nz)
    if n == 0.0:
        raise ValueError("mirror normal must be non-zero")
    nx, ny, nz = nx / n, ny / n, nz / n
    rows = [[0.0] * 4 for _ in range(4)]
    comp = (nx, ny, nz)
    for i in range(3):
        for j in range(3):
            rows[i][j] = (1.0 if i == j else 0.0) - 2.0 * comp[i] * comp[j]
    rows[3][3] = 1.0
    return TMatrix(rows, kind="mirror")


def _cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: Sequence[float]) -> Tuple[float, float, float]:
    n = math.sqrt(sum(c * c for c in v))
    if n == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return (v[0] / n, v[1] / n, v[2] / n)


def hmatrix(
    xvec: Sequence[float],
    yvec: Sequence[float],
    zvec: Optional[Sequence[float]] = None,
    pos: Sequence[float] = (0.0, 0.0, 0.0),
) -> TMatrix:
    """Build a frame from axis vectors (AngelCAD ``hmatrix``).

    ``zvec`` may be omitted, in which case it is ``xvec x yvec``.  The frame is
    orthonormalised Gram-Schmidt style (x kept, y made perpendicular to x, z
    recomputed), so slightly non-orthogonal input still yields a rigid frame.
    """
    ex = _norm(xvec)
    ey0 = tuple(float(c) for c in yvec)
    dot = sum(ex[i] * ey0[i] for i in range(3))
    ey = _norm([ey0[i] - dot * ex[i] for i in range(3)])
    ez = _cross(ex, ey)
    if zvec is not None:
        if sum(ez[i] * float(zvec[i]) for i in range(3)) < 0.0:
            # honour a left-handed z by flipping y, keeping the frame consistent
            ey = (-ey[0], -ey[1], -ey[2])
            ez = _cross(ex, ey)
    rows = [
        [ex[0], ey[0], ez[0], float(pos[0])],
        [ex[1], ey[1], ez[1], float(pos[1])],
        [ex[2], ey[2], ez[2], float(pos[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return TMatrix(rows, kind="hmatrix")


# --------------------------------------------------------------------------
# operator table
# --------------------------------------------------------------------------


class OpSpec:
    """Static signature of one CSG operator."""

    __slots__ = ("name", "result", "child", "min_children", "max_children", "params")

    def __init__(
        self,
        name: str,
        result: str,
        child: Optional[str],
        min_children: int,
        max_children: Optional[int],
        params: Mapping[str, str],
    ) -> None:
        self.name = name
        self.result = result
        self.child = child
        self.min_children = min_children
        self.max_children = max_children
        self.params = dict(params)


def _spec(name, result, child, lo, hi, **params) -> OpSpec:
    return OpSpec(name, result, child, lo, hi, params)


# parameter kinds: "num+" positive number, "num" any number, "bool", "points2",
# "points3", "faces", "path", "matrix", "angle+"
OPS: Dict[str, OpSpec] = {
    # ---- 2d primitives
    "circle": _spec("circle", SHAPE2D, None, 0, 0, r="num+"),
    "square": _spec("square", SHAPE2D, None, 0, 0, size="num+", center="bool"),
    "rectangle": _spec(
        "rectangle", SHAPE2D, None, 0, 0, dx="num+", dy="num+", center="bool"
    ),
    "polygon": _spec("polygon", SHAPE2D, None, 0, 0, points="points2"),
    # ---- 3d primitives
    "sphere": _spec("sphere", SOLID, None, 0, 0, r="num+"),
    "cube": _spec("cube", SOLID, None, 0, 0, size="num+", center="bool"),
    "cuboid": _spec(
        "cuboid", SOLID, None, 0, 0, dx="num+", dy="num+", dz="num+", center="bool"
    ),
    "cylinder": _spec("cylinder", SOLID, None, 0, 0, h="num+", r="num+", center="bool"),
    "cone": _spec(
        "cone", SOLID, None, 0, 0, h="num+", r1="num", r2="num", center="bool"
    ),
    "polyhedron": _spec("polyhedron", SOLID, None, 0, 0, points="points3", faces="faces"),
    # ---- 2d booleans
    "union2d": _spec("union2d", SHAPE2D, SHAPE2D, 1, None),
    "difference2d": _spec("difference2d", SHAPE2D, SHAPE2D, 1, None),
    "intersection2d": _spec("intersection2d", SHAPE2D, SHAPE2D, 2, None),
    "hull2d": _spec("hull2d", SHAPE2D, SHAPE2D, 1, None),
    "fill2d": _spec("fill2d", SHAPE2D, SHAPE2D, 1, 1),
    "offset2d": _spec("offset2d", SHAPE2D, SHAPE2D, 1, 1, delta="num", round="bool"),
    "minkowski2d": _spec("minkowski2d", SHAPE2D, SHAPE2D, 2, None),
    # ---- the only 3d -> 2d operator
    "projection2d": _spec("projection2d", SHAPE2D, SOLID, 1, 1, cut="bool"),
    # ---- 3d booleans
    "union3d": _spec("union3d", SOLID, SOLID, 1, None),
    "difference3d": _spec("difference3d", SOLID, SOLID, 1, None),
    "intersection3d": _spec("intersection3d", SOLID, SOLID, 2, None),
    "hull3d": _spec("hull3d", SOLID, SOLID, 1, None),
    "minkowski3d": _spec("minkowski3d", SOLID, SOLID, 2, None),
    # ---- 2d -> 3d operators
    "linear_extrude": _spec("linear_extrude", SOLID, SHAPE2D, 1, 1, dz="num+"),
    "rotate_extrude": _spec(
        "rotate_extrude", SOLID, SHAPE2D, 1, 1, angle="angle+", pitch="num"
    ),
    "transform_extrude": _spec("transform_extrude", SOLID, SHAPE2D, 2, 2),
    "sweep": _spec("sweep", SOLID, SHAPE2D, 1, 1, path="path"),
    # ---- transform (result type = child type)
    "transform": _spec("transform", "", None, 1, 1, matrix="matrix"),
}


# --------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------


class Node:
    """One typed CSG operation."""

    __slots__ = ("op", "params", "children")

    def __init__(
        self,
        op: str,
        params: Optional[Mapping[str, Any]] = None,
        children: Sequence["Node"] = (),
    ) -> None:
        self.op = op
        self.params: Dict[str, Any] = dict(params or {})
        self.children: Tuple[Node, ...] = tuple(children)

    # -- typed operators, mirroring AngelCAD's opAdd/opSub/opAnd -----------
    def _binary(self, other: "Node", kind2: str, kind3: str) -> "Node":
        if not isinstance(other, Node):
            return NotImplemented
        t = result_type(self)
        op = kind2 if t == SHAPE2D else kind3
        return Node(op, children=(self, other))

    def __add__(self, other: "Node") -> "Node":
        return self._binary(other, "union2d", "union3d")

    def __sub__(self, other: "Node") -> "Node":
        return self._binary(other, "difference2d", "difference3d")

    def __and__(self, other: "Node") -> "Node":
        return self._binary(other, "intersection2d", "intersection3d")

    def __rmul__(self, m: TMatrix) -> "Node":
        if isinstance(m, TMatrix):
            return transform(m, self)
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Node)
            and other.op == self.op
            and other.params == self.params
            and other.children == self.children
        )

    def __hash__(self) -> int:
        return hash((self.op, tuple(sorted(self.params)), self.children))

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "Node(%r, %r, %d children)" % (self.op, self.params, len(self.children))

    def walk(self):
        """Yield ``(path, node)`` in deterministic pre-order."""

        def rec(node: "Node", path: str):
            yield (path, node)
            for i, c in enumerate(node.children):
                yield from rec(c, "%s/%s[%d]" % (path, node.op, i))

        yield from rec(self, "")


def result_type(node: Node) -> str:
    """Best-effort static type of ``node`` ('solid', 'shape2d' or '')."""
    spec = OPS.get(node.op)
    if spec is None:
        return ""
    if node.op == "transform":
        return result_type(node.children[0]) if node.children else ""
    return spec.result


# ---- constructors ---------------------------------------------------------


def circle(r: float) -> Node:
    return Node("circle", {"r": r})


def square(size: float, center: bool = False) -> Node:
    return Node("square", {"size": size, "center": center})


def rectangle(dx: float, dy: float, center: bool = False) -> Node:
    return Node("rectangle", {"dx": dx, "dy": dy, "center": center})


def polygon(points: Sequence[Sequence[float]]) -> Node:
    return Node("polygon", {"points": [tuple(float(c) for c in p) for p in points]})


def sphere(r: float) -> Node:
    return Node("sphere", {"r": r})


def cube(size: float, center: bool = False) -> Node:
    return Node("cube", {"size": size, "center": center})


def cuboid(dx: float, dy: float, dz: float, center: bool = False) -> Node:
    return Node("cuboid", {"dx": dx, "dy": dy, "dz": dz, "center": center})


def cylinder(h: float, r: float, center: bool = False) -> Node:
    return Node("cylinder", {"h": h, "r": r, "center": center})


def cone(h: float, r1: float, r2: float, center: bool = False) -> Node:
    return Node("cone", {"h": h, "r1": r1, "r2": r2, "center": center})


def polyhedron(
    points: Sequence[Sequence[float]], faces: Sequence[Sequence[int]]
) -> Node:
    return Node(
        "polyhedron",
        {
            "points": [tuple(float(c) for c in p) for p in points],
            "faces": [tuple(int(i) for i in f) for f in faces],
        },
    )


def _nary(op: str):
    def make(*children: Node, **params: Any) -> Node:
        return Node(op, params, children)

    make.__name__ = op
    return make


union2d = _nary("union2d")
difference2d = _nary("difference2d")
intersection2d = _nary("intersection2d")
hull2d = _nary("hull2d")
fill2d = _nary("fill2d")
minkowski2d = _nary("minkowski2d")
union3d = _nary("union3d")
difference3d = _nary("difference3d")
intersection3d = _nary("intersection3d")
hull3d = _nary("hull3d")
minkowski3d = _nary("minkowski3d")


def offset2d(child: Node, delta: float, round: bool = False) -> Node:
    return Node("offset2d", {"delta": delta, "round": bool(round)}, (child,))


def projection2d(child: Node, cut: bool = False) -> Node:
    return Node("projection2d", {"cut": bool(cut)}, (child,))


def linear_extrude(child: Node, dz: float) -> Node:
    return Node("linear_extrude", {"dz": dz}, (child,))


def rotate_extrude(child: Node, angle: float = 360.0, pitch: float = 0.0) -> Node:
    return Node("rotate_extrude", {"angle": angle, "pitch": pitch}, (child,))


def transform_extrude(bottom: Node, top: Node) -> Node:
    return Node("transform_extrude", {}, (bottom, top))


def sweep(child: Node, path: Sequence[Sequence[float]]) -> Node:
    return Node(
        "sweep",
        {"path": [tuple(float(c) for c in p) for p in path]},
        (child,),
    )


def transform(matrix: TMatrix, child: Node) -> Node:
    return Node("transform", {"matrix": matrix}, (child,))


# --------------------------------------------------------------------------
# checker
# --------------------------------------------------------------------------


class Diagnostic:
    """One type error, with a stable path into the tree."""

    __slots__ = ("path", "op", "code", "message")

    def __init__(self, path: str, op: str, code: str, message: str) -> None:
        self.path = path
        self.op = op
        self.code = code
        self.message = message

    def key(self) -> Tuple[str, str, str]:
        return (self.path, self.code, self.message)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Diagnostic) and other.key() == self.key()

    def __hash__(self) -> int:
        return hash(self.key())

    def __repr__(self) -> str:
        return "%s: %s [%s]" % (self.path or "/", self.message, self.code)

    __str__ = __repr__


class CsgTypeError(Exception):
    """Raised by :func:`type_check` when the tree does not type-check."""

    def __init__(self, diagnostics: Sequence[Diagnostic]) -> None:
        self.diagnostics = list(diagnostics)
        super().__init__(
            "%d type error(s):\n%s"
            % (len(self.diagnostics), "\n".join(str(d) for d in self.diagnostics))
        )


def _check_param(diags, path, op, name, kind, value):
    if kind in ("num", "num+", "angle+"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            diags.append(
                Diagnostic(path, op, "param-type", "%s.%s must be a number" % (op, name))
            )
            return
        if kind == "num+" and value <= 0.0:
            diags.append(
                Diagnostic(
                    path, op, "param-range", "%s.%s must be > 0 (got %g)" % (op, name, value)
                )
            )
        if kind == "angle+" and not (0.0 < value <= 360.0):
            diags.append(
                Diagnostic(
                    path,
                    op,
                    "param-range",
                    "%s.%s must be in (0, 360] degrees (got %g)" % (op, name, value),
                )
            )
    elif kind == "bool":
        if not isinstance(value, bool):
            diags.append(
                Diagnostic(path, op, "param-type", "%s.%s must be a bool" % (op, name))
            )
    elif kind == "matrix":
        if not isinstance(value, TMatrix):
            diags.append(
                Diagnostic(path, op, "param-type", "%s.%s must be a TMatrix" % (op, name))
            )
    elif kind in ("points2", "points3", "path"):
        dim = 2 if kind == "points2" else 3
        if not isinstance(value, (list, tuple)) or not value:
            diags.append(
                Diagnostic(
                    path, op, "param-type", "%s.%s must be a non-empty point list" % (op, name)
                )
            )
            return
        for i, p in enumerate(value):
            if not isinstance(p, (list, tuple)) or len(p) != dim:
                diags.append(
                    Diagnostic(
                        path,
                        op,
                        "dim-mismatch",
                        "%s.%s[%d] must be a %dD point (got %r)" % (op, name, i, dim, p),
                    )
                )
        minimum = 3 if kind == "points2" else 2 if kind == "path" else 4
        if len(value) < minimum:
            diags.append(
                Diagnostic(
                    path,
                    op,
                    "param-range",
                    "%s.%s needs at least %d points (got %d)"
                    % (op, name, minimum, len(value)),
                )
            )
    elif kind == "faces":
        if not isinstance(value, (list, tuple)) or not value:
            diags.append(
                Diagnostic(path, op, "param-type", "%s.%s must be a non-empty face list" % (op, name))
            )


def check(root: Node) -> List[Diagnostic]:
    """Type/dimension-check a CSG tree.  Returns diagnostics in tree order."""
    diags: List[Diagnostic] = []
    for path, node in root.walk():
        spec = OPS.get(node.op)
        if spec is None:
            diags.append(
                Diagnostic(path, node.op, "unknown-op", "unknown operator %r" % node.op)
            )
            continue

        # arity
        n = len(node.children)
        if n < spec.min_children:
            diags.append(
                Diagnostic(
                    path,
                    node.op,
                    "arity",
                    "%s needs at least %d child(ren), got %d"
                    % (node.op, spec.min_children, n),
                )
            )
        if spec.max_children is not None and n > spec.max_children:
            diags.append(
                Diagnostic(
                    path,
                    node.op,
                    "arity",
                    "%s takes at most %d child(ren), got %d"
                    % (node.op, spec.max_children, n),
                )
            )

        # child types -- the 2D/3D dimension check
        expected = spec.child
        if expected is not None:
            for i, child in enumerate(node.children):
                actual = result_type(child)
                if actual and actual != expected:
                    diags.append(
                        Diagnostic(
                            path,
                            node.op,
                            "dim-mismatch",
                            "%s expects a %s child, child %d (%s) is a %s"
                            % (node.op, expected, i, child.op, actual),
                        )
                    )
        elif node.op == "transform":
            for i, child in enumerate(node.children):
                actual = result_type(child)
                if actual not in (SOLID, SHAPE2D):
                    diags.append(
                        Diagnostic(
                            path,
                            node.op,
                            "child-type",
                            "transform child %d (%s) is not a shape" % (i, child.op),
                        )
                    )
        elif node.children:
            diags.append(
                Diagnostic(
                    path,
                    node.op,
                    "arity",
                    "%s is a primitive and takes no children" % node.op,
                )
            )

        # params
        for name, kind in sorted(spec.params.items()):
            if name not in node.params:
                diags.append(
                    Diagnostic(
                        path, node.op, "param-missing", "%s.%s is required" % (node.op, name)
                    )
                )
            else:
                _check_param(diags, path, node.op, name, kind, node.params[name])
        for name in sorted(node.params):
            if name not in spec.params:
                diags.append(
                    Diagnostic(
                        path, node.op, "param-unknown", "%s has no parameter %r" % (node.op, name)
                    )
                )

        # polyhedron index validity (indices must address the point list)
        if node.op == "polyhedron":
            pts = node.params.get("points")
            faces = node.params.get("faces")
            if isinstance(pts, (list, tuple)) and isinstance(faces, (list, tuple)):
                npts = len(pts)
                for fi, face in enumerate(faces):
                    if not isinstance(face, (list, tuple)) or len(face) < 3:
                        diags.append(
                            Diagnostic(
                                path,
                                node.op,
                                "face-degenerate",
                                "polyhedron face %d needs at least 3 vertices" % fi,
                            )
                        )
                        continue
                    for iv in face:
                        if not isinstance(iv, int) or iv < 0 or iv >= npts:
                            diags.append(
                                Diagnostic(
                                    path,
                                    node.op,
                                    "index-range",
                                    "polyhedron face %d references vertex %r, valid range 0..%d"
                                    % (fi, iv, npts - 1),
                                )
                            )
                    if len(set(face)) != len(face):
                        diags.append(
                            Diagnostic(
                                path,
                                node.op,
                                "face-degenerate",
                                "polyhedron face %d repeats a vertex index" % fi,
                            )
                        )

        # cone: at least one radius must be positive, neither may be negative
        if node.op == "cone":
            r1 = node.params.get("r1")
            r2 = node.params.get("r2")
            if isinstance(r1, (int, float)) and isinstance(r2, (int, float)):
                if r1 < 0 or r2 < 0:
                    diags.append(
                        Diagnostic(path, node.op, "param-range", "cone radii must be >= 0")
                    )
                elif r1 == 0 and r2 == 0:
                    diags.append(
                        Diagnostic(
                            path, node.op, "param-range", "cone needs one radius > 0"
                        )
                    )
    return diags


def type_check(root: Node) -> str:
    """Check ``root`` and return its type; raise :class:`CsgTypeError` on error."""
    diags = check(root)
    if diags:
        raise CsgTypeError(diags)
    return result_type(root)
