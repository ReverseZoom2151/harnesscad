"""FRepBackend — a kernel-free geometry backend built entirely from SDFs.

This is a *real* geometry backend: it needs no OCCT, no CadQuery, no compiled
dependency at all.  Every CISP op is realised as a functional-representation
(F-rep) node in a small CSG tree whose leaves are exact signed-distance fields
and whose interior nodes are the SDF combinators:

    new_sketch / add_rectangle / add_circle / add_line  -> 2D profile entities
    extrude                                             -> profile x slab (exact
                                                           intersection of the 2D
                                                           polygon/circle field and
                                                           an axial slab)
    revolve                                             -> profile swept about an
                                                           in-plane axis (radial
                                                           coordinate remap)
    boolean union/cut/intersect                         -> min / max / neg
    fillet                                              -> smooth (polynomial)
                                                           combinators + Minkowski
                                                           rounding of the leaves
    chamfer                                             -> chamfer combinators
    hole                                                -> cut with a cylinder field
    shell                                               -> |f| - t/2
    mirror / linear_pattern / circular_pattern          -> field transforms + union
    tessellate / export                                 -> sample the field on a
                                                           regular grid, run
                                                           Marching Cubes, weld,
                                                           validate as a half-edge
                                                           manifold, write STL/GLB
    query('measure'|'metrics'|'validity')               -> mass properties from the
                                                           extracted mesh

The op-level semantics (id allocation, DOF bookkeeping, block-and-correct on a
bad reference, SetParam replay, digest stability) are identical to StubBackend's,
so the same HarnessSession / verifiers / CISP surface drive it unchanged; the
difference is that this backend actually produces geometry.

stdlib-only, deterministic (no wall clock, no randomness).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import (
    CONSTRAINT_DOF, PRIMITIVE_DOF,
    Op, NewSketch, AddPoint, AddLine, AddCircle, AddRectangle,
    Constrain, Extrude, Fillet, Boolean,
    Revolve, Chamfer, Hole, Shell, Draft,
    Loft, Sweep, LinearPattern, CircularPattern, Mirror,
    AddInstance, Mate, SetParam,
    canonical_json, edit_oplog,
)
from harnesscad.domain.geometry.mesh.halfedge import HalfedgeMesh
from harnesscad.domain.geometry.mesh.winding_number import (
    signed_volume as mesh_signed_volume,
    surface_area as mesh_surface_area,
)
from harnesscad.domain.geometry.sdf import combinators as comb
from harnesscad.domain.geometry.sdf import field_transforms as xf
from harnesscad.domain.geometry.sdf import polygon as poly
from harnesscad.domain.geometry.sdf import primitives as prim
from harnesscad.domain.geometry.parametric.chord_tolerance import segments_for_tolerance
from harnesscad.domain.geometry.volumes.marching_cubes import marching_cubes
from harnesscad.domain.geometry.volumes.surface_nets import (
    ScalarGrid, sample_sdf_grid, surface_nets,
)
from harnesscad.domain.numeric import quadrature
from harnesscad.eval.verifiers.assembly import mate_dof
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.backends import frep_ir
from harnesscad.io.backends.base import ApplyResult
from harnesscad.io.formats import glb as glb_fmt
from harnesscad.io.formats import stl as stl_fmt

Vec3 = Tuple[float, float, float]
Mesh = Tuple[List[Vec3], List[Tuple[int, int, int]]]

# Default number of Marching-Cubes cells along the model's longest axis.
DEFAULT_RESOLUTION = 48
_INF = float("inf")

#: The iso-surface extractors this backend can drive. They are RIVALS, never
#: blended: ``marching_cubes`` is and stays the default (it is the one every
#: existing digest/mesh expectation was recorded against); ``surface_nets`` is a
#: dual method that puts one vertex per cell and produces a different -- also
#: valid -- mesh of the same field. Choose with ``mesher=``; nothing mixes them.
MESHERS: Tuple[str, ...] = ("marching_cubes", "surface_nets")
DEFAULT_MESHER = "marching_cubes"

#: How a surface normal is obtained. ``finite_difference`` is the default (it is
#: what the mesh writers have always used, via the codecs); ``autodiff`` compiles
#: the CSG tree to the arithmetic f-rep IR and reads the exact gradient off a
#: forward-mode dual-number pass. Rivals, selectable, never blended.
NORMAL_METHODS: Tuple[str, ...] = ("finite_difference", "autodiff")
DEFAULT_NORMALS = "finite_difference"

#: Number of grid CELLS per side of an interval-pruning block. Small blocks
#: bound the field tightly (interval arithmetic loses precision over wide boxes)
#: at the cost of more interval evaluations; 4 is the point where a typical part
#: prunes most of its interior and its surrounding air.
PRUNE_BLOCK = 4


def _err(code: str, msg: str, where: Optional[str] = None) -> ApplyResult:
    return ApplyResult(False, [], [Diagnostic(Severity.ERROR, code, msg, where)])


# --------------------------------------------------------------------------
# sketch planes
# --------------------------------------------------------------------------
# A sketch plane maps a 3D point to (u, v, w): (u, v) are the in-plane sketch
# coordinates and w is the extrusion axis (the plane normal).
_PLANES: Dict[str, Tuple[int, int, int]] = {
    "XY": (0, 1, 2),
    "XZ": (0, 2, 1),
    "YZ": (1, 2, 0),
}


def _plane_axes(plane: str) -> Tuple[int, int, int]:
    return _PLANES.get(str(plane).upper(), _PLANES["XY"])


def _to_world(plane: str, u: float, v: float, w: float) -> Vec3:
    iu, iv, iw = _plane_axes(plane)
    p = [0.0, 0.0, 0.0]
    p[iu], p[iv], p[iw] = u, v, w
    return (p[0], p[1], p[2])


# --------------------------------------------------------------------------
# 2D profile entities
# --------------------------------------------------------------------------
class _Profile:
    """The 2D region of a sketch: the union of its closed entities.

    ``rects`` are (x, y, w, h) with (x, y) the *corner* (matching the CadQuery
    backend), ``circles`` are (cx, cy, r), ``polys`` are open vertex lists.
    """

    __slots__ = ("rects", "circles", "polys", "_edges")

    def __init__(self) -> None:
        self.rects: List[Tuple[float, float, float, float]] = []
        self.circles: List[Tuple[float, float, float]] = []
        self.polys: List[List[Tuple[float, float]]] = []
        self._edges: Optional[List[List[poly.Edge]]] = None

    def empty(self) -> bool:
        return not (self.rects or self.circles or self.polys)

    def _poly_edges(self) -> List[List[poly.Edge]]:
        if self._edges is None:
            self._edges = [poly.prepare_edges(v) for v in self.polys]
        return self._edges

    def sdf(self, u: float, v: float) -> float:
        """Exact signed distance to the profile boundary (negative inside)."""
        vals: List[float] = []
        for (x, y, w, h) in self.rects:
            cx, cy = x + w / 2.0, y + h / 2.0
            vals.append(prim.rect_exact((u - cx, v - cy), (w, h)))
        for (cx, cy, r) in self.circles:
            vals.append(prim.circle((u - cx, v - cy), 2.0 * r))
        for edges in self._poly_edges():
            d = poly.polygon_distance(u, v, edges)
            vals.append(-d if poly.polygon_winding(u, v, edges) != 0 else d)
        if not vals:
            return _INF
        return comb.union_all(vals)

    def bounds(self) -> Tuple[float, float, float, float]:
        lo_u = lo_v = _INF
        hi_u = hi_v = -_INF
        for (x, y, w, h) in self.rects:
            lo_u, hi_u = min(lo_u, x), max(hi_u, x + w)
            lo_v, hi_v = min(lo_v, y), max(hi_v, y + h)
        for (cx, cy, r) in self.circles:
            lo_u, hi_u = min(lo_u, cx - r), max(hi_u, cx + r)
            lo_v, hi_v = min(lo_v, cy - r), max(hi_v, cy + r)
        for verts in self.polys:
            for (x, y) in verts:
                lo_u, hi_u = min(lo_u, x), max(hi_u, x)
                lo_v, hi_v = min(lo_v, y), max(hi_v, y)
        return (lo_u, lo_v, hi_u, hi_v)


def _profile_of(sketch: dict, entities: dict) -> _Profile:
    """Collect a sketch's closed entities into a 2D profile.

    Rectangles and circles are closed on their own.  Three or more lines are
    taken as a polyline and closed into a polygon (the closing edge implied),
    matching the usual "sketch a closed loop from segments" idiom.
    """
    prof = _Profile()
    lines: List[Tuple[float, float]] = []
    for eid in sketch["entities"]:
        ent = entities[eid]
        p = ent["params"]
        kind = ent["type"]
        if kind == "rectangle":
            prof.rects.append((p["x"], p["y"], p["w"], p["h"]))
        elif kind == "circle":
            prof.circles.append((p["cx"], p["cy"], p["r"]))
        elif kind == "line":
            if not lines:
                lines.append((p["x1"], p["y1"]))
            lines.append((p["x2"], p["y2"]))
    if len(lines) >= 4 and lines[0] == lines[-1]:
        lines = lines[:-1]
    if len(lines) >= 3:
        prof.polys.append(lines)
    return prof


# --------------------------------------------------------------------------
# the F-rep CSG tree
# --------------------------------------------------------------------------
class Node:
    """One node of the F-rep tree.  ``t`` is the node kind; the rest is payload.

    Kinds:
      extrude  profile, plane, w0, w1, round, cham
      revolve  profile, plane, axis (point+dir in sketch coords), angle, round, cham
      cyl      plane, cu, cv, r, w0, w1                (hole tool)
      bool     op ('union'|'cut'|'intersect'), a, b, blend ('hard'|'smooth'|'chamfer'), k
      shell    child, thickness
      mirror   child, plane
      pattern  child, transforms (list of (dx, dy, dz, angle_deg_about_z))
    """

    __slots__ = ("t", "d")

    def __init__(self, t: str, **payload) -> None:
        self.t = t
        self.d = payload

    def spec(self) -> dict:
        """A JSON-able, deterministic description (feeds the state digest)."""
        d: dict = {"t": self.t}
        for k in sorted(self.d):
            v = self.d[k]
            if isinstance(v, Node):
                d[k] = v.spec()
            elif isinstance(v, _Profile):
                d[k] = {"rects": [list(r) for r in v.rects],
                        "circles": [list(c) for c in v.circles],
                        "polys": [[list(p) for p in q] for q in v.polys]}
            elif isinstance(v, (list, tuple)):
                d[k] = [list(x) if isinstance(x, (list, tuple)) else x for x in v]
            else:
                d[k] = v
        return d


# -- field evaluation ------------------------------------------------------
def _slab(w: float, w0: float, w1: float) -> float:
    """Exact 1D slab distance: negative between w0 and w1."""
    lo, hi = (w0, w1) if w0 <= w1 else (w1, w0)
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return abs(w - mid) - half


def _combine_prism(d2: float, dw: float, r_round: float, r_cham: float) -> float:
    """Intersect a 2D profile distance with an axial slab distance.

    With no blend this is the exact Euclidean field of the prism.  ``r_round``
    performs a genuine Minkowski rounding: erode both half-fields by ``r``,
    intersect exactly, then dilate by ``r`` -- which rounds *every* convex edge
    (the vertical profile corners and both rims).  ``r_cham`` uses the chamfer
    combinator instead (straight setback on the rim).
    """
    if r_round > 0.0:
        a, b = d2 + r_round, dw + r_round
        inside = min(max(a, b), 0.0)
        outside = math.hypot(max(a, 0.0), max(b, 0.0))
        return xf.round_field(inside + outside, r_round)
    if r_cham > 0.0:
        return comb.chamfer_intersection(d2, dw, r_cham)
    inside = min(max(d2, dw), 0.0)
    outside = math.hypot(max(d2, 0.0), max(dw, 0.0))
    return inside + outside


def _boolean_field(node: Node, a: float, b: float) -> float:
    op = node.d["op"]
    blend = node.d.get("blend", "hard")
    k = float(node.d.get("k", 0.0))
    if blend == "smooth" and k > 0.0:
        if op == "union":
            return comb.smooth_union(a, b, k)
        if op == "intersect":
            return comb.smooth_intersection(a, b, k)
        return comb.smooth_difference(a, b, k)
    if blend == "chamfer" and k > 0.0:
        if op == "union":
            return comb.chamfer_union(a, b, k)
        if op == "intersect":
            return comb.chamfer_intersection(a, b, k)
        return comb.chamfer_intersection(a, comb.complement(b), k)
    if op == "union":
        return comb.union(a, b)
    if op == "intersect":
        return comb.intersection(a, b)
    return comb.difference(a, b)


def eval_node(node: Node, p: Sequence[float]) -> float:
    """Signed distance of the F-rep ``node`` at world point ``p``."""
    t = node.t
    if t == "extrude":
        iu, iv, iw = _plane_axes(node.d["plane"])
        d2 = node.d["profile"].sdf(p[iu], p[iv])
        dw = _slab(p[iw], node.d["w0"], node.d["w1"])
        return _combine_prism(d2, dw, node.d.get("round", 0.0), node.d.get("cham", 0.0))
    if t == "cyl":
        iu, iv, iw = _plane_axes(node.d["plane"])
        d2 = math.hypot(p[iu] - node.d["cu"], p[iv] - node.d["cv"]) - node.d["r"]
        dw = _slab(p[iw], node.d["w0"], node.d["w1"])
        return _combine_prism(d2, dw, 0.0, 0.0)
    if t == "revolve":
        return _eval_revolve(node, p)
    if t == "bool":
        a = eval_node(node.d["a"], p)
        b = eval_node(node.d["b"], p)
        return _boolean_field(node, a, b)
    if t == "shell":
        return xf.shell(eval_node(node.d["child"], p), node.d["thickness"])
    if t == "mirror":
        return comb.union(eval_node(node.d["child"], p),
                          eval_node(node.d["child"], _reflect(p, node.d["plane"])))
    if t == "pattern":
        child = node.d["child"]
        vals = [eval_node(child, _untransform(p, tr)) for tr in node.d["transforms"]]
        return comb.union_all(vals)
    raise ValueError("unknown F-rep node kind '%s'" % t)  # pragma: no cover


def _reflect(p: Sequence[float], plane: str) -> Vec3:
    """Reflect a point across a named datum plane (the plane's normal flips)."""
    pl = str(plane).upper()
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    if pl == "XY":
        return (x, y, -z)
    if pl == "YZ":
        return (-x, y, z)
    return (x, -y, z)  # XZ


def _untransform(p: Sequence[float], tr: Sequence[float]) -> Vec3:
    """Inverse of a pattern instance transform (rotate about Z, then translate)."""
    dx, dy, dz, ang = tr
    x, y, z = p[0] - dx, p[1] - dy, p[2] - dz
    if ang:
        a = math.radians(-ang)
        ca, sa = math.cos(a), math.sin(a)
        x, y = ca * x - sa * y, sa * x + ca * y
    return (x, y, z)


def _eval_revolve(node: Node, p: Sequence[float]) -> float:
    """Revolve a sketch profile about an in-plane axis.

    The axis is given in sketch coordinates as a point ``(au, av)`` and a unit
    direction ``(du, dv)``.  A world point is decomposed into the coordinate
    along the axis and its perpendicular distance from it; those become the
    profile's (axial, radial) coordinates, so the profile sweeps a solid of
    revolution.  ``angle < 360`` intersects the result with an angular wedge.
    """
    iu, iv, iw = _plane_axes(node.d["plane"])
    au, av, du, dv, nu, nv = node.d["axis"]
    pu, pv, pw = p[iu], p[iv], p[iw]
    # coordinate along the axis (in-plane) and radial distance from the axis line
    s = (pu - au) * du + (pv - av) * dv
    perp = (pu - au) * nu + (pv - av) * nv
    rad = math.hypot(perp, pw)
    # back to sketch coordinates on the positive radial half-plane
    qu = au + s * du + rad * nu
    qv = av + s * dv + rad * nv
    # The profile field is already an exact SDF, so a Minkowski rounding of a
    # revolve leaf is a no-op pointwise (erode-then-dilate of an exact field is
    # the identity); revolve edges are rounded through the boolean blends only.
    d = node.d["profile"].sdf(qu, qv)
    angle = float(node.d.get("angle", 360.0))
    if abs(angle) >= 360.0:
        return d
    # wedge: intersect with the sector [0, angle) measured in the (perp, w) plane
    theta = math.atan2(pw, perp)
    half = math.radians(abs(angle)) / 2.0
    mid = half
    dt = abs(_wrap_angle(theta - mid))
    wedge = (dt - half) * max(rad, 1e-9)
    return comb.intersection(d, wedge)


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


# -- bounds ----------------------------------------------------------------
def node_bounds(node: Node) -> Tuple[Vec3, Vec3]:
    """A conservative axis-aligned bound of the node's solid."""
    t = node.t
    if t in ("extrude", "cyl", "revolve"):
        return _leaf_bounds(node)
    if t == "bool":
        ba = node_bounds(node.d["a"])
        bb = node_bounds(node.d["b"])
        op = node.d["op"]
        k = float(node.d.get("k", 0.0))
        if op == "cut":
            return _grow(ba, k)
        if op == "intersect":
            lo = tuple(max(ba[0][i], bb[0][i]) for i in range(3))
            hi = tuple(min(ba[1][i], bb[1][i]) for i in range(3))
            return _grow((lo, hi), k)  # type: ignore[arg-type]
        return _grow(_union_bounds(ba, bb), k)
    if t == "shell":
        return _grow(node_bounds(node.d["child"]), node.d["thickness"])
    if t == "mirror":
        b = node_bounds(node.d["child"])
        pl = str(node.d["plane"]).upper()
        lo = list(b[0])
        hi = list(b[1])
        ax = {"XY": 2, "YZ": 0}.get(pl, 1)
        m = max(abs(lo[ax]), abs(hi[ax]))
        lo[ax], hi[ax] = -m, m
        return ((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2]))
    if t == "pattern":
        b = node_bounds(node.d["child"])
        out = None
        for tr in node.d["transforms"]:
            corners = _corners(b)
            pts = [_apply_transform(c, tr) for c in corners]
            lo = tuple(min(p[i] for p in pts) for i in range(3))
            hi = tuple(max(p[i] for p in pts) for i in range(3))
            out = (lo, hi) if out is None else _union_bounds(out, (lo, hi))  # type: ignore
        return out  # type: ignore[return-value]
    raise ValueError("unknown F-rep node kind '%s'" % t)  # pragma: no cover


def _leaf_bounds(node: Node) -> Tuple[Vec3, Vec3]:
    plane = node.d["plane"]
    r = float(node.d.get("round", 0.0)) + float(node.d.get("cham", 0.0))
    if node.t == "cyl":
        cu, cv, rr = node.d["cu"], node.d["cv"], node.d["r"]
        lo_u, hi_u, lo_v, hi_v = cu - rr, cu + rr, cv - rr, cv + rr
        w0, w1 = sorted((node.d["w0"], node.d["w1"]))
    elif node.t == "extrude":
        lo_u, lo_v, hi_u, hi_v = node.d["profile"].bounds()
        w0, w1 = sorted((node.d["w0"], node.d["w1"]))
    else:  # revolve: the profile sweeps a disc of radius = max |perp|
        prof = node.d["profile"]
        au, av, du, dv, nu, nv = node.d["axis"]
        lo_u, lo_v, hi_u, hi_v = prof.bounds()
        rad = 0.0
        s_lo, s_hi = _INF, -_INF
        for (u, v) in ((lo_u, lo_v), (lo_u, hi_v), (hi_u, lo_v), (hi_u, hi_v)):
            perp = (u - au) * nu + (v - av) * nv
            s = (u - au) * du + (v - av) * dv
            rad = max(rad, abs(perp))
            s_lo, s_hi = min(s_lo, s), max(s_hi, s)
        lo_u = au + s_lo * du - rad * abs(nu)
        hi_u = au + s_hi * du + rad * abs(nu)
        lo_v = av + s_lo * dv - rad * abs(nv)
        hi_v = av + s_hi * dv + rad * abs(nv)
        w0, w1 = -rad, rad
    lo = list(_to_world(plane, lo_u, lo_v, w0))
    hi = list(_to_world(plane, hi_u, hi_v, w1))
    for i in range(3):
        if lo[i] > hi[i]:
            lo[i], hi[i] = hi[i], lo[i]
    return _grow(((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])), r)


def _grow(b: Tuple[Vec3, Vec3], m: float) -> Tuple[Vec3, Vec3]:
    m = abs(float(m))
    lo, hi = b
    return ((lo[0] - m, lo[1] - m, lo[2] - m), (hi[0] + m, hi[1] + m, hi[2] + m))


def _union_bounds(a: Tuple[Vec3, Vec3], b: Tuple[Vec3, Vec3]) -> Tuple[Vec3, Vec3]:
    lo = tuple(min(a[0][i], b[0][i]) for i in range(3))
    hi = tuple(max(a[1][i], b[1][i]) for i in range(3))
    return (lo, hi)  # type: ignore[return-value]


def _corners(b: Tuple[Vec3, Vec3]) -> List[Vec3]:
    lo, hi = b
    return [(x, y, z) for x in (lo[0], hi[0])
            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]


def _apply_transform(p: Vec3, tr: Sequence[float]) -> Vec3:
    dx, dy, dz, ang = tr
    x, y, z = p
    if ang:
        a = math.radians(ang)
        ca, sa = math.cos(a), math.sin(a)
        x, y = ca * x - sa * y, sa * x + ca * y
    return (x + dx, y + dy, z + dz)


# -- fillet / chamfer rewriting -------------------------------------------
def blend_tree(node: Node, kind: str, k: float) -> Node:
    """Rebuild ``node`` with every boolean blended and every leaf rounded.

    This is the SDF analogue of a fillet/chamfer feature: there are no B-rep
    edges to name, so the radius is applied uniformly to the convex edges of the
    leaves and to the (concave and convex) edges introduced by the booleans.
    """
    t = node.t
    if t in ("extrude", "revolve"):
        d = dict(node.d)
        d["round"] = k if kind == "smooth" else 0.0
        d["cham"] = k if kind == "chamfer" else 0.0
        return Node(t, **d)
    if t == "cyl":
        return node
    if t == "bool":
        return Node("bool", op=node.d["op"],
                    a=blend_tree(node.d["a"], kind, k),
                    b=blend_tree(node.d["b"], kind, k),
                    blend=kind, k=k)
    if t == "shell":
        return Node("shell", child=blend_tree(node.d["child"], kind, k),
                    thickness=node.d["thickness"])
    if t == "mirror":
        return Node("mirror", child=blend_tree(node.d["child"], kind, k),
                    plane=node.d["plane"])
    if t == "pattern":
        return Node("pattern", child=blend_tree(node.d["child"], kind, k),
                    transforms=node.d["transforms"])
    return node  # pragma: no cover


# --------------------------------------------------------------------------
# tessellation
# --------------------------------------------------------------------------
def grid_extent(bounds: Tuple[Vec3, Vec3],
                resolution: int = DEFAULT_RESOLUTION) -> Tuple[Vec3, Vec3, Tuple[int, int, int], float]:
    """The padded sampling box, its per-axis CELL counts, and the cell size."""
    lo, hi = bounds
    size = [max(hi[i] - lo[i], 1e-9) for i in range(3)]
    cell = max(size) / float(max(int(resolution), 4))
    # 3.25 (not 3) cells: an offset that is not a whole number of cells keeps
    # axis-aligned faces off the sample planes, where exact zeros would make
    # Marching Cubes emit degenerate crossings.
    pad = 3.25 * cell
    mn = (lo[0] - pad, lo[1] - pad, lo[2] - pad)
    mx = (hi[0] + pad, hi[1] + pad, hi[2] + pad)
    res = tuple(max(4, int(math.ceil((mx[i] - mn[i]) / cell))) for i in range(3))
    return mn, mx, res, cell  # type: ignore[return-value]


def resolution_for_tolerance(bounds: Tuple[Vec3, Vec3], tolerance: float) -> int:
    """Cells along the longest axis needed to hold a chord (sagitta) error.

    A tessellation is only as honest as its chord error: the sagitta of the arc
    a facet chords off. ``chord_tolerance.segments_for_tolerance`` answers that
    question for an arc of radius ``r``; the model's bounding sphere is the
    worst-case curvature radius here, so the number of segments it demands for a
    full turn is the number of cells the longest axis must carry.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be > 0")
    lo, hi = bounds
    diag = math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3)))
    radius = max(diag / 2.0, tolerance)
    segments = segments_for_tolerance(radius, 2.0 * math.pi, float(tolerance))
    return max(4, int(segments))


def sample_grid(field: Callable[[Sequence[float]], float],
                bounds: Tuple[Vec3, Vec3],
                resolution: int = DEFAULT_RESOLUTION,
                prune: Optional[frep_ir.CompiledField] = None,
                stats: Optional[dict] = None) -> ScalarGrid:
    """Sample ``field`` on a padded regular grid sized from ``bounds``.

    The grid is padded by three cells on every side so the zero level set is
    strictly interior (the extracted mesh is therefore closed), and exact zeros
    at sample points are nudged outward so that no two Marching-Cubes crossings
    can land on the same grid corner (which would produce duplicate vertices).

    ``prune`` (an IR-compiled copy of the same field) switches on interval
    pruning: blocks of cells whose interval bound proves the surface cannot pass
    through them are never sampled -- see :func:`_sample_pruned`. The extracted
    mesh is bit-for-bit the mesh of the unpruned grid; only the number of field
    evaluations changes. ``stats`` (a dict) receives the evaluation counters.
    """
    mn, mx, res, cell = grid_extent(bounds, resolution)
    if prune is None:
        grid = sample_sdf_grid(field, mn, mx, res)
        if stats is not None:
            n = (res[0] + 1) * (res[1] + 1) * (res[2] + 1)
            stats.update({"samples": n, "field_evals": n, "pruned_samples": 0,
                          "blocks": 0, "blocks_pruned": 0})
    else:
        grid = _sample_pruned(field, prune, mn, mx, res, cell, stats)
    eps = 1e-9 * cell
    vals = grid.values
    for i, v in enumerate(vals):
        if -eps < v < eps:
            vals[i] = eps
    return grid


def _sample_pruned(field: Callable[[Sequence[float]], float],
                   compiled: frep_ir.CompiledField,
                   mn: Vec3, mx: Vec3, res: Tuple[int, int, int], cell: float,
                   stats: Optional[dict]) -> ScalarGrid:
    """Sample the grid, skipping blocks the interval bound proves are uncrossed.

    The sample lattice is cut into blocks of :data:`PRUNE_BLOCK` cells. For each
    block the interval evaluator bounds the field over the block's *closed* world
    box; ``FILLED`` (bound wholly negative) and ``EMPTY`` (wholly positive) blocks
    cannot contain the zero level set, so none of the cells inside them can emit a
    triangle whatever their exact sample values are.

    Correctness: every sample on the closed boundary of an AMBIGUOUS block is
    still evaluated exactly, and every marching-cubes cell lies inside exactly one
    block. So a cell in an ambiguous block sees the same eight exact corner values
    as it would without pruning, and a cell in a pruned block sees eight
    same-signed placeholders -- which produce, as they would have, no triangles.
    The resulting mesh is therefore identical; only the work changes.
    """
    nx, ny, nz = res[0] + 1, res[1] + 1, res[2] + 1
    spacing = ((mx[0] - mn[0]) / res[0], (mx[1] - mn[1]) / res[1],
               (mx[2] - mn[2]) / res[2])

    def world(i: int, j: int, k: int) -> Vec3:
        return (mn[0] + i * spacing[0], mn[1] + j * spacing[1], mn[2] + k * spacing[2])

    # A margin big enough to swallow any float disagreement between the IR's
    # evaluation of the field and the backend's, but far below one cell.
    margin = 1e-6 * cell
    exact = bytearray(nx * ny * nz)
    fill = [0.0] * (nx * ny * nz)
    outside = float(max(mx[i] - mn[i] for i in range(3)))
    blocks = 0
    pruned = 0
    b = int(PRUNE_BLOCK)

    for k0 in range(0, res[2], b):
        k1 = min(k0 + b, res[2])
        for j0 in range(0, res[1], b):
            j1 = min(j0 + b, res[1])
            for i0 in range(0, res[0], b):
                i1 = min(i0 + b, res[0])
                blocks += 1
                verdict = frep_ir.classify_box(
                    compiled, world(i0, j0, k0), world(i1, j1, k1), margin=margin)
                ambiguous = verdict == frep_ir.AMBIGUOUS
                if not ambiguous:
                    pruned += 1
                value = -outside if verdict == frep_ir.FILLED else outside
                for k in range(k0, k1 + 1):
                    for j in range(j0, j1 + 1):
                        base = nx * (j + ny * k)
                        for i in range(i0, i1 + 1):
                            idx = base + i
                            if ambiguous:
                                exact[idx] = 1
                            elif not exact[idx]:
                                fill[idx] = value

    values = [0.0] * (nx * ny * nz)
    evals = 0
    for k in range(nz):
        for j in range(ny):
            base = nx * (j + ny * k)
            for i in range(nx):
                idx = base + i
                if exact[idx]:
                    values[idx] = float(field(world(i, j, k)))
                    evals += 1
                else:
                    values[idx] = fill[idx]
    if stats is not None:
        stats.update({
            "samples": nx * ny * nz,
            "field_evals": evals,
            "pruned_samples": nx * ny * nz - evals,
            "blocks": blocks,
            "blocks_pruned": pruned,
        })
    return ScalarGrid(values, (nx, ny, nz), mn, spacing)


def weld(verts: Sequence[Vec3], faces: Sequence[Sequence[int]],
         tol: float = 1e-9) -> Mesh:
    """Merge coincident vertices (quantised to ``tol``) and drop degenerate faces."""
    if tol <= 0.0:
        tol = 1e-12
    remap: Dict[Tuple[int, int, int], int] = {}
    index: List[int] = []
    out_v: List[Vec3] = []
    for v in verts:
        key = (int(round(v[0] / tol)), int(round(v[1] / tol)), int(round(v[2] / tol)))
        got = remap.get(key)
        if got is None:
            got = len(out_v)
            remap[key] = got
            out_v.append((float(v[0]), float(v[1]), float(v[2])))
        index.append(got)
    out_f: List[Tuple[int, int, int]] = []
    for f in faces:
        a, b, c = index[f[0]], index[f[1]], index[f[2]]
        if a == b or b == c or a == c:
            continue
        out_f.append((a, b, c))
    # drop vertices no surviving face references (keeps the half-edge check clean)
    used = sorted({i for f in out_f for i in f})
    if len(used) != len(out_v):
        renum = {old: new for new, old in enumerate(used)}
        out_v = [out_v[i] for i in used]
        out_f = [(renum[a], renum[b], renum[c]) for (a, b, c) in out_f]
    return out_v, out_f


def tessellate(field: Callable[[Sequence[float]], float],
               bounds: Tuple[Vec3, Vec3],
               resolution: int = DEFAULT_RESOLUTION,
               algorithm: str = DEFAULT_MESHER,
               prune: Optional[frep_ir.CompiledField] = None,
               stats: Optional[dict] = None) -> Mesh:
    """Sample the field and extract a welded iso-surface triangle mesh.

    ``algorithm`` selects one of :data:`MESHERS`. They are rivals: marching cubes
    puts vertices on cell EDGES (primal), surface nets puts one vertex per CELL
    (dual). Both are valid extractions of the same field; neither is a refinement
    of the other, and nothing here blends them.
    """
    if algorithm not in MESHERS:
        raise ValueError("unknown mesher %r (supported: %s)"
                         % (algorithm, ", ".join(MESHERS)))
    grid = sample_grid(field, bounds, resolution, prune=prune, stats=stats)
    if algorithm == "surface_nets":
        verts, quads = surface_nets(grid, 0.0)
        faces: List[Tuple[int, int, int]] = []
        for q in quads:
            if len(q) == 4:
                faces.append((q[0], q[1], q[2]))
                faces.append((q[0], q[2], q[3]))
            elif len(q) == 3:
                faces.append((q[0], q[1], q[2]))
    else:
        verts, faces = marching_cubes(grid, 0.0)
    tol = min(grid.spacing) * 1e-6
    return weld(verts, faces, tol)


def mesh_triangles(mesh: Mesh) -> List[stl_fmt.Triangle]:
    verts, faces = mesh
    return [stl_fmt.Triangle(verts[a], verts[b], verts[c]) for (a, b, c) in faces]


# --------------------------------------------------------------------------
# the backend
# --------------------------------------------------------------------------
class FRepBackend:
    """A GeometryBackend that realises CISP ops as signed-distance fields."""

    #: exports this backend can produce
    FORMATS = ("stl", "stl-ascii", "stl-binary", "glb", "sdf")

    def __init__(self, resolution: int = DEFAULT_RESOLUTION,
                 mesher: str = DEFAULT_MESHER,
                 normals: str = DEFAULT_NORMALS,
                 prune: bool = False) -> None:
        if mesher not in MESHERS:
            raise ValueError("unknown mesher %r (supported: %s)"
                             % (mesher, ", ".join(MESHERS)))
        if normals not in NORMAL_METHODS:
            raise ValueError("unknown normal method %r (supported: %s)"
                             % (normals, ", ".join(NORMAL_METHODS)))
        self.resolution = int(resolution)
        self.mesher = str(mesher)
        self.normals = str(normals)
        self.prune = bool(prune)
        self.reset()

    # -- state -------------------------------------------------------------
    def reset(self) -> None:
        self.sketches: dict = {}     # sid -> {plane, entities, dof}
        self.entities: dict = {}     # eid -> {type, sketch, params}
        self.features: list = []     # [{type, id, ...}]
        self.instances: list = []
        self.mates: list = []
        self.solid_present = False
        self._bodies: List[dict] = []   # [{"id": fid, "node": Node}]
        self._oplog: list = []
        self._mesh_cache: Optional[Tuple[str, Mesh]] = None
        self._ir_cache: Dict[str, Optional[frep_ir.CompiledField]] = {}
        self._n = {"sk": 0, "e": 0, "f": 0, "i": 0}

    def _new_id(self, kind: str) -> str:
        self._n[kind] += 1
        return {"sk": "sk", "e": "e", "f": "f", "i": "i"}[kind] + str(self._n[kind])

    def _invalidate(self) -> None:
        self._mesh_cache = None
        self._ir_cache = {}

    # -- op dispatch -------------------------------------------------------
    def apply(self, op: Op) -> ApplyResult:
        if isinstance(op, SetParam):
            return self._set_param(op)
        result = self._dispatch(op)
        if result.ok:
            self._oplog.append(op)
            self._invalidate()
        return result

    def _dispatch(self, op: Op) -> ApplyResult:
        if isinstance(op, NewSketch):
            if str(op.plane).upper() not in _PLANES:
                return _err("bad-value", f"unknown sketch plane '{op.plane}'")
            sid = self._new_id("sk")
            self.sketches[sid] = {"plane": str(op.plane).upper(),
                                  "entities": [], "dof": 0}
            return ApplyResult(True, [sid])
        if isinstance(op, AddPoint):
            return self._add_primitive(op.sketch, "point", {"x": op.x, "y": op.y})
        if isinstance(op, AddLine):
            return self._add_primitive(op.sketch, "line",
                                       {"x1": op.x1, "y1": op.y1,
                                        "x2": op.x2, "y2": op.y2})
        if isinstance(op, AddCircle):
            if op.r <= 0:
                return _err("bad-value", f"circle radius must be > 0 (got {op.r})")
            return self._add_primitive(op.sketch, "circle",
                                       {"cx": op.cx, "cy": op.cy, "r": op.r})
        if isinstance(op, AddRectangle):
            if op.w <= 0 or op.h <= 0:
                return _err("bad-value", "rectangle w and h must be > 0")
            return self._add_primitive(op.sketch, "rectangle",
                                       {"x": op.x, "y": op.y, "w": op.w, "h": op.h})
        if isinstance(op, Constrain):
            return self._constrain(op)
        if isinstance(op, Extrude):
            return self._extrude(op)
        if isinstance(op, Revolve):
            return self._revolve(op)
        if isinstance(op, Boolean):
            return self._boolean(op)
        if isinstance(op, Fillet):
            if op.radius <= 0:
                return _err("bad-value", f"fillet radius must be > 0 (got {op.radius})")
            return self._blend("fillet", "smooth", op.radius, op.edges)
        if isinstance(op, Chamfer):
            if op.distance <= 0:
                return _err("bad-value",
                            f"chamfer distance must be > 0 (got {op.distance})")
            return self._blend("chamfer", "chamfer", op.distance, op.edges)
        if isinstance(op, Hole):
            return self._hole(op)
        if isinstance(op, Shell):
            return self._shell(op)
        if isinstance(op, Mirror):
            return self._mirror(op)
        if isinstance(op, LinearPattern):
            return self._linear_pattern(op)
        if isinstance(op, CircularPattern):
            return self._circular_pattern(op)
        if isinstance(op, AddInstance):
            return self._add_instance(op)
        if isinstance(op, Mate):
            return self._mate(op)
        if isinstance(op, (Draft, Loft, Sweep)):
            return _err("unsupported-op",
                        f"the frep backend does not implement "
                        f"{type(op).__name__.lower()} yet")
        return _err("unknown-op", f"unhandled op {type(op).__name__}")

    # -- sketch ------------------------------------------------------------
    def _add_primitive(self, sketch: str, kind: str, params: dict) -> ApplyResult:
        if sketch not in self.sketches:
            return _err("bad-ref", f"unknown sketch '{sketch}'", sketch)
        eid = self._new_id("e")
        self.entities[eid] = {"type": kind, "sketch": sketch, "params": params}
        self.sketches[sketch]["entities"].append(eid)
        self.sketches[sketch]["dof"] += PRIMITIVE_DOF[kind]
        return ApplyResult(True, [eid])

    def _constrain(self, op: Constrain) -> ApplyResult:
        if op.kind not in CONSTRAINT_DOF:
            return _err("bad-value", f"unknown constraint kind '{op.kind}'")
        if op.kind in ("distance", "radius") and op.value is None:
            return _err("bad-value", f"'{op.kind}' constraint requires a value")
        if op.a not in self.entities:
            return _err("bad-ref", f"unknown entity '{op.a}'", op.a)
        if op.b is not None and op.b not in self.entities:
            return _err("bad-ref", f"unknown entity '{op.b}'", op.b)
        sid = self.entities[op.a]["sketch"]
        self.sketches[sid]["dof"] -= CONSTRAINT_DOF[op.kind]
        self.sketches[sid].setdefault("constraints", []).append(op.kind)
        return ApplyResult(True, [])

    def _sketch_profile(self, sid: str):
        if sid not in self.sketches:
            return None, _err("bad-ref", f"unknown sketch '{sid}'", sid)
        sk = self.sketches[sid]
        if not sk["entities"]:
            return None, _err("empty-sketch", f"sketch '{sid}' has no profile", sid)
        prof = _profile_of(sk, self.entities)
        if prof.empty():
            return None, _err("empty-sketch",
                              f"sketch '{sid}' has no closed profile to sweep", sid)
        return prof, None

    # -- solids ------------------------------------------------------------
    def _push_body(self, feature: str, node: Node, **extra) -> ApplyResult:
        fid = self._new_id("f")
        self.features.append(dict({"type": feature, "id": fid}, **extra))
        self._bodies.append({"id": fid, "node": node})
        self.solid_present = True
        return ApplyResult(True, [fid])

    def _extrude(self, op: Extrude) -> ApplyResult:
        prof, err = self._sketch_profile(op.sketch)
        if err is not None:
            return err
        if op.distance == 0:
            return _err("bad-value", "extrude distance must be non-zero")
        node = Node("extrude", profile=prof, plane=self.sketches[op.sketch]["plane"],
                    w0=0.0, w1=float(op.distance), round=0.0, cham=0.0)
        return self._push_body("extrude", node, sketch=op.sketch)

    def _revolve(self, op: Revolve) -> ApplyResult:
        prof, err = self._sketch_profile(op.sketch)
        if err is not None:
            return err
        if op.angle == 0:
            return _err("bad-value", "revolve angle must be non-zero")
        ax = list(op.axis) + [0.0] * 6
        au, av = float(ax[0]), float(ax[1])
        bu, bv = float(ax[3]), float(ax[4])
        du, dv = bu - au, bv - av
        n = math.hypot(du, dv)
        if n == 0.0:
            return _err("bad-value", "revolve axis is degenerate (zero length)")
        du, dv = du / n, dv / n
        nu, nv = -dv, du
        lo_u, lo_v, hi_u, hi_v = prof.bounds()
        mid_u, mid_v = 0.5 * (lo_u + hi_u), 0.5 * (lo_v + hi_v)
        if (mid_u - au) * nu + (mid_v - av) * nv < 0.0:
            nu, nv = dv, -du  # keep the profile on the positive radial side
        node = Node("revolve", profile=prof, plane=self.sketches[op.sketch]["plane"],
                    axis=(au, av, du, dv, nu, nv), angle=float(op.angle),
                    round=0.0, cham=0.0)
        return self._push_body("revolve", node, sketch=op.sketch)

    def _resolve_body(self, ref: str) -> Optional[int]:
        for i, b in enumerate(self._bodies):
            if b["id"] == ref:
                return i
        return None

    def _boolean(self, op: Boolean) -> ApplyResult:
        if op.kind not in ("union", "cut", "intersect"):
            return _err("bad-value", f"unknown boolean kind '{op.kind}'")
        if len(self._bodies) < 2:
            return _err("no-solid", "boolean requires two solids")
        ia = self._resolve_body(op.target) if op.target else len(self._bodies) - 2
        ib = self._resolve_body(op.tool) if op.tool else len(self._bodies) - 1
        if ia is None:
            return _err("bad-ref", f"unknown boolean target '{op.target}'", op.target)
        if ib is None:
            return _err("bad-ref", f"unknown boolean tool '{op.tool}'", op.tool)
        if ia == ib:
            return _err("bad-ref", "boolean target and tool are the same body")
        a = self._bodies[ia]["node"]
        b = self._bodies[ib]["node"]
        node = Node("bool", op=op.kind, a=a, b=b, blend="hard", k=0.0)
        # the two operands are consumed and replaced by the result
        for i in sorted((ia, ib), reverse=True):
            self._bodies.pop(i)
        return self._push_body("boolean", node, kind=op.kind)

    def _blend(self, feature: str, kind: str, k: float, edges) -> ApplyResult:
        if not self._bodies:
            return _err("no-solid", f"{feature} requires an existing solid")
        body = self._bodies[-1]
        body["node"] = blend_tree(body["node"], kind, float(k))
        fid = self._new_id("f")
        self.features.append({"type": feature, "id": fid,
                              "edges": list(edges), "value": float(k)})
        return ApplyResult(True, [fid])

    def _hole(self, op: Hole) -> ApplyResult:
        if op.diameter <= 0:
            return _err("bad-value", f"hole diameter must be > 0 (got {op.diameter})")
        if not op.through and (op.depth is None or op.depth <= 0):
            return _err("bad-value", "blind hole requires depth > 0")
        if op.kind not in ("simple", "counterbore", "countersink"):
            return _err("bad-value", f"unknown hole kind '{op.kind}'")
        ref = op.face_or_sketch
        plane = "XY"
        if ref.startswith("sk"):
            if ref not in self.sketches:
                return _err("bad-ref", f"unknown sketch '{ref}'", ref)
            plane = self.sketches[ref]["plane"]
        elif not self._bodies:
            return _err("no-solid", "hole requires an existing solid")
        if not self._bodies:
            return _err("no-solid", "hole requires an existing solid")
        target = self._bodies[-1]
        lo, hi = node_bounds(target["node"])
        _, _, iw = _plane_axes(plane)
        span = hi[iw] - lo[iw]
        if op.through:
            w0, w1 = lo[iw] - span - 1.0, hi[iw] + span + 1.0
        else:
            w0, w1 = hi[iw] - float(op.depth), hi[iw] + span + 1.0
        tool = Node("cyl", plane=plane, cu=float(op.x), cv=float(op.y),
                    r=float(op.diameter) / 2.0, w0=w0, w1=w1)
        target["node"] = Node("bool", op="cut", a=target["node"], b=tool,
                              blend="hard", k=0.0)
        fid = self._new_id("f")
        self.features.append({"type": "hole", "id": fid, "ref": ref,
                              "diameter": op.diameter, "kind": op.kind})
        self.solid_present = True
        return ApplyResult(True, [fid])

    def _shell(self, op: Shell) -> ApplyResult:
        if not self._bodies:
            return _err("no-solid", "shell requires an existing solid")
        if op.thickness <= 0:
            return _err("bad-value", f"shell thickness must be > 0 (got {op.thickness})")
        body = self._bodies[-1]
        body["node"] = Node("shell", child=body["node"],
                            thickness=float(op.thickness))
        fid = self._new_id("f")
        self.features.append({"type": "shell", "id": fid,
                              "faces": list(op.faces), "thickness": op.thickness})
        return ApplyResult(True, [fid])

    def _mirror(self, op: Mirror) -> ApplyResult:
        if not self._bodies:
            return _err("no-solid", "mirror requires an existing solid")
        if str(op.plane).upper() not in _PLANES:
            return _err("bad-value", f"unknown mirror plane '{op.plane}'")
        if op.feature_or_body and op.feature_or_body not in self._feature_ids():
            return _err("bad-ref", f"unknown feature '{op.feature_or_body}'",
                        op.feature_or_body)
        body = self._bodies[-1]
        body["node"] = Node("mirror", child=body["node"],
                            plane=str(op.plane).upper())
        fid = self._new_id("f")
        self.features.append({"type": "mirror", "id": fid, "plane": op.plane})
        return ApplyResult(True, [fid])

    def _pattern_body(self, feature: str, count: int,
                      transforms: List[Tuple[float, float, float, float]]) -> ApplyResult:
        body = self._bodies[-1]
        body["node"] = Node("pattern", child=body["node"], transforms=transforms)
        fid = self._new_id("f")
        self.features.append({"type": feature, "id": fid, "count": count})
        return ApplyResult(True, [fid])

    def _linear_pattern(self, op: LinearPattern) -> ApplyResult:
        if not self._bodies:
            return _err("no-solid", "linear_pattern requires an existing solid")
        if op.count < 2:
            return _err("bad-value", f"linear_pattern count must be >= 2 (got {op.count})")
        if op.feature and op.feature not in self._feature_ids():
            return _err("bad-ref", f"unknown feature '{op.feature}'", op.feature)
        d = list(op.direction) + [0.0] * 3
        n = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        if n == 0.0:
            return _err("bad-value", "linear_pattern direction is degenerate")
        ux, uy, uz = d[0] / n, d[1] / n, d[2] / n
        trs = [(ux * op.spacing * i, uy * op.spacing * i, uz * op.spacing * i, 0.0)
               for i in range(int(op.count))]
        return self._pattern_body("linear_pattern", int(op.count), trs)

    def _circular_pattern(self, op: CircularPattern) -> ApplyResult:
        if not self._bodies:
            return _err("no-solid", "circular_pattern requires an existing solid")
        if op.count < 2:
            return _err("bad-value",
                        f"circular_pattern count must be >= 2 (got {op.count})")
        if op.feature and op.feature not in self._feature_ids():
            return _err("bad-ref", f"unknown feature '{op.feature}'", op.feature)
        step = float(op.angle) / float(op.count)
        trs = [(0.0, 0.0, 0.0, step * i) for i in range(int(op.count))]
        return self._pattern_body("circular_pattern", int(op.count), trs)

    # -- assembly ----------------------------------------------------------
    def _feature_ids(self) -> set:
        return {f["id"] for f in self.features}

    def _instance_ids(self) -> set:
        return {inst["id"] for inst in self.instances}

    def _known_part_refs(self) -> set:
        refs = self._feature_ids() | self._instance_ids()
        if self.solid_present:
            refs |= {"solid", "body", "last"}
        return refs

    def _add_instance(self, op: AddInstance) -> ApplyResult:
        if op.part not in self._known_part_refs():
            return _err("bad-ref", f"unknown part '{op.part}'", op.part)
        iid = self._new_id("i")
        bbox = None
        if self._bodies:
            lo, hi = node_bounds(self._bodies[-1]["node"])
            bbox = [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]
        self.instances.append({
            "id": iid, "part": op.part,
            "transform": {"translate": [op.x, op.y, op.z],
                          "rotate_deg": [op.rx, op.ry, op.rz]},
            "bbox": bbox,
        })
        return ApplyResult(True, [iid])

    def _mate(self, op: Mate) -> ApplyResult:
        if mate_dof(op.kind) is None:
            return _err("bad-value", f"unknown mate kind '{op.kind}'")
        refs = self._instance_ids() | self._feature_ids()
        for ref in (op.a, op.b):
            if ref and ref not in refs:
                return _err("bad-ref", f"unknown mate ref '{ref}'", ref)
        self.mates.append({"kind": op.kind, "a": op.a, "b": op.b, "value": op.value})
        return ApplyResult(True, [])

    def _set_param(self, op: SetParam) -> ApplyResult:
        new_log, err = edit_oplog(self._oplog, op)
        if err is not None:
            return _err(*err)
        trial = type(self)(resolution=self.resolution, mesher=self.mesher,
                           normals=self.normals, prune=self.prune)
        for logged in new_log:
            r = trial.apply(logged)
            if not r.ok:
                return ApplyResult(False, [], r.diagnostics)
        self.__dict__.update(trial.__dict__)
        return ApplyResult(True, [])

    # -- geometry ----------------------------------------------------------
    def root(self) -> Optional[Node]:
        """The whole model as one F-rep node (union of all live bodies)."""
        if not self._bodies:
            return None
        node = self._bodies[0]["node"]
        for b in self._bodies[1:]:
            node = Node("bool", op="union", a=node, b=b["node"], blend="hard", k=0.0)
        return node

    def field(self) -> Optional[Callable[[Sequence[float]], float]]:
        """The model's signed-distance function, or None when there is no solid."""
        node = self.root()
        if node is None:
            return None
        return lambda p: eval_node(node, p)

    def bounds(self) -> Optional[Tuple[Vec3, Vec3]]:
        node = self.root()
        return None if node is None else node_bounds(node)

    # -- the arithmetic f-rep IR ------------------------------------------
    def ir(self, smooth: bool = False) -> Optional[frep_ir.CompiledField]:
        """The model as an arithmetic f-rep expression graph, or None.

        ``None`` means the tree is not IR-expressible (today: a sketch profile
        built from line segments -- its inside/outside test is a winding number,
        not arithmetic). Callers that want exact normals or interval pruning must
        degrade gracefully; they all do.

        ``smooth=True`` returns the graph to DIFFERENTIATE, ``smooth=False`` the
        graph to BOUND: the same function in two encodings (see
        ``frep_ir._Builder``).
        """
        node = self.root()
        if node is None:
            return None
        key = "%s|%d" % (self.state_digest(), 1 if smooth else 0)
        cache = self._ir_cache or {}
        if isinstance(cache, dict) and key in cache:
            return cache[key]
        compiled = frep_ir.try_compile(node, smooth=smooth)
        if not isinstance(cache, dict):
            cache = {}
        cache[key] = compiled
        self._ir_cache = cache
        return compiled

    def normal(self, p: Sequence[float], method: Optional[str] = None,
               h: float = 1e-6) -> Vec3:
        """Unit outward surface normal of the field at ``p``.

        ``method='finite_difference'`` (the default) uses the central-difference
        estimator; ``method='autodiff'`` compiles the model to the f-rep IR and
        reads the exact gradient off a forward-mode dual-number pass. They are
        rivals: the AD normal is exact, the FD normal is an O(h^2) approximation
        of it. Nothing blends them, and the default is unchanged.
        """
        how = self.normals if method is None else str(method)
        if how not in NORMAL_METHODS:
            raise ValueError("unknown normal method %r (supported: %s)"
                             % (how, ", ".join(NORMAL_METHODS)))
        node = self.root()
        if node is None:
            raise ValueError("no solid: the model has no field to differentiate")
        if how == "autodiff":
            compiled = self.ir(smooth=True)
            if compiled is None:
                raise ValueError(
                    "this model is not f-rep-IR-expressible (polygon sketch "
                    "profile); use method='finite_difference'")
            return frep_ir.exact_normal(compiled, p)
        gx = (eval_node(node, (p[0] + h, p[1], p[2]))
              - eval_node(node, (p[0] - h, p[1], p[2]))) / (2.0 * h)
        gy = (eval_node(node, (p[0], p[1] + h, p[2]))
              - eval_node(node, (p[0], p[1] - h, p[2]))) / (2.0 * h)
        gz = (eval_node(node, (p[0], p[1], p[2] + h))
              - eval_node(node, (p[0], p[1], p[2] - h))) / (2.0 * h)
        mag = math.sqrt(gx * gx + gy * gy + gz * gz)
        if mag == 0.0:
            return (0.0, 0.0, 0.0)
        return (gx / mag, gy / mag, gz / mag)

    # -- tessellation ------------------------------------------------------
    def mesh(self, resolution: Optional[int] = None,
             algorithm: Optional[str] = None,
             mesher: Optional[str] = None,
             prune: Optional[bool] = None,
             tolerance: Optional[float] = None,
             stats: Optional[dict] = None) -> Mesh:
        """Tessellate the model.  Cached against the state digest.

        ``mesher`` (alias: ``algorithm``) picks one of :data:`MESHERS`; the
        default stays ``marching_cubes``. ``tolerance`` sizes the grid from a
        chord-error budget instead of a cell count. ``prune`` turns on interval
        pruning, which changes only the amount of work, never the mesh.
        """
        node = self.root()
        if node is None:
            return ([], [])
        how = mesher or algorithm or self.mesher
        if how not in MESHERS:
            raise ValueError("unknown mesher %r (supported: %s)"
                             % (how, ", ".join(MESHERS)))
        bounds = node_bounds(node)
        if tolerance is not None:
            res = resolution_for_tolerance(bounds, float(tolerance))
        else:
            res = self.resolution if resolution is None else int(resolution)
        do_prune = self.prune if prune is None else bool(prune)
        compiled = self.ir() if do_prune else None
        key = "%s|%d|%s|%d" % (self.state_digest(), res, how, 1 if compiled else 0)
        if stats is None and self._mesh_cache is not None and self._mesh_cache[0] == key:
            return self._mesh_cache[1]
        m = tessellate(lambda p: eval_node(node, p), bounds, res, how,
                       prune=compiled, stats=stats)
        self._mesh_cache = (key, m)
        return m

    def regenerate(self) -> List[Diagnostic]:
        """Rebuild the tessellation and report any non-manifold output."""
        if not self._bodies:
            return []
        verts, faces = self.mesh()
        if not faces:
            return [Diagnostic(Severity.ERROR, "empty-solid",
                               "the field produced no iso-surface (empty solid)")]
        he = HalfedgeMesh(verts, faces)
        ok, issues = he.is_2manifold()
        if ok:
            return []
        codes = sorted({i.code for i in issues})
        return [Diagnostic(Severity.ERROR, "invalid-mesh",
                           "tessellated mesh is not a 2-manifold (%s; %d issues)"
                           % (", ".join(codes), len(issues)))]

    # -- queries -----------------------------------------------------------
    def query(self, q: str) -> dict:
        if q == "sketch_dof":
            return {sid: s["dof"] for sid, s in self.sketches.items()}
        if q == "summary":
            return {
                "sketch_count": len(self.sketches),
                "entity_count": len(self.entities),
                "feature_count": len(self.features),
                "solid_present": self.solid_present,
            }
        if q == "validity":
            return self._validity()
        if q == "measure":
            m = self._metrics()
            if not m:
                return {"volume": 0.0, "bbox": [0.0, 0.0, 0.0]}
            return {"volume": m["volume"], "bbox": m["bbox"]}
        if q == "metrics":
            return self._metrics()
        if q == "mass_properties":
            return self.mass_properties()
        if q == "mesh":
            verts, faces = self.mesh()
            return {"vertex_count": len(verts), "triangle_count": len(faces)}
        if q == "assembly":
            return self._assembly()
        return {}

    def _validity(self) -> dict:
        if not self._bodies:
            return {"manifold": False, "watertight": False,
                    "is_valid": False, "solid_present": False}
        verts, faces = self.mesh()
        if not faces:
            return {"manifold": False, "watertight": False,
                    "is_valid": False, "solid_present": True}
        he = HalfedgeMesh(verts, faces)
        manifold, issues = he.is_2manifold()
        watertight = he.is_closed()
        return {"manifold": bool(manifold), "watertight": bool(watertight),
                "is_valid": bool(manifold and watertight),
                "solid_present": True,
                "genus": he.genus() if watertight else None,
                "euler_characteristic": he.euler_characteristic(),
                "issues": len(issues)}

    def _metrics(self, density: float = 1.0) -> dict:
        """Mass properties read off the extracted mesh (no kernel involved)."""
        if not self._bodies:
            return {}
        verts, faces = self.mesh()
        if not faces:
            return {}
        volume = abs(mesh_signed_volume(verts, faces))
        area = mesh_surface_area(verts, faces)
        lo = [min(v[i] for v in verts) for i in range(3)]
        hi = [max(v[i] for v in verts) for i in range(3)]
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        cz = sum(v[2] for v in verts) / len(verts)
        return {
            "volume": float(volume),
            "mass": float(volume * density),
            "surface_area": float(area),
            "bbox": [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]],
            "center_of_mass": [cx, cy, cz],
            "triangle_count": len(faces),
            "vertex_count": len(verts),
        }

    def mass_properties(self, density: float = 1.0, order: int = 3,
                        resolution: Optional[int] = None,
                        mesher: Optional[str] = None) -> dict:
        """Volume, centroid and the full inertia tensor, by Gauss quadrature.

        The mesh-based ``metrics`` query reports a volume and a *vertex-average*
        "centre of mass", which is not the centre of mass of a solid at all. This
        is the real thing: each mass integral is turned into a surface integral by
        the divergence theorem

            V     = 1/3 . closed-integral(p . n) dA
            C_i   = 1/(2V) . closed-integral(p_i^2 n_i) dA
            I_ii  = 1/3 . closed-integral(p_j^3 n_j + p_k^3 n_k) dA
            P_ij  = 1/2 . closed-integral(p_i p_j^2 n_j) dA

        and each triangle's surface integral is evaluated with an ``order``-point
        Gauss-Legendre rule (:mod:`harnesscad.domain.numeric.quadrature`) on the
        unit square mapped onto the triangle by the Duffy transform. The
        integrands are cubic, so order 3 (exact through degree 5) integrates them
        exactly: the answer is the exact mass property OF THE TESSELLATION, with
        no quadrature error of its own.

        The inertia tensor is reported about the centre of mass (the parallel-axis
        shift is applied), which is the convention every CAD kernel uses.
        """
        verts, faces = self.mesh(resolution=resolution, mesher=mesher)
        if not faces:
            return {}
        nodes, weights = quadrature.nodes_and_weights(int(order))

        v_int = 0.0
        m1 = [0.0, 0.0, 0.0]            # closed-integral p_i^2 n_i dA
        m3 = [0.0, 0.0, 0.0]            # closed-integral p_i^3 n_i dA
        prod = [0.0, 0.0, 0.0]          # xy, yz, xz products
        for (ia, ib, ic) in faces:
            a, b, c = verts[ia], verts[ib], verts[ic]
            e1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
            e2 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
            cr = (e1[1] * e2[2] - e1[2] * e2[1],
                  e1[2] * e2[0] - e1[0] * e2[2],
                  e1[0] * e2[1] - e1[1] * e2[0])
            twice_area = math.sqrt(cr[0] ** 2 + cr[1] ** 2 + cr[2] ** 2)
            if twice_area == 0.0:
                continue
            n = (cr[0] / twice_area, cr[1] / twice_area, cr[2] / twice_area)
            for gu, wu in zip(nodes, weights):
                # Gauss nodes live on [-1, 1]; map to [0, 1] (weight halves).
                u = 0.5 * (gu + 1.0)
                for gv, wv in zip(nodes, weights):
                    t = 0.5 * (gv + 1.0)
                    s = t * (1.0 - u)              # Duffy: square -> triangle
                    # dA = 2A * (1-u) * (du dt);  the 0.5*0.5 folds the [-1,1] map
                    jac = 0.25 * wu * wv * twice_area * (1.0 - u)
                    px = a[0] + e1[0] * u + e2[0] * s
                    py = a[1] + e1[1] * u + e2[1] * s
                    pz = a[2] + e1[2] * u + e2[2] * s
                    v_int += jac * (px * n[0] + py * n[1] + pz * n[2])
                    m1[0] += jac * px * px * n[0]
                    m1[1] += jac * py * py * n[1]
                    m1[2] += jac * pz * pz * n[2]
                    m3[0] += jac * px ** 3 * n[0]
                    m3[1] += jac * py ** 3 * n[1]
                    m3[2] += jac * pz ** 3 * n[2]
                    prod[0] += jac * px * py * py * n[1]     # 2 * integral xy dV
                    prod[1] += jac * py * pz * pz * n[2]     # 2 * integral yz dV
                    prod[2] += jac * px * pz * pz * n[2]     # 2 * integral xz dV

        volume = v_int / 3.0
        sign = -1.0 if volume < 0.0 else 1.0          # tolerate inward normals
        volume = abs(volume)
        if volume == 0.0:
            return {}
        cx = sign * m1[0] / (2.0 * volume)
        cy = sign * m1[1] / (2.0 * volume)
        cz = sign * m1[2] / (2.0 * volume)
        ixx = sign * (m3[1] + m3[2]) / 3.0
        iyy = sign * (m3[0] + m3[2]) / 3.0
        izz = sign * (m3[0] + m3[1]) / 3.0
        pxy = sign * prod[0] / 2.0
        pyz = sign * prod[1] / 2.0
        pxz = sign * prod[2] / 2.0
        # parallel-axis shift to the centre of mass
        ixx -= volume * (cy * cy + cz * cz)
        iyy -= volume * (cx * cx + cz * cz)
        izz -= volume * (cx * cx + cy * cy)
        pxy -= volume * cx * cy
        pyz -= volume * cy * cz
        pxz -= volume * cx * cz
        d = float(density)
        tensor = [
            [ixx * d, -pxy * d, -pxz * d],
            [-pxy * d, iyy * d, -pyz * d],
            [-pxz * d, -pyz * d, izz * d],
        ]
        return {
            "volume": volume,
            "mass": volume * d,
            "center_of_mass": [cx, cy, cz],
            "inertia_tensor": tensor,
            "principal_moments": [ixx * d, iyy * d, izz * d],
            "quadrature_order": int(order),
            "triangle_count": len(faces),
        }

    def _assembly(self) -> dict:
        if not self.instances and not self.mates:
            return {}
        parts = []
        transforms = {}
        for inst in self.instances:
            part = {"id": inst["id"], "name": inst["part"],
                    "transform": inst["transform"]}
            if inst.get("bbox") is not None:
                part["bbox"] = list(inst["bbox"])
            parts.append(part)
            transforms[inst["id"]] = inst["transform"]
        return {"parts": parts, "mates": [dict(m) for m in self.mates],
                "transforms": transforms}

    # -- export ------------------------------------------------------------
    def export(self, fmt: str):
        f = str(fmt).lower()
        if f == "sdf":
            node = self.root()
            return json.dumps({} if node is None else node.spec(),
                              sort_keys=True, separators=(",", ":"))
        if f not in ("stl", "stl-ascii", "stl-binary", "stlb", "glb"):
            raise ValueError(
                "frep backend cannot export '%s' (supported: %s)"
                % (fmt, ", ".join(self.FORMATS)))
        tris = mesh_triangles(self.mesh())
        if f in ("stl", "stl-ascii"):
            return stl_fmt.write_ascii_stl(tris, name="harnesscad-frep")
        binary = stl_fmt.write_binary_stl(tris, header=b"harnesscad-frep")
        if f == "glb":
            return glb_fmt.stl_to_glb(binary, name="harnesscad-frep")
        return binary

    def write_stl(self, path: str, binary: bool = True) -> int:
        """Write the tessellated model to ``path``; returns the triangle count."""
        tris = mesh_triangles(self.mesh())
        if binary:
            with open(path, "wb") as fh:
                fh.write(stl_fmt.write_binary_stl(tris, header=b"harnesscad-frep"))
        else:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(stl_fmt.write_ascii_stl(tris, name="harnesscad-frep"))
        return len(tris)

    # -- digest ------------------------------------------------------------
    def state_digest(self) -> str:
        node = self.root()
        model = {
            "sketches": self.sketches,
            "entities": self.entities,
            "features": self.features,
            "instances": self.instances,
            "mates": self.mates,
            "solid_present": self.solid_present,
            "frep": None if node is None else node.spec(),
            "oplog": [canonical_json(o) for o in self._oplog],
        }
        blob = json.dumps(model, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()
