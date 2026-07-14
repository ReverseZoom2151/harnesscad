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
from harnesscad.domain.geometry.topology import selector_dsl
from harnesscad.domain.geometry.parametric.chord_tolerance import segments_for_tolerance
from harnesscad.domain.geometry.volumes.dual_contouring_3d import dual_contour
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
#: blended. ``dual_contouring`` is THE DEFAULT: it places the cell vertex by a
#: QEF over Hermite data (Ju, Losasso, Schaefer & Warren 2002) and is the only
#: one of the three that can put a vertex exactly on a sharp corner.
#: ``marching_cubes`` structurally CANNOT represent a sharp edge -- it can only
#: interpolate along cell edges, so it chamfers material off every corner, and
#: the resulting volume error is large and ONE-SIDED (always negative).
#: ``surface_nets`` is the same dual layout with the vertex at the cell centroid.
#:
#: Measured on a 60x40x20 block against the analytic volume (tests/io/backends/
#: test_frep.py::FRepMesherVolumeErrorTest):
#:
#:     res | marching cubes | dual contouring
#:      16 |    -3.0521%    |    -0.0082%
#:      32 |    -0.8287%    |    -0.0048%
#:      48 |    -0.3758%    |    -0.0035%
#:      96 |    -0.1008%    |    -0.0019%
#:
#: DC at res 16 beats MC at res 96: a 216x cheaper grid and 12x more accurate.
#: Choose with ``mesher=``; nothing mixes them.
MESHERS: Tuple[str, ...] = ("marching_cubes", "surface_nets", "dual_contouring")
DEFAULT_MESHER = "dual_contouring"

#: How a surface normal is obtained. ``finite_difference`` is the default (it is
#: what the mesh writers have always used, via the codecs); ``autodiff`` compiles
#: the CSG tree to the arithmetic f-rep IR and reads the exact gradient off a
#: forward-mode dual-number pass. Rivals, selectable, never blended.
NORMAL_METHODS: Tuple[str, ...] = ("finite_difference", "autodiff")
DEFAULT_NORMALS = "finite_difference"

#: How many grid CELLS a wall must span before this backend will agree to build it,
#: PER MESHER. Below the floor the wall is not in the sampled data and no extraction
#: can recover it, so the op is refused rather than silently building a smaller part.
#:
#: THE BUG THIS EXISTS TO KILL. An 80 x 30 x 5 plate shelled to t=1 at resolution 48
#: came back as 78.13 x 28.22 x 3.52 -- the OUTER surface pulled in by 2 mm on every
#: side, 75% under volume -- watertight, 2-manifold, ``is_valid`` True, and ZERO
#: diagnostics. The field is exact; the grid cannot see a 1 mm wall when the cell is
#: 80/48 = 1.67 mm. Marching cubes then meshed a different, smaller solid and
#: reported success.
#:
#: WHAT IS REFUSED, AND WHAT IS NOT. The line is SUB-CELL REPRESENTABILITY, not
#: accuracy. A wall thinner than the sample spacing is not in the data at all: the
#: two surfaces bounding it fall inside one cell, the mesher cannot separate them,
#: and it builds a smaller solid. That is a different failure from ordinary
#: discretisation error, which every sampled measurement in this backend carries and
#: which is the caller's choice of ``resolution``, not a bug. Measured, on the
#: 60x40x20 box at t=3 (analytic hollow volume 22296) and on the plate above:
#:
#:     cells/wall   marching_cubes                 dual_contouring
#:        0.60      -75.5%, bbox 78.1x28.2x3.5     -11.9%, bbox exact
#:        0.72      --                             +0.0%,  bbox exact
#:        1.00       -8.6%, bbox 59.6x39.5x19.9    +0.0%,  bbox exact
#:        1.60       -0.9%, bbox exact             +0.0%,  bbox exact
#:        2.40       -0.6%, bbox exact             +0.0%,  bbox exact
#:
#: Below one cell the part COLLAPSES (the envelope itself is destroyed); above it the
#: error is ordinary and converges. So the floor is ONE CELL for the meshers that can
#: only interpolate a crossing on a cell EDGE (marching cubes, surface nets), and
#: 0.75 for dual contouring, which places one vertex per cell by a QEF over the
#: cell's Hermite data (Ju, Losasso, Schaefer & Warren 2002) and so can hold a
#: sub-cell wall a little further down before it too falls off its cliff.
#:
#: THIS THRESHOLD HAS ALREADY BEEN WRONG ONCE, IN THE OTHER DIRECTION. It was set to
#: 3.0 ("the 2-cell Nyquist floor, plus margin"), which REFUSED a 60x40x20 box
#: shelled to t=3 -- a part frep builds correctly at every resolution tested (22293
#: against an analytic 22296, bbox exact). That refusal broke the output gate's own
#: property corpus. A refusal of a part the engine can build is not a safety
#: improvement; it is a false positive with a typed error message, and false
#: positives are how a checker gets turned off. Refusing loudly is only a feature
#: when the thing refused is genuinely unbuildable.
MIN_WALL_CELLS: Dict[str, float] = {
    "marching_cubes": 1.0,
    "surface_nets": 1.0,
    "dual_contouring": 0.75,
}

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
      cyl      plane, cu, cv, r, w0, w1                (hole shaft / counterbore)
      cone     plane, cu, cv, r0, r1, w0, w1           (countersink)
      bool     op ('union'|'cut'|'intersect'), a, b, blend ('hard'|'smooth'|'chamfer'), k
      shell    child, thickness, faces, kind ('arc'|'intersection')
      mirror   child, plane
      pattern  child, transforms (list of rigid 3x4 matrices, row-major)

    THE TREE IS THE CONTRACT. freecad, openscad and blender do not re-implement the
    CISP op semantics: they COMPOSE an :class:`FRepBackend` and lower this tree
    (see :mod:`harnesscad.io.backends.external`). So a field this tree drops is a
    field FOUR engines drop, and no amount of cross-checking them can see it --
    they agree perfectly, and are all wrong together. Every op field that decides
    geometry must therefore survive into a node payload here. It is why ``cone``
    and ``shell.kind`` exist, and why ``transforms`` is a matrix.
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
    if t == "cone":
        return _eval_cone(node, p)
    if t == "revolve":
        return _eval_revolve(node, p)
    if t == "bool":
        a = eval_node(node.d["a"], p)
        b = eval_node(node.d["b"], p)
        return _boolean_field(node, a, b)
    if t == "shell":
        return _eval_shell(node, p)
    if t == "blendcut":
        return _eval_blendcut(node, p)
    if t == "mirror":
        return comb.union(eval_node(node.d["child"], p),
                          eval_node(node.d["child"], _reflect(p, node.d["plane"])))
    if t == "pattern":
        child = node.d["child"]
        vals = [eval_node(child, _untransform(p, tr)) for tr in node.d["transforms"]]
        return comb.union_all(vals)
    raise ValueError("unknown F-rep node kind '%s'" % t)  # pragma: no cover


# --------------------------------------------------------------------------
# shell
# --------------------------------------------------------------------------
#: The face names a Shell may open, as (axis, +1 for the max face / -1 for the
#: min face). Both the CAD words and the bare signed-axis spellings are taken.
#:
#: THESE ARE ALIASES, NOT THE GRAMMAR. The canonical way to name a face (or an
#: edge) anywhere in this repo is a **CadQuery selector string** --
#: :mod:`harnesscad.domain.geometry.topology.selector_dsl`. ``ops.Shell``
#: documents ``faces`` as ``(">Z",)``, cadquery/freecad/blender all route through
#: that grammar, and frep used to be the lone holdout: it bad-valued every shell
#: written exactly as the schema specifies it. It no longer does --
#: :func:`resolve_faces` parses a selector, and this table is kept only so the old
#: word forms ("top", "+z") keep working. One grammar, every engine.
SHELL_FACES: Dict[str, Tuple[int, int]] = {
    "right": (0, +1), "+x": (0, +1), "xmax": (0, +1),
    "left": (0, -1), "-x": (0, -1), "xmin": (0, -1),
    "back": (1, +1), "+y": (1, +1), "ymax": (1, +1),
    "front": (1, -1), "-y": (1, -1), "ymin": (1, -1),
    "top": (2, +1), "+z": (2, +1), "zmax": (2, +1),
    "bottom": (2, -1), "-z": (2, -1), "zmin": (2, -1),
}

#: The canonical spelling of each (axis, sign), for the node payload. Downstream
#: lowerings (openscad, freecad, blender) look faces up in SHELL_FACES, so the
#: node always carries a NAME even when the op carried a selector.
_FACE_NAME: Dict[Tuple[int, int], str] = {
    (0, +1): "+x", (0, -1): "-x",
    (1, +1): "+y", (1, -1): "-y",
    (2, +1): "+z", (2, -1): "-z",
}

_AXIS_UNIT: Tuple[Vec3, Vec3, Vec3] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

#: Refs that mean "the current body", not a named datum. ``AddInstance.part``
#: already took these three words, and the pressure corpus writes
#: ``Hole(face_or_sketch="solid")`` throughout, so they are the incumbent spelling
#: of "the target's default face" and must keep working. They are NOT selectors.
BODY_ALIASES: frozenset = frozenset({"solid", "body", "last"})


def face_entities(bounds: Tuple[Vec3, Vec3]) -> List[selector_dsl.Entity]:
    """The six faces of a solid's bounding box, as selector-DSL entities.

    An SDF carries no B-rep, so there is no face to hand the selector engine. What
    frep DOES have is the body's axis-aligned extent, and the six planar faces of
    that box are exactly the six faces a Shell can open on the prismatic parts this
    op is defined for. Each entity is placed at its face's centre and carries the
    face's outward normal, which is all ``>Z`` / ``<X`` / ``|Z`` / ``#Z`` need.
    """
    lo, hi = bounds
    mid = [0.5 * (lo[i] + hi[i]) for i in range(3)]
    out: List[selector_dsl.Entity] = []
    for axis in range(3):
        for sign in (+1, -1):
            center = list(mid)
            center[axis] = hi[axis] if sign > 0 else lo[axis]
            normal = list(_AXIS_UNIT[axis])
            if sign < 0:
                normal = [-c for c in normal]
            out.append(selector_dsl.Entity(
                center=(center[0], center[1], center[2]),
                axis=(normal[0], normal[1], normal[2]),
                geom_type="PLANE", name=_FACE_NAME[(axis, sign)]))
    return out


#: The 12 edges of an axis-aligned box, as (edge axis, axis p, sign p, axis q, sign
#: q): the edge runs along ``d`` and lies on the ``sp`` face of ``p`` and the ``sq``
#: face of ``q``.
_BOX_EDGES: Tuple[Tuple[int, int, int, int, int], ...] = tuple(
    (d, p, sp, q, sq)
    for d in range(3)
    for (p, q) in (((d + 1) % 3, (d + 2) % 3),)
    for sp in (+1, -1)
    for sq in (+1, -1)
)


def edge_entities(bounds: Tuple[Vec3, Vec3]) -> List[selector_dsl.Entity]:
    """The twelve edges of a solid's bounding box, as selector-DSL entities.

    The same move as :func:`face_entities`, one dimension down. An SDF has no B-rep
    edge, but a prism's edges ARE the edges of its extent, and a prism is what
    ``Fillet(edges=("|Z",))`` is written for. Each entity carries the edge's
    midpoint and its tangent, which is what ``|Z`` (parallel), ``#Z``
    (perpendicular), ``>Z`` / ``<X`` (direction extremum) all read.
    """
    lo, hi = bounds
    out: List[selector_dsl.Entity] = []
    for (d, p, sp, q, sq) in _BOX_EDGES:
        center = [0.0, 0.0, 0.0]
        center[d] = 0.5 * (lo[d] + hi[d])
        center[p] = hi[p] if sp > 0 else lo[p]
        center[q] = hi[q] if sq > 0 else lo[q]
        out.append(selector_dsl.Entity(
            center=(center[0], center[1], center[2]),
            axis=_AXIS_UNIT[d], geom_type="LINE",
            name="%d%+d%+d" % (d, sp * (p + 1), sq * (q + 1))))
    return out


def resolve_edges(bounds: Tuple[Vec3, Vec3], selectors: Sequence[str]):
    """Edge selectors -> the box-edge tuples they pick, deterministically.

    Raises :class:`selector_dsl.SelectorError` on a malformed selector and
    ``ValueError`` when a well-formed one picks nothing.
    """
    entities = edge_entities(bounds)
    index = {e.name: spec for e, spec in zip(entities, _BOX_EDGES)}
    picked: List[Tuple[int, int, int, int, int]] = []
    for raw in selectors or ():
        text = str(raw).strip()
        if not text:
            continue
        hits = selector_dsl.select(text, entities)
        if not hits:
            raise ValueError("edge selector %r selected no edge" % text)
        for e in hits:
            if index[e.name] not in picked:
                picked.append(index[e.name])
    return picked


def resolve_faces(bounds: Tuple[Vec3, Vec3], selectors: Sequence[str]) -> List[str]:
    """Face selectors -> canonical face names, deterministically.

    Each entry of ``selectors`` is EITHER a CadQuery selector string (the canonical
    grammar: ``">Z"``, ``"<X or >X"``, ``"not >Z"``) OR one of the legacy alias
    words in :data:`SHELL_FACES` (``"top"``, ``"+z"``). Selectors are evaluated
    against :func:`face_entities`, so ``">Z"`` picks the same face on frep that it
    picks on CadQuery, FreeCAD and Blender.

    Raises :class:`selector_dsl.SelectorError` on a malformed selector, and
    ``ValueError`` when a well-formed selector picks no face of the bounding box.
    """
    names: List[str] = []
    entities = face_entities(bounds)
    for raw in selectors or ():
        text = str(raw).strip()
        if not text:
            continue
        alias = SHELL_FACES.get(text.lower())
        if alias is not None:
            names.append(_FACE_NAME[alias])
            continue
        picked = selector_dsl.select(text, entities)
        if not picked:
            raise ValueError("face selector %r selected no face" % text)
        names.extend(e.name for e in picked)
    # sorted+deduped so the node payload (and therefore the digest) is stable
    return sorted(set(names))


def _eval_shell(node: Node, p: Sequence[float]) -> float:
    """A CAD Shell: hollow the child INWARD, optionally opening named faces.

    Closed shell.  ``shell_inward(f, t) = max(f, -(f + t))`` -- the material
    between the original surface ``{f = 0}`` and the inward offset ``{f = -t}``.
    The outer surface is *untouched*, so the bounding box is preserved exactly.

    This is NOT ``abs(f) - t``.  That operator is Inigo Quilez's ``opOnion``
    (https://iquilezles.org/articles/distfunctions/, "Onion / opOnion"), whose
    stated purpose is "carving interiors or giving thickness to primitives" --
    it is symmetric about the boundary and therefore keeps half the wall
    *outside* the original surface, growing the part.  Curv exposes the same
    two-sided operator as ``shell``.  Both are correct for what they are and
    wrong for a CAD Shell feature, which only ever REMOVES material: SolidWorks,
    Fusion and Onshape all leave the outer faces exactly where they were.  The
    two-sided version is still available as ``field_transforms.shell``; the CAD
    one is ``field_transforms.shell_inward``, which is what this node uses.

    Open faces.  ``faces`` names the faces to delete.  Deleting a face does not
    mean cutting the solid with a half-space -- that would saw the surrounding
    walls off too, and shorten the part.  It means punching the *cavity* out
    through that face.  So the cavity ``{f <= -t}`` is swept along the face's
    outward axis by clamping the domain along that axis (the same domain-clamp
    trick as iq's Elongation operator, ``q = p - clamp(p, -h, h)``): past the
    cavity's mid-plane the cavity is evaluated *at* its mid-plane, which extrudes
    its cross-section there out through the wall and off to infinity.  The swept
    cavity is unioned with the plain one and the whole thing is subtracted from
    the solid, so the side walls keep their full height and only the named face's
    wall is removed.

    Caveats, stated rather than hidden.  (a) The swept part of the opening field
    is a distance *bound*, not the exact distance -- it ignores the sweep axis.
    That is fine here: it is only ever fed to ``max``/``min`` and to the mesher,
    both of which need the sign and the zero set, and it under-estimates, which
    is the safe direction.  (b) The cross-section taken is the one at the
    cavity's mid-plane, which is the right one for a prismatic part (the case a
    CAD Shell with open faces is defined for) and merely a defensible choice for
    a body that necks in and out along that axis.
    """
    child = node.d["child"]
    t = float(node.d["thickness"])
    f = eval_node(child, p)
    faces = node.d.get("faces") or ()
    if not faces:
        return xf.shell_inward(f, t)

    # the cavity: the inward offset of the child by t (negative inside it)
    cavity = f + t
    lo, hi = node_bounds(child)
    for name in faces:
        axis, _sign = SHELL_FACES[str(name).strip().lower()]
        mid = 0.5 * (lo[axis] + hi[axis])
        q = list(p)
        # Clamp the sweep axis onto the cavity's mid-plane on the OUTBOARD side
        # of it, so the cross-section there is extruded out through the face.
        if _sign > 0:
            if q[axis] > mid:
                q[axis] = mid
        else:
            if q[axis] < mid:
                q[axis] = mid
        cavity = comb.union(cavity, eval_node(child, q) + t)
    # solid minus the (opened) cavity
    return comb.difference(f, cavity)


def _has_shell(node: Node) -> bool:
    """Does this tree contain a node the arithmetic IR cannot encode?

    (See :meth:`FRepBackend.ir`.) ``shell`` is one because the IR still encodes the
    two-sided onion; ``blendcut`` is one because the IR has no arm for it at all,
    and an IR that disagrees with the field is worse than no IR -- interval pruning
    would bound the WRONG function and delete blocks the real surface passes
    through.
    """
    if node.t in ("shell", "blendcut", "blend"):
        return True
    for key in ("child", "a", "b"):
        kid = node.d.get(key)
        if isinstance(kid, Node) and _has_shell(kid):
            return True
    return False


def _eval_blendcut(node: Node, p: Sequence[float]) -> float:
    """A fillet / chamfer applied to NAMED EDGES, as an SDF difference.

    WHY THIS EXISTS. :func:`blend_tree` rounds EVERY convex edge of the body: it is
    a uniform operator, and it is the only blend an SDF gets for free. So
    ``Fillet(edges=("|Z",))`` -- four vertical edges -- rounded all twelve, returned
    a watertight, manifold, perfectly valid solid, and emitted no diagnostic. A
    20x10x5 box filleted r=1 on its 4 vertical edges has volume 995.708; the same
    box filleted on all 12 has 971.295. Two different parts, one reported "ok". The
    edge selector was accepted and thrown away.

    HOW. For a convex edge formed by two planar faces, let ``a`` and ``b`` be the
    depths into those two faces (positive inside the solid). The material a fillet
    of radius ``r`` removes is exactly

        { a >= 0, b >= 0, a <= r, b <= r, hypot(r - a, r - b) >= r }

    -- the square corner of side ``r`` minus the quarter-disc of the fillet arc --
    and a chamfer of setbacks ``(d1, d2)`` removes

        { a >= 0, b >= 0, a / d1 + b / d2 <= 1 }.

    Both are intersections of half-spaces and (for the fillet) the outside of a
    cylinder, so both are exact SDFs built with ``max``. Each is clamped to its own
    edge's extent, so it touches nothing else, and the tools are unioned and
    subtracted. On a prism -- which is what an edge selector is written for -- the
    result is the exact filleted solid, and it is LOCAL: an unselected edge is not
    touched at all, which is the whole point.

    The edges themselves come from the body's bounding box
    (:func:`edge_entities`), which is the same modelling assumption frep's shell
    already makes about faces, applied consistently one dimension down.
    """
    child = node.d["child"]
    f = eval_node(child, p)
    r = float(node.d["r"])
    r2 = float(node.d.get("r2") or r)
    is_fillet = node.d["kind"] == "fillet"
    cut = _INF
    for (d, lo_d, hi_d, ip, sp, cp, iq, sq, cq) in node.d["edges"]:
        a = sp * (cp - p[ip])
        b = sq * (cq - p[iq])
        parts = [-a, -b, _slab(p[d], lo_d, hi_d)]
        if is_fillet:
            parts.append(a - r)
            parts.append(b - r)
            parts.append(r - math.hypot(r - a, r - b))
        else:
            parts.append((a / r + b / r2 - 1.0) / math.hypot(1.0 / r, 1.0 / r2))
        cut = min(cut, max(parts))
    return comb.difference(f, cut)


def _eval_cone(node: Node, p: Sequence[float]) -> float:
    """A truncated cone about a sketch-plane normal: radius ``r0`` at ``w0``,
    ``r1`` at ``w1``.

    This is what a COUNTERSINK is. It exists because the F-rep tree used to have
    no way to say "cone", so ``Hole(kind='countersink')`` lowered to the same bare
    cylinder as ``Hole(kind='simple')`` -- three named manufacturing intents, one
    geometry, no diagnostic.

    The lateral distance is the radial excess ``q - r(w)`` scaled by ``cos(alpha)``
    (``alpha`` = the cone's half-angle), which is the exact perpendicular distance
    to the lateral surface; the caps are the slab. It is intersected exactly, so
    the field is a true SDF on the lateral face and the caps and a safe (never
    over-estimating) bound at the two rims -- which is all the mesher and the
    interval bound need.
    """
    iu, iv, iw = _plane_axes(node.d["plane"])
    w0, w1 = float(node.d["w0"]), float(node.d["w1"])
    r0, r1 = float(node.d["r0"]), float(node.d["r1"])
    if w1 < w0:
        w0, w1, r0, r1 = w1, w0, r1, r0
    span = w1 - w0
    q = math.hypot(p[iu] - node.d["cu"], p[iv] - node.d["cv"])
    w = p[iw]
    tau = 0.0 if span <= 0.0 else (w - w0) / span
    radius = r0 + (r1 - r0) * tau
    slope = 0.0 if span <= 0.0 else (r1 - r0) / span
    lateral = (q - radius) / math.hypot(1.0, slope)
    dw = _slab(w, w0, w1)
    inside = min(max(lateral, dw), 0.0)
    outside = math.hypot(max(lateral, 0.0), max(dw, 0.0))
    return inside + outside


def _reflect(p: Sequence[float], plane: str) -> Vec3:
    """Reflect a point across a named datum plane (the plane's normal flips)."""
    pl = str(plane).upper()
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    if pl == "XY":
        return (x, y, -z)
    if pl == "YZ":
        return (-x, y, z)
    return (x, -y, z)  # XZ


#: A pattern instance transform is a RIGID 3x4 matrix, row-major:
#: ``(r00 r01 r02 tx  r10 r11 r12 ty  r20 r21 r22 tz)``.
#:
#: It used to be ``(dx, dy, dz, angle_about_Z)``. That 4-tuple could not express a
#: rotation about any axis but Z, so ``CircularPattern.axis`` was READ, normalised,
#: and then thrown away -- every circular pattern spun about Z whatever axis it was
#: given. A rigid matrix can say what the op says, and every lowering (OpenSCAD's
#: ``multmatrix``, FreeCAD's ``Base.Matrix``, Blender's vertex transform) consumes
#: one natively.
IDENTITY_XFORM: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0, 0.0,
                                     0.0, 0.0, 1.0, 0.0)


def rigid_transform(origin: Sequence[float], axis: Sequence[float],
                    angle_deg: float, translate: Sequence[float] = (0.0, 0.0, 0.0)
                    ) -> Tuple[float, ...]:
    """Rotate ``angle_deg`` about the line (``origin``, ``axis``), then translate.

    Rodrigues' rotation formula. ``axis`` need not be unit; a zero-length axis with
    a non-zero angle is a caller error and raises.
    """
    ax, ay, az = (float(axis[0]), float(axis[1]), float(axis[2]))
    n = math.sqrt(ax * ax + ay * ay + az * az)
    ang = float(angle_deg)
    if n == 0.0:
        if ang % 360.0 != 0.0:
            raise ValueError("rotation axis is degenerate (zero length)")
        r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    else:
        ux, uy, uz = ax / n, ay / n, az / n
        a = math.radians(ang)
        c, s = math.cos(a), math.sin(a)
        k = 1.0 - c
        r = [
            [c + ux * ux * k, ux * uy * k - uz * s, ux * uz * k + uy * s],
            [uy * ux * k + uz * s, c + uy * uy * k, uy * uz * k - ux * s],
            [uz * ux * k - uy * s, uz * uy * k + ux * s, c + uz * uz * k],
        ]
    o = (float(origin[0]), float(origin[1]), float(origin[2]))
    # p -> R (p - o) + o + translate
    t = [o[i] + float(translate[i])
         - (r[i][0] * o[0] + r[i][1] * o[1] + r[i][2] * o[2])
         for i in range(3)]
    return (r[0][0], r[0][1], r[0][2], t[0],
            r[1][0], r[1][1], r[1][2], t[1],
            r[2][0], r[2][1], r[2][2], t[2])


def _apply_transform(p: Sequence[float], tr: Sequence[float]) -> Vec3:
    """Forward: the world point a pattern instance places ``p`` at."""
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    return (tr[0] * x + tr[1] * y + tr[2] * z + tr[3],
            tr[4] * x + tr[5] * y + tr[6] * z + tr[7],
            tr[8] * x + tr[9] * y + tr[10] * z + tr[11])


def _untransform(p: Sequence[float], tr: Sequence[float]) -> Vec3:
    """Inverse of a pattern instance transform.

    The rotation block is orthonormal by construction (:func:`rigid_transform`), so
    the inverse is the transpose -- no general 3x3 solve, and no accumulated error.
    """
    x = float(p[0]) - tr[3]
    y = float(p[1]) - tr[7]
    z = float(p[2]) - tr[11]
    return (tr[0] * x + tr[4] * y + tr[8] * z,
            tr[1] * x + tr[5] * y + tr[9] * z,
            tr[2] * x + tr[6] * y + tr[10] * z)


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
    if t in ("extrude", "cyl", "cone", "revolve"):
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
        # An inward shell never leaves the child's bounds.
        return node_bounds(node.d["child"])
    if t in ("blendcut", "blend"):
        # A blend REMOVES material; it never leaves the child's bounds.
        return node_bounds(node.d["child"])
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
    elif node.t == "cone":
        cu, cv = node.d["cu"], node.d["cv"]
        rr = max(float(node.d["r0"]), float(node.d["r1"]))
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


# -- fillet / chamfer rewriting -------------------------------------------
def blend_tree(node: Node, kind: str, k: float) -> Node:
    """Rebuild ``node`` with every boolean blended and every leaf rounded.

    This is the SDF analogue of a fillet/chamfer feature: there are no B-rep
    edges to name, so the radius is applied uniformly to the convex edges of the
    leaves and to the (concave and convex) edges introduced by the booleans. This
    is exactly why frep REFUSES a non-empty ``edges`` selector -- see
    :meth:`FRepBackend._blend`.
    """
    t = node.t
    if t in ("extrude", "revolve"):
        d = dict(node.d)
        d["round"] = k if kind == "smooth" else 0.0
        d["cham"] = k if kind == "chamfer" else 0.0
        return Node(t, **d)
    if t in ("cyl", "cone"):
        return node
    if t == "bool":
        return Node("bool", op=node.d["op"],
                    a=blend_tree(node.d["a"], kind, k),
                    b=blend_tree(node.d["b"], kind, k),
                    blend=kind, k=k)
    if t == "shell":
        return Node("shell", child=blend_tree(node.d["child"], kind, k),
                    thickness=node.d["thickness"],
                    faces=node.d.get("faces", ()),
                    kind=node.d.get("kind", "arc"))
    if t == "blendcut":
        return Node("blendcut", child=blend_tree(node.d["child"], kind, k),
                    kind=node.d["kind"], r=node.d["r"], r2=node.d.get("r2"),
                    edges=node.d["edges"])
    if t == "blend":
        return Node("blend", child=blend_tree(node.d["child"], kind, k),
                    kind=node.d["kind"], value=node.d["value"],
                    value2=node.d.get("value2"),
                    selectors=node.d.get("selectors", ()))
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
    if algorithm == "dual_contouring":
        # Ju, Losasso, Schaefer & Warren 2002. One vertex per cell, placed by a
        # QEF over the cell's Hermite data, so a corner where three faces meet
        # is reproduced exactly instead of being chamfered off by up to half a
        # cell the way marching cubes must.
        verts, faces = dual_contour(grid, field, 0.0)
    elif algorithm == "surface_nets":
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
# stepped holes
# --------------------------------------------------------------------------
def countersink_depth(diameter: float, csk_diameter: float, csk_angle: float) -> float:
    """Axial depth of a countersink cone.

    ``csk_angle`` is the FULL included angle (CadQuery's
    ``cskHole(diameter, cskDiameter, cskAngle)`` convention, and the convention the
    82 deg / 90 deg fastener standards are quoted in), so the cone's half-angle is
    ``csk_angle / 2`` and the depth over which the bore opens from ``diameter`` to
    ``csk_diameter`` is the adjacent side of that half-angle.
    """
    return (float(csk_diameter) - float(diameter)) / 2.0 / math.tan(
        math.radians(float(csk_angle) / 2.0))


def check_stepped_hole(op: Hole):
    """``None`` if a counterbore/countersink is fully specified, else an err triple.

    An underspecified stepped hole is REFUSED. It is tempting to fall back on a
    "conventional ratio" (1.5x the bore, say) and cut something -- FreeCAD's backend
    did, which is how a counterbore, a countersink and a plain hole all became the
    same cylinder. Inventing a dimension the caller did not give is not a default,
    it is a fabrication, and it is invisible in the output.
    """
    kind = str(op.kind)
    if kind == "counterbore":
        if op.cbore_diameter is None or op.cbore_depth is None:
            return ("bad-value",
                    "hole kind 'counterbore' needs cbore_diameter and cbore_depth; "
                    "no ratio will be substituted to quietly cut a plain hole", None)
        if float(op.cbore_diameter) <= float(op.diameter):
            return ("bad-value",
                    "cbore_diameter (%g) must exceed the hole diameter (%g)"
                    % (op.cbore_diameter, op.diameter), None)
        if float(op.cbore_depth) <= 0.0:
            return ("bad-value", "cbore_depth must be > 0", None)
        if not op.through and op.depth is not None \
                and float(op.cbore_depth) >= float(op.depth):
            return ("bad-value",
                    "cbore_depth (%g) must be less than the blind hole depth (%g)"
                    % (op.cbore_depth, op.depth), None)
    elif kind == "countersink":
        if op.csk_diameter is None:
            return ("bad-value",
                    "hole kind 'countersink' needs csk_diameter; no ratio will be "
                    "substituted to quietly cut a plain hole", None)
        if float(op.csk_diameter) <= float(op.diameter):
            return ("bad-value",
                    "csk_diameter (%g) must exceed the hole diameter (%g)"
                    % (op.csk_diameter, op.diameter), None)
        if not 0.0 < float(op.csk_angle) < 180.0:
            return ("bad-value",
                    "csk_angle (%g) must be a full included angle in (0, 180)"
                    % op.csk_angle, None)
    return None


# --------------------------------------------------------------------------
# constraints
# --------------------------------------------------------------------------
#: Constraint kinds that take NO second entity. A radius is a property of ONE
#: circle; "horizontal" is a property of ONE line. ``Constrain(kind="radius",
#: a="e1", b="e2")`` names a second entity the constraint has nothing to do with,
#: and there is no geometry it could denote -- so it is refused, which is also what
#: makes ``Constrain.b`` a field the backend demonstrably READS.
#:
#: THIS LIST IS DELIBERATELY SHORT, and it used to be a full arity table that also
#: REQUIRED a second entity for every binary kind. That was a false positive: the
#: stock generators (data/datagen/generators.py) emit
#: ``Constrain(kind="distance", a="e1", value=w)`` -- a dimensional constraint on a
#: single entity -- and the table refused every one of them. A stricter validator
#: that rejects legal, in-use input is not a safety improvement; it is a new bug
#: with a typed error message. Only the case that CANNOT mean anything is refused.
CONSTRAINT_NO_SECOND_ENTITY: frozenset = frozenset({"radius", "horizontal", "vertical"})

#: Kinds that carry a dimension and are meaningless without it.
CONSTRAINT_NEEDS_VALUE: frozenset = frozenset({"distance", "radius"})


def check_constraint_arity(op: Constrain):
    """``None`` if the op is well-formed, else an ``_err`` triple."""
    if op.kind not in CONSTRAINT_DOF:
        return ("bad-value", f"unknown constraint kind '{op.kind}'", None)
    if op.kind in CONSTRAINT_NEEDS_VALUE and op.value is None:
        return ("bad-value", f"'{op.kind}' constraint requires a value", None)
    if op.kind in CONSTRAINT_NO_SECOND_ENTITY and op.b:
        return ("bad-value",
                f"'{op.kind}' constrains a single entity: it has no second entity, "
                f"but b='{op.b}' was given", op.b)
    return None


def solve_constraint(op: Constrain, entities: dict) -> None:
    """Apply a constraint to the sketch entities, where the assignment is determined.

    A SMALL SOLVER, AND AN HONEST ONE. FreeCAD drives the real thing (planegcs);
    every other engine merely subtracted a number from a DOF counter, so
    ``Constrain(kind="radius", a="e1", value=8.0)`` on an r=6 circle left it at 6 --
    the constraint was "applied", the sketch reported one fewer degree of freedom,
    and the part came out the wrong size. Here a constraint whose resolution is a
    determined assignment (a radius on a circle; an equality between two like
    entities; a coincidence; a line swung onto an axis) is RESOLVED onto the
    entities, so ``value`` reaches the geometry.

    THIS FUNCTION NEVER REFUSES. Where the assignment is NOT determined -- a
    "horizontal" on a rectangle, a distance from one entity to nothing, two entities
    already coincident -- it leaves the geometry alone and the constraint stands as
    DOF bookkeeping, exactly as it always has. That is deliberate: those forms are
    emitted by the stock generators (``data/datagen/generators.py`` constrains a
    rectangle "horizontal") and by the corpus, and an earlier version of this solver
    rejected them, which broke every generator stream. A stricter validator that
    rejects legal, in-use input is not a safety improvement -- it is a new bug with a
    typed error message. The only refusal lives in :func:`check_constraint_arity`,
    and it covers the one case that cannot denote any geometry at all.

    ``a`` is the driving entity and is never moved; ``b`` is the driven one.
    """
    a = entities[op.a]
    b = entities[op.b] if op.b else None
    kind = op.kind
    pa = a["params"]
    pb = b["params"] if b else None

    if kind == "radius":
        if a["type"] == "circle" and float(op.value) > 0.0:
            pa["r"] = float(op.value)
        return None

    if kind == "equal":
        if b is not None and a["type"] == b["type"]:
            for key in ("r", "w", "h"):
                if key in pa and key in pb:
                    pb[key] = pa[key]
        return None

    if kind == "coincident":
        if b is None:
            return None
        ca, cb = _anchor(a), _anchor(b)
        if ca is not None and cb is not None:
            _translate(b, ca[0] - cb[0], ca[1] - cb[1])
        return None

    if kind == "distance":
        # A distance from ONE entity is a distance to WHAT? The op does not say, and
        # this solver will not invent a datum. In use (the generators emit it), so
        # accepted; it moves nothing.
        if b is None:
            return None
        ca, cb = _anchor(a), _anchor(b)
        if ca is None or cb is None:
            return None
        dx, dy = cb[0] - ca[0], cb[1] - ca[1]
        norm = math.hypot(dx, dy)
        if norm == 0.0:
            return None            # coincident: no determined direction to separate
        scale = float(op.value) / norm
        _translate(b, ca[0] + dx * scale - cb[0], ca[1] + dy * scale - cb[1])
        return None

    # horizontal / vertical / parallel / perpendicular act on LINE directions.
    _solve_angular(kind, a, b)
    return None


def _anchor(ent: dict):
    """The (x, y) a constraint moves an entity BY -- its centre / corner / start."""
    p, kind = ent["params"], ent["type"]
    if kind == "circle":
        return (p["cx"], p["cy"])
    if kind in ("rectangle", "point"):
        return (p["x"], p["y"])
    if kind == "line":
        return (p["x1"], p["y1"])
    return None


def _translate(ent: dict, dx: float, dy: float) -> None:
    p, kind = ent["params"], ent["type"]
    if kind == "circle":
        p["cx"] += dx
        p["cy"] += dy
    elif kind in ("rectangle", "point"):
        p["x"] += dx
        p["y"] += dy
    elif kind == "line":
        p["x1"] += dx
        p["y1"] += dy
        p["x2"] += dx
        p["y2"] += dy


def _solve_angular(kind: str, a: dict, b: Optional[dict]) -> None:
    """horizontal / vertical / parallel / perpendicular -- all swing a LINE.

    The driven line keeps its start point and its length and is rotated onto the
    required direction, which is what a real solver converges to from a nearby
    start. Anything that is not a line (a rectangle constrained "horizontal", which
    the stock generators emit) is left alone: it is bookkeeping, not a refusal.
    """
    driven = b if b is not None else a
    if driven["type"] != "line":
        return
    if kind == "horizontal":
        target = (1.0, 0.0)
    elif kind == "vertical":
        target = (0.0, 1.0)
    else:
        if a["type"] != "line":
            return
        pa = a["params"]
        dx, dy = pa["x2"] - pa["x1"], pa["y2"] - pa["y1"]
        n = math.hypot(dx, dy)
        if n == 0.0:
            return
        dx, dy = dx / n, dy / n
        target = (dx, dy) if kind == "parallel" else (-dy, dx)
    p = driven["params"]
    length = math.hypot(p["x2"] - p["x1"], p["y2"] - p["y1"])
    if length == 0.0:
        return
    p["x2"] = p["x1"] + target[0] * length
    p["y2"] = p["y1"] + target[1] * length


# --------------------------------------------------------------------------
# the backend
# --------------------------------------------------------------------------
class FRepBackend:
    """A GeometryBackend that realises CISP ops as signed-distance fields."""

    #: exports this backend can produce
    FORMATS = ("stl", "stl-ascii", "stl-binary", "glb", "sdf")

    #: Shell join kinds THIS op-state model's own field can evaluate.
    #:
    #: ``kind`` is the join type of the offset surface. A signed-distance field's
    #: inward offset is, by construction, the erosion by a BALL -- which is the
    #: ``"arc"`` join and nothing else. The ``"intersection"`` join extends the
    #: offset faces to meet at a sharp corner; that is an algebraic intersection of
    #: half-spaces, and a scalar distance field does not carry the half-spaces to
    #: intersect. So frep implements ``arc`` and REFUSES ``intersection``, rather
    #: than accepting the field and quietly building the arc join anyway.
    #:
    #: :class:`~harnesscad.io.backends.external.ExternalToolBackend` subclasses
    #: whose kernel HAS both joins (openscad: ``offset(r=)`` vs ``offset(delta=)``;
    #: freecad: OCCT's ``join`` argument) widen this on the composed instance.
    SHELL_JOINS: Tuple[str, ...] = ("arc",)

    #: Override for the wall floor. ``None`` derives it from the active mesher
    #: (:data:`MIN_WALL_CELLS`), which is the honest default because the floor IS a
    #: property of the extraction. ``0.0`` disables the check, which is what the
    #: external backends do: their kernels are exact and sample no grid at all.
    SHELL_MIN_WALL_CELLS: Optional[float] = None

    #: Does the CONSUMER of this op-state model select edges itself?
    #:
    #: False (frep proper): a named ``Fillet.edges`` is realised here, as a per-edge
    #: cutter over the body's bounding-box edges (:meth:`_blend`).
    #: True (freecad, blender): the external kernel has real B-rep edges and applies
    #: the blend to them from the op log, which is strictly better than a
    #: bounding-box approximation -- so this model must NOT also cut them, or the
    #: blend would be applied twice.
    EDGE_SELECTORS: bool = False

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
        #: fid -> (body id, the body's node AS OF that feature). A pattern or a
        #: mirror names a FEATURE, and "the feature" means the state of the body
        #: when that feature was made -- patterning the pad is not patterning the
        #: pad-plus-fillet. Nodes are replaced functionally, never mutated in
        #: place, so holding the reference IS holding the snapshot.
        self._snapshots: Dict[str, Tuple[str, Node]] = {}
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
            if op.distance2 is not None and op.distance2 <= 0:
                return _err("bad-value",
                            f"chamfer distance2 must be > 0 (got {op.distance2})")
            return self._blend("chamfer", "chamfer", op.distance, op.edges,
                               k2=op.distance2)
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
        err = check_constraint_arity(op)
        if err is not None:
            return _err(*err)
        if op.a not in self.entities:
            return _err("bad-ref", f"unknown entity '{op.a}'", op.a)
        if op.b is not None and op.b not in self.entities:
            return _err("bad-ref", f"unknown entity '{op.b}'", op.b)
        solve_constraint(op, self.entities)
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
        self._snapshots[fid] = (fid, node)
        self.solid_present = True
        return ApplyResult(True, [fid])

    def _record_feature(self, feature: str, body: dict, **extra) -> str:
        """Register a feature that MODIFIED ``body`` (rather than creating one).

        The snapshot it records is the body as it stands AFTER this feature, which
        is what ``LinearPattern(feature=...)`` will replicate if it names it.
        """
        fid = self._new_id("f")
        self.features.append(dict({"type": feature, "id": fid}, **extra))
        self._snapshots[fid] = (body["id"], body["node"])
        return fid

    def _feature_target(self, ref: str):
        """``(body, node)`` a pattern/mirror ``feature`` reference names.

        THE BUG THIS KILLS. Every pattern and every mirror used to operate on
        ``self._bodies[-1]`` -- the LAST solid -- whatever feature it was handed.
        In any model with more than one body, or with more than one feature on a
        body, that is a wrong part, silently, every time. ``feature`` was validated
        (an unknown id was refused) and then never used, which is the most
        misleading shape this bug can take: the reference is real, so it looks wired.

        An EMPTY ref keeps the old meaning -- the last body, as it stands now -- so
        op streams that never named a feature are unchanged.
        """
        if not self._bodies:
            return None, _err("no-solid", "this op requires an existing solid")
        if not ref:
            body = self._bodies[-1]
            return (body, body["node"]), None
        snap = self._snapshots.get(ref)
        if snap is None:
            return None, _err("bad-ref", f"unknown feature '{ref}'", ref)
        body_id, node = snap
        for body in self._bodies:
            if body["id"] == body_id:
                return (body, node), None
        return None, _err("bad-ref",
                          f"feature '{ref}' belonged to body '{body_id}', which a "
                          f"later boolean consumed", ref)

    @staticmethod
    def _graft(body: dict, node: Node, built: Node) -> None:
        """Put ``built`` (a replicated/mirrored snapshot) back onto ``body``.

        When the named feature IS the body's current state, ``built`` already
        contains it (every pattern carries the identity instance, every mirror
        unions the original), so it replaces the body outright -- which is exactly
        what the old code did, so a stream that named the last feature is
        bit-identical to before. When an EARLIER feature was named, the body's
        later features must survive, so the replicas are unioned onto it.
        """
        if node is body["node"]:
            body["node"] = built
        else:
            body["node"] = Node("bool", op="union", a=body["node"], b=built,
                                blend="hard", k=0.0)

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

    def _blend(self, feature: str, kind: str, k: float, edges,
               k2: Optional[float] = None) -> ApplyResult:
        """Fillet / chamfer, on the edges the op NAMED.

        ``edges`` is a tuple of CadQuery selector strings. It used to be recorded on
        the feature and then dropped: :func:`blend_tree` rounds every convex edge of
        the body, so a fillet of the four vertical edges rounded all twelve and
        returned a valid, watertight, WRONG solid with no diagnostic.

        * EMPTY tuple -- "every edge". That IS the uniform :func:`blend_tree`
          rewrite, so it is honoured exactly as before and existing op streams are
          bit-identical.
        * NON-EMPTY -- the selector is evaluated against the body's bounding-box
          edges and each named edge gets its own exact cutter
          (:func:`_eval_blendcut`). Only the named edges move.

        When this op-state model is driving an EXTERNAL kernel (freecad, blender),
        that kernel has REAL B-rep edges and applies the blend itself from the op
        log, against its own topology -- which is strictly better than a bounding-box
        approximation. So it keeps the uniform tree rewrite (which is only read back
        as a radius) and does its own selection. ``EDGE_SELECTORS`` is how a backend
        says which of the two it is.
        """
        if not self._bodies:
            return _err("no-solid", f"{feature} requires an existing solid")
        sels = tuple(str(e) for e in (edges or ()) if str(e).strip())
        body = self._bodies[-1]
        if sels and not self.EDGE_SELECTORS:
            try:
                picked = resolve_edges(node_bounds(body["node"]), sels)
            except selector_dsl.SelectorError as exc:
                return _err("bad-value",
                            f"{feature} edge selector is malformed: {exc}")
            except ValueError as exc:
                return _err("bad-value", str(exc))
            spec = self._edge_specs(body["node"], picked)
            body["node"] = Node("blendcut", child=body["node"],
                                kind=("fillet" if kind == "smooth" else "chamfer"),
                                r=float(k),
                                r2=(None if k2 is None else float(k2)),
                                edges=spec)
        elif self.EDGE_SELECTORS:
            # The blend becomes a NODE, at its position in the feature history.
            #
            # freecad and blender used to collect every Fillet/Chamfer from the op
            # log and apply them to the FINISHED root, after the whole tree was
            # built. That is why ``LinearPattern(feature="f1")`` and
            # ``LinearPattern(feature="f2")`` came out identical on both engines
            # even once frep resolved the reference correctly: whichever feature was
            # named, the fillet was applied last, to everything. Patterning the pad
            # and patterning the pad-plus-fillet are different parts, and the tree
            # has to be able to say which -- so the blend sits where it happened.
            body["node"] = Node("blend", child=body["node"],
                                kind=("fillet" if kind == "smooth" else "chamfer"),
                                value=float(k),
                                value2=(None if k2 is None else float(k2)),
                                selectors=tuple(str(e) for e in (edges or ())))
        else:
            body["node"] = blend_tree(body["node"], kind, float(k))
        self._record_feature(feature, body, edges=list(edges), value=float(k))
        return ApplyResult(True, [self.features[-1]["id"]])

    @staticmethod
    def _edge_specs(node: Node, picked) -> List[Tuple]:
        """Freeze each selected box edge into the numbers :func:`_eval_blendcut` needs."""
        lo, hi = node_bounds(node)
        out: List[Tuple] = []
        for (d, ip, sp, iq, sq) in picked:
            out.append((d, lo[d], hi[d],
                        ip, sp, (hi[ip] if sp > 0 else lo[ip]),
                        iq, sq, (hi[iq] if sq > 0 else lo[iq])))
        return out

    # -- hole --------------------------------------------------------------
    def _hole_frame(self, op: Hole):
        """``(plane, sign, entry_w, span, err)`` -- WHERE the bore starts and which
        way it goes.

        ``face_or_sketch`` is one of three things, and all three are now read:

        * ``""``          the target's top face (+Z). The historical default.
        * ``"skN"``       a sketch: its plane gives the bore axis (unchanged).
        * a **selector**  ``"<Z"``, ``">X"``, ``"-Y"`` -- the canonical grammar. It
          is evaluated against the body's bounding-box faces, and the bore enters
          THROUGH THAT FACE, normal to it. This is the field that made
          ``Hole(face_or_sketch="<Z")`` drill from the top anyway: the ref was
          stored on the feature record and then ignored by the geometry.
        """
        if not self._bodies:
            return None, _err("no-solid", "hole requires an existing solid")
        target = self._bodies[-1]
        bounds = node_bounds(target["node"])
        ref = str(op.face_or_sketch or "").strip()
        plane, sign = "XY", +1
        if ref.lower() in BODY_ALIASES or ref in self._feature_ids():
            # A BODY, not a datum. "solid" / "body" / "last" (the same three words
            # AddInstance.part takes) and a bare feature id like "f1" all name the
            # thing being drilled, not the face to drill through -- and they are the
            # INCUMBENT spelling: every reference op stream in the pressure corpus
            # (eval/pressure/briefs.py) writes Hole(face_or_sketch="solid"), and the
            # gate's property test writes "f1". They mean the same as an empty ref
            # (the target's default face) and they are NOT selectors.
            #
            # Parsing them as selectors -- and bad-valuing them when they failed to
            # parse -- broke every hole in the corpus. A stricter validator that
            # rejects legal, in-use input is not a safety improvement; it is a new
            # bug with a typed error message.
            ref = ""
        if ref.startswith("sk"):
            if ref not in self.sketches:
                return None, _err("bad-ref", f"unknown sketch '{ref}'", ref)
            plane = self.sketches[ref]["plane"]
        elif ref:
            try:
                names = resolve_faces(bounds, (ref,))
            except selector_dsl.SelectorError as exc:
                return None, _err("bad-value",
                                  f"hole face selector {ref!r} is malformed: {exc}")
            except ValueError as exc:
                return None, _err("bad-value", str(exc))
            if len(names) != 1:
                return None, _err(
                    "bad-value",
                    "hole face selector %r picked %d faces (%s); a hole enters "
                    "through exactly one" % (ref, len(names), ", ".join(names)))
            axis, sign = SHELL_FACES[names[0]]
            plane = {0: "YZ", 1: "XZ", 2: "XY"}[axis]
        lo, hi = bounds
        _, _, iw = _plane_axes(plane)
        entry = hi[iw] if sign > 0 else lo[iw]
        return (target, plane, sign, entry, hi[iw] - lo[iw], lo, hi), None

    def _hole(self, op: Hole) -> ApplyResult:
        """Cut a hole -- and cut the hole that was ASKED FOR.

        ``kind`` / ``cbore_*`` / ``csk_*`` used to be validated, recorded on the
        feature, and then dropped: a counterbore, a countersink and a plain hole all
        lowered to the SAME bare cylinder, on frep and on the three backends that
        lower this tree. Three distinct manufacturing intents, one geometry, no
        diagnostic.

        The tool is now built in full: the shaft, plus a flat-bottomed enlarged bore
        (``cyl``) for a counterbore, plus a truncated cone (``cone``) for a
        countersink -- both exactly expressible as SDF primitives, so there was
        never an engine on which this was impossible.

        An underspecified stepped hole is REFUSED, not silently degraded to a
        cylinder by inventing a "conventional ratio" behind the caller's back.
        """
        if op.diameter <= 0:
            return _err("bad-value", f"hole diameter must be > 0 (got {op.diameter})")
        if not op.through and (op.depth is None or op.depth <= 0):
            return _err("bad-value", "blind hole requires depth > 0")
        if op.kind not in ("simple", "counterbore", "countersink"):
            return _err("bad-value", f"unknown hole kind '{op.kind}'")
        err = check_stepped_hole(op)
        if err is not None:
            return _err(*err)
        frame, bad = self._hole_frame(op)
        if bad is not None:
            return bad
        target, plane, sign, entry, span, lo, hi = frame
        iu, iv, iw = _plane_axes(plane)
        cu, cv = float(op.x), float(op.y)
        clear = span + 1.0

        # the shaft: from the entry face, inward (-sign), through or to `depth`
        if op.through:
            w_far = (lo[iw] - clear) if sign > 0 else (hi[iw] + clear)
        else:
            w_far = entry - sign * float(op.depth)
        tool = Node("cyl", plane=plane, cu=cu, cv=cv,
                    r=float(op.diameter) / 2.0,
                    w0=w_far, w1=entry + sign * clear)

        if op.kind == "counterbore":
            depth = float(op.cbore_depth)
            tool = Node("bool", op="union", blend="hard", k=0.0, a=tool,
                        b=Node("cyl", plane=plane, cu=cu, cv=cv,
                               r=float(op.cbore_diameter) / 2.0,
                               w0=entry - sign * depth,
                               w1=entry + sign * clear))
        elif op.kind == "countersink":
            r_shaft = float(op.diameter) / 2.0
            r_csk = float(op.csk_diameter) / 2.0
            h = countersink_depth(op.diameter, op.csk_diameter, op.csk_angle)
            cone = Node("cone", plane=plane, cu=cu, cv=cv,
                        r0=r_shaft, r1=r_csk,
                        w0=entry - sign * h, w1=entry)
            mouth = Node("cyl", plane=plane, cu=cu, cv=cv, r=r_csk,
                         w0=entry, w1=entry + sign * clear)
            tool = Node("bool", op="union", blend="hard", k=0.0, a=tool,
                        b=Node("bool", op="union", blend="hard", k=0.0,
                               a=cone, b=mouth))

        target["node"] = Node("bool", op="cut", a=target["node"], b=tool,
                              blend="hard", k=0.0)
        self._record_feature("hole", target, ref=op.face_or_sketch,
                             diameter=op.diameter, kind=op.kind)
        self.solid_present = True
        return ApplyResult(True, [self.features[-1]["id"]])

    def _shell(self, op: Shell) -> ApplyResult:
        """Hollow the solid. ``faces`` is a CADQUERY SELECTOR, and always was.

        ``ops.Shell`` documents ``faces`` as selector strings (``(">Z",)``); this
        backend's vocabulary was ``("top",)``, so it answered ``bad-value`` to every
        shell written exactly as the schema specifies it. The schema and the backend
        contradicted each other, and the schema wins: the selector DSL is the one
        grammar, and :func:`resolve_faces` is how frep speaks it. The old words are
        kept as aliases.
        """
        if not self._bodies:
            return _err("no-solid", "shell requires an existing solid")
        if op.thickness <= 0:
            return _err("bad-value", f"shell thickness must be > 0 (got {op.thickness})")
        if op.kind not in ("arc", "intersection"):
            return _err("bad-value",
                        f"unknown shell join kind '{op.kind}' (arc | intersection)")
        if op.kind not in self.SHELL_JOINS:
            return _err(
                "unsupported-op",
                "the frep backend cannot build a '%s'-join shell: a signed-distance "
                "field's inward offset is the erosion by a ball, which IS the 'arc' "
                "join. An 'intersection' join extends the offset faces to a sharp "
                "corner -- an algebraic intersection of half-spaces the field does "
                "not carry. Use cadquery/freecad/openscad, whose kernels have both "
                "joins" % op.kind)
        body = self._bodies[-1]
        bounds = node_bounds(body["node"])
        try:
            faces = resolve_faces(bounds, op.faces or ())
        except selector_dsl.SelectorError as exc:
            return _err("bad-value", f"shell face selector is malformed: {exc}")
        except ValueError as exc:
            return _err("bad-value", str(exc))

        bad = self._check_wall_resolvable(bounds, float(op.thickness))
        if bad is not None:
            return bad

        body["node"] = Node("shell", child=body["node"],
                            thickness=float(op.thickness),
                            faces=tuple(faces), kind=str(op.kind))
        self._record_feature("shell", body, faces=list(op.faces),
                             thickness=op.thickness, kind=op.kind)
        return ApplyResult(True, [self.features[-1]["id"]])

    def _check_wall_resolvable(self, bounds: Tuple[Vec3, Vec3],
                               thickness: float) -> Optional[ApplyResult]:
        """Refuse a wall the sampling grid cannot represent. See :data:`MIN_WALL_CELLS`.

        This is a REPRESENTABILITY check, not an arithmetic one. The field is exact;
        the grid is not. An 80x30x5 box shelled to t=1 at resolution 48 came back
        watertight, manifold, valid, diagnostic-free -- and 75% under volume, with
        its outer surface pulled in by 2mm on every side. Building a different,
        smaller part and calling it a success is precisely the bug class this file
        is being cleaned of, so the wall is refused BEFORE anything mutates, and the
        diagnostic names the resolution that would work.
        """
        cells = (MIN_WALL_CELLS.get(self.mesher, 1.0)
                 if self.SHELL_MIN_WALL_CELLS is None
                 else float(self.SHELL_MIN_WALL_CELLS))
        if cells <= 0.0:
            return None                      # an exact kernel samples no grid
        lo, hi = bounds
        extent = max(hi[i] - lo[i] for i in range(3))
        if extent <= 0.0:
            return None
        cell = extent / float(max(int(self.resolution), 4))
        if thickness >= cells * cell:
            return None
        needed = int(math.ceil(cells * extent / thickness))
        return _err(
            "unsupported-op",
            "the frep backend cannot resolve a %g mm wall on a part %g mm across at "
            "resolution %d with mesher '%s': the grid cell is %.4g mm, so the wall "
            "spans %.2f cells and this mesher needs at least %g. Below one cell the "
            "wall is not in the sampled data at all and the mesher quietly builds a "
            "SMALLER part "
            "(watertight, manifold, and wrong). Re-run with "
            "FRepBackend(resolution=%d), with mesher='dual_contouring' (which holds "
            "a wall to ~0.75 cells), or use an exact kernel "
            "(cadquery/freecad/openscad)"
            % (thickness, extent, self.resolution, self.mesher, cell,
               thickness / cell, cells, needed))

    def _mirror(self, op: Mirror) -> ApplyResult:
        """Mirror THE NAMED FEATURE, not whatever solid happens to be last."""
        if str(op.plane).upper() not in _PLANES:
            return _err("bad-value", f"unknown mirror plane '{op.plane}'")
        target, bad = self._feature_target(op.feature_or_body)
        if bad is not None:
            return bad
        body, node = target
        built = Node("mirror", child=node, plane=str(op.plane).upper())
        self._graft(body, node, built)
        self._record_feature("mirror", body, plane=op.plane,
                             feature_ref=op.feature_or_body)
        return ApplyResult(True, [self.features[-1]["id"]])

    def _pattern_body(self, feature: str, ref: str, count: int,
                      transforms: List[Sequence[float]], **extra) -> ApplyResult:
        target, bad = self._feature_target(ref)
        if bad is not None:
            return bad
        body, node = target
        built = Node("pattern", child=node, transforms=[tuple(t) for t in transforms])
        self._graft(body, node, built)
        self._record_feature(feature, body, count=count, feature_ref=ref, **extra)
        return ApplyResult(True, [self.features[-1]["id"]])

    def _linear_pattern(self, op: LinearPattern) -> ApplyResult:
        if not self._bodies:
            return _err("no-solid", "linear_pattern requires an existing solid")
        if op.count < 2:
            return _err("bad-value", f"linear_pattern count must be >= 2 (got {op.count})")
        d = list(op.direction) + [0.0] * 3
        n = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        if n == 0.0:
            return _err("bad-value", "linear_pattern direction is degenerate")
        ux, uy, uz = d[0] / n, d[1] / n, d[2] / n
        trs = [rigid_transform((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0,
                               (ux * op.spacing * i, uy * op.spacing * i,
                                uz * op.spacing * i))
               for i in range(int(op.count))]
        return self._pattern_body("linear_pattern", op.feature, int(op.count), trs)

    def _circular_pattern(self, op: CircularPattern) -> ApplyResult:
        """Replicate about the axis the op NAMES.

        ``axis`` is a 6-tuple (point, second point) giving the rotation LINE. It was
        read, and then thrown away: every circular pattern spun about global Z,
        through the origin, whatever axis it was handed. It is now a Rodrigues
        rotation about that line (:func:`rigid_transform`), which is why the
        pattern's transforms had to become matrices.
        """
        if not self._bodies:
            return _err("no-solid", "circular_pattern requires an existing solid")
        if op.count < 2:
            return _err("bad-value",
                        f"circular_pattern count must be >= 2 (got {op.count})")
        ax = list(op.axis) + [0.0] * 6
        origin = (float(ax[0]), float(ax[1]), float(ax[2]))
        # The 6-tuple is (ax, ay, az, bx, by, bz): a POINT and a DIRECTION vector,
        # exactly as CadQuery's Workplane.rotate(axisStartPoint, axisEndPoint, angle)
        # takes them -- the cadquery backend passes (a[0..2], a[3..5]) straight
        # through, so the same 6 numbers must mean the same line here.
        direction = (float(ax[3]), float(ax[4]), float(ax[5]))
        if math.sqrt(sum(c * c for c in direction)) == 0.0:
            return _err("bad-value", "circular_pattern axis is degenerate")
        step = float(op.angle) / float(op.count)
        try:
            trs = [rigid_transform(origin, direction, step * i)
                   for i in range(int(op.count))]
        except ValueError as exc:
            return _err("bad-value", str(exc))
        return self._pattern_body("circular_pattern", op.feature, int(op.count), trs)

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
        # The replay must be governed by the SAME policy as the original, or the
        # rebuild is a different backend. These three are set per-INSTANCE by the
        # external backends (see ExternalToolBackend.__init__), and a fresh
        # FRepBackend has the frep-proper defaults -- so a SetParam on freecad
        # replayed its fillets as frep 'blendcut' nodes, which FreeCAD's driver has
        # no arm for, and the rebuilt model came back with volume 0.
        trial.EDGE_SELECTORS = self.EDGE_SELECTORS
        trial.SHELL_JOINS = self.SHELL_JOINS
        trial.SHELL_MIN_WALL_CELLS = self.SHELL_MIN_WALL_CELLS
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
        if _has_shell(node):
            # The IR still encodes a shell as the two-sided ``abs(d) - t/2``
            # (frep_ir._compile, the "shell" arm), which is NOT the function
            # ``eval_node`` samples any more -- that is ``shell_inward``, plus
            # the swept cavity when faces are open. An IR that disagrees with the
            # field is worse than no IR at all: interval pruning would be bounding
            # the WRONG function, and it is allowed to (and does) classify a block
            # that straddles the real surface as FILLED and delete it -- a 60x40x40
            # box shelled at t=12 lost 66% of its volume that way, and the
            # half-edge check passed it, because the result is a perfectly closed
            # manifold of the wrong solid.
            #
            # Refusing to compile is the sound answer: `prune` then degrades to a
            # full sample (right, merely slower) and `normals="autodiff"` raises
            # instead of returning the gradient of a function nobody asked for.
            # Both call sites already handle None. Re-enable this once
            # frep_ir grows a shell arm that matches _eval_shell.
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

    def write_stl(self, path: str, binary: bool = True, force: bool = False) -> int:
        """Write the tessellated model to ``path``; returns the triangle count.

        This is a *write path*, so it goes through the output gate exactly like
        the format registry does: an invalid model raises
        :class:`harnesscad.io.gate.InvalidArtifact` and no file appears.
        """
        from harnesscad.io import gate

        report = gate.guard(self.mesh(), str(path), source=self, force=force)
        tris = mesh_triangles(self.mesh())
        if binary:
            with open(path, "wb") as fh:
                fh.write(stl_fmt.write_binary_stl(tris, header=b"harnesscad-frep"))
        else:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(stl_fmt.write_ascii_stl(tris, name="harnesscad-frep"))
        if not report.ok:                              # forced through the gate
            gate.write_sidecar(str(path), report)
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
