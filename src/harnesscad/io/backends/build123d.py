"""Build123dBackend — a real-geometry GeometryBackend over build123d (OCCT).

build123d is the modern, Pythonic OCCT/OCP CAD library (a cleaner-API sibling of
CadQuery — same OpenCascade kernel underneath). This backend is the second OCCT
front-end in the harness: it mirrors :class:`backends.cadquery.CadQueryBackend`
op-for-op (identical id scheme, query keys, DOF bookkeeping, snapshot/pattern
semantics, block-and-correct and SetParam replay) but drives build123d's
*algebra mode* (``Plane * Pos * Rectangle`` placement arithmetic, ``+``/``-``/``&``
booleans, and the ``extrude``/``revolve``/``loft``/``sweep``/``fillet``/``chamfer``
/``offset`` free functions) instead of CadQuery's Workplane fluent chain.

Because both kernels are OCCT, this backend AGREES with the CadQuery backend to
machine precision on every shared op (see the differential oracle and
``tests/io/backends/test_build123d.py``). It exists for coverage and as a
cross-check on the CadQuery mapping: two independent front-ends that must build
the same B-rep from the same op stream.

Selectors — THE load-bearing part
---------------------------------
CISP fillet/chamfer/shell/hole ops carry CadQuery-style selector strings
(``"|Z"`` = the vertical edges, ``">Z"`` = the top face, ``"|Z and >Y"``,
``"not(<X or >X)"``, ``">Z[0]"``, ``"%CIRCLE"``, ...). build123d's *native*
selector surface is method-based — ``ShapeList.sort_by(Axis.Z)`` /
``.filter_by(GeomType.CIRCLE)`` / ``.group_by(SortBy.AREA)`` — and does not parse
strings. Rather than a lossy, ad-hoc translation of the compound string grammar
into chained method calls, this backend maps a selector string onto build123d
shapes through the harness's OWN CadQuery-compatible selector DSL
(:mod:`harnesscad.domain.geometry.topology.selector_dsl`), which is
differentially proven to agree with CadQuery's real selector engine (see
``TestEdgeSelection.test_our_selector_dsl_agrees_with_cadquerys_own_selectors``).

Each build123d ``Edge``/``Face`` is abstracted to a :class:`selector_dsl.Entity`
— centre of mass, axis (face ``normal_at`` / edge ``tangent_at``) and geometry
type (``GeomType.name``: ``"LINE"``/``"CIRCLE"``/``"PLANE"``, exactly the strings
CadQuery's ``geomType()`` returns) — the DSL filters the entity list, and the
survivors map back to their originating build123d shapes by index. So op.edges /
op.faces are HONOURED, never dropped, and the two OCCT backends resolve the same
selector to the same topology. The DSL selector maps onto build123d's native API
as: ``>Z`` ~ ``faces().sort_by(Axis.Z)[-1]``, ``<Z`` ~ ``[0]``, ``|Z`` ~
``filter_by(Axis.Z)`` (edges), ``%CIRCLE`` ~ ``filter_by(GeomType.CIRCLE)``.

build123d is imported LAZILY inside the methods that need it, so this module
imports cleanly even when build123d / OCP is not installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from typing import List, Optional

from harnesscad.core.cisp.ops import (
    Op, NewSketch, AddPoint, AddLine, AddCircle, AddRectangle,
    Constrain, Extrude, Fillet, Boolean,
    Revolve, Chamfer, Hole, Shell, Draft,
    Loft, Sweep, LinearPattern, CircularPattern, Mirror,
    AddInstance, Mate, SetParam,
    canonical_json, edit_oplog,
)
from harnesscad.core.constraints import ConstraintGraph
from harnesscad.domain.geometry.topology.selector_dsl import Entity, SelectorError
from harnesscad.domain.geometry.topology.selector_dsl import select as select_dsl
from harnesscad.eval.verifiers.assembly import mate_dof
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.backends.base import ApplyResult, BackendUnavailable
from harnesscad.io.backends.frep import check_constraint_arity, solve_constraint

#: Sketch planes this backend accepts (the CadQuery-backend set), validated at
#: NewSketch time. Each maps onto a build123d ``Plane`` class attribute.
PLANES = ("XY", "XZ", "YZ", "YX", "ZX", "ZY",
          "front", "back", "left", "right", "top", "bottom")

#: A solid at or below this volume is degenerate (OCCT can return a topologically
#: non-empty solid of ZERO volume, e.g. a revolve about an in-plane axis).
MIN_VOLUME = 1e-9


def _err(code: str, msg: str, where: Optional[str] = None) -> ApplyResult:
    return ApplyResult(False, [], [Diagnostic(Severity.ERROR, code, msg, where)])


def _b123d():
    """Lazy import of build123d, raising :class:`BackendUnavailable` (never a bare
    ImportError) so a missing kernel is never mistaken for a geometry failure."""
    try:
        import build123d  # noqa: WPS433 (deliberately local / lazy)
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(
            "build123d",
            "the build123d backend requires build123d + OCP (OCCT): %s "
            "(install with: pip install build123d)" % exc,
            ["python: import build123d"])
    return build123d


def _err_types():
    """Exceptions treated as recoverable kernel errors (block-and-correct).

    Broad on purpose: OCCT surfaces failures as StdFail_*, Standard_*, ValueError,
    RuntimeError, ... — a kernel hiccup must never mutate state or crash the loop.
    """
    return Exception


def _plane(b, name: str):
    """The build123d ``Plane`` for a validated CISP plane name."""
    return getattr(b.Plane, name)


def _volume(shape) -> float:
    """Volume of a build123d shape, or 0.0 when it cannot be measured."""
    try:
        return float(shape.volume)
    except Exception:  # noqa: BLE001
        return 0.0


def _bbox_size(shape):
    """``(dx, dy, dz)`` bounding-box extents of a build123d shape."""
    bb = shape.bounding_box()
    s = bb.size
    return (float(s.X), float(s.Y), float(s.Z))


# --- selectors -------------------------------------------------------------

def _shape_entities(b, shapes) -> List[Entity]:
    """Abstract a build123d ShapeList to :class:`selector_dsl.Entity` list.

    ``center`` is the centre of mass; ``axis`` is a face normal (``normal_at``) or
    an edge tangent (``tangent_at(0.5)``); ``geom_type`` is ``GeomType.name`` —
    the same ``"LINE"``/``"CIRCLE"``/``"PLANE"`` strings CadQuery's ``geomType()``
    returns, so the shared DSL means the same thing on both kernels. ``name`` is
    the shape's index, the key used to map the selection back to the shape.
    """
    Face = b.Face
    Edge = b.Edge
    ents: List[Entity] = []
    for i, s in enumerate(shapes):
        c = s.center()
        axis = (0.0, 0.0, 0.0)
        try:
            if isinstance(s, Face):
                a = s.normal_at()
                axis = (float(a.X), float(a.Y), float(a.Z))
            elif isinstance(s, Edge):
                a = s.tangent_at(0.5)
                axis = (float(a.X), float(a.Y), float(a.Z))
        except Exception:  # noqa: BLE001 - non-axial shape -> zero axis
            axis = (0.0, 0.0, 0.0)
        try:
            gt = s.geom_type.name
        except Exception:  # noqa: BLE001
            gt = ""
        ents.append(Entity((float(c.X), float(c.Y), float(c.Z)), axis, gt, str(i)))
    return ents


def _pick(b, shapes, selectors, default: Optional[str]):
    """Resolve a CISP selector tuple to a list of build123d shapes.

    ``default`` is used when the tuple is empty (``None`` = take every shape). A
    tuple is the UNION of its members (CadQuery's ``or`` semantics; see
    ``cadquery.join_selectors``). Raises :class:`SelectorError` for a malformed
    member and ``ValueError`` when the selection is empty.
    """
    shapes = list(shapes)
    sels = [str(s).strip() for s in (selectors or ()) if str(s).strip()]
    if not sels:
        if default is None:
            return shapes
        sels = [default]
    ents = _shape_entities(b, shapes)
    chosen: dict = {}
    for sel in sels:
        for e in select_dsl(sel, ents):   # SelectorError propagates
            chosen[int(e.name)] = True
    if not chosen:
        raise ValueError("selector %r selected no shapes" % (sels,))
    return [shapes[i] for i in sorted(chosen)]


class Build123dBackend:
    #: exports this backend can produce.
    #:
    #: The 3D B-rep / mesh formats (STEP/STL/BREP/3MF/IGES) serialise the whole
    #: solid; the 2D drawing formats (DXF/SVG) serialise a planar CROSS-SECTION of
    #: the solid taken through its centroid normal to +Z (a section view), which is
    #: what those vector-drawing formats represent. IGES reaches parity with the
    #: CadQuery backend (both drive the same OCP ``IGESControl_Writer``); DXF/SVG
    #: cover build123d's ``ExportDXF`` / ``ExportSVG`` 2D exporters, which the first
    #: pass did not expose. build123d natively lacks IGES, so it is driven through
    #: OCP directly (the same kernel build123d itself is built on).
    FORMATS = ("step", "stl", "brep", "3mf", "iges", "dxf", "svg", "gltf")

    def __init__(self) -> None:
        self.reset()

    @staticmethod
    def available() -> bool:
        """Whether build123d/OCP can be imported here. Never raises."""
        try:
            _b123d()
        except BackendUnavailable:
            return False
        return True

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        self.sketches: dict = {}
        self.entities: dict = {}
        self.features: list = []
        self.instances: list = []
        self.mates: list = []
        self.solid_present = False
        self._solids: list = []       # list of build123d solids (each a body)
        self._snapshots: dict = {}    # fid -> (index into _solids, solid snapshot)
        self._oplog: list = []
        self._n = {"sk": 0, "e": 0, "f": 0, "i": 0}

    def _new_id(self, kind: str) -> str:
        self._n[kind] += 1
        return {"sk": "sk", "e": "e", "f": "f", "i": "i"}[kind] + str(self._n[kind])

    # -- sketch primitives --------------------------------------------------
    def _add_primitive(self, sketch: str, kind: str, params: dict) -> ApplyResult:
        if sketch not in self.sketches:
            return _err("bad-ref", f"unknown sketch '{sketch}'", sketch)
        eid = self._new_id("e")
        self.entities[eid] = {"type": kind, "sketch": sketch, "params": params}
        self.sketches[sketch]["entities"].append(eid)
        graph = self.sketches[sketch]["graph"]
        graph.add_entity(eid, kind)
        self.sketches[sketch]["dof"] = graph.residual_dof()
        return ApplyResult(True, [eid])

    # -- op dispatch --------------------------------------------------------
    def apply(self, op: Op) -> ApplyResult:
        if isinstance(op, SetParam):
            return self._set_param(op)
        before = len(self.features)
        result = self._dispatch(op)
        if result.ok:
            self._oplog.append(op)
            if self._solids:
                for feat in self.features[before:]:
                    self._snapshots[feat["id"]] = (len(self._solids) - 1,
                                                   self._solids[-1])
        return result

    def _feature_target(self, ref: str):
        """``(index into _solids, snapshot solid)`` a pattern/mirror ``feature``
        names. An empty ref keeps the historical meaning — the last solid."""
        if not self._solids:
            return None, _err("no-solid", "this op requires an existing solid")
        if not ref:
            return (len(self._solids) - 1, self._solids[-1]), None
        snap = self._snapshots.get(ref)
        if snap is None:
            return None, _err("bad-ref", f"unknown feature '{ref}'", ref)
        index, solid = snap
        if index >= len(self._solids):
            return None, _err("bad-ref",
                              f"feature '{ref}' belonged to a body a later boolean "
                              f"consumed", ref)
        return (index, solid), None

    def _dispatch(self, op: Op) -> ApplyResult:
        if isinstance(op, NewSketch):
            if str(op.plane) not in PLANES:
                return _err("bad-value", f"unknown sketch plane '{op.plane}' "
                                         f"(supported: {', '.join(PLANES)})")
            sid = self._new_id("sk")
            self.sketches[sid] = {
                "plane": op.plane, "entities": [], "dof": 0,
                "graph": ConstraintGraph(),
            }
            return ApplyResult(True, [sid])
        if isinstance(op, AddPoint):
            return self._add_primitive(op.sketch, "point", {"x": op.x, "y": op.y})
        if isinstance(op, AddLine):
            return self._add_primitive(
                op.sketch, "line",
                {"x1": op.x1, "y1": op.y1, "x2": op.x2, "y2": op.y2})
        if isinstance(op, AddCircle):
            if op.r <= 0:
                return _err("bad-value", f"circle radius must be > 0 (got {op.r})")
            return self._add_primitive(
                op.sketch, "circle", {"cx": op.cx, "cy": op.cy, "r": op.r})
        if isinstance(op, AddRectangle):
            if op.w <= 0 or op.h <= 0:
                return _err("bad-value", "rectangle w and h must be > 0")
            return self._add_primitive(
                op.sketch, "rectangle",
                {"x": op.x, "y": op.y, "w": op.w, "h": op.h})
        if isinstance(op, Constrain):
            return self._constrain(op)
        if isinstance(op, Extrude):
            return self._extrude(op)
        if isinstance(op, Fillet):
            return self._fillet(op)
        if isinstance(op, Boolean):
            return self._boolean(op)
        if isinstance(op, Revolve):
            return self._revolve(op)
        if isinstance(op, Chamfer):
            return self._chamfer(op)
        if isinstance(op, Hole):
            return self._hole(op)
        if isinstance(op, Shell):
            return self._shell(op)
        if isinstance(op, Draft):
            return self._draft(op)
        if isinstance(op, Loft):
            return self._loft(op)
        if isinstance(op, Sweep):
            return self._sweep(op)
        if isinstance(op, LinearPattern):
            return self._linear_pattern(op)
        if isinstance(op, CircularPattern):
            return self._circular_pattern(op)
        if isinstance(op, Mirror):
            return self._mirror(op)
        if isinstance(op, AddInstance):
            return self._add_instance(op)
        if isinstance(op, Mate):
            return self._mate(op)
        return _err("unknown-op", f"unhandled op {type(op).__name__}")

    def _feature_ids(self) -> set:
        return {f["id"] for f in self.features}

    def _instance_ids(self) -> set:
        return {inst["id"] for inst in self.instances}

    def _known_part_refs(self) -> set:
        refs = self._feature_ids() | self._instance_ids()
        if self.solid_present:
            refs |= {"solid", "body", "last"}
        return refs

    # -- assembly -----------------------------------------------------------
    def _add_instance(self, op: AddInstance) -> ApplyResult:
        if op.part not in self._known_part_refs():
            return _err("bad-ref", f"unknown part '{op.part}'", op.part)
        transform = {"translate": [op.x, op.y, op.z],
                     "rotate_deg": [op.rx, op.ry, op.rz]}
        shape = None
        bbox = None
        try:
            base = self._combined()
            if base is not None:
                placed = self._place_shape(base, op)
                shape = placed
                bb = placed.bounding_box()
                bbox = [float(bb.min.X), float(bb.min.Y), float(bb.min.Z),
                        float(bb.max.X), float(bb.max.Y), float(bb.max.Z)]
        except BackendUnavailable:
            raise
        except _err_types() as exc:  # noqa: BLE001
            return _err("kernel-error", f"instance placement failed: {exc}")
        iid = self._new_id("i")
        self.instances.append({"id": iid, "part": op.part, "transform": transform,
                               "shape": shape, "bbox": bbox})
        return ApplyResult(True, [iid])

    def _place_shape(self, base, op: AddInstance):
        """Intrinsic X-Y-Z rotation then translation applied to a copy of ``base``
        (a build123d shape), returning the placed shape."""
        b = _b123d()
        placed = base
        if op.rx:
            placed = placed.rotate(b.Axis.X, op.rx)
        if op.ry:
            placed = placed.rotate(b.Axis.Y, op.ry)
        if op.rz:
            placed = placed.rotate(b.Axis.Z, op.rz)
        if op.x or op.y or op.z:
            placed = placed.translate((op.x, op.y, op.z))
        return placed

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
        trial = type(self)()
        for logged in new_log:
            r = trial.apply(logged)
            if not r.ok:
                return ApplyResult(False, [], r.diagnostics)
        self.__dict__.update(trial.__dict__)
        return ApplyResult(True, [])

    def _constrain(self, op: Constrain) -> ApplyResult:
        bad = check_constraint_arity(op)
        if bad is not None:
            return _err(*bad)
        if op.a not in self.entities:
            return _err("bad-ref", f"unknown entity '{op.a}'", op.a)
        if op.b is not None and op.b not in self.entities:
            return _err("bad-ref", f"unknown entity '{op.b}'", op.b)
        solve_constraint(op, self.entities)
        sid = self.entities[op.a]["sketch"]
        graph = self.sketches[sid]["graph"]
        b = op.b if (op.b is not None
                     and self.entities[op.b]["sketch"] == sid) else None
        graph.add_constraint(op.kind, op.a, b, op.value)
        self.sketches[sid]["dof"] = graph.residual_dof()
        self.sketches[sid].setdefault("constraints", []).append(op.kind)
        return ApplyResult(True, [])

    # -- profile realisation ------------------------------------------------
    def _build_profile(self, b, sketch: dict, offset: float = 0.0):
        """The sketch's closed profiles as one build123d Sketch (a sum of faces),
        or None. ``offset`` shifts the profile along its plane normal (for loft)."""
        plane = _plane(b, sketch["plane"])
        faces = None
        for eid in sketch["entities"]:
            ent = self.entities[eid]
            p = ent["params"]
            if ent["type"] == "rectangle":
                cx, cy = p["x"] + p["w"] / 2.0, p["y"] + p["h"] / 2.0
                f = plane * b.Pos(cx, cy) * b.Rectangle(p["w"], p["h"])
            elif ent["type"] == "circle":
                f = plane * b.Pos(p["cx"], p["cy"]) * b.Circle(p["r"])
            else:
                continue  # points / lines are not closed profiles
            faces = f if faces is None else (faces + f)
        if faces is None:
            return None
        if offset:
            n = plane.z_dir
            faces = faces.translate((n.X * offset, n.Y * offset, n.Z * offset))
        return faces

    def _extrude(self, op: Extrude) -> ApplyResult:
        if op.sketch not in self.sketches:
            return _err("bad-ref", f"unknown sketch '{op.sketch}'", op.sketch)
        if not self.sketches[op.sketch]["entities"]:
            return _err("empty-sketch", f"sketch '{op.sketch}' has no profile",
                        op.sketch)
        if op.distance == 0:
            return _err("bad-value", "extrude distance must be non-zero")
        try:
            b = _b123d()
            profile = self._build_profile(b, self.sketches[op.sketch])
            if profile is None:
                return _err("empty-sketch",
                            f"sketch '{op.sketch}' has no closed profile to extrude",
                            op.sketch)
            solid = b.extrude(profile, amount=op.distance)
            if _volume(solid) <= MIN_VOLUME:
                return _err("degenerate", "extrude produced a zero-volume solid")
        except BackendUnavailable:
            raise
        except _err_types() as exc:
            return _err("kernel-error", f"extrude failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "extrude", "id": fid, "sketch": op.sketch})
        self._solids.append(solid)
        self.solid_present = True
        return ApplyResult(True, [fid])

    def _fillet(self, op: Fillet) -> ApplyResult:
        if not self.solid_present or not self._solids:
            return _err("no-solid", "fillet requires an existing solid")
        if op.radius <= 0:
            return _err("bad-value", f"fillet radius must be > 0 (got {op.radius})")
        try:
            b = _b123d()
            target = self._solids[-1]
            # op.edges is a tuple of CadQuery selector strings; empty = every edge.
            edges = _pick(b, target.edges(), op.edges, None)
            n_before = len(target.faces())
            filleted = b.fillet(edges, radius=op.radius)
            if _volume(filleted) <= MIN_VOLUME:
                return _err("degenerate", "fillet produced a zero-volume solid")
            if len(filleted.faces()) <= n_before:
                return _err("degenerate",
                            "fillet did not change the solid: the selected edges "
                            "could not be blended")
        except BackendUnavailable:
            raise
        except SelectorError as exc:
            return _err("bad-value", f"fillet edge selector is malformed: {exc}")
        except _err_types() as exc:
            return _err("kernel-error", f"fillet failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "fillet", "id": fid, "edges": list(op.edges)})
        self._solids[-1] = filleted
        return ApplyResult(True, [fid])

    def _solid_index(self, ref: str) -> Optional[int]:
        """Index into ``self._solids`` of the body a feature id names."""
        bodies = [f["id"] for f in self.features
                  if f["type"] in ("extrude", "revolve", "boolean",
                                   "loft", "sweep")]
        try:
            i = bodies.index(ref)
        except ValueError:
            return None
        return i if i < len(self._solids) else None

    def _boolean(self, op: Boolean) -> ApplyResult:
        if op.kind not in ("union", "cut", "intersect"):
            return _err("bad-value", f"unknown boolean kind '{op.kind}'")
        if len(self.features) < 2 or len(self._solids) < 2:
            return _err("no-solid", "boolean requires two solids")
        ia = self._solid_index(op.target) if op.target else len(self._solids) - 2
        ib = self._solid_index(op.tool) if op.tool else len(self._solids) - 1
        if ia is None:
            return _err("bad-ref", f"unknown boolean target '{op.target}'", op.target)
        if ib is None:
            return _err("bad-ref", f"unknown boolean tool '{op.tool}'", op.tool)
        if ia == ib:
            return _err("bad-ref", "boolean target and tool are the same body")
        try:
            _b123d()
            a = self._solids[ia]
            bsol = self._solids[ib]
            if op.kind == "union":
                result = a + bsol
            elif op.kind == "cut":
                result = a - bsol
            else:
                result = a & bsol
            if _volume(result) <= MIN_VOLUME:
                return _err("degenerate", f"boolean '{op.kind}' produced a "
                                          "zero-volume solid")
        except BackendUnavailable:
            raise
        except _err_types() as exc:
            return _err("kernel-error", f"boolean failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "boolean", "id": fid, "kind": op.kind})
        for i in sorted((ia, ib), reverse=True):
            self._solids.pop(i)
        self._solids.append(result)
        return ApplyResult(True, [fid])

    # -- extended mechanical features --------------------------------------
    def _revolve(self, op: Revolve) -> ApplyResult:
        if op.sketch not in self.sketches:
            return _err("bad-ref", f"unknown sketch '{op.sketch}'", op.sketch)
        if not self.sketches[op.sketch]["entities"]:
            return _err("empty-sketch", f"sketch '{op.sketch}' has no profile",
                        op.sketch)
        if op.angle == 0:
            return _err("bad-value", "revolve angle must be non-zero")
        try:
            b = _b123d()
            sketch = self.sketches[op.sketch]
            profile = self._build_profile(b, sketch)
            if profile is None:
                return _err("empty-sketch",
                            f"sketch '{op.sketch}' has no closed profile to revolve",
                            op.sketch)
            # The two axis points are given in the sketch plane's LOCAL frame (as
            # CadQuery's Workplane.revolve interprets them), so map them to world.
            plane = _plane(b, sketch["plane"])
            a = op.axis
            o = plane.from_local_coords((a[0], a[1], a[2]))
            p2 = plane.from_local_coords((a[3], a[4], a[5]))
            direction = (p2.X - o.X, p2.Y - o.Y, p2.Z - o.Z)
            if abs(direction[0]) + abs(direction[1]) + abs(direction[2]) < 1e-12:
                return _err("bad-value", "revolve axis is degenerate")
            axis = b.Axis((o.X, o.Y, o.Z), direction)
            solid = b.revolve(profile, axis=axis, revolution_arc=abs(op.angle))
            if _volume(solid) <= MIN_VOLUME:
                return _err("degenerate", "revolve produced a zero-volume solid")
        except BackendUnavailable:
            raise
        except _err_types() as exc:
            return _err("kernel-error", f"revolve failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "revolve", "id": fid, "sketch": op.sketch})
        self._solids.append(solid)
        self.solid_present = True
        return ApplyResult(True, [fid])

    def _chamfer(self, op: Chamfer) -> ApplyResult:
        if not self.solid_present or not self._solids:
            return _err("no-solid", "chamfer requires an existing solid")
        if op.distance <= 0:
            return _err("bad-value", f"chamfer distance must be > 0 (got {op.distance})")
        if op.distance2 is not None and op.distance2 <= 0:
            return _err("bad-value",
                        f"chamfer distance2 must be > 0 (got {op.distance2})")
        try:
            b = _b123d()
            target = self._solids[-1]
            edges = _pick(b, target.edges(), op.edges, None)
            n_before = len(target.faces())
            chamfered = b.chamfer(edges, length=op.distance, length2=op.distance2)
            if _volume(chamfered) <= MIN_VOLUME:
                return _err("degenerate", "chamfer produced a zero-volume solid")
            if len(chamfered.faces()) <= n_before:
                return _err("degenerate",
                            "chamfer did not change the solid: the selected edges "
                            "could not be blended")
        except BackendUnavailable:
            raise
        except SelectorError as exc:
            return _err("bad-value", f"chamfer edge selector is malformed: {exc}")
        except _err_types() as exc:
            return _err("kernel-error", f"chamfer failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "chamfer", "id": fid, "edges": list(op.edges)})
        self._solids[-1] = chamfered
        return ApplyResult(True, [fid])

    def _hole(self, op: Hole) -> ApplyResult:
        if op.diameter <= 0:
            return _err("bad-value", f"hole diameter must be > 0 (got {op.diameter})")
        if not op.through and (op.depth is None or op.depth <= 0):
            return _err("bad-value", "blind hole requires depth > 0")
        if op.kind not in ("simple", "counterbore", "countersink"):
            return _err("bad-value", f"unknown hole kind '{op.kind}' "
                                     "(simple | counterbore | countersink)")
        cbore_d = op.cbore_diameter if op.cbore_diameter is not None \
            else 2.0 * op.diameter
        cbore_z = op.cbore_depth if op.cbore_depth is not None else op.diameter
        csk_d = op.csk_diameter if op.csk_diameter is not None else 2.0 * op.diameter
        if op.kind == "counterbore":
            if cbore_d <= op.diameter:
                return _err("bad-value", "counterbore diameter must exceed the "
                                         "hole diameter")
            if cbore_z <= 0:
                return _err("bad-value", "counterbore depth must be > 0")
        if op.kind == "countersink":
            if csk_d <= op.diameter:
                return _err("bad-value", "countersink diameter must exceed the "
                                         "hole diameter")
            if not 0.0 < op.csk_angle < 180.0:
                return _err("bad-value", "countersink angle must be in (0, 180)")
        ref = op.face_or_sketch
        if ref.startswith("sk") and ref not in self.sketches:
            return _err("bad-ref", f"unknown sketch '{ref}'", ref)
        if not self.solid_present or not self._solids:
            return _err("no-solid", "hole requires an existing solid to cut")
        try:
            b = _b123d()
            target = self._solids[-1]
            # A body alias or a sketch id means the default drilling face (">Z");
            # anything else is a CadQuery face selector (honoured, never dropped).
            if ref in ("", "solid", "body", "last") or ref.startswith("sk"):
                face_sel = ">Z"
            else:
                face_sel = ref
            face = _pick(b, target.faces(), (face_sel,), ">Z")[0]
            tool = self._hole_tool(b, target, face, op,
                                   cbore_d, cbore_z, csk_d)
            result = target - tool
            if _volume(result) <= MIN_VOLUME:
                return _err("degenerate", "hole removed the whole solid")
            if _volume(result) >= _volume(target) - MIN_VOLUME:
                return _err("degenerate", "hole removed no material")
        except BackendUnavailable:
            raise
        except SelectorError as exc:
            return _err("bad-value", f"hole face selector is malformed: {exc}")
        except _err_types() as exc:
            return _err("kernel-error", f"hole failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "hole", "id": fid, "ref": ref,
                              "diameter": op.diameter, "kind": op.kind})
        self._solids[-1] = result
        return ApplyResult(True, [fid])

    def _hole_tool(self, b, solid, face, op: Hole, cbore_d, cbore_z, csk_d):
        """The material a hole removes: a plain bore, plus a counterbore cylinder
        or a countersink cone, positioned at world (op.x, op.y) on ``face`` and
        drilling along -normal (into the solid). Built in a canonical +Z frame
        (drilling toward -Z from the surface at z=0) then transformed onto the
        face by ``Plane(origin=entry, z_dir=outward_normal)``.
        """
        Align = b.Align
        MAX = (Align.CENTER, Align.CENTER, Align.MAX)
        n = face.normal_at()
        nvec = (float(n.X), float(n.Y), float(n.Z))
        r = op.diameter / 2.0
        margin = 1.0
        span = max(_bbox_size(solid)) * 1.5 + 2.0 * margin
        # Entry point: world (x, y) projected onto the face plane. For an
        # axis-aligned drilling face (normal ±Z / ±X / ±Y — every face our
        # selectors resolve) this places the hole at the model's (x, y).
        entry = self._face_entry(face, nvec, op.x, op.y)
        length = span if op.through else (op.depth + margin)
        tool = b.Pos(0, 0, margin) * b.Cylinder(r, length, align=MAX)
        if op.kind == "counterbore":
            tool = tool + b.Pos(0, 0, margin) * b.Cylinder(
                cbore_d / 2.0, cbore_z + margin, align=MAX)
        elif op.kind == "countersink":
            csk_h = (csk_d / 2.0 - r) / math.tan(math.radians(op.csk_angle / 2.0))
            tool = tool + b.Cone(bottom_radius=r, top_radius=csk_d / 2.0,
                                 height=csk_h, align=MAX)
        drill_plane = b.Plane(origin=entry, z_dir=nvec)
        return drill_plane * tool

    @staticmethod
    def _face_entry(face, nvec, x: float, y: float):
        """World point on ``face`` at model coordinates (x, y).

        The outward normal is axis-aligned for the faces our selectors resolve, so
        the two axes orthogonal to it carry the model's (x, y) and the normal axis
        carries the face's own coordinate — the CadQuery ProjectedOrigin rule.
        """
        c = face.center()
        ax, ay, az = abs(nvec[0]), abs(nvec[1]), abs(nvec[2])
        if az >= ax and az >= ay:          # ±Z face: (x, y, face_z)
            return (x, y, float(c.Z))
        if ax >= ay:                       # ±X face: (face_x, x, y) — u=Y, v=Z
            return (float(c.X), x, y)
        return (x, float(c.Y), y)          # ±Y face: (x, face_y, y) — u=X, v=Z

    def _shell(self, op: Shell) -> ApplyResult:
        if not self.solid_present or not self._solids:
            return _err("no-solid", "shell requires an existing solid")
        if op.thickness <= 0:
            return _err("bad-value", f"shell thickness must be > 0 (got {op.thickness})")
        if op.kind not in ("arc", "intersection"):
            return _err("bad-value",
                        f"unknown shell kind '{op.kind}' (arc | intersection)")
        try:
            b = _b123d()
            target = self._solids[-1]
            before_size = _bbox_size(target)
            before_v = _volume(target)
            kind = b.Kind.ARC if op.kind == "arc" else b.Kind.INTERSECTION
            # SHELL SEMANTICS. build123d's offset(solid, amount<0) with no openings
            # does NOT hollow — it returns the solid inset inward. So an empty faces
            # tuple (CISP's CLOSED HOLLOW: a sealed void, no face opened) is built as
            # target MINUS its inward inset, giving a sealed shell whose OUTER
            # surface is unchanged. With openings, build123d's offset hollows
            # directly (removing the named faces). Either way the wall is `thickness`
            # and the outer bounding box never grows.
            if not any(str(s).strip() for s in (op.faces or ())):
                inner = b.offset(target, amount=-abs(op.thickness), kind=kind)
                if _volume(inner) <= MIN_VOLUME:
                    return _err("degenerate", "shell wall is too thick for this body")
                shelled = target - inner
            else:
                openings = _pick(b, target.faces(), op.faces, None)
                shelled = b.offset(target, amount=-abs(op.thickness),
                                   openings=openings, kind=kind)
            if _volume(shelled) <= MIN_VOLUME:
                return _err("degenerate", "shell produced a zero-volume solid")
            after_size = _bbox_size(shelled)
            grew = [after_size[i] - before_size[i] for i in range(3)]
            if max(grew) > 1e-6:
                return _err("degenerate",
                            "shell grew the part's bounding box by %s — a shell "
                            "must hollow inward" % [round(g, 6) for g in grew])
            if _volume(shelled) >= before_v - MIN_VOLUME:
                return _err("degenerate",
                            "shell removed no material (wall thickness %g is too "
                            "large for this body)" % op.thickness)
        except BackendUnavailable:
            raise
        except SelectorError as exc:
            return _err("bad-value", f"shell face selector is malformed: {exc}")
        except _err_types() as exc:
            return _err("kernel-error", f"shell failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "shell", "id": fid,
                              "faces": list(op.faces), "thickness": op.thickness})
        self._solids[-1] = shelled
        return ApplyResult(True, [fid])

    def _linear_pattern(self, op: LinearPattern) -> ApplyResult:
        if not self.solid_present or not self._solids:
            return _err("no-solid", "linear_pattern requires an existing solid")
        if op.count < 2:
            return _err("bad-value", f"linear_pattern count must be >= 2 (got {op.count})")
        target, bad = self._feature_target(op.feature)
        if bad is not None:
            return bad
        index, base = target
        d = list(op.direction) + [0.0, 0.0, 0.0]
        norm = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        if norm == 0.0:
            return _err("bad-value", "linear_pattern direction is degenerate")
        ux, uy, uz = d[0] / norm, d[1] / norm, d[2] / norm
        try:
            _b123d()
            result = self._solids[index]
            for i in range(1, op.count):
                off = (ux * op.spacing * i, uy * op.spacing * i, uz * op.spacing * i)
                result = result + base.translate(off)
            if _volume(result) <= MIN_VOLUME:
                return _err("degenerate", "linear_pattern produced no solid")
        except BackendUnavailable:
            raise
        except _err_types() as exc:
            return _err("kernel-error", f"linear_pattern failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "linear_pattern", "id": fid, "count": op.count})
        self._solids[index] = result
        return ApplyResult(True, [fid])

    def _circular_pattern(self, op: CircularPattern) -> ApplyResult:
        if not self.solid_present or not self._solids:
            return _err("no-solid", "circular_pattern requires an existing solid")
        if op.count < 2:
            return _err("bad-value", f"circular_pattern count must be >= 2 (got {op.count})")
        target, bad = self._feature_target(op.feature)
        if bad is not None:
            return bad
        index, base = target
        try:
            b = _b123d()
            a = op.axis
            axis = b.Axis((a[0], a[1], a[2]), (a[3], a[4], a[5]))
            # PITCH RULE (CadQuery Workplane.polarArray): a full turn divides by
            # count (else the last copy lands on the first); any other arc is
            # spanned INCLUSIVELY (angle / (count - 1)).
            if abs(math.remainder(float(op.angle), 360.0)) < 1e-9:
                step = float(op.angle) / float(op.count)
            else:
                step = float(op.angle) / float(op.count - 1)
            result = self._solids[index]
            for i in range(1, op.count):
                result = result + base.rotate(axis, step * i)
            if _volume(result) <= MIN_VOLUME:
                return _err("degenerate", "circular_pattern produced no solid")
        except BackendUnavailable:
            raise
        except _err_types() as exc:
            return _err("kernel-error", f"circular_pattern failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "circular_pattern", "id": fid, "count": op.count})
        self._solids[index] = result
        return ApplyResult(True, [fid])

    def _mirror(self, op: Mirror) -> ApplyResult:
        if not self.solid_present or not self._solids:
            return _err("no-solid", "mirror requires an existing solid")
        if op.plane not in ("XY", "XZ", "YZ"):
            return _err("bad-value", f"unknown mirror plane '{op.plane}'")
        target, bad = self._feature_target(op.feature_or_body)
        if bad is not None:
            return bad
        index, base = target
        try:
            b = _b123d()
            mirrored = b.mirror(base, about=_plane(b, op.plane))
            result = self._solids[index] + mirrored
            if _volume(result) <= MIN_VOLUME:
                return _err("degenerate", "mirror produced no solid")
        except BackendUnavailable:
            raise
        except _err_types() as exc:
            return _err("kernel-error", f"mirror failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "mirror", "id": fid, "plane": op.plane})
        self._solids[index] = result
        return ApplyResult(True, [fid])

    # -- loft / sweep / draft ----------------------------------------------
    def _loft(self, op: Loft) -> ApplyResult:
        if len(op.sketches) < 2:
            return _err("bad-value", "loft requires at least two sketches")
        for sid in op.sketches:
            if sid not in self.sketches:
                return _err("bad-ref", f"unknown sketch '{sid}'", sid)
            if not self.sketches[sid]["entities"]:
                return _err("empty-sketch", f"sketch '{sid}' has no profile", sid)
        try:
            b = _b123d()
            offsets = list(op.offsets) + [0.0] * len(op.sketches)
            sections = []
            for i, sid in enumerate(op.sketches):
                sec = self._build_profile(b, self.sketches[sid], float(offsets[i]))
                if sec is None:
                    return _err("empty-sketch",
                                f"sketch '{sid}' has no closed profile to loft", sid)
                sections.append(sec)
            solid = b.loft(sections, ruled=bool(op.ruled))
            if _volume(solid) <= MIN_VOLUME:
                return _err("degenerate", "loft produced a zero-volume solid")
        except BackendUnavailable:
            raise
        except _err_types() as exc:
            return _err("kernel-error", f"loft failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "loft", "id": fid,
                              "sketches": list(op.sketches)})
        self._solids.append(solid)
        self.solid_present = True
        return ApplyResult(True, [fid])

    def _build_path(self, b, sketch: dict):
        """A sketch's path as a build123d Wire (chained lines) or a circle wire,
        in world coordinates via the sketch plane's local->world map."""
        plane = _plane(b, sketch["plane"])
        lines = [self.entities[e] for e in sketch["entities"]
                 if self.entities[e]["type"] == "line"]
        circles = [self.entities[e] for e in sketch["entities"]
                   if self.entities[e]["type"] == "circle"]
        if lines:
            edges = []
            for ent in lines:
                q = ent["params"]
                p1 = plane.from_local_coords((q["x1"], q["y1"]))
                p2 = plane.from_local_coords((q["x2"], q["y2"]))
                edges.append(b.Edge.make_line((p1.X, p1.Y, p1.Z),
                                              (p2.X, p2.Y, p2.Z)))
            return b.Wire(edges) if len(edges) > 1 else edges[0]
        if circles:
            p = circles[0]["params"]
            face = plane * b.Pos(p["cx"], p["cy"]) * b.Circle(p["r"])
            return face.wire()
        return None

    def _sweep(self, op: Sweep) -> ApplyResult:
        for sid in (op.sketch, op.path):
            if sid not in self.sketches:
                return _err("bad-ref", f"unknown sketch '{sid}'", sid)
            if not self.sketches[sid]["entities"]:
                return _err("empty-sketch", f"sketch '{sid}' has no profile", sid)
        try:
            b = _b123d()
            profile = self._build_profile(b, self.sketches[op.sketch])
            if profile is None:
                return _err("empty-sketch",
                            f"sweep profile sketch '{op.sketch}' has no closed "
                            f"profile", op.sketch)
            path = self._build_path(b, self.sketches[op.path])
            if path is None:
                return _err("empty-sketch",
                            f"sweep path sketch '{op.path}' has no line or circle "
                            f"path", op.path)
            solid = b.sweep(profile, path=path)
            if _volume(solid) <= MIN_VOLUME:
                return _err("degenerate", "sweep produced a zero-volume solid")
        except BackendUnavailable:
            raise
        except _err_types() as exc:
            return _err("kernel-error", f"sweep failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "sweep", "id": fid, "sketch": op.sketch,
                              "path": op.path})
        self._solids.append(solid)
        self.solid_present = True
        return ApplyResult(True, [fid])

    def _draft(self, op: Draft) -> ApplyResult:
        """Taper faces about a neutral plane (OCCT BRepOffsetAPI_DraftAngle).

        build123d has no draft *operation*, so — as the CadQuery backend does —
        this drives the OCCT algorithm directly (both kernels share OCP), then
        wraps the result back into a build123d Solid. ``op.faces`` are CadQuery
        face selectors; ``op.neutral_plane`` is a datum name (XY/XZ/YZ) or a face
        selector (default "<Z"); the pull direction is that plane's inward normal.
        """
        if not self.solid_present or not self._solids:
            return _err("no-solid", "draft requires an existing solid")
        if op.angle == 0:
            return _err("bad-value", "draft angle must be non-zero")
        if abs(op.angle) >= 90.0:
            return _err("bad-value", f"draft angle must be < 90 deg (got {op.angle})")
        try:
            b = _b123d()
            from OCP.BRepOffsetAPI import BRepOffsetAPI_DraftAngle
            from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

            target = self._solids[-1]
            spec = str(op.neutral_plane).strip() or "<Z"
            datum = {"XY": (0.0, 0.0, 1.0), "YX": (0.0, 0.0, 1.0),
                     "XZ": (0.0, 1.0, 0.0), "ZX": (0.0, 1.0, 0.0),
                     "YZ": (1.0, 0.0, 0.0), "ZY": (1.0, 0.0, 0.0)}
            if spec.upper() in datum:
                pull_v = datum[spec.upper()]
                origin = (0.0, 0.0, 0.0)
            else:
                neutral = _pick(b, target.faces(), (spec,), "<Z")[0]
                nc, nn = neutral.center(), neutral.normal_at()
                origin = (nc.X, nc.Y, nc.Z)
                pull_v = (-nn.X, -nn.Y, -nn.Z)  # into the body
            pull = gp_Dir(*pull_v)
            plane = gp_Pln(gp_Pnt(*origin), pull)
            # Default face set: the side walls — every face whose normal is
            # perpendicular to the pull direction (what a mould release needs).
            if any(str(s).strip() for s in (op.faces or ())):
                faces = _pick(b, target.faces(), op.faces, None)
            else:
                faces = [f for f in target.faces()
                         if abs(f.normal_at().X * pull_v[0]
                                + f.normal_at().Y * pull_v[1]
                                + f.normal_at().Z * pull_v[2]) <= 1e-6]
            builder = BRepOffsetAPI_DraftAngle(target.wrapped)
            added = 0
            for f in faces:
                builder.Add(f.wrapped, pull, math.radians(abs(op.angle)), plane)
                added += 1
            if not added:
                return _err("bad-value", "draft selected no faces to taper")
            builder.Build()
            if not builder.IsDone():
                return _err("kernel-error", "OCCT draft (BRepOffsetAPI_DraftAngle) "
                                            "could not build the tapered solid")
            drafted = b.Solid(b.downcast(builder.Shape()))
            if _volume(drafted) <= MIN_VOLUME:
                return _err("degenerate", "draft produced a zero-volume solid")
        except BackendUnavailable:
            raise
        except SelectorError as exc:
            return _err("bad-value", f"draft face selector is malformed: {exc}")
        except _err_types() as exc:
            return _err("kernel-error", f"draft failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "draft", "id": fid,
                              "faces": list(op.faces), "angle": op.angle})
        self._solids[-1] = drafted
        return ApplyResult(True, [fid])

    def regenerate(self) -> List[Diagnostic]:
        return []  # incremental backend; nothing to rebuild

    # -- combined-shape helpers --------------------------------------------
    def _combined(self):
        """A single build123d Compound of all current solids, or None."""
        if not self._solids:
            return None
        b = _b123d()
        solids = []
        for s in self._solids:
            solids.extend(s.solids())
        if not solids:
            return None
        if len(solids) == 1:
            return solids[0]
        return b.Compound(solids)

    def _validity(self) -> dict:
        shape = None
        try:
            shape = self._combined()
        except Exception:  # noqa: BLE001 - validity must never raise
            shape = None
        if shape is None:
            return {"manifold": False, "watertight": False,
                    "is_valid": False, "solid_present": False}
        is_valid = False
        watertight = False
        try:
            is_valid = bool(shape.is_valid())
        except Exception:  # noqa: BLE001
            is_valid = False
        try:
            watertight = is_valid and _volume(shape) > 1e-12
        except Exception:  # noqa: BLE001
            watertight = is_valid
        return {
            "manifold": is_valid,
            "watertight": watertight,
            "is_valid": is_valid,
            "solid_present": True,
        }

    # -- queries ------------------------------------------------------------
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
            return self._measure()
        if q == "metrics":
            return self._metrics()
        if q == "assembly":
            return self._assembly()
        return {}

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
            if inst.get("shape") is not None:
                part["shape"] = inst["shape"]
            parts.append(part)
            transforms[inst["id"]] = inst["transform"]
        return {"parts": parts, "mates": [dict(m) for m in self.mates],
                "transforms": transforms}

    def _measure(self) -> dict:
        try:
            shape = self._combined()
        except Exception:  # noqa: BLE001
            shape = None
        if shape is None:
            return {"volume": 0.0, "bbox": [0.0, 0.0, 0.0]}
        try:
            return {"volume": _volume(shape), "bbox": list(_bbox_size(shape))}
        except Exception:  # noqa: BLE001
            return {"volume": 0.0, "bbox": [0.0, 0.0, 0.0]}

    def _metrics(self, density: float = 1.0) -> dict:
        try:
            shape = self._combined()
        except Exception:  # noqa: BLE001
            shape = None
        if shape is None:
            return {}
        try:
            b = _b123d()
            volume = _volume(shape)
            com = shape.center(b.CenterOf.MASS)
            return {
                "volume": volume,
                "mass": volume * density,
                "surface_area": float(shape.area),
                "bbox": list(_bbox_size(shape)),
                "center_of_mass": [float(com.X), float(com.Y), float(com.Z)],
                "faces": len(shape.faces()),
                "edges": len(shape.edges()),
                "vertices": len(shape.vertices()),
                "solids": len(shape.solids()),
            }
        except Exception:  # noqa: BLE001
            return {}

    # -- export -------------------------------------------------------------
    #: Tessellation deflection for meshed exports (STL / 3MF). Matches the
    #: CadQuery backend's pinned defaults so the two kernels mesh comparably.
    LINEAR_DEFLECTION = 1e-3      # mm
    ANGULAR_DEFLECTION = 0.05     # radians (~2.9 deg)

    def export(self, fmt: str, tolerance: Optional[float] = None,
               angular_tolerance: Optional[float] = None):
        fmt = str(fmt).lower()
        if fmt not in self.FORMATS:
            raise ValueError("the build123d backend cannot export '%s' (supported: "
                             "%s)" % (fmt, ", ".join(self.FORMATS)))
        b = _b123d()
        shape = self._combined()
        if shape is None:
            raise ValueError("nothing to export: no solid present")
        lin = self.LINEAR_DEFLECTION if tolerance is None else float(tolerance)
        ang = (self.ANGULAR_DEFLECTION if angular_tolerance is None
               else float(angular_tolerance))
        if lin <= 0 or ang <= 0:
            raise ValueError("export tolerances must be > 0")
        if fmt == "step":
            return self._export_file(b, shape, ".step",
                                     lambda p: b.export_step(shape, p))
        if fmt == "stl":
            return self._export_file(
                b, shape, ".stl",
                lambda p: b.export_stl(shape, p, tolerance=lin,
                                       angular_tolerance=ang, ascii_format=True))
        if fmt == "brep":
            return self._export_file(b, shape, ".brep",
                                     lambda p: b.export_brep(shape, p))
        if fmt == "iges":
            return self._export_iges(shape)
        if fmt in ("dxf", "svg"):
            return self._export_2d(b, shape, fmt)
        if fmt == "gltf":
            return self._export_file(
                b, shape, ".gltf",
                lambda p: b.export_gltf(shape, p, binary=False,
                                        linear_deflection=lin,
                                        angular_deflection=ang))
        return self._export_3mf(b, shape, lin, ang)

    @staticmethod
    def _export_iges(shape) -> str:
        """IGES via the OCP ``IGESControl_Writer``.

        build123d ships no IGES exporter, but it is built on OCP (OCCT), so the
        kernel's own IGES writer is available and produces the SAME AP-neutral
        surface serialisation the CadQuery backend does — this is why the two
        OCCT front-ends now export the same format set. ``shape.wrapped`` is the
        underlying ``TopoDS_Shape``.
        """
        from OCP.IGESControl import IGESControl_Writer

        fd, path = tempfile.mkstemp(suffix=".iges")
        os.close(fd)
        try:
            writer = IGESControl_Writer()
            writer.AddShape(shape.wrapped)
            writer.ComputeModel()
            if not writer.Write(path):
                raise ValueError("IGES writer reported failure")
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    @staticmethod
    def _export_2d(b, shape, fmt: str) -> str:
        """A 2D drawing (DXF or SVG) of a planar section of the solid.

        DXF and SVG are vector *drawing* formats, not solid formats: build123d's
        ``ExportDXF`` / ``ExportSVG`` draw wires/edges/faces that lie in a plane.
        Feeding them a 3D solid draws every edge flattened onto Z=0 (build123d
        warns "points found outside the XY plane"), which is not a meaningful
        drawing. Instead this takes a real CROSS-SECTION of the solid — the plane
        through the solid's centroid with a +Z normal — and moves that planar
        section onto Z=0 so the exporter receives clean in-plane geometry. The
        result is a section VIEW of the part, which is exactly what a DXF/SVG of a
        solid conventionally means.
        """
        bb = shape.bounding_box()
        z_mid = float((bb.min.Z + bb.max.Z) / 2.0)
        section = b.section(shape, section_by=b.Plane.XY.offset(z_mid))
        if not section.faces() and not section.edges():
            raise ValueError("nothing to draw: the section plane missed the solid")
        planar = section.moved(b.Location((0.0, 0.0, -z_mid)))
        exporter = (b.ExportDXF() if fmt == "dxf" else b.ExportSVG())
        suffix = "." + fmt
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            exporter.add_shape(planar)
            exporter.write(path)
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    @staticmethod
    def _export_file(b, shape, suffix: str, writer) -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            writer(path)
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    @staticmethod
    def _export_3mf(b, shape, lin: float, ang: float) -> str:
        fd, path = tempfile.mkstemp(suffix=".3mf")
        os.close(fd)
        try:
            mesher = b.Mesher()
            mesher.add_shape(shape, linear_deflection=lin, angular_deflection=ang)
            mesher.write(path)
            with open(path, "rb") as fh:
                return fh.read().hex()   # 3MF is a binary zip; hex-encode for text
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    # -- content digest -----------------------------------------------------
    def state_digest(self) -> str:
        descriptor = {
            "sketch_count": len(self.sketches),
            "entity_count": len(self.entities),
            "feature_count": len(self.features),
            "solid_present": self.solid_present,
            "instances": [
                {"part": inst["part"], "transform": inst["transform"],
                 "bbox": ([round(v, 6) for v in inst["bbox"]]
                          if inst.get("bbox") is not None else None)}
                for inst in self.instances
            ],
            "mates": self.mates,
            "oplog": [canonical_json(o) for o in self._oplog],
        }
        shape = None
        try:
            shape = self._combined()
        except Exception:  # noqa: BLE001
            shape = None
        if shape is not None:
            try:
                size = _bbox_size(shape)
                descriptor["geom"] = {
                    "volume": round(_volume(shape), 6),
                    "bbox": [round(size[0], 6), round(size[1], 6), round(size[2], 6)],
                    "faces": len(shape.faces()),
                    "edges": len(shape.edges()),
                    "vertices": len(shape.vertices()),
                    "solids": len(shape.solids()),
                }
            except Exception:  # noqa: BLE001
                descriptor["geom"] = None
        blob = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()
