"""CadQueryBackend — a real-geometry GeometryBackend over CadQuery (OCCT).

This mirrors :class:`backends.stub.StubBackend` op-for-op (identical id scheme,
query keys, DOF bookkeeping and block-and-correct semantics) but produces *real*
B-rep solids through the OpenCascade kernel:

  * NewSketch      -> a logical sketch bound to a named cq plane (XY/YZ/XZ/...).
  * AddRectangle / AddCircle / AddLine / AddPoint -> real 2D sketch geometry
    (recorded as profile entities and realised as cq wires at extrude time).
  * Extrude        -> a real OCCT solid (cq extrude of the sketch profile).
  * Fillet         -> a real edge fillet on the current solid.
  * Boolean        -> a real union / cut / intersect of two solids.

Block-and-correct: on an invalid reference *or* any kernel exception we return
``ApplyResult(ok=False, ...)`` WITHOUT mutating state — geometry is only
committed to ``self`` after the kernel op succeeds.

Constraints: CadQuery ships no full 2D constraint solver, so per-sketch DOF is
tracked by :class:`constraints.ConstraintGraph` — a genuine rank-style DOF
analysis (per-entity DOF, per-constraint DOF removal, redundancy detection,
under/well/over classification) over the CISP abstract sketch model, replacing
the old inline additive heuristic. ``query('sketch_dof')`` reports the graph's
signed residual DOF (so it stays consistent with the harness conventions and can
still go negative when over-determined). A concrete geometric solve is available
via :class:`constraints.SolveSpaceSketch` (the optional ``constraints`` extra,
python-solvespace).

CadQuery is imported LAZILY inside the methods that need it, so this module
imports cleanly even when cadquery / cadquery-ocp (OCCT) is not installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import os
from typing import List, Optional

from harnesscad.core.cisp.ops import (
    CONSTRAINT_DOF, PRIMITIVE_DOF,
    Op, NewSketch, AddPoint, AddLine, AddCircle, AddRectangle,
    AddArc, AddEllipse, AddPolygon, AddSpline,
    Constrain, Extrude, Fillet, Boolean,
    Primitive, Split, Thicken, Hull, Minkowski,
    Transform, Scale, PatternTransform,
    Revolve, Chamfer, Hole, Shell, Draft,
    Loft, Sweep, LinearPattern, CircularPattern, Mirror,
    AddInstance, Mate, SetParam,
    canonical_json, check_mate_ports, edit_oplog, mate_record,
    thicken_delta,
)
from harnesscad.eval.verifiers.assembly import mate_dof
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.backends.base import ApplyResult, BackendUnavailable
from harnesscad.io.backends.frep import check_constraint_arity, solve_constraint
from harnesscad.core.constraints import ConstraintGraph
from harnesscad.domain.geometry.topology.selector_dsl import SelectorError
from harnesscad.domain.geometry.topology.selector_dsl import parse as parse_selector

#: Sketch planes this backend accepts. Validated at NewSketch time (as the frep
#: backend does) rather than exploding later inside cq.Workplane at extrude time.
PLANES = ("XY", "XZ", "YZ", "YX", "ZX", "ZY",
          "front", "back", "left", "right", "top", "bottom")

#: A solid whose volume is at or below this is degenerate. OCCT will happily
#: return a *topologically* non-empty solid with ZERO volume (e.g. a revolve
#: about an axis normal to the sketch plane), so counting solids is not enough.
MIN_VOLUME = 1e-9


def _err(code: str, msg: str, where: Optional[str] = None) -> ApplyResult:
    return ApplyResult(False, [], [Diagnostic(Severity.ERROR, code, msg, where)])


def _cq():
    """Lazy import of cadquery so this module loads without OCCT installed.

    Raises :class:`BackendUnavailable` (not a bare ImportError) so that a missing
    kernel is never mistaken for a *geometry* failure: the op handlers catch
    kernel exceptions broadly to implement block-and-correct, and without this a
    missing install surfaced as a per-op ``kernel-error`` diagnostic and the
    harness silently produced an empty model instead of refusing to start.
    """
    try:
        import cadquery  # noqa: WPS433 (deliberately local / lazy)
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(
            "cadquery",
            "the cadquery backend requires cadquery + cadquery-ocp (OCCT): %s "
            "(install with: pip install 'harnesscad[cadquery]')" % exc,
            ["python: import cadquery"])
    return cadquery


def solid_volume(wp) -> float:
    """Total volume of every solid in ``wp``, or 0.0 when there is none."""
    try:
        return float(sum(s.Volume() for s in wp.solids().vals()))
    except Exception:  # noqa: BLE001
        return 0.0


def _degenerate(wp, what: str) -> Optional[ApplyResult]:
    """``ApplyResult`` when ``wp`` is not a real, positive-volume solid, else None."""
    try:
        if not wp.solids().vals():
            return _err("degenerate", "%s produced no solid" % what)
    except Exception as exc:  # noqa: BLE001
        return _err("kernel-error", "%s produced no readable solid: %s" % (what, exc))
    if solid_volume(wp) <= MIN_VOLUME:
        return _err("degenerate", "%s produced a zero-volume solid" % what)
    return None


def join_selectors(selectors) -> Optional[str]:
    """Combine CISP selector strings into ONE CadQuery selector string, or None.

    CISP carries a *tuple* of selectors (``Fillet.edges``, ``Shell.faces``);
    CadQuery's ``Workplane.edges(selector)`` / ``.faces(selector)`` take a single
    string. Per the selectors doc (selectors.html, "Combining Selectors"), string
    selectors compose with ``and`` / ``or`` / ``not`` / ``exc``, so a tuple is the
    ``or`` (set union) of its members — each parenthesised so that a member which
    is itself a compound expression cannot bind across the join.

    Returns None for an empty tuple, which every caller reads as "no filter".
    Raises :class:`SelectorError` for a malformed member, so the backend reports a
    typed ``bad-value`` diagnostic instead of letting a typo reach the kernel.
    """
    sels = [str(s).strip() for s in (selectors or ()) if str(s).strip()]
    if not sels:
        return None
    joined = " or ".join("(%s)" % s for s in sels) if len(sels) > 1 else sels[0]
    parse_selector(joined)  # validate against our CadQuery-compatible grammar
    return joined


def _blend_took_effect(before, after, what: str) -> Optional[ApplyResult]:
    """``ApplyResult`` when a fillet/chamfer silently did NOTHING, else None.

    ``_degenerate`` only asks "is there a positive-volume solid?", which an
    UNCHANGED solid passes. OCCT usually throws on an impossible blend, but not
    always (a degenerate radius, or an edge list that resolved to a seam edge, can
    return the input shape untouched). A blend that really happened MUST change the
    topology: BRepFilletAPI_MakeFillet adds one face per rounded edge (and a patch
    per corner), and BRepFilletAPI_MakeChamfer adds one planar face per edge. So a
    face count that did not rise means the op was a no-op and must be refused
    rather than committed as a phantom feature.
    """
    try:
        n_before = len(before.val().Faces())
        n_after = len(after.val().Faces())
    except Exception:  # noqa: BLE001 - can't tell -> don't block
        return None
    if n_after <= n_before:
        return _err("degenerate",
                    "%s did not change the solid (face count stayed at %d): the "
                    "selected edges could not be blended" % (what, n_before))
    return None


def _sub_shapes(wp, kind: str, selectors, default: Optional[str] = None):
    """``wp.edges(sel)`` / ``wp.faces(sel)`` for a CISP selector tuple.

    ``default`` is the selector used when the tuple is empty (None = take them
    all). Returns the resulting Workplane with the sub-shapes on the stack.
    Raises SelectorError (malformed) or ValueError (selected nothing).
    """
    sel = join_selectors(selectors)
    if sel is None:
        sel = default
    picked = getattr(wp, kind)(sel) if sel else getattr(wp, kind)()
    if not picked.vals():
        raise ValueError("selector %r selected no %s" % (sel, kind))
    return picked


class CadQueryBackend:
    #: exports this backend can produce (the only backend with real B-rep/STEP)
    FORMATS = ("step", "stl", "iges", "brep")

    def __init__(self) -> None:
        self.reset()

    @staticmethod
    def available() -> bool:
        """Whether cadquery/OCCT can be imported here. Never raises."""
        try:
            _cq()
        except BackendUnavailable:
            return False
        return True

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        self.sketches: dict = {}      # sid -> {plane, entities:[eid], dof}
        self.entities: dict = {}      # eid -> {type, sketch, params}
        self.features: list = []      # [{type, id, ...}]
        self.instances: list = []     # [{id, part, transform, shape, bbox}]
        self.mates: list = []         # [{kind, a, b, value}]
        self.solid_present = False
        self._solids: list = []       # list of cq.Workplane, each a real solid
        #: fid -> (index into _solids, the solid AS OF that feature). A pattern or a
        #: mirror names a FEATURE, and it used to replicate self._solids[-1] instead
        #: -- the last body, whatever it was handed. Patterning the pad is not
        #: patterning the pad-plus-fillet, and in a model with two bodies it is
        #: simply the wrong one. Recorded centrally in apply(), so no handler can
        #: forget to.
        self._snapshots: dict = {}
        self._oplog: list = []        # successfully applied mutating ops (no SetParam)
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
        # SetParam edits + replays the recorded op stream; it manages its own
        # logging (and must never itself be logged, to keep replay finite).
        if isinstance(op, SetParam):
            return self._set_param(op)
        before = len(self.features)
        result = self._dispatch(op)
        if result.ok:
            self._oplog.append(op)
            # Snapshot every feature this op created, against the solid it left
            # behind. cq.Workplane objects are replaced functionally (a fillet
            # produces a NEW Workplane), so holding the reference IS the snapshot.
            if self._solids:
                for feat in self.features[before:]:
                    self._snapshots[feat["id"]] = (len(self._solids) - 1,
                                                   self._solids[-1])
        return result

    def _feature_target(self, ref: str):
        """``(index into _solids, the solid)`` a pattern/mirror ``feature`` names.

        An empty ref keeps the historical meaning -- the last solid, as it stands --
        so streams that never named a feature are unchanged.
        """
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
            # Validate the plane HERE (as frep does). cq.Workplane accepts only a
            # fixed set of names; an unknown one used to sail through NewSketch and
            # only blow up later inside _extrude, where it was reported as a
            # 'kernel-error' on the extrude rather than a 'bad-value' on the sketch.
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
        if isinstance(op, AddArc):
            if op.r <= 0:
                return _err("bad-value", f"arc radius must be > 0 (got {op.r})")
            if float(op.start) == float(op.end):
                return _err("bad-value", "arc start and end angle must differ")
            return self._add_primitive(
                op.sketch, "arc",
                {"cx": op.cx, "cy": op.cy, "r": op.r,
                 "start": op.start, "end": op.end})
        if isinstance(op, AddEllipse):
            if op.rx <= 0 or op.ry <= 0:
                return _err("bad-value", "ellipse rx and ry must be > 0")
            return self._add_primitive(
                op.sketch, "ellipse",
                {"cx": op.cx, "cy": op.cy, "rx": op.rx, "ry": op.ry,
                 "rotation": op.rotation})
        if isinstance(op, AddPolygon):
            if len(op.points) < 6 or len(op.points) % 2 != 0:
                return _err("bad-value", "polygon needs >= 3 vertices")
            return self._add_primitive(op.sketch, "polygon",
                                       {"points": tuple(op.points)})
        if isinstance(op, AddSpline):
            if len(op.points) < 4 or len(op.points) % 2 != 0:
                return _err("bad-value", "spline needs >= 2 points")
            return self._add_primitive(op.sketch, "spline",
                                       {"points": tuple(op.points),
                                        "closed": bool(op.closed)})
        if isinstance(op, Primitive):
            return self._primitive(op)
        if isinstance(op, Split):
            return self._split(op)
        if isinstance(op, Thicken):
            return self._thicken(op)
        if isinstance(op, Hull):
            return _err("unsupported-op",
                        "the cadquery backend cannot build a convex hull: OCCT has "
                        "no convex-hull operation exposed through CadQuery. A hull "
                        "is built by the openscad or manifold backend")
        if isinstance(op, Minkowski):
            return _err("unsupported-op",
                        "the cadquery backend cannot build a Minkowski sum: OCCT "
                        "has no 3D Minkowski / ball-dilation operation exposed "
                        "through CadQuery. A ball dilation is built by the frep "
                        "SDF kernel or by OpenSCAD's minkowski()")
        if isinstance(op, Transform):
            return self._transform(op)
        if isinstance(op, Scale):
            return _err("unsupported-op",
                        "the cadquery backend does not wire a scale transform: a "
                        "uniform scale is OCCT gp_Trsf.SetScale and a non-uniform "
                        "scale BRepBuilderAPI_GTransform, neither yet exposed here. "
                        "Scale is built by the frep, openscad or manifold backend")
        if isinstance(op, PatternTransform):
            return self._pattern_transform(op)
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

    # -- assembly (real placed geometry) -----------------------------------
    def _add_instance(self, op: AddInstance) -> ApplyResult:
        if op.part not in self._known_part_refs():
            return _err("bad-ref", f"unknown part '{op.part}'", op.part)
        transform = {"translate": [op.x, op.y, op.z],
                     "rotate_deg": [op.rx, op.ry, op.rz]}
        shape = None
        bbox = None
        try:
            base = self._combined()  # current body snapshot for this instance
            if base is not None:
                placed = self._place_shape(base, op)
                shape = placed
                b = placed.BoundingBox()
                bbox = [float(b.xmin), float(b.ymin), float(b.zmin),
                        float(b.xmax), float(b.ymax), float(b.zmax)]
        except BackendUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - placement issue -> block-and-correct
            return _err("kernel-error", f"instance placement failed: {exc}")
        iid = self._new_id("i")
        self.instances.append({"id": iid, "part": op.part, "transform": transform,
                               "shape": shape, "bbox": bbox})
        return ApplyResult(True, [iid])

    @staticmethod
    def _place_shape(base, op: AddInstance):
        """Apply the instance's intrinsic X-Y-Z rotation then translation to a
        copy of ``base`` (a cq Shape), returning the placed shape."""
        placed = base
        if op.rx:
            placed = placed.rotate((0, 0, 0), (1, 0, 0), op.rx)
        if op.ry:
            placed = placed.rotate((0, 0, 0), (0, 1, 0), op.ry)
        if op.rz:
            placed = placed.rotate((0, 0, 0), (0, 0, 1), op.rz)
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
        bad = check_mate_ports(op)
        if bad is not None:
            return _err(*bad)
        self.mates.append(mate_record(op))
        return ApplyResult(True, [])

    def _set_param(self, op: SetParam) -> ApplyResult:
        new_log, err = edit_oplog(self._oplog, op)
        if err is not None:
            return _err(*err)  # block-and-correct: self untouched
        # Replay the edited op stream onto a fresh backend; only adopt it if the
        # whole stream re-applies cleanly (block-and-correct, self untouched).
        trial = type(self)()
        for logged in new_log:
            r = trial.apply(logged)
            if not r.ok:
                return ApplyResult(False, [], r.diagnostics)
        self.__dict__.update(trial.__dict__)
        return ApplyResult(True, [])

    def _constrain(self, op: Constrain) -> ApplyResult:
        """Constrain the sketch -- and MOVE it.

        Two bugs here, and they are the same bug. ``b`` was accepted for kinds that
        have no second entity (``Constrain(kind="radius", a="e1", b="e2")`` is
        malformed, and was taken); and ``value`` was fed to a DOF counter and never
        to the geometry, so a radius constraint of 8 on an r=6 circle left it at 6
        and reported one fewer degree of freedom. The arity is now declared
        (:data:`~harnesscad.io.backends.frep.CONSTRAINT_ARITY`) and enforced, and
        the constraint is RESOLVED onto the sketch entities by the same solver frep
        uses -- so the same op moves the same geometry on both engines, which is the
        only way the differential oracle can mean anything here.
        """
        # Real DOF analysis via constraints.ConstraintGraph (rank-style, with
        # redundancy detection) rather than a bare additive heuristic.
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
        # Only couple a second entity when it lives in the same sketch's graph.
        b = op.b if (op.b is not None
                     and self.entities[op.b]["sketch"] == sid) else None
        graph.add_constraint(op.kind, op.a, b, op.value)
        self.sketches[sid]["dof"] = graph.residual_dof()
        self.sketches[sid].setdefault("constraints", []).append(op.kind)
        return ApplyResult(True, [])

    # -- features (real geometry) ------------------------------------------
    def _build_profile(self, cq, sketch: dict):
        """Realise a sketch's closed profiles as a cq.Workplane, or None."""
        wp = cq.Workplane(sketch["plane"])
        n_profiles = 0
        for eid in sketch["entities"]:
            ent = self.entities[eid]
            p = ent["params"]
            if ent["type"] == "rectangle":
                cx, cy = p["x"] + p["w"] / 2.0, p["y"] + p["h"] / 2.0
                wp = wp.moveTo(cx, cy).rect(p["w"], p["h"])
                n_profiles += 1
            elif ent["type"] == "circle":
                wp = wp.moveTo(p["cx"], p["cy"]).circle(p["r"])
                n_profiles += 1
            elif ent["type"] == "ellipse":
                wp = wp.moveTo(p["cx"], p["cy"]).ellipse(
                    p["rx"], p["ry"], rotation_angle=p.get("rotation", 0.0))
                n_profiles += 1
            elif ent["type"] == "polygon":
                pts = [(p["points"][i], p["points"][i + 1])
                       for i in range(0, len(p["points"]) - 1, 2)]
                wp = wp.polyline(pts).close()
                n_profiles += 1
            # points/lines/arcs/splines do not form closed profiles on their own
            # (a mixed line+arc wire is not chained by this profile builder) -> ignored
        return wp if n_profiles else None

    def _extrude(self, op: Extrude) -> ApplyResult:
        if op.sketch not in self.sketches:
            return _err("bad-ref", f"unknown sketch '{op.sketch}'", op.sketch)
        if not self.sketches[op.sketch]["entities"]:
            return _err("empty-sketch", f"sketch '{op.sketch}' has no profile", op.sketch)
        if op.distance == 0:
            return _err("bad-value", "extrude distance must be non-zero")
        try:
            cq = _cq()
            profile = self._build_profile(cq, self.sketches[op.sketch])
            if profile is None:
                return _err("empty-sketch",
                            f"sketch '{op.sketch}' has no closed profile to extrude",
                            op.sketch)
            solid = profile.extrude(op.distance)
            bad = _degenerate(solid, "extrude")
            if bad is not None:
                return bad
        except BackendUnavailable:
            raise
        except _err_types() as exc:  # kernel exception -> block-and-correct
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
            _cq()
            target = self._solids[-1]
            # BUG: op.edges was IGNORED and EVERY edge was filleted. A fillet on
            # the wrong edge set is a silent correctness bug (a 20x10x5 box
            # filleted r=1 on its 4 vertical edges has 10 faces / V=995.708; the
            # same box filleted on all 12 edges has 26 faces / V=971.295 -- two
            # different parts, both "ok"). op.edges is a tuple of CadQuery
            # selector strings (selectors.html); an empty tuple still means every
            # edge, so existing op streams are unchanged.
            edges = _sub_shapes(target, "edges", op.edges)
            filleted = edges.fillet(op.radius)
            bad = _degenerate(filleted, "fillet") \
                or _blend_took_effect(target, filleted, "fillet")
            if bad is not None:
                return bad
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
        """Index into ``self._solids`` of the body a feature id names.

        ``self._solids`` and the solid-producing features are pushed in lockstep,
        so the n-th solid-producing feature is the n-th solid.
        """
        # Every feature type that PUSHES a new entry onto self._solids must be
        # listed here, or Boolean(target=...) resolves to the wrong body. loft and
        # sweep are body-producing now, so they belong in this set.
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
        # BUG: op.target / op.tool were IGNORED -- a Boolean that explicitly named
        # its operands silently operated on the last two solids instead, which is
        # wrong geometry with no diagnostic. Resolve them as the frep backend does.
        ia = self._solid_index(op.target) if op.target else len(self._solids) - 2
        ib = self._solid_index(op.tool) if op.tool else len(self._solids) - 1
        if ia is None:
            return _err("bad-ref", f"unknown boolean target '{op.target}'", op.target)
        if ib is None:
            return _err("bad-ref", f"unknown boolean tool '{op.tool}'", op.tool)
        if ia == ib:
            return _err("bad-ref", "boolean target and tool are the same body")
        try:
            _cq()
            b = self._solids[ib]
            a = self._solids[ia]
            if op.kind == "union":
                result = a.union(b)
            elif op.kind == "cut":
                result = a.cut(b)
            else:
                result = a.intersect(b)
            bad = _degenerate(result, f"boolean '{op.kind}'")
            if bad is not None:
                return bad
        except BackendUnavailable:
            raise
        except _err_types() as exc:
            return _err("kernel-error", f"boolean failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "boolean", "id": fid, "kind": op.kind})
        # the two operands are consumed and replaced by their combination
        for i in sorted((ia, ib), reverse=True):
            self._solids.pop(i)
        self._solids.append(result)
        return ApplyResult(True, [fid])

    # -- extended mechanical features --------------------------------------
    def _revolve(self, op: Revolve) -> ApplyResult:
        if op.sketch not in self.sketches:
            return _err("bad-ref", f"unknown sketch '{op.sketch}'", op.sketch)
        if not self.sketches[op.sketch]["entities"]:
            return _err("empty-sketch", f"sketch '{op.sketch}' has no profile", op.sketch)
        if op.angle == 0:
            return _err("bad-value", "revolve angle must be non-zero")
        try:
            cq = _cq()
            profile = self._build_profile(cq, self.sketches[op.sketch])
            if profile is None:
                return _err("empty-sketch",
                            f"sketch '{op.sketch}' has no closed profile to revolve",
                            op.sketch)
            a = op.axis
            solid = profile.revolve(op.angle, (a[0], a[1], a[2]), (a[3], a[4], a[5]))
            bad = _degenerate(solid, "revolve")
            if bad is not None:
                return bad
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
            _cq()
            target = self._solids[-1]
            # Same bug as _fillet: op.edges was ignored and EVERY edge chamfered.
            # Workplane.chamfer(length, length2=None) (classreference.html) takes
            # an optional second setback for an asymmetric chamfer.
            edges = _sub_shapes(target, "edges", op.edges)
            chamfered = edges.chamfer(op.distance, op.distance2)
            bad = _degenerate(chamfered, "chamfer") \
                or _blend_took_effect(target, chamfered, "chamfer")
            if bad is not None:
                return bad
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
        # CAPABILITY: counterbore / countersink used to be refused outright. They
        # are first-class CadQuery ops -- Workplane.cboreHole(diameter,
        # cboreDiameter, cboreDepth, depth=None) and Workplane.cskHole(diameter,
        # cskDiameter, cskAngle, depth=None) (classreference.html); depth=None is
        # through-all. Both drill along the -normal of the workplane face.
        cbore_d = op.cbore_diameter if op.cbore_diameter is not None \
            else 2.0 * op.diameter
        cbore_z = op.cbore_depth if op.cbore_depth is not None else op.diameter
        csk_d = op.csk_diameter if op.csk_diameter is not None \
            else 2.0 * op.diameter
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
            _cq()
            target = self._solids[-1]
            # BUG: op.face_or_sketch was IGNORED -- every hole was drilled into the
            # TOP face, so Hole(face_or_sketch="<Z", ...) silently drilled the wrong
            # side. The ref may be a body alias ("solid"/"body"/"last") or a sketch
            # id, both of which mean "the default drilling face" (">Z"); anything
            # else is a CadQuery face selector.
            if ref in ("", "solid", "body", "last") or ref.startswith("sk"):
                face_sel = ">Z"
            else:
                face_sel = ref
            # centerOption="ProjectedOrigin" projects the GLOBAL origin onto the
            # face (classreference.html#cadquery.Workplane.workplane), so the local
            # (x, y) we push ARE the model's x/y. This is deliberate: CISP's
            # Hole(x, y) are absolute model coordinates, and the other two options
            # ("CenterOfMass", "CenterOfBoundBox") would re-origin the hole on the
            # face's own centre and silently move it on any off-centre face.
            wp = (_sub_shapes(target, "faces", (face_sel,))
                        .workplane(centerOption="ProjectedOrigin")
                        .pushPoints([(op.x, op.y)]))
            depth = None if op.through else op.depth
            if op.kind == "counterbore":
                result = wp.cboreHole(op.diameter, cbore_d, cbore_z, depth)
            elif op.kind == "countersink":
                result = wp.cskHole(op.diameter, csk_d, op.csk_angle, depth)
            else:
                result = wp.hole(op.diameter, depth)
            bad = _degenerate(result, "hole")
            if bad is not None:
                return bad
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

    def _shell(self, op: Shell) -> ApplyResult:
        if not self.solid_present or not self._solids:
            return _err("no-solid", "shell requires an existing solid")
        if op.thickness <= 0:
            return _err("bad-value", f"shell thickness must be > 0 (got {op.thickness})")
        if op.kind not in ("arc", "intersection"):
            return _err("bad-value",
                        f"unknown shell kind '{op.kind}' (arc | intersection)")
        try:
            _cq()
            target = self._solids[-1]
            # SHELL SIGN CONVENTION -- this is documented, not luck.
            # Workplane.shell(thickness, kind='arc') (classreference.html):
            #   "Negative values shell inwards, positive values shell outwards."
            # examples.html#shelling-to-create-thin-features is blunter: "To shell
            # an object and 'hollow out' the inside pass a NEGATIVE thickness"; a
            # positive one "wraps an object ... and the original object will be the
            # 'hollowed out' portion" -- i.e. it GROWS the part (60x40x20 -> 66x46x23).
            # So we pass -thickness. That is the whole reason this backend got the
            # bbox right while frep/blender did not.
            #
            # WHICH FACES: Workplane.shell removes the faces ON THE STACK
            # (`faces = [f for f in self.objects if isinstance(f, Face)]`).
            #   * op.faces non-empty -> those faces are REMOVED (opened).
            #   * op.faces EMPTY     -> a CLOSED HOLLOW: a sealed internal void with
            #     no opening. The free function `hollow` documents exactly this --
            #     "if no faces provided a watertight solid will be constructed".
            # BUG: we used to hardcode ">Z" for the empty case, silently opening the
            # top face on every shell that did not name one. That is what made this
            # backend disagree with frep/blender; it was OUR bug, not a semantic
            # difference. Never default a face open.
            before_bb = target.val().BoundingBox()
            before_v = solid_volume(target)
            if join_selectors(op.faces) is None:
                shelled = target.shell(-abs(op.thickness), kind=op.kind)
            else:
                faces = _sub_shapes(target, "faces", op.faces)
                shelled = faces.shell(-abs(op.thickness), kind=op.kind)
            bad = _degenerate(shelled, "shell")
            if bad is not None:
                return bad
            # POSTCONDITION -- a shell HOLLOWS, it never grows and never adds
            # material. OCCT does not always refuse an over-thick wall: shelling a
            # 20mm cube by 50mm returns a solid rather than raising, and it used to
            # be committed. Assert the invariant instead of trusting the kernel.
            after_bb = shelled.val().BoundingBox()
            grew = [
                after_bb.xlen - before_bb.xlen,
                after_bb.ylen - before_bb.ylen,
                after_bb.zlen - before_bb.zlen,
            ]
            if max(grew) > 1e-6:
                return _err("degenerate",
                            "shell grew the part's bounding box by %s -- a shell "
                            "must hollow inward and leave the outer surface where "
                            "it was" % [round(g, 6) for g in grew])
            if solid_volume(shelled) >= before_v - MIN_VOLUME:
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
        # BUG: the direction was used RAW, so a non-unit direction scaled the
        # spacing (direction=(2,0,0), spacing=10 stepped 20mm here but 10mm on
        # frep). Normalise, exactly as frep does, so spacing means spacing.
        d = list(op.direction) + [0.0, 0.0, 0.0]
        norm = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        if norm == 0.0:
            return _err("bad-value", "linear_pattern direction is degenerate")
        ux, uy, uz = d[0] / norm, d[1] / norm, d[2] / norm
        try:
            _cq()
            result = self._solids[index]
            for i in range(1, op.count):
                off = (ux * op.spacing * i, uy * op.spacing * i, uz * op.spacing * i)
                result = result.union(base.translate(off))
            bad = _degenerate(result, "linear_pattern")
            if bad is not None:
                return bad
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
            _cq()
            a = op.axis
            # PITCH RULE -- copied from CadQuery's own Workplane.polarArray(fill=True)
            # (cq.py; classreference.html#cadquery.Workplane.polarArray):
            #   if abs(math.remainder(angle, 360)) < TOL: angle = angle / count
            #   else:                                     angle = angle / (count - 1)
            # A full turn divides evenly (the last copy would land on the first);
            # any other arc is spanned INCLUSIVELY, start and end. We used
            # angle/count unconditionally, so a 180-degree 4-up pattern spanned
            # 135 degrees where CadQuery (and every CAD package) spans 180.
            if abs(math.remainder(float(op.angle), 360.0)) < 1e-9:
                step = float(op.angle) / float(op.count)
            else:
                step = float(op.angle) / float(op.count - 1)
            result = self._solids[index]
            for i in range(1, op.count):
                rotated = base.rotate((a[0], a[1], a[2]), (a[3], a[4], a[5]), step * i)
                result = result.union(rotated)
            bad = _degenerate(result, "circular_pattern")
            if bad is not None:
                return bad
        except BackendUnavailable:
            raise
        except _err_types() as exc:
            return _err("kernel-error", f"circular_pattern failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "circular_pattern", "id": fid, "count": op.count})
        self._solids[index] = result
        return ApplyResult(True, [fid])

    @staticmethod
    def _place(base, tx, ty, tz, rx, ry, rz):
        """Rotate ``base`` about X, then Y, then Z (world axes through the origin),
        then translate -- the one convention ops.Transform / ops.PatternTransform
        document, and the same sequence :meth:`_place_shape` uses for AddInstance."""
        placed = base
        if rx:
            placed = placed.rotate((0, 0, 0), (1, 0, 0), rx)
        if ry:
            placed = placed.rotate((0, 0, 0), (0, 1, 0), ry)
        if rz:
            placed = placed.rotate((0, 0, 0), (0, 0, 1), rz)
        if tx or ty or tz:
            placed = placed.translate((tx, ty, tz))
        return placed

    def _transform(self, op: Transform) -> ApplyResult:
        if not self.solid_present or not self._solids:
            return _err("no-solid", "transform requires an existing solid")
        target, bad = self._feature_target(op.feature_or_body)
        if bad is not None:
            return bad
        index, base = target
        try:
            _cq()
            placed = self._place(base, op.tx, op.ty, op.tz, op.rx, op.ry, op.rz)
            # Match frep's _graft: moving the body's CURRENT state moves it in
            # place; moving an earlier named feature adds a moved copy.
            if base is self._solids[index]:
                result = placed
            else:
                result = self._solids[index].union(placed)
            bad = _degenerate(result, "transform")
            if bad is not None:
                return bad
        except BackendUnavailable:
            raise
        except _err_types() as exc:
            return _err("kernel-error", f"transform failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "transform", "id": fid})
        self._solids[index] = result
        return ApplyResult(True, [fid])

    def _pattern_transform(self, op: PatternTransform) -> ApplyResult:
        if not self.solid_present or not self._solids:
            return _err("no-solid", "pattern_transform requires an existing solid")
        pts = tuple(float(v) for v in op.placements)
        if len(pts) < 6 or len(pts) % 6 != 0:
            return _err("bad-value",
                        "pattern_transform placements must be a non-empty flat "
                        "tuple of six-float (tx,ty,tz,rx,ry,rz) instances")
        target, bad = self._feature_target(op.feature)
        if bad is not None:
            return bad
        index, base = target
        try:
            _cq()
            result = None
            for i in range(0, len(pts), 6):
                placed = self._place(base, *pts[i:i + 6])
                result = placed if result is None else result.union(placed)
            if base is not self._solids[index]:
                result = self._solids[index].union(result)
            bad = _degenerate(result, "pattern_transform")
            if bad is not None:
                return bad
        except BackendUnavailable:
            raise
        except _err_types() as exc:
            return _err("kernel-error", f"pattern_transform failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "pattern_transform", "id": fid,
                              "count": len(pts) // 6})
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
            _cq()
            result = self._solids[index].union(base.mirror(op.plane))
            bad = _degenerate(result, "mirror")
            if bad is not None:
                return bad
        except BackendUnavailable:
            raise
        except _err_types() as exc:
            return _err("kernel-error", f"mirror failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "mirror", "id": fid, "plane": op.plane})
        self._solids[index] = result
        return ApplyResult(True, [fid])

    # -- loft / sweep / draft (real geometry) -------------------------------
    def _outer_wire(self, cq, sketch: dict, offset: float = 0.0):
        """The sketch's outer closed wire, on its own plane, offset along its normal.

        ``Workplane.workplane(offset=...)`` (classreference.html) shifts the plane
        along its normal, which is how two profiles sketched on the SAME plane get
        the separation a loft needs.
        """
        wp = cq.Workplane(sketch["plane"])
        if offset:
            wp = wp.workplane(offset=offset)
        n = 0
        for eid in sketch["entities"]:
            ent = self.entities[eid]
            p = ent["params"]
            if ent["type"] == "rectangle":
                wp = wp.moveTo(p["x"] + p["w"] / 2.0, p["y"] + p["h"] / 2.0)
                wp = wp.rect(p["w"], p["h"])
                n += 1
            elif ent["type"] == "circle":
                wp = wp.moveTo(p["cx"], p["cy"]).circle(p["r"])
                n += 1
        if not n:
            return None
        wires = wp.wires().vals()
        return wires[0] if wires else None

    def _loft(self, op: Loft) -> ApplyResult:
        if len(op.sketches) < 2:
            return _err("bad-value", "loft requires at least two sketches")
        for sid in op.sketches:
            if sid not in self.sketches:
                return _err("bad-ref", f"unknown sketch '{sid}'", sid)
            if not self.sketches[sid]["entities"]:
                return _err("empty-sketch", f"sketch '{sid}' has no profile", sid)
        try:
            cq = _cq()
            offsets = list(op.offsets) + [0.0] * len(op.sketches)
            wires = []
            for i, sid in enumerate(op.sketches):
                w = self._outer_wire(cq, self.sketches[sid], float(offsets[i]))
                if w is None:
                    return _err("empty-sketch",
                                f"sketch '{sid}' has no closed profile to loft", sid)
                wires.append(w)
            # Solid.makeLoft(listOfWire, ruled=False) (classreference.html).
            solid = cq.Workplane("XY").newObject(
                [cq.Solid.makeLoft(wires, bool(op.ruled))])
            # Coincident profiles (all offsets 0 on one plane) loft to zero volume
            # -- caught here rather than committed as a phantom body.
            bad = _degenerate(solid, "loft")
            if bad is not None:
                return bad
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

    def _build_path(self, cq, sketch: dict):
        """A sketch's open/closed path wire: chained lines, or a single circle."""
        lines = [self.entities[e] for e in sketch["entities"]
                 if self.entities[e]["type"] == "line"]
        circles = [self.entities[e] for e in sketch["entities"]
                   if self.entities[e]["type"] == "circle"]
        wp = cq.Workplane(sketch["plane"])
        if lines:
            p = lines[0]["params"]
            wp = wp.moveTo(p["x1"], p["y1"]).lineTo(p["x2"], p["y2"])
            for ent in lines[1:]:
                q = ent["params"]
                wp = wp.lineTo(q["x2"], q["y2"])
            return wp.wire()
        if circles:
            p = circles[0]["params"]
            return wp.moveTo(p["cx"], p["cy"]).circle(p["r"])
        return None

    def _sweep(self, op: Sweep) -> ApplyResult:
        for sid in (op.sketch, op.path):
            if sid not in self.sketches:
                return _err("bad-ref", f"unknown sketch '{sid}'", sid)
            if not self.sketches[sid]["entities"]:
                return _err("empty-sketch", f"sketch '{sid}' has no profile", sid)
        try:
            cq = _cq()
            profile = self._build_profile(cq, self.sketches[op.sketch])
            if profile is None:
                return _err("empty-sketch",
                            f"sweep profile sketch '{op.sketch}' has no closed "
                            f"profile", op.sketch)
            path = self._build_path(cq, self.sketches[op.path])
            if path is None:
                return _err("empty-sketch",
                            f"sweep path sketch '{op.path}' has no line or circle "
                            f"path", op.path)
            # Workplane.sweep(path, ...) (classreference.html): sweeps the
            # un-extruded wires in the chain along `path`. combine defaults to True
            # but there is no context solid here, so it just returns the swept body.
            solid = profile.sweep(path)
            bad = _degenerate(solid, "sweep")
            if bad is not None:
                return bad
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

        CadQuery's Workplane has no draft method (only ``extrude(taper=)`` /
        ``cutBlind(taper=)``, which draft an extrusion as it is created, not an
        existing solid), so we drive the OCCT algorithm the kernel exposes
        directly. ``op.faces`` are CadQuery selector strings for the faces to
        taper. ``op.neutral_plane`` is the surface that stays put; it accepts
        either a datum-plane NAME ("XY" / "XZ" / "YZ", through the global origin)
        or a CadQuery face selector ("<Z" = the bottom face). It defaults to
        "<Z". The pull direction is that plane's normal, pointing into the body.
        """
        if not self.solid_present or not self._solids:
            return _err("no-solid", "draft requires an existing solid")
        if op.angle == 0:
            return _err("bad-value", "draft angle must be non-zero")
        if abs(op.angle) >= 90.0:
            return _err("bad-value", f"draft angle must be < 90 deg (got {op.angle})")
        try:
            cq = _cq()
            from OCP.BRepOffsetAPI import BRepOffsetAPI_DraftAngle
            from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

            target = self._solids[-1]
            spec = str(op.neutral_plane).strip() or "<Z"
            # A datum-plane NAME (through the global origin) or a face selector.
            datum = {"XY": (0.0, 0.0, 1.0), "YX": (0.0, 0.0, 1.0),
                     "XZ": (0.0, 1.0, 0.0), "ZX": (0.0, 1.0, 0.0),
                     "YZ": (1.0, 0.0, 0.0), "ZY": (1.0, 0.0, 0.0)}
            if spec.upper() in datum:
                d = datum[spec.upper()]
                origin = (0.0, 0.0, 0.0)
                pull_v = d                      # taper away from the datum plane
            else:
                neutral = _sub_shapes(target, "faces", (spec,)).vals()[0]
                nc, nn = neutral.Center(), neutral.normalAt()
                origin = (nc.x, nc.y, nc.z)
                pull_v = (-nn.x, -nn.y, -nn.z)  # the face's normal, INTO the body
            pull = gp_Dir(*pull_v)
            # The neutral plane is perpendicular to the pull direction.
            plane = gp_Pln(gp_Pnt(*origin), pull)

            # Default face set: the side walls -- every face whose normal is
            # PERPENDICULAR to the pull direction (those are the faces a mould
            # release needs; the neutral face and its opposite are untouched).
            # Computed here rather than as a selector string because CadQuery's own
            # string-selector grammar cannot express it: it parses "not(a or b)"
            # but rejects "not(a) and not(b)" ("Expected end of text, found 'and'").
            if join_selectors(op.faces) is not None:
                faces = _sub_shapes(target, "faces", op.faces).vals()
            else:
                faces = [f for f in target.faces().vals()
                         if abs(f.normalAt().dot(
                             cq.Vector(*pull_v))) <= 1e-6]
            builder = BRepOffsetAPI_DraftAngle(target.val().wrapped)
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
            drafted = cq.Workplane("XY").newObject(
                [cq.Shape.cast(builder.Shape())])
            bad = _degenerate(drafted, "draft")
            if bad is not None:
                return bad
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

    # -- primitives / split / thicken (real geometry) ----------------------
    def _primitive(self, op: Primitive) -> ApplyResult:
        """A solid primitive via OCCT's own builders (BRepPrimAPI, exposed as
        ``cq.Solid.make*``). Placed at the origin; the standalone body's absolute
        position is immaterial (only its size and volume are measured)."""
        shape = str(op.shape).lower()
        if shape in ("box", "wedge") and (op.dx <= 0 or op.dy <= 0 or op.dz <= 0):
            return _err("bad-value", f"{shape} dx, dy, dz must be > 0")
        if shape in ("sphere", "torus") and op.r <= 0:
            return _err("bad-value", f"{shape} radius r must be > 0")
        if shape == "torus" and op.r2 <= 0:
            return _err("bad-value", "torus minor radius r2 must be > 0")
        if shape in ("cylinder", "cone") and (op.r <= 0 or op.h <= 0):
            return _err("bad-value", f"{shape} r and h must be > 0")
        if shape not in ("box", "sphere", "cylinder", "cone", "torus", "wedge"):
            return _err("bad-value",
                        f"unknown primitive shape '{op.shape}' (box | sphere | "
                        "cylinder | cone | torus | wedge)")
        try:
            cq = _cq()
            if shape == "box":
                solid = cq.Workplane("XY").newObject(
                    [cq.Solid.makeBox(op.dx, op.dy, op.dz)])
            elif shape == "sphere":
                solid = cq.Workplane("XY").newObject([cq.Solid.makeSphere(op.r)])
            elif shape == "cylinder":
                solid = cq.Workplane("XY").newObject(
                    [cq.Solid.makeCylinder(op.r, op.h)])
            elif shape == "cone":
                solid = cq.Workplane("XY").newObject(
                    [cq.Solid.makeCone(op.r, op.r2, op.h)])
            elif shape == "torus":
                solid = cq.Workplane("XY").newObject(
                    [cq.Solid.makeTorus(op.r, op.r2)])
            else:  # wedge: a right-triangular prism (triangle in X-Z, extruded +Y)
                solid = (cq.Workplane("XZ").polyline(
                    [(0.0, 0.0), (op.dx, 0.0), (0.0, op.dz)]).close()
                    .extrude(op.dy))
            bad = _degenerate(solid, f"primitive {shape}")
            if bad is not None:
                return bad
        except BackendUnavailable:
            raise
        except _err_types() as exc:
            return _err("kernel-error", f"primitive {shape} failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "primitive", "id": fid, "shape": shape})
        self._solids.append(solid)
        self.solid_present = True
        return ApplyResult(True, [fid])

    @staticmethod
    def _halfspace_box(cq, plane: str, offset: float, positive: bool, size: float):
        """A large box filling one side of a datum plane at ``offset``."""
        box = cq.Solid.makeBox(2.0 * size, 2.0 * size, 2.0 * size)
        near = offset if positive else offset - 2.0 * size
        if plane == "XY":
            box = box.translate((-size, -size, near))
        elif plane == "XZ":
            box = box.translate((-size, near, -size))
        else:  # YZ
            box = box.translate((near, -size, -size))
        return cq.Workplane("XY").newObject([box])

    def _split(self, op: Split) -> ApplyResult:
        if not self.solid_present or not self._solids:
            return _err("no-solid", "split requires an existing solid")
        plane = str(op.plane).upper()
        if plane not in ("XY", "XZ", "YZ"):
            return _err("bad-value", f"unknown split plane '{op.plane}'")
        if op.keep not in ("positive", "negative", "both"):
            return _err("bad-value",
                        f"unknown split keep '{op.keep}' (positive|negative|both)")
        try:
            cq = _cq()
            target = self._solids[-1]
            bb = target.val().BoundingBox()
            size = 2.0 * max(bb.xlen, bb.ylen, bb.zlen) + 100.0
            pos_cut = self._halfspace_box(cq, plane, float(op.offset), True, size)
            neg_cut = self._halfspace_box(cq, plane, float(op.offset), False, size)
            keep_pos = target.cut(neg_cut)   # remove the negative side
            keep_neg = target.cut(pos_cut)   # remove the positive side
            primary = keep_pos if op.keep in ("positive", "both") else keep_neg
            bad = _degenerate(primary, "split")
            if bad is not None:
                return bad
            if op.keep == "both":
                bad = _degenerate(keep_neg, "split (negative half)")
                if bad is not None:
                    return bad
        except BackendUnavailable:
            raise
        except _err_types() as exc:
            return _err("kernel-error", f"split failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "split", "id": fid, "plane": plane,
                              "keep": op.keep})
        self._solids[-1] = primary
        if op.keep == "both":
            self._solids.append(keep_neg)
        return ApplyResult(True, [fid])

    def _thicken(self, op: Thicken) -> ApplyResult:
        """Grow / shrink the whole solid by ``thickness`` (OCCT MakeOffsetShape)."""
        if not self.solid_present or not self._solids:
            return _err("no-solid", "thicken requires an existing solid")
        if op.thickness == 0:
            return _err("bad-value", "thicken thickness must be non-zero")
        if any(str(f).strip() for f in (op.faces or ())):
            return _err("unsupported-op",
                        "this backend offsets the whole outer surface; a per-face "
                        "thicken (Thicken.faces) is not wired here")
        try:
            cq = _cq()
            target = self._solids[-1]
            from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape
            from OCP.BRepOffset import BRepOffset_Mode
            from OCP.GeomAbs import GeomAbs_JoinType
            mk = BRepOffsetAPI_MakeOffsetShape()
            mk.PerformByJoin(target.val().wrapped, thicken_delta(op), 1e-3,
                             BRepOffset_Mode.BRepOffset_Skin, False, False,
                             GeomAbs_JoinType.GeomAbs_Arc, False)
            result = cq.Workplane("XY").newObject([cq.Shape.cast(mk.Shape())])
            bad = _degenerate(result, "thicken")
            if bad is not None:
                return bad
        except BackendUnavailable:
            raise
        except _err_types() as exc:
            return _err("kernel-error", f"thicken failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "thicken", "id": fid,
                              "thickness": op.thickness,
                              "both": bool(op.both)})
        self._solids[-1] = result
        return ApplyResult(True, [fid])

    def regenerate(self) -> List[Diagnostic]:
        return []  # incremental backend; nothing to rebuild

    # -- combined-shape helpers --------------------------------------------
    def _combined(self):
        """A single cq Shape (Compound) of all current solids, or None."""
        if not self._solids:
            return None
        cq = _cq()
        shapes = []
        for wp in self._solids:
            shapes.extend(wp.solids().vals())
        if not shapes:
            return None
        if len(shapes) == 1:
            return shapes[0]
        return cq.Compound.makeCompound(shapes)

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
            # Real OCCT topology check.
            from OCP.BRepCheck import BRepCheck_Analyzer
            is_valid = bool(BRepCheck_Analyzer(shape.wrapped).IsValid())
        except Exception:  # noqa: BLE001 - fall back to cq's own check
            try:
                is_valid = bool(shape.isValid())
            except Exception:  # noqa: BLE001
                is_valid = False
        try:
            # A closed, positive-volume OCCT solid is watertight/manifold.
            watertight = is_valid and shape.Volume() > 1e-12
        except Exception:  # noqa: BLE001
            watertight = is_valid
        return {
            "manifold": is_valid,     # OCCT valid solids are 2-manifold
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
        """The parts + mates view consumed by AssemblyCheck / InterferenceCheck.

        Each placed instance is a part carrying its placement ``transform``, a
        real OCCT ``shape`` (the body snapshot, placed) and its axis-aligned
        ``bbox`` — so InterferenceCheck can run the exact boolean-common narrow
        phase and AssemblyCheck can count residual DOF. Empty ({}) when nothing
        has been placed, so verifiers INFO-skip exactly like before."""
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
                part["shape"] = inst["shape"].val() if hasattr(inst["shape"], "val") \
                    else inst["shape"]
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
            bb = shape.BoundingBox()
            return {
                "volume": float(shape.Volume()),
                "bbox": [float(bb.xlen), float(bb.ylen), float(bb.zlen)],
            }
        except Exception:  # noqa: BLE001
            return {"volume": 0.0, "bbox": [0.0, 0.0, 0.0]}

    def _metrics(self, density: float = 1.0) -> dict:
        """Mass properties via OCCT GProp_GProps.

        Returns volume, mass (= volume * density), surface_area, bbox and
        center_of_mass. Returns {} when there is no solid (so callers INFO-skip,
        matching the stub which always returns {}).
        """
        try:
            shape = self._combined()
        except Exception:  # noqa: BLE001
            shape = None
        if shape is None:
            return {}
        try:
            from OCP.GProp import GProp_GProps
            from OCP.BRepGProp import BRepGProp
            vprops = GProp_GProps()
            BRepGProp.VolumeProperties_s(shape.wrapped, vprops)
            volume = float(vprops.Mass())
            com = vprops.CentreOfMass()
            sprops = GProp_GProps()
            BRepGProp.SurfaceProperties_s(shape.wrapped, sprops)
            surface_area = float(sprops.Mass())
            bb = shape.BoundingBox()
            return {
                "volume": volume,
                "mass": volume * density,
                "surface_area": surface_area,
                "bbox": [float(bb.xlen), float(bb.ylen), float(bb.zlen)],
                "center_of_mass": [float(com.X()), float(com.Y()), float(com.Z())],
                # Topology counts: how a fillet/chamfer PROVES it rounded an edge
                # (a blended box gains a face per edge and a patch per corner).
                "faces": len(shape.Faces()),
                "edges": len(shape.Edges()),
                "vertices": len(shape.Vertices()),
                "solids": len(shape.Solids()),
            }
        except Exception:  # noqa: BLE001
            return {}

    # -- export -------------------------------------------------------------
    #: Tessellation deflection used for every meshed export (STL, ...).
    #:
    #: exporters.export(w, fname, exportType, tolerance=0.1, angularTolerance=0.1)
    #: (importexport.html): ``tolerance`` is the LINEAR deflection in model units,
    #: ``angularTolerance`` is in RADIANS.
    #:
    #: THE TRAP: exporters.export defaults tolerance=0.1, but the Shape.exportStl
    #: it delegates to defaults tolerance=1e-3 ("a good starting point for a range
    #: of cases"). export() passes its own coarse default straight through, so any
    #: mesh produced via export() is 100x coarser than the documented default --
    #: and we were passing NO tolerance at all, so every mesh-based verifier and
    #: the differential oracle silently ran on that coarse tessellation.
    #:
    #: Both bounds must be pinned. The angular one is what actually binds on
    #: curved faces: at cq's 0.1 rad (5.7 deg) our reference plate meshes to 520
    #: facets whether the linear tolerance is 0.1 or 0.01 -- tightening the linear
    #: bound ALONE changes nothing. Measured mesh-volume error vs analytic on a
    #: 40x24x8 plate with a 6mm bore:
    #:     cq export() default (0.1 / 0.1 rad)  ->  520 facets, 0.0013% error
    #:     ours              (1e-3 / 0.05 rad)  -> 1024 facets, 0.0003% error
    LINEAR_DEFLECTION = 1e-3      # mm  (Shape.exportStl's own documented default)
    ANGULAR_DEFLECTION = 0.05     # radians (~2.9 deg)

    #: STEP application protocol. OCCT's default is AP214IS, but it is a *global*
    #: Interface_Static setting that any other OCCT user in-process can change, so
    #: we set it explicitly on every export rather than inheriting whatever the
    #: process happens to be in (importexport.html / OCCT STEP writer).
    STEP_SCHEMA = "AP214IS"

    def export(self, fmt: str, tolerance: Optional[float] = None,
               angular_tolerance: Optional[float] = None):
        fmt = str(fmt).lower()
        if fmt not in self.FORMATS:
            raise ValueError("the cadquery backend cannot export '%s' (supported: %s)"
                             % (fmt, ", ".join(self.FORMATS)))
        cq = _cq()  # raises BackendUnavailable when OCCT is missing
        shape = self._combined()
        if shape is None:
            raise ValueError("nothing to export: no solid present")
        from cadquery import exporters
        lin = self.LINEAR_DEFLECTION if tolerance is None else float(tolerance)
        ang = (self.ANGULAR_DEFLECTION if angular_tolerance is None
               else float(angular_tolerance))
        if lin <= 0 or ang <= 0:
            raise ValueError("export tolerances must be > 0")
        wp = cq.Workplane("XY").add(shape)
        if fmt == "step":
            from OCP.Interface import Interface_Static
            Interface_Static.SetCVal_s("write.step.schema", self.STEP_SCHEMA)
            # write_pcurves=True keeps the parametric curves (a lossless B-rep
            # round-trip); precision_mode=0 uses the shape's own tolerances.
            return self._export_text(exporters, wp, "STEP", ".step",
                                     opt={"write_pcurves": True,
                                          "precision_mode": 0})
        if fmt == "stl":
            return self._export_stl(shape, lin, ang)
        if fmt == "brep":
            return self._export_brep(shape)
        return self._export_iges(shape)

    @staticmethod
    def _export_stl(shape, tolerance: float, angular_tolerance: float) -> str:
        """ASCII STL at a KNOWN, ABSOLUTE tessellation deflection.

        We drive ``Shape.exportStl(fileName, tolerance, angularTolerance, ascii,
        relative, parallel)`` directly instead of ``exporters.export(...,
        exportType="STL")``, because that wrapper does
        ``shape.exportStl(fname, tolerance, angularTolerance, useascii)`` and so
        leaves TWO defaults we must not accept:

        * ``relative=True`` -- the linear tolerance is then scaled per edge rather
          than being the deflection in model units importexport.html documents. It
          is also barely monotonic: on our reference plate, tolerance 0.5 and 0.05
          both produced 44 facets. With relative=False the same sweep gives
          48 -> 116 -> 324 -> 992 facets, which is the predictable behaviour the
          downstream mesh checks need.
        * no triangulation reset -- OCCT CACHES the mesh on the TopoDS_Shape, so
          the SECOND export of a shape silently reuses the FIRST export's mesh and
          ignores the new tolerance entirely (measured: 992 facets returned for
          every tolerance from 0.0005 to 0.5). BRepTools.Clean_s drops the cached
          triangulation so each export really re-tessellates.

        ASCII (not cq's default binary) because the caller reads the result back as
        text; binary bytes do not survive the utf-8 decode.
        """
        from OCP.BRepTools import BRepTools

        fd, path = tempfile.mkstemp(suffix=".stl")
        os.close(fd)
        try:
            BRepTools.Clean_s(shape.wrapped)   # drop any cached triangulation
            shape.exportStl(path, tolerance, angular_tolerance, True, False)
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    @staticmethod
    def _export_brep(shape) -> str:
        """The native OCCT B-rep serialisation (lossless: the real kernel data)."""
        fd, path = tempfile.mkstemp(suffix=".brep")
        os.close(fd)
        try:
            shape.exportBrep(path)
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    @staticmethod
    def _export_iges(shape) -> str:
        """IGES via the OCCT IGESControl_Writer (cq's exporters have no IGES)."""
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
    def _export_text(exporters, wp, export_type: str, suffix: str, opt=None,
                     tolerance: Optional[float] = None,
                     angular_tolerance: Optional[float] = None) -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        kw = {}
        if tolerance is not None:
            kw["tolerance"] = tolerance
        if angular_tolerance is not None:
            kw["angularTolerance"] = angular_tolerance
        try:
            exporters.export(wp, path, exportType=export_type, opt=opt, **kw)
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    # -- content digest -----------------------------------------------------
    def state_digest(self) -> str:
        """Stable hash of a canonical geometry descriptor.

        Uses rounded volume + bbox dims + face/edge/vertex/solid counts, so
        replaying identical ops yields an identical digest (deterministic
        replay), while floating-point noise is rounded away.
        """
        descriptor = {
            "sketch_count": len(self.sketches),
            "entity_count": len(self.entities),
            "feature_count": len(self.features),
            "solid_present": self.solid_present,
            # Assembly placements + joints, plus the canonical op stream so any
            # SetParam edit changes the digest deterministically.
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
                bb = shape.BoundingBox()
                descriptor["geom"] = {
                    "volume": round(float(shape.Volume()), 6),
                    "bbox": [round(float(bb.xlen), 6),
                             round(float(bb.ylen), 6),
                             round(float(bb.zlen), 6)],
                    "faces": len(shape.Faces()),
                    "edges": len(shape.Edges()),
                    "vertices": len(shape.Vertices()),
                    "solids": len(shape.Solids()),
                }
            except Exception:  # noqa: BLE001
                descriptor["geom"] = None
        blob = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


def _err_types():
    """Exception types treated as recoverable kernel errors (block-and-correct).

    Broad on purpose: OCCT surfaces failures as a wide variety of exception
    types (StdFail_*, Standard_*, ValueError, RuntimeError, ...). We never want
    a kernel hiccup to mutate state or crash the loop.
    """
    return Exception
