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
import tempfile
import os
from typing import List, Optional

from cisp.ops import (
    CONSTRAINT_DOF, PRIMITIVE_DOF,
    Op, NewSketch, AddPoint, AddLine, AddCircle, AddRectangle,
    Constrain, Extrude, Fillet, Boolean,
    Revolve, Chamfer, Hole, Shell, Draft,
    Loft, Sweep, LinearPattern, CircularPattern, Mirror,
    AddInstance, Mate, SetParam,
    canonical_json, edit_oplog,
)
from checks_assembly import mate_dof
from verify import Diagnostic, Severity
from backends.base import ApplyResult
from constraints import ConstraintGraph


def _err(code: str, msg: str, where: Optional[str] = None) -> ApplyResult:
    return ApplyResult(False, [], [Diagnostic(Severity.ERROR, code, msg, where)])


def _cq():
    """Lazy import of cadquery so this module loads without OCCT installed."""
    import cadquery  # noqa: WPS433 (deliberately local / lazy)
    return cadquery


class CadQueryBackend:
    def __init__(self) -> None:
        self.reset()

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        self.sketches: dict = {}      # sid -> {plane, entities:[eid], dof}
        self.entities: dict = {}      # eid -> {type, sketch, params}
        self.features: list = []      # [{type, id, ...}]
        self.instances: list = []     # [{id, part, transform, shape, bbox}]
        self.mates: list = []         # [{kind, a, b, value}]
        self.solid_present = False
        self._solids: list = []       # list of cq.Workplane, each a real solid
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
        result = self._dispatch(op)
        if result.ok:
            self._oplog.append(op)
        return result

    def _dispatch(self, op: Op) -> ApplyResult:
        if isinstance(op, NewSketch):
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
            return self._unsupported_feature(
                "draft", faces_ok=True,
                msg="draft angle is not yet wired on this CadQuery/OCCT build")
        if isinstance(op, Loft):
            return self._loft_unsupported(op)
        if isinstance(op, Sweep):
            return self._sweep_unsupported(op)
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
        self.mates.append({"kind": op.kind, "a": op.a, "b": op.b, "value": op.value})
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
        # Real DOF analysis via constraints.ConstraintGraph (rank-style, with
        # redundancy detection) rather than a bare additive heuristic.
        if op.kind not in CONSTRAINT_DOF:
            return _err("bad-value", f"unknown constraint kind '{op.kind}'")
        if op.kind in ("distance", "radius") and op.value is None:
            return _err("bad-value", f"'{op.kind}' constraint requires a value")
        if op.a not in self.entities:
            return _err("bad-ref", f"unknown entity '{op.a}'", op.a)
        if op.b is not None and op.b not in self.entities:
            return _err("bad-ref", f"unknown entity '{op.b}'", op.b)
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
            # points/lines do not form closed profiles on their own -> ignored
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
            if solid.val() is None or not solid.solids().vals():
                return _err("degenerate", "extrude produced no solid")
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
            # Edge ids are opaque across backends; fillet all edges of the
            # current solid (a real OCCT fillet). Too-large radius -> kernel
            # exception -> block-and-correct below.
            filleted = target.edges().fillet(op.radius)
            if not filleted.solids().vals():
                return _err("degenerate", "fillet produced no solid")
        except _err_types() as exc:
            return _err("kernel-error", f"fillet failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "fillet", "id": fid, "edges": list(op.edges)})
        self._solids[-1] = filleted
        return ApplyResult(True, [fid])

    def _boolean(self, op: Boolean) -> ApplyResult:
        if op.kind not in ("union", "cut", "intersect"):
            return _err("bad-value", f"unknown boolean kind '{op.kind}'")
        if len(self.features) < 2 or len(self._solids) < 2:
            return _err("no-solid", "boolean requires two solids")
        try:
            _cq()
            b = self._solids[-1]
            a = self._solids[-2]
            if op.kind == "union":
                result = a.union(b)
            elif op.kind == "cut":
                result = a.cut(b)
            else:
                result = a.intersect(b)
            if not result.solids().vals():
                return _err("degenerate", f"boolean '{op.kind}' produced no solid")
        except _err_types() as exc:
            return _err("kernel-error", f"boolean failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "boolean", "id": fid, "kind": op.kind})
        # replace the two operands with their combination
        self._solids[-2:] = [result]
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
            if not solid.solids().vals():
                return _err("degenerate", "revolve produced no solid")
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
        try:
            _cq()
            target = self._solids[-1]
            # Edge ids are opaque across backends; chamfer all edges (a real OCCT
            # chamfer). Too-large a setback -> kernel exception -> block-and-correct.
            chamfered = target.edges().chamfer(op.distance)
            if not chamfered.solids().vals():
                return _err("degenerate", "chamfer produced no solid")
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
        if op.kind != "simple":
            return _err("not-yet-supported",
                        f"hole kind '{op.kind}' is not yet realised (only 'simple')")
        ref = op.face_or_sketch
        if ref.startswith("sk") and ref not in self.sketches:
            return _err("bad-ref", f"unknown sketch '{ref}'", ref)
        if not self.solid_present or not self._solids:
            return _err("no-solid", "hole requires an existing solid to cut")
        try:
            _cq()
            target = self._solids[-1]
            wp = (target.faces(">Z")
                        .workplane(centerOption="ProjectedOrigin")
                        .pushPoints([(op.x, op.y)]))
            if op.through:
                result = wp.hole(op.diameter)
            else:
                result = wp.hole(op.diameter, op.depth)
            if not result.solids().vals():
                return _err("degenerate", "hole produced no solid")
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
        try:
            _cq()
            target = self._solids[-1]
            # Remove the top face and hollow inward by `thickness` (a real OCCT
            # MakeThickSolid). Too-thick a wall -> kernel exception -> rollback.
            shelled = target.faces(">Z").shell(-abs(op.thickness))
            if not shelled.solids().vals():
                return _err("degenerate", "shell produced no solid")
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
        if op.feature and op.feature not in self._feature_ids():
            return _err("bad-ref", f"unknown feature '{op.feature}'", op.feature)
        try:
            _cq()
            base = self._solids[-1]
            d = op.direction
            result = base
            for i in range(1, op.count):
                off = (d[0] * op.spacing * i, d[1] * op.spacing * i, d[2] * op.spacing * i)
                result = result.union(base.translate(off))
            if not result.solids().vals():
                return _err("degenerate", "linear_pattern produced no solid")
        except _err_types() as exc:
            return _err("kernel-error", f"linear_pattern failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "linear_pattern", "id": fid, "count": op.count})
        self._solids[-1] = result
        return ApplyResult(True, [fid])

    def _circular_pattern(self, op: CircularPattern) -> ApplyResult:
        if not self.solid_present or not self._solids:
            return _err("no-solid", "circular_pattern requires an existing solid")
        if op.count < 2:
            return _err("bad-value", f"circular_pattern count must be >= 2 (got {op.count})")
        if op.feature and op.feature not in self._feature_ids():
            return _err("bad-ref", f"unknown feature '{op.feature}'", op.feature)
        try:
            _cq()
            base = self._solids[-1]
            a = op.axis
            step = op.angle / op.count
            result = base
            for i in range(1, op.count):
                rotated = base.rotate((a[0], a[1], a[2]), (a[3], a[4], a[5]), step * i)
                result = result.union(rotated)
            if not result.solids().vals():
                return _err("degenerate", "circular_pattern produced no solid")
        except _err_types() as exc:
            return _err("kernel-error", f"circular_pattern failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "circular_pattern", "id": fid, "count": op.count})
        self._solids[-1] = result
        return ApplyResult(True, [fid])

    def _mirror(self, op: Mirror) -> ApplyResult:
        if not self.solid_present or not self._solids:
            return _err("no-solid", "mirror requires an existing solid")
        if op.plane not in ("XY", "XZ", "YZ"):
            return _err("bad-value", f"unknown mirror plane '{op.plane}'")
        if op.feature_or_body and op.feature_or_body not in self._feature_ids():
            return _err("bad-ref", f"unknown feature '{op.feature_or_body}'",
                        op.feature_or_body)
        try:
            _cq()
            base = self._solids[-1]
            mirrored = base.mirror(op.plane)
            result = base.union(mirrored)
            if not result.solids().vals():
                return _err("degenerate", "mirror produced no solid")
        except _err_types() as exc:
            return _err("kernel-error", f"mirror failed: {exc}")
        fid = self._new_id("f")
        self.features.append({"type": "mirror", "id": fid, "plane": op.plane})
        self._solids[-1] = result
        return ApplyResult(True, [fid])

    # -- honestly-unsupported features -------------------------------------
    # These validate references (so a bad ref still reports 'bad-ref') but then
    # return a typed 'not-yet-supported' diagnostic instead of fabricating
    # geometry the current CadQuery/OCCT build cannot produce reliably.
    def _unsupported_feature(self, name: str, faces_ok: bool, msg: str) -> ApplyResult:
        if not self.solid_present or not self._solids:
            return _err("no-solid", f"{name} requires an existing solid")
        return _err("not-yet-supported", msg)

    def _loft_unsupported(self, op: Loft) -> ApplyResult:
        if len(op.sketches) < 2:
            return _err("bad-value", "loft requires at least two sketches")
        for sid in op.sketches:
            if sid not in self.sketches:
                return _err("bad-ref", f"unknown sketch '{sid}'", sid)
            if not self.sketches[sid]["entities"]:
                return _err("empty-sketch", f"sketch '{sid}' has no profile", sid)
        return _err("not-yet-supported",
                    "loft over abstract coplanar profiles is not yet wired on this "
                    "CadQuery/OCCT build")

    def _sweep_unsupported(self, op: Sweep) -> ApplyResult:
        for sid in (op.sketch, op.path):
            if sid not in self.sketches:
                return _err("bad-ref", f"unknown sketch '{sid}'", sid)
            if not self.sketches[sid]["entities"]:
                return _err("empty-sketch", f"sketch '{sid}' has no profile", sid)
        return _err("not-yet-supported",
                    "sweep along an abstract path sketch is not yet wired on this "
                    "CadQuery/OCCT build")

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
            }
        except Exception:  # noqa: BLE001
            return {}

    # -- export -------------------------------------------------------------
    def export(self, fmt: str):
        fmt = fmt.lower()
        shape = self._combined()
        if shape is None:
            raise ValueError("nothing to export: no solid present")
        cq = _cq()
        from cadquery import exporters
        wp = cq.Workplane("XY").add(shape)
        if fmt == "step":
            return self._export_text(exporters, wp, "STEP", ".step")
        if fmt == "stl":
            return self._export_text(exporters, wp, "STL", ".stl")
        if fmt == "iges":
            return self._export_iges(shape)
        raise ValueError(f"unsupported export format '{fmt}'")

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
    def _export_text(exporters, wp, export_type: str, suffix: str) -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            exporters.export(wp, path, exportType=export_type)
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
