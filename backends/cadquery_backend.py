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
)
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
        self.solid_present = False
        self._solids: list = []       # list of cq.Workplane, each a real solid
        self._n = {"sk": 0, "e": 0, "f": 0}

    def _new_id(self, kind: str) -> str:
        self._n[kind] += 1
        return {"sk": "sk", "e": "e", "f": "f"}[kind] + str(self._n[kind])

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
        return _err("unknown-op", f"unhandled op {type(op).__name__}")

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
        return {}

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
        raise ValueError(f"unsupported export format '{fmt}'")

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
