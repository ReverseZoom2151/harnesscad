"""StubBackend — a dependency-free backend that models the op semantics well
enough to exercise the whole harness spine (DOF tracking, references, digests,
deterministic replay) without a geometry kernel. It is NOT geometry: extrude
just marks a solid present. The CadQuery/OCCT backend replaces it.
"""

from __future__ import annotations

import hashlib
import json
from typing import List

from cisp.ops import (
    CONSTRAINT_DOF, PRIMITIVE_DOF,
    Op, NewSketch, AddPoint, AddLine, AddCircle, AddRectangle,
    Constrain, Extrude, Fillet, Boolean,
    Revolve, Chamfer, Hole, Shell, Draft,
    Loft, Sweep, LinearPattern, CircularPattern, Mirror,
)
from verify import Diagnostic, Severity
from backends.base import ApplyResult


def _err(code: str, msg: str, where: str = None) -> ApplyResult:
    return ApplyResult(False, [], [Diagnostic(Severity.ERROR, code, msg, where)])


class StubBackend:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sketches: dict = {}      # sid -> {plane, entities:[eid], dof}
        self.entities: dict = {}      # eid -> {type, sketch}
        self.features: list = []      # [{type, ...}]
        self.solid_present = False
        self._n = {"sk": 0, "e": 0, "f": 0}

    def _new_id(self, kind: str) -> str:
        self._n[kind] += 1
        return {"sk": "sk", "e": "e", "f": "f"}[kind] + str(self._n[kind])

    def _add_primitive(self, sketch: str, kind: str) -> ApplyResult:
        if sketch not in self.sketches:
            return _err("bad-ref", f"unknown sketch '{sketch}'", sketch)
        eid = self._new_id("e")
        self.entities[eid] = {"type": kind, "sketch": sketch}
        self.sketches[sketch]["entities"].append(eid)
        self.sketches[sketch]["dof"] += PRIMITIVE_DOF[kind]
        return ApplyResult(True, [eid])

    def apply(self, op: Op) -> ApplyResult:
        if isinstance(op, NewSketch):
            sid = self._new_id("sk")
            self.sketches[sid] = {"plane": op.plane, "entities": [], "dof": 0}
            return ApplyResult(True, [sid])
        if isinstance(op, AddPoint):
            return self._add_primitive(op.sketch, "point")
        if isinstance(op, AddLine):
            return self._add_primitive(op.sketch, "line")
        if isinstance(op, AddCircle):
            if op.r <= 0:
                return _err("bad-value", f"circle radius must be > 0 (got {op.r})")
            return self._add_primitive(op.sketch, "circle")
        if isinstance(op, AddRectangle):
            if op.w <= 0 or op.h <= 0:
                return _err("bad-value", "rectangle w and h must be > 0")
            return self._add_primitive(op.sketch, "rectangle")
        if isinstance(op, Constrain):
            return self._constrain(op)
        if isinstance(op, Extrude):
            if op.sketch not in self.sketches:
                return _err("bad-ref", f"unknown sketch '{op.sketch}'", op.sketch)
            if not self.sketches[op.sketch]["entities"]:
                return _err("empty-sketch", f"sketch '{op.sketch}' has no profile", op.sketch)
            if op.distance == 0:
                return _err("bad-value", "extrude distance must be non-zero")
            fid = self._new_id("f")
            self.features.append({"type": "extrude", "id": fid, "sketch": op.sketch})
            self.solid_present = True
            return ApplyResult(True, [fid])
        if isinstance(op, Fillet):
            if not self.solid_present:
                return _err("no-solid", "fillet requires an existing solid")
            if op.radius <= 0:
                return _err("bad-value", f"fillet radius must be > 0 (got {op.radius})")
            fid = self._new_id("f")
            self.features.append({"type": "fillet", "id": fid, "edges": list(op.edges)})
            return ApplyResult(True, [fid])
        if isinstance(op, Boolean):
            if op.kind not in ("union", "cut", "intersect"):
                return _err("bad-value", f"unknown boolean kind '{op.kind}'")
            if len(self.features) < 2:
                return _err("no-solid", "boolean requires two solids")
            fid = self._new_id("f")
            self.features.append({"type": "boolean", "id": fid, "kind": op.kind})
            return ApplyResult(True, [fid])
        if isinstance(op, Revolve):
            return self._solid_from_sketch(op.sketch, "revolve",
                                           bad_value=(op.angle == 0),
                                           bad_value_msg="revolve angle must be non-zero")
        if isinstance(op, Chamfer):
            if not self.solid_present:
                return _err("no-solid", "chamfer requires an existing solid")
            if op.distance <= 0:
                return _err("bad-value", f"chamfer distance must be > 0 (got {op.distance})")
            fid = self._new_id("f")
            self.features.append({"type": "chamfer", "id": fid, "edges": list(op.edges)})
            return ApplyResult(True, [fid])
        if isinstance(op, Hole):
            return self._hole(op)
        if isinstance(op, Shell):
            if not self.solid_present:
                return _err("no-solid", "shell requires an existing solid")
            if op.thickness <= 0:
                return _err("bad-value", f"shell thickness must be > 0 (got {op.thickness})")
            fid = self._new_id("f")
            self.features.append({"type": "shell", "id": fid,
                                  "faces": list(op.faces), "thickness": op.thickness})
            return ApplyResult(True, [fid])
        if isinstance(op, Draft):
            if not self.solid_present:
                return _err("no-solid", "draft requires an existing solid")
            if not op.neutral_plane:
                return _err("bad-value", "draft requires a neutral_plane")
            fid = self._new_id("f")
            self.features.append({"type": "draft", "id": fid,
                                  "faces": list(op.faces), "angle": op.angle,
                                  "neutral_plane": op.neutral_plane})
            return ApplyResult(True, [fid])
        if isinstance(op, Loft):
            if len(op.sketches) < 2:
                return _err("bad-value", "loft requires at least two sketches")
            for sid in op.sketches:
                if sid not in self.sketches:
                    return _err("bad-ref", f"unknown sketch '{sid}'", sid)
                if not self.sketches[sid]["entities"]:
                    return _err("empty-sketch", f"sketch '{sid}' has no profile", sid)
            fid = self._new_id("f")
            self.features.append({"type": "loft", "id": fid, "sketches": list(op.sketches)})
            self.solid_present = True
            return ApplyResult(True, [fid])
        if isinstance(op, Sweep):
            for sid in (op.sketch, op.path):
                if sid not in self.sketches:
                    return _err("bad-ref", f"unknown sketch '{sid}'", sid)
                if not self.sketches[sid]["entities"]:
                    return _err("empty-sketch", f"sketch '{sid}' has no profile", sid)
            fid = self._new_id("f")
            self.features.append({"type": "sweep", "id": fid,
                                  "sketch": op.sketch, "path": op.path})
            self.solid_present = True
            return ApplyResult(True, [fid])
        if isinstance(op, LinearPattern):
            return self._pattern(op.feature, "linear_pattern", op.count)
        if isinstance(op, CircularPattern):
            return self._pattern(op.feature, "circular_pattern", op.count)
        if isinstance(op, Mirror):
            if not self.solid_present:
                return _err("no-solid", "mirror requires an existing solid")
            if op.feature_or_body and op.feature_or_body not in self._feature_ids():
                return _err("bad-ref", f"unknown feature '{op.feature_or_body}'",
                            op.feature_or_body)
            fid = self._new_id("f")
            self.features.append({"type": "mirror", "id": fid, "plane": op.plane})
            return ApplyResult(True, [fid])
        return _err("unknown-op", f"unhandled op {type(op).__name__}")

    def _feature_ids(self) -> set:
        return {f["id"] for f in self.features}

    def _solid_from_sketch(self, sketch: str, kind: str,
                           bad_value: bool = False,
                           bad_value_msg: str = "") -> ApplyResult:
        if sketch not in self.sketches:
            return _err("bad-ref", f"unknown sketch '{sketch}'", sketch)
        if not self.sketches[sketch]["entities"]:
            return _err("empty-sketch", f"sketch '{sketch}' has no profile", sketch)
        if bad_value:
            return _err("bad-value", bad_value_msg)
        fid = self._new_id("f")
        self.features.append({"type": kind, "id": fid, "sketch": sketch})
        self.solid_present = True
        return ApplyResult(True, [fid])

    def _hole(self, op: Hole) -> ApplyResult:
        if op.diameter <= 0:
            return _err("bad-value", f"hole diameter must be > 0 (got {op.diameter})")
        if not op.through and (op.depth is None or op.depth <= 0):
            return _err("bad-value", "blind hole requires depth > 0")
        if op.kind not in ("simple", "counterbore", "countersink"):
            return _err("bad-value", f"unknown hole kind '{op.kind}'")
        ref = op.face_or_sketch
        if ref.startswith("sk"):
            if ref not in self.sketches:
                return _err("bad-ref", f"unknown sketch '{ref}'", ref)
        elif not self.solid_present:
            return _err("no-solid", "hole requires an existing solid")
        fid = self._new_id("f")
        self.features.append({"type": "hole", "id": fid, "ref": ref,
                              "diameter": op.diameter, "kind": op.kind})
        self.solid_present = True
        return ApplyResult(True, [fid])

    def _pattern(self, feature: str, kind: str, count: int) -> ApplyResult:
        if not self.solid_present:
            return _err("no-solid", f"{kind} requires an existing solid")
        if count < 2:
            return _err("bad-value", f"{kind} count must be >= 2 (got {count})")
        if feature and feature not in self._feature_ids():
            return _err("bad-ref", f"unknown feature '{feature}'", feature)
        fid = self._new_id("f")
        self.features.append({"type": kind, "id": fid, "count": count})
        return ApplyResult(True, [fid])

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

    def regenerate(self) -> List[Diagnostic]:
        return []  # incremental backend; nothing to rebuild

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
        return {}

    def export(self, fmt: str):
        s = self.query("summary")
        return f"# stub-{fmt}\n# {json.dumps(s, sort_keys=True)}\n"

    def state_digest(self) -> str:
        model = {
            "sketches": self.sketches,
            "entities": self.entities,
            "features": self.features,
            "solid_present": self.solid_present,
        }
        blob = json.dumps(model, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()
