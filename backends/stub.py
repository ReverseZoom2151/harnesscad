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
        return _err("unknown-op", f"unhandled op {type(op).__name__}")

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
