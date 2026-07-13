"""Sketch2CAD (SIGGRAPH Asia 2020) incremental modelling state machine.

Sketch2CAD is *sequential*: every stroke is interpreted against the current
shape, the predicted operation is applied to a face of that shape, and the
result becomes the context for the next stroke.  The bookkeeping that makes this
loop work — which faces of the current model are available to stitch onto, which
face a decoded operation actually refers to, what each operation consumes and
produces, and how to undo/redo/replay the history — is deterministic, and is
what this module implements.

The session tracks *topology and plane bookkeeping*, not an exact B-rep (the
harness has kernels for that): each face carries a supporting plane, and each
operation declares the faces it consumes and the faces it produces, with the
produced planes estimated by translating along the decoded offset vector.  A
real kernel refines the geometry; the state machine's job is to keep the
operation sequence consistent and replayable.

Per-operation effects (see :mod:`reconstruction.s2cadsig_op_router`):

  * ``extrusion`` — consumes the stitching face, produces a cap face translated
    by the offset vector (same normal) plus a lateral shell entry.
  * ``addSub`` — with a positive sign behaves like ``extrusion``; with a negative
    sign it is a pocket: the stitching face survives (material removed inside the
    base curve) and a floor face is produced at ``point + offset_vector`` with an
    inverted normal.
  * ``sweep`` — consumes the stitching face and produces an end-cap face at the
    end of the sweep path (offset vector), same normal.
  * ``bevel`` — modifies the stitching face: the face survives (shrunk) and one
    bevel face is produced, anchored at the base-curve centroid with the same
    supporting normal (the kernel resolves its true slope from the adjacent face).

Stdlib-only, deterministic.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.reconstruction.fitting.s2cadsig_op_router import spec_for

Vec3 = Tuple[float, float, float]

EPS = 1e-9


class SessionError(ValueError):
    """Raised on an inapplicable operation or a malformed history."""


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _neg(a: Vec3) -> Vec3:
    return (-a[0], -a[1], -a[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _round(v: Vec3, nd: int = 9) -> Vec3:
    return (round(v[0], nd), round(v[1], nd), round(v[2], nd))


@dataclass(frozen=True)
class SessionFace:
    face_id: int
    point: Vec3
    normal: Vec3
    #: index of the step that produced it (-1 for the seed shape)
    created_by: int
    kind: str = "planar"

    def plane_distance(self, point: Vec3, normal: Vec3) -> float:
        """Plane mismatch: |offset along normal| + angular disagreement."""
        off = abs(_dot((point[0] - self.point[0],
                        point[1] - self.point[1],
                        point[2] - self.point[2]), self.normal))
        ang = 1.0 - _dot(self.normal, normal)
        return off + ang


@dataclass
class OpRecord:
    """One applied operation, enough to replay it exactly."""
    op_name: str
    face_id: int
    offset: Optional[Vec3] = None
    curve: Tuple[Vec3, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "op": self.op_name,
            "face_id": self.face_id,
            "offset": None if self.offset is None else list(self.offset),
            "curve": [list(p) for p in self.curve],
        }

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "OpRecord":
        off = d.get("offset")
        return OpRecord(
            op_name=str(d["op"]),
            face_id=int(d["face_id"]),
            offset=None if off is None else tuple(float(x) for x in off),  # type: ignore[arg-type]
            curve=tuple(tuple(float(x) for x in p) for p in d.get("curve", [])),  # type: ignore[arg-type]
        )


@dataclass
class Step:
    index: int
    record: OpRecord
    consumed: Tuple[int, ...]
    produced: Tuple[int, ...]


class ModelingSession:
    """Sequential Sketch2CAD modelling state: faces + operation history."""

    def __init__(self, seed_faces: Sequence[Tuple[Vec3, Vec3]] = ()) -> None:
        self._faces: Dict[int, SessionFace] = {}
        self._active: List[int] = []
        self._next_id = 0
        self._steps: List[Step] = []
        self._redo: List[OpRecord] = []
        for point, normal in seed_faces:
            self._new_face(point, normal, created_by=-1)

    # -- faces ------------------------------------------------------------
    def _new_face(self, point: Vec3, normal: Vec3, created_by: int, kind: str = "planar") -> SessionFace:
        n = math.sqrt(_dot(normal, normal))
        if n < EPS:
            raise SessionError("face normal must be non-zero")
        unit = (normal[0] / n, normal[1] / n, normal[2] / n)
        face = SessionFace(self._next_id, _round(point), _round(unit), created_by, kind)
        self._faces[face.face_id] = face
        self._active.append(face.face_id)
        self._next_id += 1
        return face

    @property
    def faces(self) -> List[SessionFace]:
        return [self._faces[i] for i in self._active]

    def face(self, face_id: int) -> SessionFace:
        if face_id not in self._faces:
            raise SessionError("unknown face id: {}".format(face_id))
        return self._faces[face_id]

    def is_active(self, face_id: int) -> bool:
        return face_id in self._active

    def match_face(self, point: Vec3, normal: Vec3, tolerance: float = 1e-3) -> int:
        """Nearest active face to a decoded stitching plane, or raise."""
        if not self._active:
            raise SessionError("the shape has no active face")
        best_id = -1
        best = float("inf")
        for fid in self._active:
            d = self._faces[fid].plane_distance(point, normal)
            if d < best - 1e-12:
                best = d
                best_id = fid
        if best > tolerance:
            raise SessionError(
                "no active face within tolerance (best mismatch {:.6f})".format(best)
            )
        return best_id

    # -- operations -------------------------------------------------------
    @property
    def steps(self) -> List[Step]:
        return list(self._steps)

    @property
    def step_count(self) -> int:
        return len(self._steps)

    def history(self) -> List[OpRecord]:
        return [s.record for s in self._steps]

    def validate(self, record: OpRecord) -> None:
        """Check an operation is applicable to the current shape."""
        spec = spec_for(record.op_name)
        if not self.is_active(record.face_id):
            raise SessionError(
                "face {} is not an active face of the current shape".format(record.face_id)
            )
        if spec.needs_offset:
            if record.offset is None:
                raise SessionError("{} requires an offset vector".format(spec.name))
            if math.sqrt(_dot(record.offset, record.offset)) < 1e-9:
                raise SessionError("{} offset is degenerate".format(spec.name))
        if not record.curve:
            raise SessionError("{} requires a guiding curve".format(spec.name))

    def apply(self, record: OpRecord) -> Step:
        """Apply an operation, consuming/producing faces per its spec."""
        self.validate(record)
        spec = spec_for(record.op_name)
        base = self._faces[record.face_id]
        index = len(self._steps)
        consumed: List[int] = []
        produced: List[int] = []

        if spec.name == "bevel":
            centroid = _centroid(record.curve)
            produced.append(
                self._new_face(centroid, base.normal, index, kind="bevel").face_id
            )
        elif spec.name == "addSub" and record.offset is not None and _dot(record.offset, base.normal) < 0:
            floor = _add(base.point, record.offset)
            produced.append(
                self._new_face(floor, _neg(base.normal), index, kind="pocket_floor").face_id
            )
        else:
            # extrusion / sweep / additive addSub: the base face becomes the cap
            offset = record.offset or (0.0, 0.0, 0.0)
            cap = _add(base.point, offset)
            kind = "cap" if spec.name != "sweep" else "sweep_cap"
            produced.append(self._new_face(cap, base.normal, index, kind=kind).face_id)
            consumed.append(base.face_id)

        for fid in consumed:
            self._active.remove(fid)
        step = Step(index=index, record=record, consumed=tuple(consumed), produced=tuple(produced))
        self._steps.append(step)
        self._redo.clear()
        return step

    def apply_decoded(self, params: object, tolerance: float = 1e-3) -> Step:
        """Apply an :class:`OperationParameters` from ``s2cadsig_param_decode``."""
        face = getattr(params, "face")
        face_id = self.match_face(face.point, face.normal, tolerance)
        offset = getattr(params, "offset_vector", None)
        record = OpRecord(
            op_name=getattr(params, "op_name"),
            face_id=face_id,
            offset=None if offset is None else tuple(float(x) for x in offset),
            curve=tuple(tuple(float(x) for x in p) for p in getattr(params, "curve_points", ())),
        )
        return self.apply(record)

    # -- undo / redo / replay ---------------------------------------------
    def undo(self) -> OpRecord:
        if not self._steps:
            raise SessionError("nothing to undo")
        step = self._steps.pop()
        for fid in step.produced:
            self._active.remove(fid)
            del self._faces[fid]
        for fid in step.consumed:
            self._active.append(fid)
        self._active.sort()
        self._next_id = min(step.produced) if step.produced else self._next_id
        self._redo.append(step.record)
        return step.record

    def redo(self) -> Step:
        if not self._redo:
            raise SessionError("nothing to redo")
        record = self._redo.pop()
        redo_stack = list(self._redo)
        step = self.apply(record)
        self._redo = redo_stack
        return step

    def state_signature(self) -> str:
        """Stable hash of the active faces + history (for regression tests)."""
        h = hashlib.sha256()
        for fid in sorted(self._active):
            f = self._faces[fid]
            h.update("{}|{}|{}|{}\n".format(f.face_id, f.point, f.normal, f.kind).encode())
        for s in self._steps:
            h.update(repr(s.record.to_dict()).encode())
        return h.hexdigest()

    def summary(self) -> List[Dict[str, object]]:
        return [
            {
                "index": s.index,
                "op": s.record.op_name,
                "face": s.record.face_id,
                "consumed": list(s.consumed),
                "produced": list(s.produced),
            }
            for s in self._steps
        ]


def _centroid(points: Sequence[Vec3]) -> Vec3:
    if not points:
        raise SessionError("empty curve")
    n = float(len(points))
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
        sum(p[2] for p in points) / n,
    )


def replay(
    seed_faces: Sequence[Tuple[Vec3, Vec3]], records: Sequence[OpRecord]
) -> ModelingSession:
    """Rebuild a session by replaying a history from the seed shape."""
    session = ModelingSession(seed_faces)
    for rec in records:
        session.apply(rec)
    return session


def serialize_history(session: ModelingSession) -> List[Dict[str, object]]:
    return [r.to_dict() for r in session.history()]


def deserialize_history(data: Sequence[Dict[str, object]]) -> List[OpRecord]:
    return [OpRecord.from_dict(d) for d in data]
