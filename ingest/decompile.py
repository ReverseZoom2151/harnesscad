"""decompile — best-effort B-rep -> CISP feature tree recovery.

Given an imported reference solid (or any backend that can measure itself),
recover a plausible CISP op list that would *rebuild* it. This is deliberately
scoped: we recover **prismatic** features (a rectangular profile extruded) and
**revolved** features (an axisymmetric profile revolved) — the two families a
face-classification pass can identify with confidence. Free-form / sculpted
geometry is reported as *unrecovered* in the note rather than faked.

Strategy:
  * With OCCT: classify every face as planar / cylindrical / other, then
      - a box-like all-planar solid  -> NewSketch + AddRectangle + Extrude,
      - a dominantly-cylindrical solid -> a revolved / circular-extrude recovery,
      - otherwise                     -> bbox prismatic block, low confidence,
        with a note flagging the unrecovered free-form portion.
  * Without OCCT (or a shapeless backend): fall back to metrics. If a bounding
    box is measurable, emit a bbox prismatic block (low confidence, clearly
    noted as metrics-only); if nothing is measurable, return an empty op list
    with a clear metrics-only note. Never raises.

The op vocabulary is exactly ``cisp.ops`` (NewSketch / AddRectangle / AddCircle
/ Extrude / Revolve), so the recovered tree replays through any backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from cisp.ops import (
    Op, NewSketch, AddRectangle, AddCircle, Extrude, Revolve,
)


@dataclass
class DecompileResult:
    """A recovered feature tree (best effort).

    - ``ops``        : the recovered CISP op list (possibly empty).
    - ``confidence`` : 0.0..1.0 heuristic confidence in the recovery.
    - ``note``       : scope / degradation notes (what was and wasn't recovered).
    - ``method``     : "brep-faces" | "metrics-bbox" | "none".
    - ``face_summary``: face-classification counts when a shape was traversed.
    """

    ops: List[Op] = field(default_factory=list)
    confidence: float = 0.0
    note: str = ""
    method: str = "none"
    face_summary: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.ops)

    def to_dict(self) -> dict:
        return {
            "ops": [op.to_dict() for op in self.ops],
            "confidence": self.confidence,
            "note": self.note,
            "method": self.method,
            "face_summary": dict(self.face_summary),
        }


def decompile(imported_or_backend) -> DecompileResult:
    """Recover a best-effort CISP op list from a reference solid or backend."""
    shape = _get_shape(imported_or_backend)
    if shape is not None:
        result = _decompile_shape(shape)
        if result is not None:
            return result
    # No shape (no OCCT, or a shapeless backend) -> metrics-only fallback.
    metrics = _get_metrics(imported_or_backend)
    return _decompile_from_metrics(metrics)


# --------------------------------------------------------------------------- #
# Input adaptation
# --------------------------------------------------------------------------- #
def _get_shape(obj):
    """Extract a cq/OCP shape from an ImportedPart or a geometry backend."""
    shape = getattr(obj, "shape", None)
    if shape is not None:
        return shape
    combined = getattr(obj, "_combined", None)
    if callable(combined):
        try:
            return combined()
        except Exception:  # noqa: BLE001 - never crash on a kernel hiccup
            return None
    return None


def _get_metrics(obj) -> dict:
    """Best-available numeric metrics: ImportedPart.metrics or backend queries."""
    metrics = getattr(obj, "metrics", None)
    if isinstance(metrics, dict) and metrics:
        return metrics
    query = getattr(obj, "query", None)
    if callable(query):
        for q in ("metrics", "measure"):
            try:
                res = query(q)
            except Exception:  # noqa: BLE001
                res = None
            if res:
                return res
    return {}


# --------------------------------------------------------------------------- #
# Metrics-only recovery
# --------------------------------------------------------------------------- #
def _bbox_of(metrics: dict) -> Optional[list]:
    bbox = metrics.get("bbox") if metrics else None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 3:
        try:
            vals = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
        if all(v > 0 for v in vals):
            return vals
    return None


def _prismatic_ops(dx: float, dy: float, dz: float) -> List[Op]:
    """A rectangle-in-XY extruded by dz — the canonical prismatic block."""
    return [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", x=-dx / 2.0, y=-dy / 2.0, w=dx, h=dy),
        Extrude(sketch="sk1", distance=dz),
    ]


def _decompile_from_metrics(metrics: dict) -> DecompileResult:
    bbox = _bbox_of(metrics)
    if bbox is None:
        return DecompileResult(
            ops=[], confidence=0.0, method="none",
            note="metrics-only: no measurable bounding box available; "
                 "no feature tree recovered (need OCCT or a measurable backend)")
    dx, dy, dz = bbox
    ops = _prismatic_ops(dx, dy, dz)
    return DecompileResult(
        ops=ops, confidence=0.2, method="metrics-bbox",
        note="metrics-only recovery: emitted a bounding-box prismatic block "
             "(no B-rep face data; actual profile unverified)")


# --------------------------------------------------------------------------- #
# B-rep face-classification recovery (OCCT)
# --------------------------------------------------------------------------- #
def _decompile_shape(shape) -> Optional[DecompileResult]:
    """Classify faces and recover prismatic / revolved features. Guarded."""
    try:
        classes = _classify_faces(shape)
    except Exception:  # noqa: BLE001 - fall back to metrics on any kernel error
        return None
    if classes is None:
        return None

    planar = classes.get("planar", 0)
    cylindrical = classes.get("cylindrical", 0)
    other = classes.get("other", 0)
    total = planar + cylindrical + other
    summary = {"planar": planar, "cylindrical": cylindrical, "other": other}

    try:
        bb = shape.BoundingBox()
        bbox = [float(bb.xlen), float(bb.ylen), float(bb.zlen)]
    except Exception:  # noqa: BLE001
        bbox = None
    if bbox is None or not all(v > 0 for v in bbox):
        # Shape but no usable bbox -> metrics fallback handles it.
        return None
    dx, dy, dz = bbox

    # Box: six planar faces, no curvature -> confident prismatic recovery.
    if total > 0 and cylindrical == 0 and other == 0 and planar == 6:
        return DecompileResult(
            ops=_prismatic_ops(dx, dy, dz), confidence=0.9,
            method="brep-faces", face_summary=summary,
            note="recovered prismatic block (6 planar faces, box topology)")

    # Purely cylindrical/planar and axisymmetric -> revolved (or circular)
    # recovery: a full solid of revolution of a rectangular profile == cylinder.
    if cylindrical >= 1 and other == 0 and _is_axisymmetric(bbox):
        radius = min(dx, dy) / 2.0
        ops: List[Op] = [
            NewSketch(plane="XY"),
            AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=radius),
            Extrude(sketch="sk1", distance=dz),
        ]
        return DecompileResult(
            ops=ops, confidence=0.7, method="brep-faces",
            face_summary=summary,
            note="recovered revolved/cylindrical feature (axisymmetric, "
                 f"{cylindrical} cylindrical face(s)); modelled as a circular "
                 "extrude")

    # Mixed / free-form: emit a bbox prismatic envelope but flag low confidence.
    note = ("partial recovery: prismatic bounding envelope only; "
            f"{cylindrical} cylindrical + {other} non-planar/cylindrical "
            "face(s) not decompiled (free-form features unrecovered)")
    return DecompileResult(
        ops=_prismatic_ops(dx, dy, dz), confidence=0.3,
        method="brep-faces", face_summary=summary, note=note)


def _is_axisymmetric(bbox: list, tol: float = 1e-6) -> bool:
    dx, dy, _dz = bbox
    return abs(dx - dy) <= tol * max(dx, dy, 1.0)


def _classify_faces(shape) -> Optional[dict]:
    """Count faces by surface type: planar / cylindrical / other. Guarded."""
    faces = shape.Faces()
    if not faces:
        return None
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType
    counts = {"planar": 0, "cylindrical": 0, "other": 0}
    for face in faces:
        wrapped = getattr(face, "wrapped", face)
        try:
            surf = BRepAdaptor_Surface(wrapped)
            stype = surf.GetType()
        except Exception:  # noqa: BLE001 - unclassifiable face
            counts["other"] += 1
            continue
        if stype == GeomAbs_SurfaceType.GeomAbs_Plane:
            counts["planar"] += 1
        elif stype == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            counts["cylindrical"] += 1
        else:
            counts["other"] += 1
    return counts
