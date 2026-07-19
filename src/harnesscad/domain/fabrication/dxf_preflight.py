"""Structural preflight for an already-parsed neutral DXF drawing.

This is deliberately *not* a DXF reader.  ``io.formats.dxf`` defines the
neutral :class:`~harnesscad.io.formats.dxf.DxfDocument` contract, while its
parser is only a protocol.  A caller must first establish that a file parsed
into that contract and that its units are resolved; this module then measures
the declared 2-D entities without silently inventing a scale.

The checks are a clean-room response to the unlicensed ``Printability-main``
repository's useful, narrow idea: reject degenerate drawing geometry and make
circle/arc size and drawing density available to a manufacturing rule.  Its
code, rule files, labels, and dataset are not redistributed here.  In
particular, this module does not inherit its source-specific thresholds or its
rule-generated PASS/FAIL labels.  Thresholds are explicit call-site policy.

It also does not promise what a 2-D drawing cannot show: wall thickness,
overhangs, watertightness, or that a circle is actually a drilled/printed hole.
Those remain unmeasured rather than quietly green.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from harnesscad.io.formats.dxf import DxfDocument, Entity

__all__ = [
    "DxfPreflightFinding", "DxfPreflightMetrics", "DxfPreflightReport",
    "dxf_preflight", "main",
]

_SCALE_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "ft": 304.8}
_MEASURED_KINDS = frozenset(("LINE", "ARC", "CIRCLE", "LWPOLYLINE", "POLYLINE"))
_EPSILON = 1e-12


@dataclass(frozen=True)
class DxfPreflightFinding:
    """One evidence-backed structural or policy finding."""

    code: str
    severity: str  # ``error`` | ``warning`` | ``info``
    message: str
    entity_ids: Tuple[str, ...] = ()
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity,
                "message": self.message, "entity_ids": list(self.entity_ids),
                "recommendation": self.recommendation}


@dataclass(frozen=True)
class DxfPreflightMetrics:
    """Measurements that are actually derivable from the neutral entities."""

    units: str
    entity_count: int
    entity_kinds: Mapping[str, int]
    zero_length_segments: int
    circle_diameters_mm: Tuple[float, ...]
    arc_radii_mm: Tuple[float, ...]
    bbox_mm: Optional[Tuple[float, float, float, float]]
    total_path_length_mm: Optional[float]
    density_per_mm2: Optional[float]
    unmeasured_entity_ids: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "units": self.units, "entity_count": self.entity_count,
            "entity_kinds": dict(sorted(self.entity_kinds.items())),
            "zero_length_segments": self.zero_length_segments,
            "circle_diameters_mm": list(self.circle_diameters_mm),
            "arc_radii_mm": list(self.arc_radii_mm),
            "bbox_mm": list(self.bbox_mm) if self.bbox_mm is not None else None,
            "total_path_length_mm": self.total_path_length_mm,
            "density_per_mm2": self.density_per_mm2,
            "unmeasured_entity_ids": list(self.unmeasured_entity_ids),
        }


@dataclass(frozen=True)
class DxfPreflightReport:
    """A deterministic preflight result; ``REVIEW`` means evidence is missing."""

    verdict: str  # PASS | REVIEW | FAIL
    metrics: DxfPreflightMetrics
    findings: Tuple[DxfPreflightFinding, ...]

    @property
    def ok(self) -> bool:
        return self.verdict == "PASS"

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "ok": self.ok,
                "metrics": self.metrics.to_dict(),
                "findings": [f.to_dict() for f in self.findings]}


def _number(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s is not a finite number" % field) from exc
    if not math.isfinite(result):
        raise ValueError("%s is not a finite number" % field)
    return result


def _point(value: object, field: str) -> Tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise ValueError("%s must be a two-coordinate sequence" % field)
    return _number(value[0], field + "[0]"), _number(value[1], field + "[1]")


def _entity_bounds_and_length(entity: Entity, scale: float) -> Tuple[
        Tuple[float, float, float, float], float, int, Tuple[float, ...], Tuple[float, ...]]:
    """Return bounds, path length, degenerate-edge count, circle Ds, arc Rs."""
    values = entity.values
    kind = entity.kind.upper()
    if kind == "LINE":
        a, b = _point(values.get("start"), "start"), _point(values.get("end"), "end")
        length = math.dist(a, b) * scale
        return (_bounds((a, b), scale), length, int(length <= _EPSILON), (), ())
    if kind == "CIRCLE":
        center, radius = _point(values.get("center"), "center"), _number(values.get("radius"), "radius")
        radius *= scale
        if radius <= _EPSILON:
            raise ValueError("circle radius must be positive")
        x, y = center[0] * scale, center[1] * scale
        return ((x - radius, y - radius, x + radius, y + radius), 2.0 * math.pi * radius,
                0, (2.0 * radius,), ())
    if kind == "ARC":
        center, radius = _point(values.get("center"), "center"), _number(values.get("radius"), "radius")
        radius *= scale
        if radius <= _EPSILON:
            raise ValueError("arc radius must be positive")
        start = _number(values.get("start_angle_deg", values.get("start_angle", 0.0)), "start_angle_deg")
        end = _number(values.get("end_angle_deg", values.get("end_angle", 360.0)), "end_angle_deg")
        sweep = (end - start) % 360.0
        if abs(end - start) > _EPSILON and sweep <= _EPSILON:
            sweep = 360.0
        angles = [start, end] + [a for a in (0.0, 90.0, 180.0, 270.0)
                                 if _angle_in_sweep(a, start, sweep)]
        points = [(center[0] + (radius / scale) * math.cos(math.radians(a)),
                   center[1] + (radius / scale) * math.sin(math.radians(a))) for a in angles]
        return (_bounds(points, scale), radius * math.radians(sweep), 0, (), (radius,))
    if kind in ("LWPOLYLINE", "POLYLINE"):
        raw = values.get("points")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("points must be a sequence")
        points = tuple(_point(point, "points[%d]" % i) for i, point in enumerate(raw))
        if not points:
            raise ValueError("points must not be empty")
        closed = bool(values.get("closed", False))
        pairs = list(zip(points, points[1:]))
        if closed and len(points) > 1:
            pairs.append((points[-1], points[0]))
        lengths = [math.dist(a, b) * scale for a, b in pairs]
        return (_bounds(points, scale), sum(lengths), sum(x <= _EPSILON for x in lengths), (), ())
    raise ValueError("unsupported kind %s" % entity.kind)


def _bounds(points: Iterable[Tuple[float, float]], scale: float) -> Tuple[float, float, float, float]:
    scaled = tuple((x * scale, y * scale) for x, y in points)
    return (min(x for x, _ in scaled), min(y for _, y in scaled),
            max(x for x, _ in scaled), max(y for _, y in scaled))


def _angle_in_sweep(angle: float, start: float, sweep: float) -> bool:
    return ((angle - start) % 360.0) <= sweep + _EPSILON


def _threshold(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    value = _number(value, name)
    if value <= 0:
        raise ValueError("%s must be > 0" % name)
    return value


def dxf_preflight(document: DxfDocument, *, min_circle_diameter_mm: Optional[float] = None,
                  min_arc_radius_mm: Optional[float] = None,
                  max_entity_density_per_mm2: Optional[float] = None) -> DxfPreflightReport:
    """Measure and preflight one resolved neutral DXF document.

    Optional thresholds are policy supplied by the caller, never silently taken
    from a source-specific dataset.  An unsupported or malformed entity is a
    ``REVIEW`` finding: no numeric policy based on an incomplete drawing may
    pass.  A zero-length segment or non-positive radius is a ``FAIL``.
    """
    min_circle_diameter_mm = _threshold(min_circle_diameter_mm, "min_circle_diameter_mm")
    min_arc_radius_mm = _threshold(min_arc_radius_mm, "min_arc_radius_mm")
    max_entity_density_per_mm2 = _threshold(max_entity_density_per_mm2, "max_entity_density_per_mm2")
    scale = _SCALE_TO_MM[document.units]
    kinds: Dict[str, int] = {}
    findings = []
    bounds = []
    path_length = 0.0
    zero_length = 0
    circles = []
    arcs = []
    unmeasured = []
    for entity_id, entity in sorted(document.entities.items()):
        kind = entity.kind.upper()
        kinds[kind] = kinds.get(kind, 0) + 1
        try:
            b, length, degenerate, circle_ds, arc_rs = _entity_bounds_and_length(entity, scale)
        except ValueError as exc:
            unmeasured.append(entity_id)
            findings.append(DxfPreflightFinding(
                "DXF_ENTITY_UNMEASURABLE", "warning",
                "%s (%s) was not measured: %s" % (entity_id, entity.kind, exc), (entity_id,),
                "Use a supported geometric entity or provide a preflight adapter for this entity kind."))
            continue
        bounds.append(b)
        path_length += length
        zero_length += degenerate
        circles.extend(circle_ds)
        arcs.extend(arc_rs)
    bbox = None
    density = None
    if bounds:
        bbox = (min(b[0] for b in bounds), min(b[1] for b in bounds),
                max(b[2] for b in bounds), max(b[3] for b in bounds))
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area > _EPSILON and not unmeasured:
            density = len(document.entities) / area
    if not document.entities:
        findings.append(DxfPreflightFinding(
            "DXF_EMPTY", "warning", "Drawing declares no entities.", (),
            "Supply model-space geometry before manufacturing review."))
    if zero_length:
        findings.append(DxfPreflightFinding(
            "DXF_ZERO_LENGTH", "error", "%d zero-length segment(s) found." % zero_length, (),
            "Remove or repair degenerate line/polyline segments."))
    if min_circle_diameter_mm is not None and circles and min(circles) < min_circle_diameter_mm:
        findings.append(DxfPreflightFinding(
            "DXF_CIRCLE_BELOW_MIN", "error",
            "Smallest declared circle is %.6g mm; policy minimum is %.6g mm." %
            (min(circles), min_circle_diameter_mm), (),
            "Increase the circle diameter or select a process policy that supports it."))
    if min_arc_radius_mm is not None and arcs and min(arcs) < min_arc_radius_mm:
        findings.append(DxfPreflightFinding(
            "DXF_ARC_BELOW_MIN", "warning",
            "Smallest declared arc radius is %.6g mm; policy minimum is %.6g mm." %
            (min(arcs), min_arc_radius_mm), (),
            "Increase the arc radius or confirm the selected process can resolve it."))
    if max_entity_density_per_mm2 is not None:
        if unmeasured or density is None:
            findings.append(DxfPreflightFinding(
                "DXF_DENSITY_UNCERTIFIED", "warning",
                "Entity density cannot be evaluated from incomplete or zero-area bounds.",
                tuple(unmeasured), "Resolve unmeasured entities and provide non-zero drawing extents."))
        elif density > max_entity_density_per_mm2:
            findings.append(DxfPreflightFinding(
                "DXF_DENSITY_ABOVE_MAX", "warning",
                "Entity density is %.6g/mm^2; policy maximum is %.6g/mm^2." %
                (density, max_entity_density_per_mm2), (),
                "Simplify closely packed drawing entities or review the process policy."))
    metrics = DxfPreflightMetrics(
        document.units, len(document.entities), kinds, zero_length, tuple(sorted(circles)), tuple(sorted(arcs)),
        bbox, path_length if not unmeasured else None, density, tuple(unmeasured))
    verdict = "FAIL" if any(f.severity == "error" for f in findings) else (
        "REVIEW" if any(f.severity == "warning" for f in findings) else "PASS")
    return DxfPreflightReport(verdict, metrics, tuple(findings))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Small executable proof; the public API operates on ``DxfDocument``."""
    del argv
    from harnesscad.io.formats.dxf import Entity, Layer
    clean = DxfDocument("mm", (Layer("0"),), {
        "outline": Entity("LWPOLYLINE", {"points": ((0, 0), (20, 0), (20, 10), (0, 10)), "closed": True}),
        "hole": Entity("CIRCLE", {"center": (10, 5), "radius": 2}),
    })
    assert dxf_preflight(clean, min_circle_diameter_mm=3).verdict == "PASS"
    bad = DxfDocument("mm", (Layer("0"),), {"zero": Entity("LINE", {"start": (1, 1), "end": (1, 1)})})
    assert dxf_preflight(bad).verdict == "FAIL"
    unknown = DxfDocument("mm", (Layer("0"),), {"spline": Entity("SPLINE", {})})
    assert dxf_preflight(unknown, max_entity_density_per_mm2=1).verdict == "REVIEW"
    print("[selfcheck] DXF preflight: clean PASS; degenerate FAIL; unmeasured REVIEW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
