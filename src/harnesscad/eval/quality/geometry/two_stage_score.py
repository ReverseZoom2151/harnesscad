"""Two-stage design scoring: Quality (spec-agnostic) x SpecMatch (Studio-OSS).

Mined from **Studio-OSS** (``lib/scoring.ts``, Evaluation Agent v3.0). Studio's
key scoring insight is the factoring:

    overall = QualityGate x (0.7 x SpecMatch + 0.3 x Quality)

with three orthogonal parts:

  * **QualityGate** -- a hard multiplicative cap on trust: an invalid B-rep
    caps the score at 0.4, zero volume at 0.3, no matter how well the tree
    "looks". A wrong solid can never score well on style points.
  * **Quality** (spec-agnostic): is this a well-formed solid at all --
    kernel validity, manifoldness, Euler-characteristic component count,
    slenderness/compactness sanity, and a face-count *complexity budget* that
    penalises both boolean explosions and oversimplification.
  * **SpecMatch** (spec-conditioned): does the geometry match the prompt's
    target spec -- proportions via Gaussian similarity in log-ratio space
    (ratios are multiplicative, so the deviation is ``|log a - log t|``),
    symmetry actual-vs-wanted weighted by how confidently symmetry was
    requested, features checked against *face-type evidence* (holes need >= 2
    cylindrical faces, fillets need toroidal/B-spline faces, teeth and
    patterns need face-count budgets), and parameter bounds + richness.

The geometry side consumes a :class:`GeometryMetrics` record whose derived
fields (aspect ratio, sphere-normalised compactness, centre-of-mass symmetry
hint, Euler characteristic) are computed here from raw kernel counts -- the
same ~20 metrics Studio extracts from OpenCascade after compilation
(``app/api/compile/route.ts``). Supply them from any backend.

Target specs come from
:mod:`harnesscad.domain.spec.prompt_spec_extract`. stdlib-only, deterministic.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.spec.prompt_spec_extract import TargetSpec, extract_target_spec

__all__ = [
    "GeometryMetrics",
    "QualityResult",
    "SpecMatchResult",
    "TwoStageScore",
    "compute_quality",
    "compute_spec_match",
    "score_design",
    "main",
]


# --------------------------------------------------------------------------- #
# Geometry metrics with derived fields
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GeometryMetrics:
    """Raw kernel measurements plus deterministic derived metrics."""

    volume: float
    surface_area: float
    dimensions: Tuple[float, float, float]          # bbox extents (mm)
    face_count: int
    edge_count: int
    vertex_count: int
    is_valid: bool
    center_of_mass: Optional[Tuple[float, float, float]] = None
    bbox_center: Optional[Tuple[float, float, float]] = None
    face_types: Dict[str, int] = field(default_factory=dict)
    edge_types: Dict[str, int] = field(default_factory=dict)

    @property
    def max_dimension(self) -> float:
        return max(self.dimensions) if self.dimensions else 0.0

    @property
    def min_dimension(self) -> float:
        positive = [d for d in self.dimensions if d > 1e-3]
        return min(positive) if positive else 0.0

    @property
    def aspect_ratio(self) -> float:
        return self.max_dimension / self.min_dimension if self.min_dimension > 0 else 0.0

    @property
    def compactness(self) -> float:
        """Sphere-normalised compactness: 1.0 for a sphere, -> 0 degenerate."""
        if self.volume <= 0 or self.surface_area <= 0:
            return 0.0
        return (math.pi ** (1.0 / 3.0) * (6.0 * self.volume) ** (2.0 / 3.0)) \
            / self.surface_area

    @property
    def euler_characteristic(self) -> int:
        return self.vertex_count - self.edge_count + self.face_count

    @property
    def symmetry_hint(self) -> float:
        """1.0 when the centre of mass sits at the bbox centre (Studio metric)."""
        if self.center_of_mass is None or self.bbox_center is None:
            return 0.5
        offset = sum(abs(c - b) for c, b in zip(self.center_of_mass, self.bbox_center))
        return max(0.0, 1.0 - offset / max(self.max_dimension, 1e-3))


# --------------------------------------------------------------------------- #
# Stage 1: Quality (spec-agnostic)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QualityResult:
    overall: float
    validity: float
    manifold: float
    component_count: float
    slenderness: float
    complexity_budget: float
    issues: Tuple[str, ...]


def compute_quality(metrics: Optional[GeometryMetrics]) -> QualityResult:
    if metrics is None:
        return QualityResult(0.65, 0.65, 0.65, 0.8, 0.7, 0.7,
                             ("no geometry metrics; using neutral defaults",))
    issues: List[str] = []

    validity = 1.0 if metrics.is_valid else 0.0
    if not metrics.is_valid:
        issues.append("kernel validity check failed")

    manifold = 1.0
    if metrics.volume <= 0:
        manifold = 0.0
        issues.append("zero volume")
    elif metrics.surface_area <= 0:
        manifold = 0.2
        issues.append("zero surface area")
    elif metrics.face_count < 4:
        manifold = 0.3
        issues.append("too few faces for a solid")

    euler = metrics.euler_characteristic
    if euler == 2:
        component_count = 1.0
    elif euler > 2:
        component_count = 0.7
        issues.append(f"multiple components suspected (Euler={euler})")
    else:
        component_count = 0.8       # genus > 0: holes/tunnels are acceptable

    aspect = metrics.aspect_ratio
    slenderness = 1.0
    if aspect > 100:
        slenderness = 0.2
        issues.append(f"extreme slenderness: aspect ratio {aspect:.0f}")
    elif aspect > 50:
        slenderness = 0.4
        issues.append(f"very high aspect ratio: {aspect:.0f}")
    elif aspect > 20:
        slenderness = 0.7
    if metrics.compactness < 0.01 and metrics.volume > 0:
        slenderness = min(slenderness, 0.5)
        issues.append("very low compactness")

    fc = metrics.face_count
    complexity = 1.0
    if fc > 500:
        complexity = 0.6
        issues.append(f"high face count: {fc} (possible boolean explosion)")
    elif fc > 200:
        complexity = 0.8
    elif fc < 4 and metrics.volume > 0:
        complexity = 0.4
        issues.append(f"too few faces: {fc}")

    overall = (validity * 0.30 + manifold * 0.25 + component_count * 0.15
               + slenderness * 0.15 + complexity * 0.15)
    return QualityResult(round(overall, 2), validity, manifold, component_count,
                         slenderness, complexity, tuple(issues))


# --------------------------------------------------------------------------- #
# Stage 2: SpecMatch (spec-conditioned)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpecMatchResult:
    overall: float
    proportions: float
    symmetry: float
    features: float
    param_bounds: float
    issues: Tuple[str, ...]


_RADIAL_FACE_TYPES = ("CYLINDER", "SPHERE", "CONE", "TORUS")


def _log_gaussian(actual: float, target: float, sharpness: float = 3.0) -> float:
    dev = abs(math.log(actual) - math.log(target))
    return math.exp(-sharpness * dev * dev)


def compute_spec_match(
    spec: TargetSpec,
    metrics: Optional[GeometryMetrics],
    parameters: Optional[Dict[str, Dict[str, object]]] = None,
) -> SpecMatchResult:
    """Score the geometry against the extracted target spec.

    ``parameters`` (optional) follows Studio's tree-parameter shape:
    name -> record with ``value`` and optional ``min``/``max`` bounds.
    """
    issues: List[str] = []

    # 1. Proportions.
    proportions = 0.75
    if metrics is not None:
        sorted_dims = sorted((d for d in metrics.dimensions if d > 1e-3), reverse=True)
        if len(sorted_dims) >= 2:
            actual_ratio = sorted_dims[0] / sorted_dims[1]
            target_ratio = spec.target_ratios.get("height_width")
            if target_ratio:
                proportions = _log_gaussian(actual_ratio, target_ratio)
                if proportions < 0.5:
                    issues.append(f"ratio mismatch: actual={actual_ratio:.2f} "
                                  f"target={target_ratio:.2f}")
            else:
                lo, hi = spec.ideal_ratio_min, spec.ideal_ratio_max
                if lo <= actual_ratio <= hi:
                    proportions = 1.0
                else:
                    edge = lo if actual_ratio < lo else hi
                    proportions = max(0.3, _log_gaussian(actual_ratio, edge, 2.0))

    # 2. Symmetry.
    symmetry = 0.7
    if metrics is not None:
        hint = metrics.symmetry_hint
        has_radial = sum(metrics.face_types.get(t, 0) for t in _RADIAL_FACE_TYPES) > 0
        geo_symmetry = 0.4
        if has_radial:
            geo_symmetry += 0.25
        if hint > 0.9:
            geo_symmetry += 0.2
        elif hint > 0.7:
            geo_symmetry += 0.1
        if metrics.compactness > 0.5:
            geo_symmetry += 0.1
        geo_symmetry = min(1.0, geo_symmetry)
        if spec.wants_symmetry:
            symmetry = (geo_symmetry * spec.symmetry_confidence
                        + (1 - spec.symmetry_confidence) * 0.7)
            if geo_symmetry < 0.5 and spec.symmetry_confidence > 0.7:
                issues.append(f"symmetry required but geometry is asymmetric "
                              f"(hint={hint:.2f})")
        else:
            symmetry = 0.7 + geo_symmetry * 0.3

    # 3. Features against face-type evidence.
    features = 0.7
    if spec.target_features:
        total = len(spec.target_features)
        matched = 0.0
        if metrics is not None:
            ft = metrics.face_types
            et = metrics.edge_types
            fc = metrics.face_count
            for feat in spec.target_features:
                if feat in ("holes", "bore"):
                    if ft.get("CYLINDER", 0) >= 2:
                        matched += 1
                    else:
                        issues.append(f"expected {feat} but no cylindrical faces")
                elif feat == "ribs":
                    if fc > 20 and ft.get("PLANE", 0) > 10:
                        matched += 1
                    elif spec.texture_hint == "ribbed" and fc > 15:
                        matched += 1
                    else:
                        issues.append(f"expected ribs but low face complexity "
                                      f"(faces={fc})")
                elif feat == "teeth":
                    if fc > 30:
                        matched += 1
                    else:
                        issues.append(f"expected teeth but face count low (faces={fc})")
                elif feat == "fillet":
                    if ft.get("TORUS", 0) > 0 or ft.get("BSPLINE", 0) > 0:
                        matched += 1
                    elif et.get("CIRCLE", 0) > 0:
                        matched += 0.5
                    else:
                        issues.append("expected fillets but no curved transition faces")
                elif feat == "chamfer":
                    matched += 1 if ft.get("PLANE", 0) > 8 else 0.5
                elif feat in ("slots", "cutout"):
                    if fc > 12:
                        matched += 1
                    else:
                        issues.append(f"expected {feat} but geometry too simple")
                elif feat == "pattern":
                    if fc > 20:
                        matched += 1
                    else:
                        issues.append("expected pattern but face count low")
                else:
                    matched += 0.5
            features = min(1.0, matched / total) if total else 0.7
            if parameters:
                for name, target_count in spec.feature_counts.items():
                    for key, record in parameters.items():
                        value = record.get("value")
                        if name[:4].lower() in key.lower() \
                                and isinstance(value, (int, float)) \
                                and abs(float(value) - target_count) < 2:
                            features = min(1.0, features + 0.05)
                            break
        else:
            features = 0.5
    elif metrics is not None:
        fc = metrics.face_count
        variety = len(metrics.face_types)
        features = min(1.0, 0.5 + math.log2(max(fc, 1)) / 12 + variety * 0.05)

    # 4. Parameter bounds + richness.
    param_bounds = 0.8
    if parameters:
        in_bounds = 0
        total = 0
        for record in parameters.values():
            value = record.get("value")
            if not isinstance(value, (int, float)):
                continue
            total += 1
            lo = record.get("min")
            hi = record.get("max")
            ok_lo = lo is None or float(value) >= float(lo)      # type: ignore[arg-type]
            ok_hi = hi is None or float(value) <= float(hi)      # type: ignore[arg-type]
            if ok_lo and ok_hi:
                in_bounds += 1
        if total:
            bounds_ratio = in_bounds / total
            richness = min(total / 5.0, 1.0)
            param_bounds = bounds_ratio * 0.75 + richness * 0.25

    overall = (proportions * 0.30 + symmetry * 0.20
               + features * 0.30 + param_bounds * 0.20)
    return SpecMatchResult(round(overall, 2), round(proportions, 2),
                           round(symmetry, 2), round(features, 2),
                           round(param_bounds, 2), tuple(issues))


# --------------------------------------------------------------------------- #
# Combined
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TwoStageScore:
    overall: float
    quality_gate: float
    quality: QualityResult
    spec_match: SpecMatchResult
    breakdown: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "overall": self.overall,
            "quality_gate": self.quality_gate,
            "quality": self.quality.overall,
            "spec_match": self.spec_match.overall,
            "proportions": self.spec_match.proportions,
            "symmetry": self.spec_match.symmetry,
            "features": self.spec_match.features,
            "param_bounds": self.spec_match.param_bounds,
            "breakdown": list(self.breakdown),
        }


def score_design(
    prompt: str,
    metrics: Optional[GeometryMetrics],
    parameters: Optional[Dict[str, Dict[str, object]]] = None,
    *,
    spec: Optional[TargetSpec] = None,
) -> TwoStageScore:
    """QualityGate x (0.7 x SpecMatch + 0.3 x Quality)."""
    spec = spec if spec is not None else extract_target_spec(prompt, parameters)
    quality = compute_quality(metrics)
    spec_match = compute_spec_match(spec, metrics, parameters)

    quality_gate = 1.0
    breakdown: List[str] = []
    if metrics is not None:
        if not metrics.is_valid:
            quality_gate = 0.4
            breakdown.append("B-rep topology invalid; QualityGate applied (x0.4)")
        elif metrics.volume <= 0:
            quality_gate = 0.3
            breakdown.append("zero volume; degenerate geometry (x0.3)")
    if quality.overall < 0.6:
        breakdown.append("Quality: " + "; ".join(quality.issues))
    if spec_match.overall < 0.6:
        breakdown.append("SpecMatch: " + "; ".join(spec_match.issues))

    raw = 0.7 * spec_match.overall + 0.3 * quality.overall
    return TwoStageScore(
        overall=round(quality_gate * raw, 2),
        quality_gate=quality_gate,
        quality=quality,
        spec_match=spec_match,
        breakdown=tuple(breakdown),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _good_gear_metrics() -> GeometryMetrics:
    return GeometryMetrics(
        volume=7850.0, surface_area=4200.0, dimensions=(44.0, 44.0, 10.0),
        face_count=68, edge_count=204, vertex_count=138, is_valid=True,
        center_of_mass=(0.0, 0.0, 5.0), bbox_center=(0.0, 0.0, 5.0),
        face_types={"PLANE": 46, "CYLINDER": 22}, edge_types={"CIRCLE": 8})


def _invalid_metrics() -> GeometryMetrics:
    return GeometryMetrics(
        volume=100.0, surface_area=200.0, dimensions=(10.0, 10.0, 10.0),
        face_count=6, edge_count=12, vertex_count=8, is_valid=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.eval.quality.geometry.two_stage_score",
        description="Two-stage Quality x SpecMatch design scoring (Studio-OSS).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="score a well-formed gear and an invalid solid and "
                             "verify the quality gate bites.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    prompt = "A symmetric spur gear with 20 teeth, thickness of 10mm and a 5mm bore"
    params = {
        "teeth": {"value": 20, "unit": "count", "min": 6, "max": 120},
        "thickness": {"value": 10, "unit": "mm", "min": 2, "max": 50},
        "bore_radius": {"value": 2.5, "unit": "mm", "min": 1, "max": 20},
    }
    good = score_design(prompt, _good_gear_metrics(), params)
    assert good.quality_gate == 1.0 and good.overall >= 0.8, good.to_dict()
    print(f"[selfcheck] valid gear: overall={good.overall:.2f} "
          f"(quality={good.quality.overall:.2f}, "
          f"spec_match={good.spec_match.overall:.2f})")

    bad = score_design(prompt, _invalid_metrics(), params)
    assert bad.quality_gate == 0.4 and bad.overall <= 0.4, bad.to_dict()
    print(f"[selfcheck] invalid solid: overall={bad.overall:.2f} "
          f"(gate x{bad.quality_gate}) -> {bad.breakdown[0]}")

    no_metrics = score_design(prompt, None, params)
    assert 0.3 < no_metrics.overall < 0.9
    print(f"[selfcheck] no metrics: overall={no_metrics.overall:.2f} "
          "(neutral defaults, no gate)")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
