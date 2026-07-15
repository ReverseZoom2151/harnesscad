"""Parameter-aware topological feature recognition with confidence tiers.

Mined from **IntentForge** (``src/harness/topology/feature_recognizer.py``),
whose insight is that feature recognition is far more reliable when it is
*parameter-aware*: instead of guessing "is this cylinder a hole", it checks the
model's faces against the **expected** holes/cutouts/corners implied by the
part's parameter table, matching diameter, through-length, and hole centres.
That is what stops it from mistaking a rounded outside corner for a hole. Each
recognized feature carries a confidence tier and an expected-vs-recognized
count, and low-confidence failures are surfaced as warnings rather than turned
into a brittle release gate.

IntentForge runs this against live CadQuery topology. The transferable,
kernel-free core is the *recognition logic*, so this module operates on a plain
face list -- each face a light record of ``(kind, bbox, axis, radius)`` -- which
any backend (OCC, CadQuery, a mesh face-fitter, a synthetic fixture) can
produce. That makes the recognizer a deterministic, checkable oracle: given the
faces and the expected parameters, does the geometry actually contain the
features the spec promised.

Recognizers provided (IntentForge's set, generalized):

*   :func:`recognize_through_holes` -- cylindrical faces along an axis, matched
    on diameter / through-length / expected centres.
*   :func:`recognize_center_cutout` -- centred rectangular through-cutout from
    its four internal planar walls.
*   :func:`recognize_rounded_corners` -- >=4 vertical corner-radius cylinders.

stdlib-only (``math``, ``dataclasses``), deterministic, absolute imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "CONFIDENCE_HIGH",
    "CONFIDENCE_MEDIUM",
    "CONFIDENCE_LOW",
    "CONFIDENCE_UNKNOWN",
    "Face",
    "FeatureResult",
    "recognize_through_holes",
    "recognize_center_cutout",
    "recognize_rounded_corners",
]

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
CONFIDENCE_UNKNOWN = "unknown"

_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Face:
    """A minimal topological face record.

    ``kind`` is ``"cylinder"`` or ``"plane"``. ``bbox`` is a dict with keys
    ``xmin,ymin,zmin,xmax,ymax,zmax``; extents/centres are derived. ``axis`` is
    the cylinder axis (``"x"/"y"/"z"``) when known; ``radius`` is the cylinder
    radius when known.
    """

    kind: str
    bbox: Dict[str, float]
    axis: Optional[str] = None
    radius: Optional[float] = None

    def extent(self, axis: str) -> float:
        return float(self.bbox[f"{axis}max"] - self.bbox[f"{axis}min"])

    def center(self, axis: str) -> float:
        return float(self.bbox[f"{axis}max"] + self.bbox[f"{axis}min"]) / 2.0

    def cross_extents(self, axis: str) -> Tuple[float, float]:
        others = [a for a in _AXES if a != axis]
        return self.extent(others[0]), self.extent(others[1])

    def radius_estimate(self, axis: Optional[str]) -> Optional[float]:
        if self.radius is not None:
            return float(self.radius)
        if axis is not None:
            a, b = self.cross_extents(axis)
            return (a + b) / 4.0
        return None


@dataclass(frozen=True)
class FeatureResult:
    """Outcome of one recognizer.

    ``passed`` is the pass/fail verdict; ``confidence`` is the tier;
    ``expected_count`` / ``recognized_count`` are the parameter-derived and
    observed counts; ``warnings`` explain any gap.
    """

    passed: bool
    confidence: str
    expected_count: Optional[int] = None
    recognized_count: Optional[int] = None
    warnings: Tuple[str, ...] = ()
    matched: Tuple[int, ...] = ()


def _close(value: float, expected: float, tolerance: float) -> bool:
    return abs(value - expected) <= tolerance


def _cylindrical(faces: Sequence[Face], expected_axis: Optional[str]) -> List[Tuple[int, Face, str]]:
    out: List[Tuple[int, Face, str]] = []
    for i, f in enumerate(faces):
        if f.kind != "cylinder":
            continue
        possible = [a for a in _AXES if f.extent(a) > 0]
        if expected_axis:
            possible = [a for a in possible if a == expected_axis]
        if not possible:
            continue
        axis = expected_axis or max(possible, key=f.extent)
        out.append((i, f, axis))
    return out


def recognize_through_holes(
    faces: Sequence[Face],
    expected_axis: Optional[str] = None,
    *,
    expected_count: Optional[int] = None,
    expected_diameter: Optional[float] = None,
    expected_centers: Optional[Sequence[Tuple[float, float]]] = None,
    center_axes: Tuple[str, str] = ("x", "y"),
    through_length: Optional[float] = None,
    tolerance: Optional[float] = None,
) -> FeatureResult:
    """Recognize through-hole candidates from cylindrical faces.

    Parameter-aware: a candidate must match the expected diameter (within a
    fraction), through-length along ``expected_axis``, and one of the expected
    centres (in ``center_axes``) when those are supplied. Matching expected
    centres is what avoids counting rounded outside corners as holes.

    Confidence is ``high`` when centres and count both match, ``medium`` when the
    count matches without centres, ``low`` on a count mismatch.
    """
    candidates = _cylindrical(faces, expected_axis)
    warnings: List[str] = []
    tol = tolerance if tolerance is not None else max((expected_diameter or 0) * 0.75, 1.0)
    dia_tol = max((expected_diameter or 0) * 0.35, 0.75)
    len_tol = max((through_length or 0) * 0.35, 0.75)

    matched: List[int] = []
    for idx, face, axis in candidates:
        if expected_diameter is not None:
            r = face.radius_estimate(axis)
            if r is not None and not _close(2 * r, expected_diameter, dia_tol):
                continue
        if expected_axis and through_length is not None:
            if not _close(face.extent(expected_axis), through_length, len_tol):
                continue
        if expected_centers:
            c0, c1 = face.center(center_axes[0]), face.center(center_axes[1])
            if not any(
                _close(c0, ex, tol) and _close(c1, ey, tol) for ex, ey in expected_centers
            ):
                continue
        matched.append(idx)

    recognized = len(matched)
    passed = expected_count is None or recognized == expected_count
    confidence = CONFIDENCE_MEDIUM if passed else CONFIDENCE_LOW
    if expected_centers and expected_count is not None and passed:
        confidence = CONFIDENCE_HIGH
    if expected_count is not None and not passed:
        warnings.append(
            f"expected {expected_count} through-hole(s) but recognized {recognized}"
        )
    return FeatureResult(
        passed=passed,
        confidence=confidence,
        expected_count=expected_count,
        recognized_count=recognized,
        warnings=tuple(warnings),
        matched=tuple(matched),
    )


def recognize_center_cutout(
    faces: Sequence[Face],
    *,
    cutout_width: float,
    cutout_height: float,
    plate_thickness: float,
) -> FeatureResult:
    """Recognize a centred rectangular through-cutout from its planar walls.

    Looks for thin, centred vertical planar faces whose z-extent matches the
    plate thickness: two x-walls near +/-width/2 and two y-walls near
    +/-height/2. Recognized when at least one wall of each orientation is found.
    """
    x_walls = 0
    y_walls = 0
    matched: List[int] = []
    for i, f in enumerate(faces):
        if f.kind != "plane":
            continue
        if not _close(f.extent("z"), plate_thickness, max(1.0, plate_thickness * 0.35)):
            continue
        near_center = abs(f.center("x")) <= cutout_width / 2 + 2.0 and abs(
            f.center("y")
        ) <= cutout_height / 2 + 2.0
        if not near_center:
            continue
        if f.extent("x") <= 0.25 and _close(
            abs(f.center("x")), cutout_width / 2, max(2.0, cutout_width * 0.08)
        ):
            x_walls += 1
            matched.append(i)
        elif f.extent("y") <= 0.25 and _close(
            abs(f.center("y")), cutout_height / 2, max(2.0, cutout_height * 0.12)
        ):
            y_walls += 1
            matched.append(i)

    recognized = x_walls >= 1 and y_walls >= 1
    warnings: Tuple[str, ...] = ()
    if not recognized:
        warnings = ("could not find enough centred vertical planar cutout walls",)
    return FeatureResult(
        passed=recognized,
        confidence=CONFIDENCE_MEDIUM if recognized else CONFIDENCE_LOW,
        recognized_count=len(matched),
        warnings=warnings,
        matched=tuple(matched),
    )


def recognize_rounded_corners(
    faces: Sequence[Face],
    *,
    corner_radius: float,
    plate_width: float,
    plate_height: float,
    plate_thickness: float,
) -> FeatureResult:
    """Recognize >=4 vertical corner-radius cylinders on a plate.

    A corner candidate is a z-axis cylinder whose z-extent matches the plate
    thickness, whose radius matches ``corner_radius``, and whose centre lies out
    toward a corner (not near the plate centre).
    """
    candidates = _cylindrical(faces, "z")
    matched: List[int] = []
    for idx, face, _axis in candidates:
        if not _close(face.extent("z"), plate_thickness, max(1.0, plate_thickness * 0.35)):
            continue
        r = face.radius_estimate("z")
        if r is not None and not _close(r, corner_radius, max(0.75, corner_radius * 0.35)):
            continue
        if abs(face.center("x")) < plate_width * 0.35 or abs(face.center("y")) < plate_height * 0.35:
            continue
        matched.append(idx)

    recognized = len(matched) >= 4
    warnings: Tuple[str, ...] = ()
    if not recognized:
        warnings = (
            f"expected >=4 outside corner-radius candidates but recognized {len(matched)}",
        )
    return FeatureResult(
        passed=recognized,
        confidence=CONFIDENCE_MEDIUM if recognized else CONFIDENCE_LOW,
        expected_count=4,
        recognized_count=len(matched),
        warnings=warnings,
        matched=tuple(matched),
    )
