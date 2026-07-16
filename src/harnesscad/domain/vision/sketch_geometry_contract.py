"""Two-pass constrained sketch analysis: geometry contract checker (Studio-OSS).

Mined from **Studio-OSS** (``app/api/analyze-image/route.ts``, the "Two-Pass
Constrained Vision Pipeline"). Studio's fix for VLM geometry hallucination is
architectural, and its deterministic half is fully portable:

  * **Pass 1** asks the vision model only what is *literally drawn* -- outline,
    topology (flat 2D vs perspective/orthographic/isometric), thickness cues,
    internal features with orientation and a subtractive flag, symmetry types,
    and any visible profile/cross-section. No interpretation allowed.
  * **Pass 2** produces a Design Intent Representation, but *locked* to the
    Pass-1 analysis by hard constraints: a flat 2D sketch may only become a
    thin extrusion (family ``panel_pattern`` / ``extrude_profile``), every
    observed internal feature must appear in the DIR, no features may be
    invented, grid/crosshatch patterns must be marked subtractive, and the
    DIR's symmetry must be among the observed symmetry types.

The vision calls are learned and stay outside. What this module builds is the
deterministic *contract enforcement* between the passes: the typed
:class:`GeometricAnalysis` record for Pass 1 (with tolerant JSON parsing), the
:func:`check_dir_against_analysis` verifier that returns named violations of
every hard constraint (Studio only stated them in the prompt and trusted the
model; here they are actually checked), and the thin-extrusion thickness rule
(5 percent of the largest dimension, clamped to 2-5 mm) for flat sketches.

Pairs with :mod:`harnesscad.domain.vision.design_intent`. stdlib-only,
deterministic.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.vision.design_intent import DesignIntent, DirFeature

__all__ = [
    "TOPOLOGIES",
    "InternalFeature",
    "GeometricAnalysis",
    "parse_geometric_analysis",
    "thin_extrusion_thickness",
    "ContractViolation",
    "check_dir_against_analysis",
    "main",
]

TOPOLOGIES = ("flat_2d", "3d_perspective", "3d_orthographic", "isometric", "unclear")

#: Feature words in a literal Pass-1 description mapped to DIR feature types.
_FEATURE_SYNONYMS: Tuple[Tuple[str, str], ...] = (
    ("crosshatch", "crosshatch"),
    ("cross-hatch", "crosshatch"),
    ("grid", "grid"),
    ("lattice", "grid"),
    ("spoke", "spokes"),
    ("concentric", "concentric_rings"),
    ("teeth", "teeth"),
    ("tooth", "teeth"),
    ("hole", "holes"),
    ("slot", "slots"),
    ("rib", "ribs"),
    ("ridge", "ribs"),
    ("bore", "bore"),
    ("pattern", "pattern"),
)

#: Pattern types that must be marked subtractive in the DIR.
_SUBTRACTIVE_TYPES = frozenset({"crosshatch", "grid", "slots", "holes", "bore"})

_FLAT_2D_FAMILIES = frozenset({"panel_pattern", "extrude_profile"})


@dataclass(frozen=True)
class InternalFeature:
    """One literally-observed internal feature from Pass 1."""

    type: str                      # literal description, e.g. "crosshatch grid"
    orientation: str = "mixed"
    count_or_spacing: str = ""
    suggests_depth: bool = False
    is_subtractive: bool = False

    def canonical_types(self) -> Tuple[str, ...]:
        """DIR feature types this literal description maps onto."""
        lower = self.type.lower()
        found = []
        for word, canonical in _FEATURE_SYNONYMS:
            if word in lower and canonical not in found:
                found.append(canonical)
        return tuple(found)


@dataclass(frozen=True)
class GeometricAnalysis:
    """Typed Pass-1 record: what is literally visible in the sketch."""

    outline: str
    topology: str
    is_3d_drawing: bool
    internal_features: Tuple[InternalFeature, ...] = ()
    has_thickness: bool = False
    thickness_evidence: str = "none visible"
    symmetry: Tuple[str, ...] = ()
    profile_shape: str = "N/A"
    width_to_height: float = 1.0
    drawing_style: str = "unknown"

    def __post_init__(self) -> None:
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"unknown topology: {self.topology!r}")

    @property
    def is_flat_2d(self) -> bool:
        return self.topology == "flat_2d" and not self.is_3d_drawing

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outline": self.outline,
            "topology": self.topology,
            "is_3d_drawing": self.is_3d_drawing,
            "internal_features": [
                {"type": f.type, "orientation": f.orientation,
                 "count_or_spacing": f.count_or_spacing,
                 "suggests_depth": f.suggests_depth,
                 "is_subtractive": f.is_subtractive}
                for f in self.internal_features
            ],
            "thickness_cues": {"has_thickness": self.has_thickness,
                               "evidence": self.thickness_evidence},
            "symmetry": list(self.symmetry),
            "profile_shape": self.profile_shape,
            "estimated_dimensions_ratio": {"width_to_height": self.width_to_height},
            "drawing_style": self.drawing_style,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GeometricAnalysis":
        cues = payload.get("thickness_cues", {}) or {}
        ratio = payload.get("estimated_dimensions_ratio", {}) or {}
        features = tuple(
            InternalFeature(
                type=str(f.get("type", "")),
                orientation=str(f.get("orientation", "mixed")),
                count_or_spacing=str(f.get("count_or_spacing", "")),
                suggests_depth=bool(f.get("suggests_depth", False)),
                is_subtractive=bool(f.get("is_subtractive", False)),
            )
            for f in payload.get("internal_features", []) or []
        )
        topology = payload.get("topology", "unclear")
        return cls(
            outline=str(payload.get("outline", "irregular")),
            topology=topology if topology in TOPOLOGIES else "unclear",
            is_3d_drawing=bool(payload.get("is_3d_drawing", False)),
            internal_features=features,
            has_thickness=bool(cues.get("has_thickness", False)),
            thickness_evidence=str(cues.get("evidence", "none visible")),
            symmetry=tuple(str(s) for s in payload.get("symmetry", []) or []),
            profile_shape=str(payload.get("profile_shape", "N/A")),
            width_to_height=float(ratio.get("width_to_height", 1.0)),
            drawing_style=str(payload.get("drawing_style", "unknown")),
        )


def parse_geometric_analysis(text: str) -> Optional[GeometricAnalysis]:
    """Tolerantly parse a Pass-1 JSON blob; None on failure (fallback path)."""
    cleaned = re.sub(r"```(?:json)?\n?", "", text).replace("```", "").strip()
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict) or not payload.get("outline"):
        return None
    try:
        return GeometricAnalysis.from_dict(payload)
    except (TypeError, ValueError):
        return None


def thin_extrusion_thickness(largest_dimension_mm: float) -> float:
    """Studio's flat-sketch rule: 5% of the largest dimension, clamped 2-5 mm."""
    if largest_dimension_mm <= 0:
        return 2.0
    return min(5.0, max(2.0, 0.05 * largest_dimension_mm))


# --------------------------------------------------------------------------- #
# The contract checker
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContractViolation:
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def check_dir_against_analysis(
    dir_record: DesignIntent,
    analysis: GeometricAnalysis,
) -> Tuple[ContractViolation, ...]:
    """All hard-constraint violations of a Pass-2 DIR against the Pass-1 record.

    An empty result means the DIR is admissible. Mirrors Studio's Pass-2
    "HARD CONSTRAINTS" and "SELF-CHECK" lists, but as an actual verifier
    instead of prompt text the model is trusted to obey.
    """
    violations: List[ContractViolation] = []

    # 1. Flat 2D sketches may only become thin extrusions.
    if analysis.is_flat_2d and dir_record.family not in _FLAT_2D_FAMILIES:
        violations.append(ContractViolation(
            "FLAT_2D_FAMILY",
            f"topology is flat_2d but DIR family is '{dir_record.family}'; "
            f"only {sorted(_FLAT_2D_FAMILIES)} are admissible"))

    # 2. Every observed internal feature must appear in the DIR.
    dir_types = {f.type for f in dir_record.features}
    for observed in analysis.internal_features:
        expected = observed.canonical_types()
        if expected and not any(t in dir_types for t in expected):
            violations.append(ContractViolation(
                "MISSING_FEATURE",
                f"observed feature '{observed.type}' (maps to "
                f"{list(expected)}) is absent from the DIR"))

    # 3. No invented features: every DIR feature must trace to an observation.
    observed_types = set()
    for observed in analysis.internal_features:
        observed_types.update(observed.canonical_types())
    for feature in dir_record.features:
        if feature.likelihood >= 0.4 and feature.type not in observed_types:
            violations.append(ContractViolation(
                "INVENTED_FEATURE",
                f"DIR feature '{feature.type}' (likelihood "
                f"{feature.likelihood:.2f}) matches no observed feature"))

    # 4. Grid/crosshatch/hole patterns must be marked subtractive.
    for feature in dir_record.features:
        if feature.type in _SUBTRACTIVE_TYPES and not feature.is_subtractive:
            violations.append(ContractViolation(
                "NOT_SUBTRACTIVE",
                f"feature '{feature.type}' must be marked subtractive "
                "(cut into the base shape, not added)"))

    # 5. Symmetry must be among the observed types.
    if dir_record.symmetry_score > 0.7 and analysis.symmetry:
        observed_symmetry = {s.lower() for s in analysis.symmetry}
        claimed = dir_record.symmetry_type.lower()
        recognised = (
            any(claimed in s or s in claimed for s in observed_symmetry)
            or (claimed in ("radial", "rotational")
                and any(s.startswith(("radial", "rotational")) for s in observed_symmetry))
            or (claimed.startswith("mirror")
                and any(s.startswith("bilateral") for s in observed_symmetry))
        )
        if not recognised and "none" not in observed_symmetry:
            violations.append(ContractViolation(
                "SYMMETRY_MISMATCH",
                f"DIR claims '{dir_record.symmetry_type}' symmetry but the "
                f"analysis observed {sorted(observed_symmetry)}"))
        if "none" in observed_symmetry and len(observed_symmetry) == 1:
            violations.append(ContractViolation(
                "SYMMETRY_INVENTED",
                f"DIR claims '{dir_record.symmetry_type}' symmetry at "
                f"{dir_record.symmetry_score:.2f} but none was observed"))

    # 6. Flat sketches must not claim significant hollow depth.
    if analysis.is_flat_2d and dir_record.hollow_likelihood > 0.5:
        violations.append(ContractViolation(
            "DEPTH_INVENTED",
            "flat 2D sketch cannot support hollow-interior claims"))

    # 7. Thickness of a flat sketch must follow the thin-extrusion rule.
    if analysis.is_flat_2d:
        hints = dir_record.size_hint_mm
        thickness = hints.get("thickness")
        largest = max((v for k, v in hints.items() if k != "thickness"),
                      default=0.0)
        if thickness is not None and largest > 0:
            allowed = thin_extrusion_thickness(largest)
            if thickness > allowed + 1e-9:
                violations.append(ContractViolation(
                    "THICKNESS_EXCESSIVE",
                    f"flat sketch thickness {thickness:g}mm exceeds the "
                    f"thin-extrusion limit {allowed:g}mm"))

    return tuple(violations)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
_DEMO_ANALYSIS = {
    "outline": "circle",
    "outline_details": "single circle, uniform",
    "is_3d_drawing": False,
    "topology": "flat_2d",
    "internal_features": [
        {"type": "crosshatch grid of diagonal lines", "orientation": "diagonal_45",
         "count_or_spacing": "8 lines each way", "suggests_depth": False,
         "is_subtractive": True},
    ],
    "thickness_cues": {"has_thickness": False, "evidence": "none visible"},
    "symmetry": ["radial"],
    "profile_shape": "N/A",
    "estimated_dimensions_ratio": {"width_to_height": 1.0},
    "drawing_style": "sketch",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.vision.sketch_geometry_contract",
        description="Two-pass sketch-analysis geometry contract checker "
                    "(Studio-OSS).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="verify an admissible DIR and reject a "
                             "hallucinated one against the same analysis.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    analysis = parse_geometric_analysis(json.dumps(_DEMO_ANALYSIS))
    assert analysis is not None and analysis.is_flat_2d

    good = DesignIntent(
        family="panel_pattern", confidence=0.85,
        symmetry_type="radial", symmetry_score=0.9,
        roundness=0.9,
        features=(DirFeature("crosshatch", 0.9, 8, "diagonal", True),),
        size_hint_mm={"diameter": 25.0, "thickness": 2.0})
    violations = check_dir_against_analysis(good, analysis)
    assert not violations, [str(v) for v in violations]
    print("[selfcheck] admissible DIR: 0 violations")

    bad = DesignIntent(
        family="cylindrical_part", confidence=0.85,
        symmetry_type="mirror_y", symmetry_score=0.9,
        hollow_likelihood=0.8,
        features=(DirFeature("crosshatch", 0.9, 8, "diagonal", False),
                  DirFeature("teeth", 0.8, 20)),
        size_hint_mm={"diameter": 25.0, "thickness": 25.0})
    violations = check_dir_against_analysis(bad, analysis)
    codes = sorted(v.code for v in violations)
    for v in violations:
        print(f"  [reject] {v}")
    for expected in ("FLAT_2D_FAMILY", "INVENTED_FEATURE", "NOT_SUBTRACTIVE",
                     "SYMMETRY_MISMATCH", "DEPTH_INVENTED", "THICKNESS_EXCESSIVE"):
        assert expected in codes, (expected, codes)
    print(f"[selfcheck] hallucinated DIR: {len(violations)} violations caught")

    assert thin_extrusion_thickness(100.0) == 5.0
    assert thin_extrusion_thickness(25.0) == 2.0
    assert abs(thin_extrusion_thickness(60.0) - 3.0) < 1e-9
    print("[selfcheck] thin-extrusion thickness rule OK")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
