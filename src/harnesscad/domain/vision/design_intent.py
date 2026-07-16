"""Design Intent Representation (DIR): image perception to prompt (Studio-OSS).

Mined from **Studio-OSS** (``lib/image-dir.ts`` and
``app/api/analyze-image/route.ts``). Studio's image-to-CAD architecture rests
on a structured intermediate format between vision and generation: a vision
model handles *perception only* and must emit a Design Intent Representation
-- family classification, global proportions/symmetry, shape character
(taper, roundness, rectangularity, hollowness), a feature list with
likelihoods and subtractive flags, and constraint suggestions. The DIR is
then converted to a generation prompt by *pure template logic with no model*,
so the model in the loop never hallucinates geometry specs: perception is
learned, prompt assembly is deterministic and auditable.

This module carries the deterministic halves:

  * the typed :class:`DesignIntent` record with the seven-family taxonomy and
    validation (likelihoods and scores in [0, 1], families closed-vocabulary);
  * :func:`parse_design_intent` -- tolerant JSON extraction from raw model
    text (markdown-fence stripping, minimum-field validation) that returns
    ``None`` instead of raising, mirroring Studio's fall-back behaviour;
  * :func:`design_intent_to_prompt` -- the full deterministic DIR-to-prompt
    template: family phrasing, size hints, ratio, symmetry sentences, taper /
    hollow / cross-section characteristics, per-feature sentences with
    explicit SUBTRACTIVE marking (cut material, do not add), construction
    strategy and notes pass-through, preferred axis, and detail-level
    guidance.

The vision call that produces the DIR JSON stays outside this module.
stdlib-only, deterministic.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DIR_FAMILIES",
    "DirFeature",
    "DesignIntent",
    "parse_design_intent",
    "design_intent_to_prompt",
    "main",
]

DIR_FAMILIES = (
    "revolve_profile",
    "extrude_profile",
    "boxy_enclosure",
    "cylindrical_part",
    "panel_pattern",
    "gear_mechanism",
    "bracket_mount",
    "unknown",
)

_FAMILY_PHRASES: Dict[str, str] = {
    "revolve_profile": "revolved profile shape (like a vase or bottle)",
    "extrude_profile": "extruded profile shape (like a plate or bracket)",
    "boxy_enclosure": "rectangular enclosure box",
    "cylindrical_part": "cylindrical part",
    "panel_pattern": "flat panel with pattern",
    "gear_mechanism": "gear mechanism",
    "bracket_mount": "mounting bracket",
    "unknown": "3D object",
}


@dataclass(frozen=True)
class DirFeature:
    """One perceived feature with likelihood and additive/subtractive nature."""

    type: str
    likelihood: float
    count_estimate: Optional[int] = None
    direction: Optional[str] = None
    is_subtractive: bool = False

    def __post_init__(self) -> None:
        if not self.type:
            raise ValueError("feature type must be non-empty")
        if not 0.0 <= self.likelihood <= 1.0:
            raise ValueError(f"feature '{self.type}' likelihood out of [0,1]")


@dataclass(frozen=True)
class DesignIntent:
    """The structured perception record a vision pass must produce."""

    family: str
    confidence: float
    height_width_ratio: float = 0.0
    symmetry_type: str = "asymmetric"
    symmetry_score: float = 0.0
    orientation: str = "upright"
    detail_level: float = 0.5
    taper_ratio: float = 1.0
    roundness: float = 0.0
    rectangularity: float = 0.0
    hollow_likelihood: float = 0.0
    features: Tuple[DirFeature, ...] = ()
    prefer_symmetry_axis: str = "Z"
    size_hint_mm: Dict[str, float] = field(default_factory=dict)
    construction_strategy: Optional[str] = None
    construction_notes: Optional[str] = None

    def __post_init__(self) -> None:
        if self.family not in DIR_FAMILIES:
            raise ValueError(f"unknown DIR family: {self.family!r}")
        for name in ("confidence", "symmetry_score", "detail_level",
                     "roundness", "rectangularity", "hollow_likelihood"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} out of [0,1]: {value}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "confidence": self.confidence,
            "construction_strategy": self.construction_strategy,
            "global": {
                "height_width_ratio": self.height_width_ratio,
                "symmetry": {"type": self.symmetry_type, "score": self.symmetry_score},
                "orientation": self.orientation,
                "detail_level": self.detail_level,
            },
            "shape": {
                "taper_ratio": self.taper_ratio,
                "roundness": self.roundness,
                "rectangularity": self.rectangularity,
                "hollow_likelihood": self.hollow_likelihood,
            },
            "features": [
                {"type": f.type, "likelihood": f.likelihood,
                 "count_estimate": f.count_estimate, "direction": f.direction,
                 "is_subtractive": f.is_subtractive}
                for f in self.features
            ],
            "constraints_suggestions": {
                "prefer_symmetry_axis": self.prefer_symmetry_axis,
                "size_hint_mm": dict(self.size_hint_mm),
                "construction_notes": self.construction_notes,
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DesignIntent":
        glob = payload.get("global", {}) or {}
        shape = payload.get("shape", {}) or {}
        sym = glob.get("symmetry", {}) or {}
        cons = payload.get("constraints_suggestions", {}) or {}
        features = []
        for f in payload.get("features", []) or []:
            count = f.get("count_estimate")
            features.append(DirFeature(
                type=str(f.get("type", "")),
                likelihood=float(f.get("likelihood", 0.0)),
                count_estimate=int(count) if count is not None else None,
                direction=f.get("direction"),
                is_subtractive=bool(f.get("is_subtractive", False)),
            ))
        family = payload.get("family", "unknown")
        return cls(
            family=family if family in DIR_FAMILIES else "unknown",
            confidence=float(payload.get("confidence", 0.0)),
            height_width_ratio=float(glob.get("height_width_ratio", 0.0)),
            symmetry_type=str(sym.get("type", "asymmetric")),
            symmetry_score=float(sym.get("score", 0.0)),
            orientation=str(glob.get("orientation", "upright")),
            detail_level=float(glob.get("detail_level", 0.5)),
            taper_ratio=float(shape.get("taper_ratio", 1.0)),
            roundness=float(shape.get("roundness", 0.0)),
            rectangularity=float(shape.get("rectangularity", 0.0)),
            hollow_likelihood=float(shape.get("hollow_likelihood", 0.0)),
            features=tuple(features),
            prefer_symmetry_axis=str(cons.get("prefer_symmetry_axis", "Z") or "Z"),
            size_hint_mm={k: float(v) for k, v in
                          (cons.get("size_hint_mm", {}) or {}).items()},
            construction_strategy=payload.get("construction_strategy"),
            construction_notes=cons.get("construction_notes"),
        )


def parse_design_intent(text: str) -> Optional[DesignIntent]:
    """Tolerantly parse DIR JSON from raw model output.

    Strips markdown code fences, requires at least ``family`` and ``global``
    fields, and returns ``None`` on any failure instead of raising -- the
    caller falls back to another perception pass.
    """
    cleaned = re.sub(r"```(?:json)?\n?", "", text).replace("```", "").strip()
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if not payload.get("family") or not payload.get("global"):
        return None
    try:
        return DesignIntent.from_dict(payload)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Deterministic DIR -> prompt template
# --------------------------------------------------------------------------- #
_FEATURE_SENTENCES: Dict[str, str] = {
    "bore": "Include a center bore hole (SUBTRACTIVE).",
    "fillet": "Apply fillets to edges for smooth transitions.",
    "chamfer": "Apply chamfers to exposed edges.",
}


def _feature_sentence(f: DirFeature) -> str:
    sub = " (SUBTRACTIVE: cut/remove material, do NOT add)" if f.is_subtractive else ""
    if f.type in _FEATURE_SENTENCES:
        return _FEATURE_SENTENCES[f.type]
    if f.type == "ribs":
        count = f.count_estimate if f.count_estimate is not None else "several"
        direction = f.direction or "vertical"
        return (f"Add {count} {direction} ribs evenly spaced around the "
                f"circumference{sub}.")
    if f.type == "teeth":
        count = f.count_estimate if f.count_estimate is not None else 20
        return f"Include {count} gear teeth evenly distributed around the circumference{sub}."
    if f.type == "holes":
        count = f.count_estimate if f.count_estimate is not None else 4
        return f"Add {count} mounting holes{sub}."
    if f.type == "slots":
        return f"Include ventilation slots{sub}."
    if f.type == "pattern":
        direction = f.direction or "surface"
        count = f.count_estimate if f.count_estimate is not None else "multiple"
        return f"Add a repeating {direction} pattern with ~{count} elements{sub}."
    if f.type in ("crosshatch", "grid"):
        return (f"Cut a {f.type} grid pattern through the surface using boolean "
                f"SUBTRACTION with slots in two perpendicular directions{sub}.")
    return f"Include {f.type} feature{sub}."


def design_intent_to_prompt(
    dir_record: DesignIntent,
    *,
    likelihood_floor: float = 0.4,
    flat_2d: Optional[bool] = None,
    profile_shape: Optional[str] = None,
) -> str:
    """Assemble a generation prompt from a DIR with pure template logic.

    ``flat_2d`` and ``profile_shape`` carry findings from a prior geometric
    analysis pass (see
    :mod:`harnesscad.domain.vision.sketch_geometry_contract`): a flat 2D
    sketch forces a thin-extrusion instruction, and a detected profile forces
    an exact-profile revolve/extrude instruction.
    """
    parts: List[str] = []

    if dir_record.construction_strategy:
        parts.append(dir_record.construction_strategy)
    else:
        parts.append(f"Create a {_FAMILY_PHRASES[dir_record.family]}.")

    hints = dir_record.size_hint_mm
    for key, label in (("height", "Target height"), ("width", "Target width"),
                       ("diameter", "Target diameter"), ("thickness", "Thickness")):
        if hints.get(key):
            parts.append(f"{label}: {hints[key]:g}mm.")

    if dir_record.height_width_ratio > 0:
        parts.append("Height-to-width ratio approximately "
                     f"{dir_record.height_width_ratio:.1f}.")

    if dir_record.symmetry_score > 0.7:
        if "radial" in dir_record.symmetry_type or "rotational" in dir_record.symmetry_type:
            parts.append("Radially symmetric around the central axis.")
        else:
            parts.append("Mirror symmetry should be high.")

    if 0 < dir_record.taper_ratio < 0.8:
        parts.append(f"Tapers toward the top with taper ratio "
                     f"~{dir_record.taper_ratio:.2f}.")
    if dir_record.hollow_likelihood > 0.5:
        parts.append("The object appears hollow: include wall thickness and "
                     "an interior cavity.")
    if dir_record.roundness > 0.7:
        parts.append("Predominantly round/circular cross-section.")
    elif dir_record.rectangularity > 0.7:
        parts.append("Predominantly rectangular/boxy cross-section.")

    for feature in dir_record.features:
        if feature.likelihood >= likelihood_floor:
            parts.append(_feature_sentence(feature))

    if dir_record.construction_notes:
        parts.append(dir_record.construction_notes)
    if dir_record.prefer_symmetry_axis:
        parts.append(f"Prefer {dir_record.prefer_symmetry_axis}-axis as the "
                     "primary symmetry axis.")
    if dir_record.detail_level > 0.7:
        parts.append("The model should be detailed with precise geometry.")
    elif dir_record.detail_level < 0.3:
        parts.append("Keep the geometry simple and clean.")

    if flat_2d:
        parts.append("IMPORTANT: the sketch is a FLAT 2D drawing with no depth "
                     "cues. Create this as a thin extruded shape; do NOT "
                     "interpret it as a solid 3D primitive.")
    if profile_shape and profile_shape not in ("N/A", "none"):
        parts.append(f"The sketch shows a profile/cross-section: {profile_shape}. "
                     "Use this exact profile for revolution or extrusion.")

    return " ".join(parts)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
_DEMO_JSON = """```json
{
  "family": "panel_pattern",
  "confidence": 0.86,
  "global": {
    "height_width_ratio": 1.0,
    "symmetry": {"type": "radial", "score": 0.9},
    "orientation": "upright",
    "detail_level": 0.6
  },
  "shape": {"taper_ratio": 1.0, "roundness": 0.9, "rectangularity": 0.1,
            "hollow_likelihood": 0.1},
  "features": [
    {"type": "crosshatch", "likelihood": 0.9, "count_estimate": 8,
     "direction": "diagonal", "is_subtractive": true},
    {"type": "bore", "likelihood": 0.2}
  ],
  "constraints_suggestions": {
    "prefer_symmetry_axis": "Z",
    "size_hint_mm": {"diameter": 25, "thickness": 3},
    "construction_notes": "Use boolean subtraction loops for the grid."
  }
}
```"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.vision.design_intent",
        description="Design Intent Representation and deterministic "
                    "DIR-to-prompt template (Studio-OSS).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="parse a fenced DIR JSON, round-trip it, and "
                             "render the deterministic prompt.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    dir_record = parse_design_intent(_DEMO_JSON)
    assert dir_record is not None
    assert dir_record.family == "panel_pattern"
    assert dir_record.features[0].is_subtractive
    round_tripped = DesignIntent.from_dict(dir_record.to_dict())
    assert round_tripped == dir_record
    print(f"[selfcheck] parsed family={dir_record.family} "
          f"confidence={dir_record.confidence}")

    prompt = design_intent_to_prompt(dir_record, flat_2d=True)
    assert "SUBTRACTION" in prompt
    assert "FLAT 2D" in prompt
    assert "bore" not in prompt.lower(), "low-likelihood feature must be dropped"
    print(f"[selfcheck] prompt ({len(prompt)} chars): {prompt[:120]}...")

    assert parse_design_intent("not json at all") is None
    assert parse_design_intent('{"global": {}}') is None
    print("[selfcheck] malformed inputs return None")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
