"""Target-spec extraction from a free-form CAD prompt.

An evaluation agent should never score a design in a vacuum: it first parses the
user's prompt into a structured *target specification* -- explicit dimensions,
target aspect ratios, whether symmetry is wanted and with what confidence, which
features are requested and in what counts, and a coarse texture hint -- and
then scores the produced geometry *against that spec*. The extraction itself
is entirely deterministic (regex + lexicons), which is what makes the
spec-conditioned half of the score reproducible.

Key ideas:

  * dimension regexes in both orders ("30 mm height" and "height of 30mm")
    for height/width/diameter/thickness/depth/bore;
  * target height/width ratio derived from the extracted dimensions;
  * a two-tier symmetry lexicon: explicit words ("symmetric", "centered", ...)
    at 0.95 confidence, and object families that imply symmetry ("gear",
    "vase", "flange", ...) at 0.75;
  * feature patterns with count capture ("12 ribs", "20 teeth", "4 mounting
    holes") plus keyword features (bore, fillet, chamfer, slots, cutout,
    pattern);
  * texture hint (ribbed / smooth / patterned / unknown);
  * per-family *ideal aspect-ratio ranges* used when the prompt states no
    ratio (gears and plates are flat, vases and columns are tall, brackets
    are in between).

This complements ``harnesscad.domain.spec.part_brief_parser`` (which parses two
fixed part families into full build specs); this module extracts a *scoring
target* from any prompt.

stdlib-only (``re``, ``dataclasses``), deterministic.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "TargetSpec",
    "extract_target_spec",
    "ideal_ratio_range",
    "main",
]


# --------------------------------------------------------------------------- #
# Lexicons and patterns
# --------------------------------------------------------------------------- #
_NUM = r"(\d+(?:\.\d+)?)"

_DIM_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (rf"{_NUM}\s*mm\s*(?:height|tall|high)\b", "height"),
    (rf"height\s*(?:of|:)?\s*{_NUM}\s*mm", "height"),
    (rf"{_NUM}\s*mm\s*(?:width|wide)\b", "width"),
    (rf"width\s*(?:of|:)?\s*{_NUM}\s*mm", "width"),
    (rf"{_NUM}\s*mm\s*(?:diameter|dia)\b", "diameter"),
    (rf"diameter\s*(?:of|:)?\s*{_NUM}\s*mm", "diameter"),
    (rf"{_NUM}\s*mm\s*(?:thick|thickness)\b", "thickness"),
    (rf"thickness\s*(?:of|:)?\s*{_NUM}\s*mm", "thickness"),
    (rf"{_NUM}\s*mm\s*(?:depth|deep)\b", "depth"),
    (rf"{_NUM}\s*mm\s*(?:bore|hole)\b", "bore_diameter"),
)

_EXPLICIT_SYMMETRY_WORDS = (
    "symmetric", "symmetrical", "centered", "centred", "balanced",
    "uniform", "round", "circular", "cylindrical", "radial",
)
_IMPLICIT_SYMMETRY_OBJECTS = (
    "gear", "vase", "bottle", "cylinder", "sphere", "torus", "wheel",
    "disc", "disk", "ring", "flange", "bearing", "pulley",
)

_FEATURE_PATTERNS: Tuple[Tuple[str, str, bool], ...] = (
    # (regex, feature name, regex captures a count)
    (r"(\d+)\s*(?:vertical\s+|horizontal\s+)?ribs?\b", "ribs", True),
    (r"\bribs?\s*(?:count)?\s*(\d+)", "ribs", True),
    (r"\bribs?\b|\bribbed\b", "ribs", False),
    (r"(\d+)\s*teeth\b", "teeth", True),
    (r"\bteeth\s*(\d+)", "teeth", True),
    (r"\bteeth\b", "teeth", False),
    (r"(\d+)\s*(?:mounting\s+)?holes?\b", "holes", True),
    (r"\bholes?\s*(\d+)", "holes", True),
    (r"\bholes?\b", "holes", False),
    (r"\bbore\b", "bore", False),
    (r"\bfillet", "fillet", False),
    (r"\bchamfer", "chamfer", False),
    (r"(\d+)\s*slots?\b", "slots", True),
    (r"\bslots?\b|\bventilation\b", "slots", False),
    (r"\bcutout", "cutout", False),
    (r"\bpattern", "pattern", False),
)

#: Design-family keyword -> ideal (min, max) height/width ratio range,
#: used when no explicit ratio target exists.
_FAMILY_RATIO_RANGES: Tuple[Tuple[Tuple[str, ...], Tuple[float, float]], ...] = (
    (("gear", "disc", "disk", "plate", "washer", "flange"), (0.05, 1.5)),
    (("vase", "bottle", "column", "tower"), (1.5, 8.0)),
    (("bracket", "mount", "frame"), (0.3, 4.0)),
)
_DEFAULT_RATIO_RANGE = (0.1, 6.0)


# --------------------------------------------------------------------------- #
# The spec record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TargetSpec:
    """Structured scoring target extracted from a prompt."""

    target_dims: Dict[str, float] = field(default_factory=dict)
    target_ratios: Dict[str, float] = field(default_factory=dict)
    wants_symmetry: bool = False
    symmetry_confidence: float = 0.3
    target_features: Tuple[str, ...] = ()
    feature_counts: Dict[str, int] = field(default_factory=dict)
    texture_hint: str = "unknown"          # ribbed | smooth | patterned | unknown
    ideal_ratio_min: float = _DEFAULT_RATIO_RANGE[0]
    ideal_ratio_max: float = _DEFAULT_RATIO_RANGE[1]

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_dims": dict(self.target_dims),
            "target_ratios": dict(self.target_ratios),
            "wants_symmetry": self.wants_symmetry,
            "symmetry_confidence": self.symmetry_confidence,
            "target_features": list(self.target_features),
            "feature_counts": dict(self.feature_counts),
            "texture_hint": self.texture_hint,
            "ideal_ratio_range": [self.ideal_ratio_min, self.ideal_ratio_max],
        }


def ideal_ratio_range(prompt: str) -> Tuple[float, float]:
    """The design-family ideal height/width ratio range for a prompt."""
    lower = prompt.lower()
    for keywords, bounds in _FAMILY_RATIO_RANGES:
        if any(k in lower for k in keywords):
            return bounds
    return _DEFAULT_RATIO_RANGE


def extract_target_spec(
    prompt: str,
    tree_parameters: Optional[Dict[str, Dict[str, object]]] = None,
) -> TargetSpec:
    """Parse the prompt (and optionally tree parameters) into a TargetSpec.

    ``tree_parameters`` has the shape: name -> record with at least
    ``value`` and ``unit`` ("mm" or "count"); mm values enter target_dims and
    count values whose key looks like a feature enter feature_counts.
    """
    lower = prompt.lower()

    target_dims: Dict[str, float] = {}
    for pattern, key in _DIM_PATTERNS:
        m = re.search(pattern, prompt, re.IGNORECASE)
        if m and key not in target_dims:
            target_dims[key] = float(m.group(1))

    if tree_parameters:
        for key, record in tree_parameters.items():
            unit = record.get("unit")
            value = record.get("value")
            if unit == "mm" and isinstance(value, (int, float)) and key not in target_dims:
                target_dims[key] = float(value)

    target_ratios: Dict[str, float] = {}
    h = target_dims.get("height") or target_dims.get("thickness") or target_dims.get("depth")
    w = target_dims.get("width") or target_dims.get("diameter")
    if h and w and w > 0:
        target_ratios["height_width"] = h / w

    explicit = any(word in lower for word in _EXPLICIT_SYMMETRY_WORDS)
    implied = any(word in lower for word in _IMPLICIT_SYMMETRY_OBJECTS)
    wants_symmetry = explicit or implied
    symmetry_confidence = 0.95 if explicit else 0.75 if implied else 0.3

    features: List[str] = []
    feature_counts: Dict[str, int] = {}
    for pattern, name, has_count in _FEATURE_PATTERNS:
        m = re.search(pattern, prompt, re.IGNORECASE)
        if not m:
            continue
        if name not in features:
            features.append(name)
        if has_count and m.group(1) and name not in feature_counts:
            feature_counts[name] = int(m.group(1))

    if tree_parameters:
        for key, record in tree_parameters.items():
            if record.get("unit") != "count":
                continue
            value = record.get("value")
            if not isinstance(value, (int, float)):
                continue
            name = None
            if re.search(r"rib", key, re.IGNORECASE):
                name = "ribs"
            elif re.search(r"teeth|tooth", key, re.IGNORECASE):
                name = "teeth"
            elif re.search(r"hole", key, re.IGNORECASE):
                name = "holes"
            elif re.search(r"segment", key, re.IGNORECASE):
                name = "segments"
            if name:
                feature_counts.setdefault(name, int(value))
                if name not in features:
                    features.append(name)

    texture = "unknown"
    if "ribs" in features or "ribbed" in lower or "groov" in lower:
        texture = "ribbed"
    elif "smooth" in lower or "plain" in lower:
        texture = "smooth"
    elif "pattern" in features or "repeating" in lower:
        texture = "patterned"

    ratio_min, ratio_max = ideal_ratio_range(prompt)
    return TargetSpec(
        target_dims=target_dims,
        target_ratios=target_ratios,
        wants_symmetry=wants_symmetry,
        symmetry_confidence=symmetry_confidence,
        target_features=tuple(features),
        feature_counts=feature_counts,
        texture_hint=texture,
        ideal_ratio_min=ratio_min,
        ideal_ratio_max=ratio_max,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.spec.prompt_spec_extract",
        description="Deterministic target-spec extraction from a CAD prompt "
                    "(Studio-OSS).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="extract specs from three representative prompts "
                             "and verify the parsed targets.")
    parser.add_argument("prompt", nargs="?", help="prompt to parse (prints JSON)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.prompt and not args.selfcheck:
        print(json.dumps(extract_target_spec(args.prompt).to_dict(),
                         indent=2, sort_keys=True))
        return 0
    if not args.selfcheck:
        parser.print_help()
        return 0

    spec = extract_target_spec(
        "Create a ribbed vase 200mm height and 60mm diameter with 12 vertical ribs")
    assert spec.target_dims == {"height": 200.0, "diameter": 60.0}
    assert abs(spec.target_ratios["height_width"] - 200.0 / 60.0) < 1e-9
    assert spec.wants_symmetry and spec.symmetry_confidence == 0.75
    assert spec.feature_counts.get("ribs") == 12
    assert spec.texture_hint == "ribbed"
    assert (spec.ideal_ratio_min, spec.ideal_ratio_max) == (1.5, 8.0)
    print(f"[selfcheck] vase: dims={spec.target_dims} ribs={spec.feature_counts['ribs']} "
          f"ratio={spec.target_ratios['height_width']:.2f}")

    spec = extract_target_spec(
        "A symmetric spur gear with 20 teeth, thickness of 10mm and a 5mm bore")
    assert spec.symmetry_confidence == 0.95
    assert spec.feature_counts.get("teeth") == 20
    assert "bore" in spec.target_features
    assert spec.target_dims.get("thickness") == 10.0
    assert spec.target_dims.get("bore_diameter") == 5.0
    assert (spec.ideal_ratio_min, spec.ideal_ratio_max) == (0.05, 1.5)
    print(f"[selfcheck] gear: teeth={spec.feature_counts['teeth']} "
          f"family range=({spec.ideal_ratio_min}, {spec.ideal_ratio_max})")

    spec = extract_target_spec("a smooth mounting bracket with 4 mounting holes")
    assert spec.feature_counts.get("holes") == 4
    assert spec.texture_hint == "smooth"
    assert (spec.ideal_ratio_min, spec.ideal_ratio_max) == (0.3, 4.0)
    print(f"[selfcheck] bracket: holes={spec.feature_counts['holes']} "
          f"texture={spec.texture_hint}")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
