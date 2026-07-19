"""annotation_schema -- typed enriched-annotation schema + tolerant LLM-JSON parser.

Ported from the CAD-Annotator reference repo (artifacts/api-server/src/lib/
compliance-engine.ts type definitions and gdt-prompts.ts response parsing).

CAD-Annotator's vision pipeline asks a model to return structured drawing
annotations as JSON: five annotation types (dimension, fcf, datum,
surface_finish, note) each with a percentage-coordinate bounding box, a
confidence score, and type-specific sub-fields. The model output is untrusted,
so every field is validated tolerantly: malformed annotations are silently
dropped, missing ids/colors/confidences receive deterministic fallbacks, and
numeric ranges are clamped.

Harness gap filled: harnesscad already parses raw OCR callout *text*
(harnesscad.domain.drawings.annotation_parser) and validates individual GD&T
feature-control frames (harnesscad.domain.drawings.gdt -- per-frame checks live
there and are NOT duplicated here). What was missing is the typed schema for a
*set* of enriched annotations as returned by a vision model, plus the tolerant
JSON parsing layer that turns raw LLM text into validated objects. Set-level
compliance rules over these objects live in
harnesscad.domain.drawings.annotation_set_compliance.

Pure stdlib, deterministic, no LLM calls; input is text/dicts, output is data.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# --------------------------------------------------------------------------- #
# Vocabularies (mirroring gdt-prompts.ts constants)
# --------------------------------------------------------------------------- #

VALID_ANNOTATION_TYPES: Tuple[str, ...] = (
    "dimension",
    "fcf",
    "datum",
    "surface_finish",
    "note",
)

VALID_DIMENSION_TYPES: Tuple[str, ...] = ("linear", "angular", "radius", "diameter")

# The 14 geometric characteristics, camelCase exactly as CAD-Annotator emits
# them (per-frame legality rules for these live in harnesscad.domain.drawings.gdt).
VALID_GEOMETRIC_CHARACTERISTICS: Tuple[str, ...] = (
    "position",
    "flatness",
    "straightness",
    "circularity",
    "cylindricity",
    "perpendicularity",
    "parallelism",
    "angularity",
    "profileOfLine",
    "profileOfSurface",
    "circularRunout",
    "totalRunout",
    "symmetry",
    "concentricity",
)

VALID_MATERIAL_CONDITIONS: Tuple[str, ...] = ("MMC", "LMC", "RFS")

# Colour palette for annotation bounding boxes, cycled by annotation index.
ANNOTATION_COLORS: Tuple[str, ...] = (
    "green",
    "blue",
    "red",
    "orange",
    "purple",
    "cyan",
    "yellow",
)

#: A datum letter is exactly one capital -- and `$` is not the way to say so.
#: Python's `$` also matches just before a trailing newline, so `^[A-Z]$`
#: accepted "A\n" while correctly rejecting "AB". These strings arrive as JSON
#: from a vision model, so "A\n" is representable and reachable; it then flowed
#: into DatumAnnotation.datum_letter and through the FCF datumReferences filter,
#: where it would never match the "A" it was meant to reference. \A and \Z are
#: absolute anchors and have no newline exception.
_DATUM_LETTER_RE = re.compile(r"\A[A-Z]\Z")


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in PERCENTAGES (0-100) of the image dimensions.

    ``x`` / ``y`` are the top-left corner; ``width`` / ``height`` are the box
    dimensions, also as percentages.
    """

    x: float
    y: float
    width: float
    height: float
    color: str

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "color": self.color,
        }


@dataclass(frozen=True, kw_only=True)
class AnnotationBase:
    """Fields common to every enriched annotation."""

    id: str
    label: str
    value: str
    view: str
    bounding_box: BoundingBox
    confidence: float
    needs_review: bool = False
    description: Optional[str] = None

    #: overridden by each subclass
    type: str = ""

    def _base_dict(self) -> dict:
        d = {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "value": self.value,
            "view": self.view,
            "bounding_box": self.bounding_box.to_dict(),
            "confidence": self.confidence,
            "needs_review": self.needs_review,
        }
        if self.description is not None:
            d["description"] = self.description
        return d

    def to_dict(self) -> dict:
        return self._base_dict()


@dataclass(frozen=True, kw_only=True)
class DimensionAnnotation(AnnotationBase):
    dimension_type: str  # linear | angular | radius | diameter
    nominal_value: float
    plus_tolerance: Optional[float] = None
    minus_tolerance: Optional[float] = None
    unit: Optional[str] = None
    type: str = "dimension"

    def to_dict(self) -> dict:
        d = self._base_dict()
        d["dimension_type"] = self.dimension_type
        d["nominal_value"] = self.nominal_value
        if self.plus_tolerance is not None:
            d["plus_tolerance"] = self.plus_tolerance
        if self.minus_tolerance is not None:
            d["minus_tolerance"] = self.minus_tolerance
        if self.unit is not None:
            d["unit"] = self.unit
        return d


@dataclass(frozen=True, kw_only=True)
class FcfAnnotation(AnnotationBase):
    geometric_characteristic: str  # one of VALID_GEOMETRIC_CHARACTERISTICS
    tolerance_value: float
    material_condition: Optional[str] = None  # MMC | LMC | RFS | None
    datum_references: Tuple[str, ...] = ()
    type: str = "fcf"

    def to_dict(self) -> dict:
        d = self._base_dict()
        d["geometric_characteristic"] = self.geometric_characteristic
        d["tolerance_value"] = self.tolerance_value
        d["material_condition"] = self.material_condition
        d["datum_references"] = list(self.datum_references)
        return d


@dataclass(frozen=True, kw_only=True)
class DatumAnnotation(AnnotationBase):
    datum_letter: str  # single uppercase letter A-Z
    type: str = "datum"

    def to_dict(self) -> dict:
        d = self._base_dict()
        d["datum_letter"] = self.datum_letter
        return d


@dataclass(frozen=True, kw_only=True)
class SurfaceFinishAnnotation(AnnotationBase):
    roughness_value: float
    process_note: Optional[str] = None
    type: str = "surface_finish"

    def to_dict(self) -> dict:
        d = self._base_dict()
        d["roughness_value"] = self.roughness_value
        if self.process_note is not None:
            d["process_note"] = self.process_note
        return d


@dataclass(frozen=True, kw_only=True)
class NoteAnnotation(AnnotationBase):
    type: str = "note"


Annotation = Union[
    DimensionAnnotation,
    FcfAnnotation,
    DatumAnnotation,
    SurfaceFinishAnnotation,
    NoteAnnotation,
]


# --------------------------------------------------------------------------- #
# Tolerant validation helpers (mirroring gdt-prompts.ts exactly)
# --------------------------------------------------------------------------- #


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` to the inclusive range [lo, hi]."""
    return max(lo, min(hi, value))


def _num(raw: Any) -> float:
    """Return a finite float, or NaN if ``raw`` is not a usable number.

    Booleans are rejected (Python bools are ints, but ``true`` is not a
    coordinate).
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return float("nan")
    v = float(raw)
    return v if not math.isnan(v) else float("nan")


def _get(raw: Dict[str, Any], camel: str, snake: str) -> Any:
    """Fetch a raw field accepting both camelCase (as the TS pipeline emits)
    and snake_case (as harness-native dicts use). camelCase wins."""
    if camel in raw:
        return raw[camel]
    return raw.get(snake)


def validate_bounding_box(raw: Any, fallback_color: str) -> Optional[BoundingBox]:
    """Validate and normalise a bounding box from raw LLM output.

    Rejects non-dict input, non-numeric or NaN coordinates and non-positive
    width/height. Clamps x, y to 0-100 and width, height to 0.1-100. Uses
    ``fallback_color`` when the box carries no string color.
    Returns None if the bounding box is invalid.
    """
    if not isinstance(raw, dict):
        return None

    x = _num(raw.get("x"))
    y = _num(raw.get("y"))
    width = _num(raw.get("width"))
    height = _num(raw.get("height"))

    if any(math.isnan(v) for v in (x, y, width, height)):
        return None
    if width <= 0 or height <= 0:
        return None

    color = raw.get("color")
    return BoundingBox(
        x=clamp(x, 0.0, 100.0),
        y=clamp(y, 0.0, 100.0),
        width=clamp(width, 0.1, 100.0),
        height=clamp(height, 0.1, 100.0),
        color=color if isinstance(color, str) else fallback_color,
    )


def validate_confidence(raw: Any) -> float:
    """Clamp a confidence score to [0, 1]; default 0.5 for non-numbers."""
    v = _num(raw)
    if math.isnan(v):
        return 0.5
    return clamp(v, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Per-type parsers (mirroring parseDimension / parseFcf / parseDatum /
# parseSurfaceFinish)
# --------------------------------------------------------------------------- #


def _parse_dimension(raw: Dict[str, Any], base: Dict[str, Any]) -> Optional[DimensionAnnotation]:
    dimension_type = _get(raw, "dimensionType", "dimension_type")
    if not isinstance(dimension_type, str) or dimension_type not in VALID_DIMENSION_TYPES:
        return None

    nominal_value = _num(_get(raw, "nominalValue", "nominal_value"))
    if math.isnan(nominal_value):
        return None

    plus_tolerance = _num(_get(raw, "plusTolerance", "plus_tolerance"))
    minus_tolerance = _num(_get(raw, "minusTolerance", "minus_tolerance"))
    unit = raw.get("unit")

    return DimensionAnnotation(
        dimension_type=dimension_type,
        nominal_value=nominal_value,
        plus_tolerance=None if math.isnan(plus_tolerance) else plus_tolerance,
        minus_tolerance=None if math.isnan(minus_tolerance) else minus_tolerance,
        unit=unit if isinstance(unit, str) else None,
        **base,
    )


def _parse_fcf(raw: Dict[str, Any], base: Dict[str, Any]) -> Optional[FcfAnnotation]:
    gc = _get(raw, "geometricCharacteristic", "geometric_characteristic")
    if not isinstance(gc, str) or gc not in VALID_GEOMETRIC_CHARACTERISTICS:
        return None

    tolerance_value = _num(_get(raw, "toleranceValue", "tolerance_value"))
    if math.isnan(tolerance_value):
        return None

    material_condition = None
    raw_mc = _get(raw, "materialCondition", "material_condition")
    if isinstance(raw_mc, str) and raw_mc in VALID_MATERIAL_CONDITIONS:
        material_condition = raw_mc

    datum_references: Tuple[str, ...] = ()
    raw_refs = _get(raw, "datumReferences", "datum_references")
    if isinstance(raw_refs, list):
        datum_references = tuple(
            ref for ref in raw_refs
            if isinstance(ref, str) and _DATUM_LETTER_RE.match(ref)
        )[:3]

    return FcfAnnotation(
        geometric_characteristic=gc,
        tolerance_value=tolerance_value,
        material_condition=material_condition,
        datum_references=datum_references,
        **base,
    )


def _parse_datum(raw: Dict[str, Any], base: Dict[str, Any]) -> Optional[DatumAnnotation]:
    datum_letter = _get(raw, "datumLetter", "datum_letter")
    if not isinstance(datum_letter, str) or not _DATUM_LETTER_RE.match(datum_letter):
        return None
    return DatumAnnotation(datum_letter=datum_letter, **base)


def _parse_surface_finish(
    raw: Dict[str, Any], base: Dict[str, Any]
) -> Optional[SurfaceFinishAnnotation]:
    roughness_value = _num(_get(raw, "roughnessValue", "roughness_value"))
    if math.isnan(roughness_value):
        return None
    process_note = _get(raw, "processNote", "process_note")
    return SurfaceFinishAnnotation(
        roughness_value=roughness_value,
        process_note=process_note if isinstance(process_note, str) else None,
        **base,
    )


# --------------------------------------------------------------------------- #
# parse_annotation / parse_annotation_response
# --------------------------------------------------------------------------- #


def parse_annotation(raw: Any, index: int) -> Optional[Annotation]:
    """Parse a single raw annotation dict into a validated Annotation.

    Mirrors parseAnnotation in gdt-prompts.ts: unknown types and missing
    required per-type fields drop the annotation (return None); missing id
    falls back to ``ann_{index+1}``; missing colors cycle a 7-color palette by
    index; confidence is clamped to [0, 1] with a 0.5 default.
    """
    if not isinstance(raw, dict):
        return None

    ann_type = raw.get("type")
    if not isinstance(ann_type, str) or ann_type not in VALID_ANNOTATION_TYPES:
        return None

    raw_id = raw.get("id")
    ann_id = raw_id if isinstance(raw_id, str) and raw_id else "ann_%d" % (index + 1)

    label = raw.get("label") if isinstance(raw.get("label"), str) else ""
    value = raw.get("value") if isinstance(raw.get("value"), str) else ""
    view = raw.get("view") if isinstance(raw.get("view"), str) else "View 1"
    description = raw.get("description")
    description = description if isinstance(description, str) else None
    confidence = validate_confidence(raw.get("confidence"))
    raw_review = _get(raw, "needsReview", "needs_review")
    needs_review = raw_review if isinstance(raw_review, bool) else False

    fallback_color = ANNOTATION_COLORS[index % len(ANNOTATION_COLORS)]
    bounding_box = validate_bounding_box(
        _get(raw, "boundingBox", "bounding_box"), fallback_color
    )
    if bounding_box is None:
        return None

    base = {
        "id": ann_id,
        "label": label,
        "value": value,
        "view": view,
        "bounding_box": bounding_box,
        "confidence": confidence,
        "needs_review": needs_review,
        "description": description,
    }

    if ann_type == "dimension":
        return _parse_dimension(raw, base)
    if ann_type == "fcf":
        return _parse_fcf(raw, base)
    if ann_type == "datum":
        return _parse_datum(raw, base)
    if ann_type == "surface_finish":
        return _parse_surface_finish(raw, base)
    if ann_type == "note":
        return NoteAnnotation(**base)
    return None


def extract_json_blob(content: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the first ``{...}`` JSON object from possibly-fenced
    LLM text. Returns None on failure."""
    match = re.search(r"\{[\s\S]*\}", content or "")
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_annotation_response(
    content: str,
) -> Tuple[List[Annotation], List[str], Optional[str]]:
    """Parse a full LLM JSON response into (annotations, views, description).

    Mirrors parseGdtResponse in gdt-prompts.ts: tolerates markdown code
    fences, silently drops malformed annotations, defaults views to
    ["View 1"], and returns description only when it is a string.
    """
    parsed = extract_json_blob(content)
    if parsed is None:
        return [], ["View 1"], None

    raw_annotations = parsed.get("annotations")
    if not isinstance(raw_annotations, list):
        raw_annotations = []

    annotations: List[Annotation] = []
    for i, raw in enumerate(raw_annotations):
        ann = parse_annotation(raw, i)
        if ann is not None:
            annotations.append(ann)

    raw_views = parsed.get("views")
    if isinstance(raw_views, list):
        views = [v for v in raw_views if isinstance(v, str) and v]
    else:
        views = ["View 1"]
    if not views:
        views = ["View 1"]

    description = parsed.get("description")
    description = description if isinstance(description, str) else None

    return annotations, views, description


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_SYNTHETIC_RESPONSE = """Here is my analysis:
```json
{
  "annotations": [
    {"id": "ann_1", "type": "dimension", "label": "40.2 +/-0.1", "value": "40.2",
     "view": "Front View",
     "boundingBox": {"x": 10, "y": 20, "width": 15, "height": 8, "color": "green"},
     "confidence": 0.95, "dimensionType": "linear", "nominalValue": 40.2,
     "plusTolerance": 0.1, "minusTolerance": -0.1, "unit": "mm"},
    {"type": "fcf", "label": "Position 0.05 MMC A B C", "value": "0.05",
     "view": "Front View",
     "boundingBox": {"x": 30, "y": 40, "width": 200, "height": 6},
     "confidence": 1.7, "geometricCharacteristic": "position",
     "toleranceValue": 0.05, "materialCondition": "MMC",
     "datumReferences": ["A", "B", "C", "D", "bogus"]},
    {"id": "ann_3", "type": "datum", "label": "Datum A", "value": "A",
     "view": "Front View",
     "boundingBox": {"x": 50, "y": 60, "width": 5, "height": 5, "color": "red"},
     "confidence": 0.97, "datumLetter": "A"},
    {"id": "bad_1", "type": "datum", "label": "Datum ?", "value": "?",
     "boundingBox": {"x": 1, "y": 1, "width": 2, "height": 2},
     "datumLetter": "abc"},
    {"id": "bad_2", "type": "surface_finish", "label": "Ra ?", "value": "?",
     "boundingBox": {"x": 1, "y": 1, "width": 0, "height": 2},
     "roughnessValue": 1.6},
    {"id": "ann_4", "type": "surface_finish", "label": "Ra 1.6", "value": "1.6",
     "view": "Side View",
     "boundingBox": {"x": 70, "y": 30, "width": 8, "height": 8, "color": "orange"},
     "confidence": 0.82, "roughnessValue": 1.6, "processNote": "Ground"},
    {"id": "ann_5", "type": "note", "label": "GENERAL NOTE",
     "value": "DIMENSIONS ARE IN MM", "view": "Title Block",
     "boundingBox": {"x": 5, "y": 90, "width": 30, "height": 5},
     "confidence": "high"}
  ],
  "views": ["Front View", "Side View", "Title Block"],
  "description": "Synthetic bracket drawing"
}
```"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` parses a synthetic fenced LLM response
    containing valid, salvageable, and invalid annotations and asserts the
    tolerant-parsing invariants."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.drawings.annotation_schema",
        description="Typed enriched-annotation schema + tolerant LLM-JSON "
        "parser (ported from CAD-Annotator).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="parse a synthetic fenced LLM response and print the validated "
        "annotations.",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit parsed annotations as JSON."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    annotations, views, description = parse_annotation_response(_SYNTHETIC_RESPONSE)

    if args.json:
        print(
            json.dumps(
                {
                    "annotations": [a.to_dict() for a in annotations],
                    "views": views,
                    "description": description,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("parsed %d annotation(s); views=%s" % (len(annotations), views))
        for a in annotations:
            print(
                "  %-6s %-15s id=%-6s conf=%.2f color=%s"
                % (a.type, a.label[:15], a.id, a.confidence, a.bounding_box.color)
            )

    fcf = next((a for a in annotations if a.type == "fcf"), None)
    note = next((a for a in annotations if a.type == "note"), None)
    ok = (
        len(annotations) == 5  # two invalid annotations dropped
        and fcf is not None
        and fcf.id == "ann_2"  # fallback id by index
        and fcf.confidence == 1.0  # clamped from 1.7
        and fcf.bounding_box.width == 100.0  # clamped from 200
        and fcf.bounding_box.color == "blue"  # palette cycled by index 1
        and fcf.datum_references == ("A", "B", "C")  # sliced to 3, bogus dropped
        and note is not None
        and note.confidence == 0.5  # non-numeric confidence default
        and views == ["Front View", "Side View", "Title Block"]
        and description == "Synthetic bracket drawing"
        and parse_annotation_response("no json here")[1] == ["View 1"]
    )
    if not ok:
        print("SELFCHECK FAILED")
        return 1
    print("selfcheck OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
