"""dfm_review -- Design for Manufacturability review over enriched annotations.

The LLM path is optional (an injected callable), and deterministic checks keep
the review useful with no model at all:

  * datum scheme completeness (< 3 unique datum letters -> warning);
  * missing tolerance (dimensions with neither plus nor minus tolerance);
  * surface finish consistency (Ra-vs-tightest-tolerance rule-of-thumb table);
  * over-tolerancing (FCF tolerance below a typical machining capability floor).

Harness gap filled: harnesscad had compliance checks (per-frame in
harnesscad.domain.drawings.gdt, which is NOT duplicated here, and set-level in
harnesscad.domain.drawings.annotation_set_compliance) but no manufacturability
feedback layer. Deterministic findings always run; LLM findings are merged
afterwards with the deterministic datum_scheme_completeness finding taking
precedence over any LLM duplicate, and LLM failures are swallowed.

Pure stdlib, deterministic core; the only nondeterminism is an injected
``llm`` callable which the caller controls.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from harnesscad.domain.drawings.annotation_schema import (
    Annotation,
    BoundingBox,
    DatumAnnotation,
    DimensionAnnotation,
    FcfAnnotation,
    SurfaceFinishAnnotation,
    extract_json_blob,
)

# --------------------------------------------------------------------------- #
# Types and vocabularies
# --------------------------------------------------------------------------- #

VALID_CATEGORIES = frozenset(
    {
        "over_tolerancing",
        "missing_tolerance",
        "datum_scheme_completeness",
        "surface_finish_consistency",
        "general",
    }
)

VALID_SEVERITIES = frozenset({"error", "warning", "info"})


@dataclass(frozen=True)
class DfmFinding:
    """A single Design for Manufacturability finding."""

    id: str
    category: str  # one of VALID_CATEGORIES
    severity: str  # "error" | "warning" | "info"
    description: str
    recommendation: str
    related_annotation_ids: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "category": self.category,
            "severity": self.severity,
            "description": self.description,
            "recommendation": self.recommendation,
        }
        if self.related_annotation_ids:
            d["related_annotation_ids"] = list(self.related_annotation_ids)
        return d


# --------------------------------------------------------------------------- #
# Deterministic pre-check tables
# --------------------------------------------------------------------------- #

# Rule-of-thumb pairing of the tightest total tolerance band (mm) on the
# drawing with the roughest acceptable surface finish (Ra, um). A band below
# the threshold in column 0 needs a finish at or below column 1.
SURFACE_FINISH_TABLE: Tuple[Tuple[float, float], ...] = (
    (0.025, 0.4),
    (0.1, 0.8),
    (0.25, 1.6),
)

# Typical machining capability floor (mm). Geometric tolerances tighter than
# this are flagged as probable over-tolerancing.
MACHINING_CAPABILITY_FLOOR_MM = 0.01


# --------------------------------------------------------------------------- #
# Deterministic pre-checks
# --------------------------------------------------------------------------- #


def check_datum_scheme_completeness(
    annotations: Sequence[Annotation],
) -> Optional[DfmFinding]:
    """Warn when fewer than 3 unique datum letters are declared.

    Ported verbatim from checkDatumSchemeCompleteness in dfm-reviewer.ts: per
    ASME Y14.5-2018 a fully constrained part typically requires primary,
    secondary, and tertiary datums.
    """
    unique = sorted(
        {ann.datum_letter for ann in annotations if isinstance(ann, DatumAnnotation)}
    )
    if len(unique) >= 3:
        return None

    datum_ids = tuple(
        ann.id for ann in annotations if isinstance(ann, DatumAnnotation)
    )
    if not unique:
        description = (
            "No datums detected. A fully constrained part typically requires "
            "at least 3 datums (primary, secondary, tertiary)."
        )
    else:
        description = (
            "Only %d unique datum(s) detected (%s). A fully constrained part "
            "typically requires at least 3 datums (primary, secondary, "
            "tertiary)." % (len(unique), ", ".join(unique))
        )
    return DfmFinding(
        id="dfm_datum_scheme_completeness",
        category="datum_scheme_completeness",
        severity="warning",
        description=description,
        recommendation="Review the drawing and add datum references to "
        "establish a complete datum reference frame with primary, secondary, "
        "and tertiary datums.",
        related_annotation_ids=datum_ids,
    )


def check_missing_tolerance(annotations: Sequence[Annotation]) -> List[DfmFinding]:
    """Flag dimensions carrying neither a plus nor a minus tolerance.

    Severity is "warning" for size-controlling dimension types (linear,
    diameter) and "info" for angular/radius dimensions, which are more often
    basic or reference.
    """
    findings: List[DfmFinding] = []
    for ann in annotations:
        if not isinstance(ann, DimensionAnnotation):
            continue
        if ann.plus_tolerance is not None or ann.minus_tolerance is not None:
            continue
        severity = "warning" if ann.dimension_type in ("linear", "diameter") else "info"
        findings.append(
            DfmFinding(
                id="dfm_missing_tolerance_%s" % ann.id,
                category="missing_tolerance",
                severity=severity,
                description='Dimension "%s" (%s, nominal %s) has no plus or '
                "minus tolerance; its permitted variation is undefined unless "
                "covered by a general tolerance note."
                % (ann.label or ann.id, ann.dimension_type, ann.nominal_value),
                recommendation="Add an explicit tolerance to the dimension, or "
                "confirm it is covered by a title-block general tolerance or "
                "marked as basic/reference.",
                related_annotation_ids=(ann.id,),
            )
        )
    return findings


def _dimension_band(ann: DimensionAnnotation) -> Optional[float]:
    """Total tolerance band (mm) of a dimension, or None if untoleranced."""
    if ann.plus_tolerance is None and ann.minus_tolerance is None:
        return None
    plus = abs(ann.plus_tolerance) if ann.plus_tolerance is not None else 0.0
    minus = abs(ann.minus_tolerance) if ann.minus_tolerance is not None else 0.0
    band = plus + minus
    return band if band > 0 else None


def _tightest_band(annotations: Sequence[Annotation]) -> Optional[Tuple[float, str]]:
    """The tightest positive tolerance band on the drawing and the id of the
    annotation carrying it, over dimensions and FCFs."""
    best: Optional[Tuple[float, str]] = None
    for ann in annotations:
        band: Optional[float] = None
        if isinstance(ann, DimensionAnnotation):
            band = _dimension_band(ann)
        elif isinstance(ann, FcfAnnotation) and ann.tolerance_value > 0:
            band = ann.tolerance_value
        if band is not None and (best is None or band < best[0]):
            best = (band, ann.id)
    return best


def required_max_roughness(tolerance_band: float) -> Optional[float]:
    """The roughest acceptable Ra (um) for a given total tolerance band (mm)
    per SURFACE_FINISH_TABLE, or None when the band imposes no constraint."""
    for band_threshold, max_ra in SURFACE_FINISH_TABLE:
        if tolerance_band < band_threshold:
            return max_ra
    return None


def check_surface_finish_consistency(
    annotations: Sequence[Annotation],
) -> List[DfmFinding]:
    """Flag rough surface finishes paired with tight tolerances.

    Uses the tightest tolerance band on the drawing (over dimensions and
    FCFs) and the SURFACE_FINISH_TABLE rule of thumb: a band < 0.025 mm needs
    Ra <= 0.4, < 0.1 mm needs Ra <= 0.8, < 0.25 mm needs Ra <= 1.6.
    """
    tightest = _tightest_band(annotations)
    if tightest is None:
        return []
    band, band_ann_id = tightest
    max_ra = required_max_roughness(band)
    if max_ra is None:
        return []

    findings: List[DfmFinding] = []
    for ann in annotations:
        if not isinstance(ann, SurfaceFinishAnnotation):
            continue
        if ann.roughness_value <= max_ra:
            continue
        findings.append(
            DfmFinding(
                id="dfm_surface_finish_%s" % ann.id,
                category="surface_finish_consistency",
                severity="warning",
                description="Surface finish Ra %s is rougher than the Ra <= %s "
                "rule-of-thumb for the tightest tolerance band on the drawing "
                "(%s mm, annotation %s). Tight tolerances typically require "
                "finer finishes."
                % (ann.roughness_value, max_ra, band, band_ann_id),
                recommendation="Specify a finer surface finish (Ra <= %s) on "
                "the toleranced feature, or relax the tolerance if the finish "
                "is functionally adequate." % max_ra,
                related_annotation_ids=(ann.id, band_ann_id),
            )
        )
    return findings


def check_over_tolerancing(annotations: Sequence[Annotation]) -> List[DfmFinding]:
    """Flag FCF tolerances below the typical machining capability floor."""
    findings: List[DfmFinding] = []
    for ann in annotations:
        if not isinstance(ann, FcfAnnotation):
            continue
        if not (0 < ann.tolerance_value < MACHINING_CAPABILITY_FLOOR_MM):
            continue
        findings.append(
            DfmFinding(
                id="dfm_over_tolerancing_%s" % ann.id,
                category="over_tolerancing",
                severity="warning",
                description="%s tolerance %s mm is tighter than the typical "
                "machining capability floor of %s mm and will significantly "
                "increase manufacturing cost."
                % (
                    ann.geometric_characteristic,
                    ann.tolerance_value,
                    MACHINING_CAPABILITY_FLOOR_MM,
                ),
                recommendation="Verify the functional requirement; relax the "
                "tolerance toward >= %s mm or plan for grinding/lapping and "
                "100 percent inspection." % MACHINING_CAPABILITY_FLOOR_MM,
                related_annotation_ids=(ann.id,),
            )
        )
    return findings


def deterministic_dfm_findings(annotations: Sequence[Annotation]) -> List[DfmFinding]:
    """Run every deterministic DFM check, in a fixed order."""
    findings: List[DfmFinding] = []
    datum_finding = check_datum_scheme_completeness(annotations)
    if datum_finding is not None:
        findings.append(datum_finding)
    findings.extend(check_missing_tolerance(annotations))
    findings.extend(check_surface_finish_consistency(annotations))
    findings.extend(check_over_tolerancing(annotations))
    return findings


# --------------------------------------------------------------------------- #
# LLM prompt construction (buildDfmPrompt in dfm-reviewer.ts)
# --------------------------------------------------------------------------- #


def _summarize_annotation(ann: Annotation) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "id": ann.id,
        "type": ann.type,
        "label": ann.label,
        "value": ann.value,
        "confidence": ann.confidence,
    }
    if isinstance(ann, DimensionAnnotation):
        base.update(
            {
                "dimensionType": ann.dimension_type,
                "nominalValue": ann.nominal_value,
                "plusTolerance": ann.plus_tolerance,
                "minusTolerance": ann.minus_tolerance,
                "unit": ann.unit,
            }
        )
    elif isinstance(ann, FcfAnnotation):
        base.update(
            {
                "geometricCharacteristic": ann.geometric_characteristic,
                "toleranceValue": ann.tolerance_value,
                "materialCondition": ann.material_condition,
                "datumReferences": list(ann.datum_references),
            }
        )
    elif isinstance(ann, DatumAnnotation):
        base["datumLetter"] = ann.datum_letter
    elif isinstance(ann, SurfaceFinishAnnotation):
        base.update(
            {
                "roughnessValue": ann.roughness_value,
                "processNote": ann.process_note,
            }
        )
    return base


def build_dfm_prompt(annotations: Sequence[Annotation]) -> str:
    """Build the text-only DFM analysis prompt (structured annotation JSON
    summary, category definitions, and the output JSON contract)."""
    annotation_json = json.dumps(
        [_summarize_annotation(a) for a in annotations], indent=2
    )
    return (
        "You are an expert manufacturing engineer reviewing GD&T (Geometric "
        "Dimensioning and Tolerancing) annotations extracted from an "
        "engineering drawing. Analyze the following annotations for Design "
        "for Manufacturability (DFM) concerns.\n"
        "\n"
        "Annotations:\n"
        "%s\n"
        "\n"
        "Evaluate the annotations for the following DFM categories:\n"
        "\n"
        "1. **over_tolerancing**: Identify cases where multiple tight "
        "tolerances are specified that would significantly increase "
        "manufacturing cost. Look for unnecessarily tight geometric "
        "tolerances, redundant tolerance specifications, or tolerance values "
        "that are tighter than typical manufacturing capabilities.\n"
        "\n"
        "2. **missing_tolerance**: Identify critical features that appear to "
        "lack dimensional control. Look for dimensions without tolerances, "
        "features that should have geometric tolerances but don't, or "
        "incomplete tolerance specifications.\n"
        "\n"
        "3. **datum_scheme_completeness**: Evaluate whether the datum "
        "reference frame is complete and well-defined. Check if datums are "
        "properly ordered (primary, secondary, tertiary) and if the datum "
        "scheme adequately constrains the part.\n"
        "\n"
        "4. **surface_finish_consistency**: Check if surface finish values "
        "are consistent with the specified tolerances. Tight tolerances "
        "typically require finer surface finishes. Flag inconsistencies where "
        "rough surface finishes are paired with tight tolerances.\n"
        "\n"
        "For each finding, return a JSON object in this exact format:\n"
        "{\n"
        '  "findings": [\n'
        "    {\n"
        '      "id": "dfm_1",\n'
        '      "category": "over_tolerancing",\n'
        '      "severity": "warning",\n'
        '      "description": "Clear description of the issue",\n'
        '      "recommendation": "Specific corrective action",\n'
        '      "relatedAnnotationIds": ["ann_1", "ann_2"]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "Rules:\n"
        '- category MUST be one of: "over_tolerancing", "missing_tolerance", '
        '"datum_scheme_completeness", "surface_finish_consistency", "general"\n'
        '- severity MUST be one of: "error", "warning", "info"\n'
        "- Each finding MUST have a non-empty description and recommendation\n"
        "- relatedAnnotationIds should reference actual annotation IDs from "
        "the input\n"
        "- Only return valid JSON, no other text\n"
        "- Be specific and actionable in your recommendations\n"
        "- If no issues are found for a category, do not include empty findings"
    ) % annotation_json


# --------------------------------------------------------------------------- #
# Response parsing (parseSingleFinding / parseDfmResponse in dfm-reviewer.ts)
# --------------------------------------------------------------------------- #


def parse_single_finding(
    raw: Any, index: int, valid_annotation_ids: Set[str]
) -> Optional[DfmFinding]:
    """Validate a single raw finding dict; returns None when invalid.

    Identical rules to the TS source: drop invalid category/severity and
    empty description or recommendation; fall back to ``dfm_{index+1}`` ids;
    filter related annotation ids to the valid set.
    """
    if not isinstance(raw, dict):
        return None

    category = raw.get("category")
    if not isinstance(category, str) or category not in VALID_CATEGORIES:
        return None

    severity = raw.get("severity")
    if not isinstance(severity, str) or severity not in VALID_SEVERITIES:
        return None

    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        return None

    recommendation = raw.get("recommendation")
    if not isinstance(recommendation, str) or not recommendation.strip():
        return None

    raw_id = raw.get("id")
    finding_id = raw_id if isinstance(raw_id, str) and raw_id else "dfm_%d" % (index + 1)

    related: Tuple[str, ...] = ()
    raw_related = raw.get("relatedAnnotationIds")
    if raw_related is None:
        raw_related = raw.get("related_annotation_ids")
    if isinstance(raw_related, list):
        related = tuple(
            ref
            for ref in raw_related
            if isinstance(ref, str) and ref in valid_annotation_ids
        )

    return DfmFinding(
        id=finding_id,
        category=category,
        severity=severity,
        description=description.strip(),
        recommendation=recommendation.strip(),
        related_annotation_ids=related,
    )


def parse_dfm_response(content: str, valid_annotation_ids: Set[str]) -> List[DfmFinding]:
    """Parse LLM response content into validated DfmFinding objects,
    tolerating markdown fences and silently dropping malformed findings."""
    parsed = extract_json_blob(content)
    if parsed is None:
        return []

    raw_findings = parsed.get("findings")
    if not isinstance(raw_findings, list):
        return []

    findings: List[DfmFinding] = []
    for i, raw in enumerate(raw_findings):
        finding = parse_single_finding(raw, i, valid_annotation_ids)
        if finding is not None:
            findings.append(finding)
    return findings


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def review_dfm(
    annotations: Sequence[Annotation],
    llm: Optional[Callable[[str], str]] = None,
) -> List[DfmFinding]:
    """Generate DFM findings for an annotation set.

    1. Deterministic pre-checks always run first (datum scheme completeness,
       missing tolerances, surface finish consistency, over-tolerancing).
    2. If ``llm`` is provided (a callable prompt -> response text), its
       findings are parsed, validated, and merged. A deterministic
       datum_scheme_completeness finding takes precedence over any LLM
       duplicate of the same category.
    3. Any LLM failure is swallowed; the deterministic findings are still
       returned.
    """
    findings = deterministic_dfm_findings(annotations)
    has_det_datum_finding = any(
        f.category == "datum_scheme_completeness" for f in findings
    )

    if llm is not None:
        try:
            content = llm(build_dfm_prompt(annotations))
            valid_ids = {a.id for a in annotations}
            for llm_finding in parse_dfm_response(content, valid_ids):
                if (
                    llm_finding.category == "datum_scheme_completeness"
                    and has_det_datum_finding
                ):
                    continue
                findings.append(llm_finding)
        except Exception:
            # LLM call failed -- return only deterministic findings.
            pass

    return findings


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _bbox(i: int) -> BoundingBox:
    return BoundingBox(x=float(i), y=float(i), width=5.0, height=5.0, color="green")


def _base(i: int, label: str, value: str) -> dict:
    return {
        "id": "ann_%d" % i,
        "label": label,
        "value": value,
        "view": "Front View",
        "bounding_box": _bbox(i),
        "confidence": 0.9,
    }


def _synthetic_annotations() -> List[Annotation]:
    return [
        DatumAnnotation(datum_letter="A", **_base(1, "Datum A", "A")),
        # tight dimension: total band 0.02 mm
        DimensionAnnotation(
            dimension_type="linear",
            nominal_value=40.2,
            plus_tolerance=0.01,
            minus_tolerance=0.01,
            unit="mm",
            **_base(2, "40.2 +/-0.01", "40.2"),
        ),
        # untoleranced diameter -> missing_tolerance warning
        DimensionAnnotation(
            dimension_type="diameter",
            nominal_value=10.0,
            unit="mm",
            **_base(3, "D10", "10"),
        ),
        # rough finish paired with the tight band -> surface_finish_consistency
        SurfaceFinishAnnotation(
            roughness_value=3.2, **_base(4, "Ra 3.2", "3.2")
        ),
        # sub-capability FCF tolerance -> over_tolerancing
        FcfAnnotation(
            geometric_characteristic="flatness",
            tolerance_value=0.005,
            material_condition=None,
            datum_references=(),
            **_base(5, "Flatness 0.005", "0.005"),
        ),
    ]


_SYNTHETIC_LLM_RESPONSE = """```json
{
  "findings": [
    {"id": "dfm_llm_1", "category": "datum_scheme_completeness",
     "severity": "warning", "description": "duplicate of deterministic",
     "recommendation": "should be skipped"},
    {"id": "dfm_llm_2", "category": "general", "severity": "info",
     "description": "Consider a general tolerance note.",
     "recommendation": "Add an ISO 2768 note to the title block.",
     "relatedAnnotationIds": ["ann_3", "not_a_real_id"]},
    {"category": "bogus_category", "severity": "info",
     "description": "x", "recommendation": "y"}
  ]
}
```"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` runs the deterministic checks on a
    synthetic annotation set, then re-runs with a fake LLM callable and with a
    failing LLM callable, asserting the merge and swallow behaviour."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.drawings.dfm_review",
        description="DFM review over enriched annotations (ported from "
        "CAD-Annotator, deterministic side expanded). LLM optional.",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="run deterministic + fake-LLM + failing-LLM reviews on synthetic "
        "annotations and print the findings.",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit findings as JSON."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    annotations = _synthetic_annotations()
    det = review_dfm(annotations)
    merged = review_dfm(annotations, llm=lambda prompt: _SYNTHETIC_LLM_RESPONSE)

    def _boom(prompt: str) -> str:
        raise RuntimeError("llm down")

    swallowed = review_dfm(annotations, llm=_boom)

    if args.json:
        print(
            json.dumps(
                {
                    "deterministic": [f.to_dict() for f in det],
                    "merged": [f.to_dict() for f in merged],
                    "llm_failure": [f.to_dict() for f in swallowed],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("deterministic findings (%d):" % len(det))
        for f in det:
            print("  [%s] %s: %s" % (f.severity, f.category, f.description[:70]))
        print("merged with fake LLM (%d):" % len(merged))
        for f in merged:
            print("  [%s] %s: %s" % (f.severity, f.category, f.description[:70]))
        print("with failing LLM (%d findings, error swallowed)" % len(swallowed))

    det_categories = sorted(f.category for f in det)
    llm_general = [f for f in merged if f.id == "dfm_llm_2"]
    ok = (
        det_categories
        == [
            "datum_scheme_completeness",
            "missing_tolerance",
            "over_tolerancing",
            "surface_finish_consistency",
        ]
        and len(merged) == len(det) + 1  # duplicate + bogus dropped, general kept
        and len(llm_general) == 1
        and llm_general[0].related_annotation_ids == ("ann_3",)  # invalid id filtered
        and swallowed == det  # failure swallowed, deterministic intact
        and required_max_roughness(0.02) == 0.4
        and required_max_roughness(0.05) == 0.8
        and required_max_roughness(0.2) == 1.6
        and required_max_roughness(0.5) is None
    )
    if not ok:
        print("SELFCHECK FAILED")
        return 1
    print("selfcheck OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
