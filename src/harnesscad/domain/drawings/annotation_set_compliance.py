"""annotation_set_compliance -- set-level ASME Y14.5-2018 compliance rules.

Harness gap filled: harnesscad.domain.drawings.gdt already validates a SINGLE
GD&T feature-control frame (datum-less form controls, modifier legality per
characteristic, datum precedence within one frame) -- those per-frame checks
are NOT rebuilt here. What was missing are the SET-LEVEL rules that only make
sense over a whole annotation set from a drawing:

  * FCF_DATUM_COUNT -- datum reference count within the per-characteristic
    [min, max] range;
  * DATUM_REF_EXISTS -- every datum letter referenced by an FCF must be
    declared by a datum annotation somewhere in the set;
  * MMC_LMC_APPLICABILITY -- MMC/LMC only for position/concentricity/symmetry;
  * TOLERANCE_POSITIVE -- FCF tolerance values must be positive;
  * DUPLICATE_DATUM_LETTER -- two datum
    annotations declaring the same letter is an error;

plus the compliance summary counts (errors / warnings / passing per
annotation) with the invariant errors + warnings + passing == len(annotations).

Pure stdlib, deterministic, no LLM calls.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from harnesscad.domain.drawings.annotation_schema import (
    Annotation,
    BoundingBox,
    DatumAnnotation,
    FcfAnnotation,
)

# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"


@dataclass(frozen=True)
class ComplianceIssue:
    """A single set-level compliance violation tied to one annotation."""

    annotation_id: str
    rule_id: str
    severity: str  # "error" | "warning"
    description: str

    def to_dict(self) -> dict:
        return {
            "annotation_id": self.annotation_id,
            "rule_id": self.rule_id,
            "severity": self.severity,
            "description": self.description,
        }


@dataclass(frozen=True)
class ComplianceSummaryCounts:
    """Per-annotation triage counts; errors + warnings + passing == total."""

    errors: int
    warnings: int
    passing: int

    def to_dict(self) -> dict:
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "passing": self.passing,
        }


# --------------------------------------------------------------------------- #
# ASME Y14.5-2018 lookup tables (compliance-engine.ts)
# --------------------------------------------------------------------------- #

# Valid datum reference count ranges per geometric characteristic, [min, max]
# inclusive.
DATUM_COUNT_RANGE: Dict[str, Tuple[int, int]] = {
    "position": (2, 3),
    "flatness": (0, 0),
    "straightness": (0, 0),
    "circularity": (0, 0),
    "cylindricity": (0, 0),
    "perpendicularity": (1, 2),
    "parallelism": (1, 2),
    "angularity": (1, 2),
    "profileOfLine": (0, 3),
    "profileOfSurface": (0, 3),
    "circularRunout": (1, 2),
    "totalRunout": (1, 2),
    "symmetry": (3, 3),
    "concentricity": (1, 1),
}

# Geometric characteristics that permit MMC / LMC material condition modifiers.
MMC_LMC_PERMITTED = frozenset({"position", "concentricity", "symmetry"})


# --------------------------------------------------------------------------- #
# Individual rules
# --------------------------------------------------------------------------- #


def check_fcf_datum_count(annotations: Sequence[Annotation]) -> List[ComplianceIssue]:
    """FCF_DATUM_COUNT -- datum reference count per geometric characteristic."""
    issues: List[ComplianceIssue] = []
    for ann in annotations:
        if not isinstance(ann, FcfAnnotation):
            continue
        rng = DATUM_COUNT_RANGE.get(ann.geometric_characteristic)
        if rng is None:
            continue
        lo, hi = rng
        count = len(ann.datum_references)
        if count < lo or count > hi:
            expected = str(lo) if lo == hi else "%d-%d" % (lo, hi)
            issues.append(
                ComplianceIssue(
                    annotation_id=ann.id,
                    rule_id="FCF_DATUM_COUNT",
                    severity=SEVERITY_ERROR,
                    description="%s requires %s datum reference(s), but %d provided."
                    % (ann.geometric_characteristic, expected, count),
                )
            )
    return issues


def check_datum_ref_exists(annotations: Sequence[Annotation]) -> List[ComplianceIssue]:
    """DATUM_REF_EXISTS -- all referenced datums must be declared in the set."""
    declared: Set[str] = {
        ann.datum_letter for ann in annotations if isinstance(ann, DatumAnnotation)
    }
    issues: List[ComplianceIssue] = []
    for ann in annotations:
        if not isinstance(ann, FcfAnnotation):
            continue
        for ref in ann.datum_references:
            if ref not in declared:
                issues.append(
                    ComplianceIssue(
                        annotation_id=ann.id,
                        rule_id="DATUM_REF_EXISTS",
                        severity=SEVERITY_ERROR,
                        description='FCF references datum "%s" which is not '
                        "declared in the annotation set." % ref,
                    )
                )
    return issues


def check_mmc_lmc_applicability(
    annotations: Sequence[Annotation],
) -> List[ComplianceIssue]:
    """MMC_LMC_APPLICABILITY -- modifier permitted only for position,
    concentricity, symmetry."""
    issues: List[ComplianceIssue] = []
    for ann in annotations:
        if not isinstance(ann, FcfAnnotation):
            continue
        if ann.material_condition is None or ann.material_condition == "RFS":
            continue
        if ann.geometric_characteristic not in MMC_LMC_PERMITTED:
            issues.append(
                ComplianceIssue(
                    annotation_id=ann.id,
                    rule_id="MMC_LMC_APPLICABILITY",
                    severity=SEVERITY_ERROR,
                    description="%s is not permitted for %s per ASME Y14.5-2018."
                    % (ann.material_condition, ann.geometric_characteristic),
                )
            )
    return issues


def check_tolerance_positive(
    annotations: Sequence[Annotation],
) -> List[ComplianceIssue]:
    """TOLERANCE_POSITIVE -- flag FCFs with zero or negative tolerance values."""
    issues: List[ComplianceIssue] = []
    for ann in annotations:
        if not isinstance(ann, FcfAnnotation):
            continue
        if ann.tolerance_value <= 0:
            issues.append(
                ComplianceIssue(
                    annotation_id=ann.id,
                    rule_id="TOLERANCE_POSITIVE",
                    severity=SEVERITY_ERROR,
                    description="Tolerance value must be positive, but got %s."
                    % _fmt_num(ann.tolerance_value),
                )
            )
    return issues


def check_duplicate_datum_letters(
    annotations: Sequence[Annotation],
) -> List[ComplianceIssue]:
    """DUPLICATE_DATUM_LETTER -- two datum annotations declaring the same
    letter is an error (new rule, not in the TS source)."""
    seen: Dict[str, str] = {}  # letter -> first annotation id
    issues: List[ComplianceIssue] = []
    for ann in annotations:
        if not isinstance(ann, DatumAnnotation):
            continue
        first_id = seen.get(ann.datum_letter)
        if first_id is None:
            seen[ann.datum_letter] = ann.id
        else:
            issues.append(
                ComplianceIssue(
                    annotation_id=ann.id,
                    rule_id="DUPLICATE_DATUM_LETTER",
                    severity=SEVERITY_ERROR,
                    description='Datum letter "%s" is already declared by '
                    'annotation "%s"; datum letters must be unique on a drawing.'
                    % (ann.datum_letter, first_id),
                )
            )
    return issues


def _fmt_num(v: float) -> str:
    """Format a number the way JS template literals would (no trailing .0
    for integral values)."""
    if float(v).is_integer():
        return str(int(v))
    return repr(float(v))


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def validate_compliance(annotations: Sequence[Annotation]) -> List[ComplianceIssue]:
    """Validate an annotation set against all set-level ASME Y14.5-2018 rules.

    Returns the concatenated list of compliance issues from every rule.
    """
    return (
        check_fcf_datum_count(annotations)
        + check_datum_ref_exists(annotations)
        + check_mmc_lmc_applicability(annotations)
        + check_tolerance_positive(annotations)
        + check_duplicate_datum_letters(annotations)
    )


def compute_compliance_summary(
    annotations: Sequence[Annotation], issues: Sequence[ComplianceIssue]
) -> ComplianceSummaryCounts:
    """Compute per-annotation triage counts from annotations and issues.

    An annotation counts as:
      * "error" if it has any issue with severity "error";
      * "warning" if it has only warning-severity issues;
      * "passing" if it has no issues at all.

    Invariant: errors + warnings + passing == len(annotations).
    """
    severities_by_id: Dict[str, Set[str]] = {}
    for issue in issues:
        severities_by_id.setdefault(issue.annotation_id, set()).add(issue.severity)

    errors = warnings = passing = 0
    for ann in annotations:
        severities = severities_by_id.get(ann.id)
        if not severities:
            passing += 1
        elif SEVERITY_ERROR in severities:
            errors += 1
        else:
            warnings += 1

    return ComplianceSummaryCounts(errors=errors, warnings=warnings, passing=passing)


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
        # valid datum trio
        DatumAnnotation(datum_letter="A", **_base(1, "Datum A", "A")),
        DatumAnnotation(datum_letter="B", **_base(2, "Datum B", "B")),
        DatumAnnotation(datum_letter="C", **_base(3, "Datum C", "C")),
        # duplicate datum letter -> DUPLICATE_DATUM_LETTER
        DatumAnnotation(datum_letter="A", **_base(4, "Datum A (dup)", "A")),
        # valid position FCF
        FcfAnnotation(
            geometric_characteristic="position",
            tolerance_value=0.05,
            material_condition="MMC",
            datum_references=("A", "B", "C"),
            **_base(5, "Position 0.05 MMC A B C", "0.05"),
        ),
        # flatness with a datum and MMC -> FCF_DATUM_COUNT + MMC_LMC_APPLICABILITY
        FcfAnnotation(
            geometric_characteristic="flatness",
            tolerance_value=0.02,
            material_condition="MMC",
            datum_references=("A",),
            **_base(6, "Flatness 0.02 MMC A", "0.02"),
        ),
        # references undeclared datum D, negative tolerance
        FcfAnnotation(
            geometric_characteristic="perpendicularity",
            tolerance_value=-0.1,
            material_condition=None,
            datum_references=("D",),
            **_base(7, "Perpendicularity -0.1 D", "-0.1"),
        ),
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` runs every set-level rule on a
    synthetic annotation set and asserts the expected issues and the summary
    invariant."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.drawings.annotation_set_compliance",
        description="Set-level ASME Y14.5-2018 compliance over enriched "
        "annotations (ported from CAD-Annotator; per-frame checks live in "
        "harnesscad.domain.drawings.gdt).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="validate a synthetic annotation set and print the issues and "
        "summary counts.",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit issues and summary as JSON."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    annotations = _synthetic_annotations()
    issues = validate_compliance(annotations)
    summary = compute_compliance_summary(annotations, issues)

    if args.json:
        print(
            json.dumps(
                {
                    "issues": [i.to_dict() for i in issues],
                    "summary": summary.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("%d issue(s) on %d annotation(s):" % (len(issues), len(annotations)))
        for issue in issues:
            print(
                "  [%s] %s %s: %s"
                % (issue.severity, issue.annotation_id, issue.rule_id, issue.description)
            )
        print(
            "summary: errors=%d warnings=%d passing=%d"
            % (summary.errors, summary.warnings, summary.passing)
        )

    rule_ids = sorted({i.rule_id for i in issues})
    ok = (
        rule_ids
        == [
            "DATUM_REF_EXISTS",
            "DUPLICATE_DATUM_LETTER",
            "FCF_DATUM_COUNT",
            "MMC_LMC_APPLICABILITY",
            "TOLERANCE_POSITIVE",
        ]
        and summary.errors + summary.warnings + summary.passing == len(annotations)
        and summary.errors == 3  # ann_4, ann_6, ann_7
        and summary.passing == 4
        and validate_compliance([]) == []
    )
    if not ok:
        print("SELFCHECK FAILED")
        return 1
    print("selfcheck OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
