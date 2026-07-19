"""Feature-typed FDM printability minima and a feature-level rule checker.

The transferable core is a *typed rule table*: per-feature minimum and
recommended sizes (wall, rib, through hole, blind hole, boss, text, clearance
gap), the overhang/bridge maxima, and the boolean-robustness tolerances
(union merge overlap, difference cut extension). This module makes those
numbers machine-checkable.

Where :mod:`harnesscad.domain.fabrication.printability_verdict` judges a
*whole-model metric bundle* (min wall anywhere, overhang area ratio, fit)
against a printer profile, this module judges *individual named features*
("this boss is 2.4 mm across") against feature-specific minima that are much
stricter than the nozzle-width floor. The two compose: feature findings
convert into that module's :class:`~harnesscad.domain.fabrication.printability_verdict.Issue`
records and merge into its verdict/score flow via :func:`feature_verdict`.

Severity semantics:

* below the feature minimum (or above a feature maximum) -> ``violation``;
* between minimum and recommended -> ``warning``;
* at or above recommended (or at/above minimum when no recommended value
  exists) -> ``ok``.

stdlib-only, deterministic, absolute imports.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from harnesscad.domain.fabrication.printability_verdict import (
    Issue,
    Measurements,
    PrinterProfile,
    classify_issues,
    score_issues,
)

__all__ = [
    "FeatureRule",
    "FEATURE_MINIMA",
    "MeasuredFeature",
    "Finding",
    "check_feature",
    "check_features",
    "findings_to_issues",
    "feature_verdict",
    "main",
]

# Severity labels used by this checker (Finding.severity).
SEVERITY_VIOLATION = "violation"
SEVERITY_WARNING = "warning"
SEVERITY_OK = "ok"

# Rule kinds: "minimum" rules fail when the measured value is below the
# threshold; "maximum" rules fail when it is above.
KIND_MINIMUM = "minimum"
KIND_MAXIMUM = "maximum"


@dataclass(frozen=True)
class FeatureRule:
    """One feature-typed printability rule.

    ``threshold`` is the hard limit (a floor for ``kind == "minimum"`` rules,
    a ceiling for ``kind == "maximum"`` rules). ``recommended`` is the
    comfortable value where one is defined, else ``None``. ``rule`` is the
    rationale text.
    """

    feature: str
    threshold: float
    recommended: Optional[float]
    units: str
    kind: str
    rule: str


# Feature-typed FDM minima and maxima.
FEATURE_MINIMA: Dict[str, FeatureRule] = {
    "wall": FeatureRule(
        feature="wall",
        threshold=1.2,
        recommended=2.0,
        units="mm",
        kind=KIND_MINIMUM,
        rule="Wall thickness: minimum 1.2 mm, recommended 2.0 mm. "
        "Features below nozzle width will not print.",
    ),
    "rib": FeatureRule(
        feature="rib",
        threshold=1.2,
        recommended=1.6,
        units="mm",
        kind=KIND_MINIMUM,
        rule="Decorative rib/relief width: minimum 1.2 mm, recommended 1.6 mm.",
    ),
    "through_hole": FeatureRule(
        feature="through_hole",
        threshold=2.0,
        recommended=3.0,
        units="mm",
        kind=KIND_MINIMUM,
        rule="Hole diameter (through): minimum 2.0 mm, recommended 3.0 mm.",
    ),
    "blind_hole": FeatureRule(
        feature="blind_hole",
        threshold=3.0,
        recommended=4.0,
        units="mm",
        kind=KIND_MINIMUM,
        rule="Hole diameter (blind): minimum 3.0 mm, recommended 4.0 mm.",
    ),
    "boss": FeatureRule(
        feature="boss",
        threshold=3.0,
        recommended=5.0,
        units="mm",
        kind=KIND_MINIMUM,
        rule="Boss/post diameter: minimum 3.0 mm, recommended 5.0 mm.",
    ),
    "text": FeatureRule(
        feature="text",
        threshold=0.5,
        recommended=1.0,
        units="mm",
        kind=KIND_MINIMUM,
        rule="Text feature height/depth (embossed or engraved): "
        "minimum 0.5 mm, recommended 1.0 mm.",
    ),
    "clearance_gap": FeatureRule(
        feature="clearance_gap",
        threshold=0.2,
        recommended=0.4,
        units="mm",
        kind=KIND_MINIMUM,
        rule="Gap/clearance between parts: minimum 0.2 mm, "
        "recommended 0.4 mm.",
    ),
    "bridge": FeatureRule(
        feature="bridge",
        threshold=20.0,
        recommended=None,
        units="mm",
        kind=KIND_MAXIMUM,
        rule="Maximum bridge span 20 mm without supports.",
    ),
    "overhang_angle": FeatureRule(
        feature="overhang_angle",
        threshold=45.0,
        recommended=None,
        units="deg",
        kind=KIND_MAXIMUM,
        rule="Maximum unsupported overhang: 45 degrees from vertical. "
        "Orient parts to minimize overhangs; avoid horizontal holes.",
    ),
    "merge_overlap": FeatureRule(
        feature="merge_overlap",
        threshold=0.2,
        recommended=None,
        units="mm",
        kind=KIND_MINIMUM,
        rule="Always overlap unioned solids by the merge tolerance "
        "(0.2 mm); never rely on coplanar faces.",
    ),
    "cut_extension": FeatureRule(
        feature="cut_extension",
        threshold=0.5,
        recommended=None,
        units="mm",
        kind=KIND_MINIMUM,
        rule="Avoid coincident faces in difference(): extend subtracted "
        "geometry past surfaces by at least 0.5 mm.",
    ),
}


@dataclass(frozen=True)
class MeasuredFeature:
    """A single measured feature: its type key and measured dimension.

    ``feature`` must be a key of :data:`FEATURE_MINIMA`. ``value`` is the
    measured dimension in the rule's units (mm, except ``overhang_angle`` in
    degrees). ``label`` optionally names the specific instance
    (e.g. "mounting boss NE").
    """

    feature: str
    value: float
    label: str = ""


@dataclass(frozen=True)
class Finding:
    """The checker's judgement of one measured feature against its rule."""

    feature: str
    label: str
    value: float
    severity: str  # "violation" | "warning" | "ok"
    threshold: float
    recommended: Optional[float]
    units: str
    rule: str
    message: str


def _describe(measured: MeasuredFeature) -> str:
    name = measured.feature.replace("_", " ")
    return f"{name} '{measured.label}'" if measured.label else name


def check_feature(measured: MeasuredFeature) -> Finding:
    """Judge one measured feature against :data:`FEATURE_MINIMA`.

    Raises :class:`KeyError` for an unknown feature type -- an unknown
    feature is a caller bug, not a printability pass.
    """
    rule = FEATURE_MINIMA[measured.feature]
    value = measured.value
    what = _describe(measured)

    if rule.kind == KIND_MAXIMUM:
        if value > rule.threshold:
            severity = SEVERITY_VIOLATION
            message = (
                f"{what} is {round(value, 3)} {rule.units}, above the "
                f"{round(rule.threshold, 3)} {rule.units} maximum."
            )
        else:
            severity = SEVERITY_OK
            message = (
                f"{what} is {round(value, 3)} {rule.units}, within the "
                f"{round(rule.threshold, 3)} {rule.units} maximum."
            )
    else:
        if value < rule.threshold:
            severity = SEVERITY_VIOLATION
            message = (
                f"{what} is {round(value, 3)} {rule.units}, below the "
                f"{round(rule.threshold, 3)} {rule.units} minimum."
            )
        elif rule.recommended is not None and value < rule.recommended:
            severity = SEVERITY_WARNING
            message = (
                f"{what} is {round(value, 3)} {rule.units}, above the "
                f"{round(rule.threshold, 3)} {rule.units} minimum but below "
                f"the {round(rule.recommended, 3)} {rule.units} recommended."
            )
        else:
            severity = SEVERITY_OK
            message = (
                f"{what} is {round(value, 3)} {rule.units}, at or above "
                + (
                    f"the {round(rule.recommended, 3)} {rule.units} recommended."
                    if rule.recommended is not None
                    else f"the {round(rule.threshold, 3)} {rule.units} minimum."
                )
            )

    return Finding(
        feature=measured.feature,
        label=measured.label,
        value=value,
        severity=severity,
        threshold=rule.threshold,
        recommended=rule.recommended,
        units=rule.units,
        rule=rule.rule,
        message=message,
    )


def check_features(features: Sequence[MeasuredFeature]) -> List[Finding]:
    """Judge every measured feature, in input order (deterministic)."""
    return [check_feature(f) for f in features]


# --- composition with printability_verdict ---

# Finding severity -> printability_verdict Issue severity.
_ISSUE_SEVERITY: Dict[str, str] = {
    SEVERITY_VIOLATION: "error",
    SEVERITY_WARNING: "warning",
    SEVERITY_OK: "info",
}


def findings_to_issues(
    findings: Sequence[Finding], include_ok: bool = False
) -> List[Issue]:
    """Convert findings into printability_verdict :class:`Issue` records.

    Violations map to error severity, warnings to warning, ok to info.
    ``ok`` findings are dropped unless ``include_ok`` is set, matching
    printability_verdict's convention of only reporting problems (its
    PRINT_OK sentinel covers the clean case).
    """
    issues: List[Issue] = []
    for f in findings:
        if f.severity == SEVERITY_OK and not include_ok:
            continue
        detail: Dict[str, float] = {
            "value": round(f.value, 3),
            "threshold": round(f.threshold, 3),
        }
        if f.recommended is not None:
            detail["recommended"] = round(f.recommended, 3)
        issues.append(
            Issue(
                code="FEATURE_" + f.feature.upper(),
                severity=_ISSUE_SEVERITY[f.severity],
                message=f.message + " Rule: " + f.rule,
                detail=detail,
            )
        )
    return issues


def feature_verdict(
    features: Sequence[MeasuredFeature],
    measurements: Optional[Measurements] = None,
    profile: PrinterProfile = PrinterProfile(),
) -> Dict[str, object]:
    """Merged verdict: feature findings plus (optionally) the metric bundle.

    Runs this module's feature checker, converts its findings to
    printability_verdict issues, appends the whole-model issues from
    :func:`classify_issues` when ``measurements`` is given, and scores the
    combined list with :func:`score_issues` -- returning the same
    ``{printable, score, issues}`` contract dict as
    :func:`printability_verdict.printability_verdict`.
    """
    issues = findings_to_issues(check_features(features))
    if measurements is not None:
        model_issues = classify_issues(measurements, profile)
        # Drop the PRINT_OK sentinel when feature issues exist.
        if issues:
            model_issues = [i for i in model_issues if i.code != "PRINT_OK"]
        issues.extend(model_issues)
    if not issues:
        issues.append(
            Issue(
                "PRINT_OK",
                "info",
                "No feature-minima or printability issues found.",
            )
        )
    printable, score = score_issues(issues)
    return {
        "printable": printable,
        "score": score,
        "issues": [
            {
                "code": i.code,
                "severity": i.severity,
                "message": i.message,
                "detail": i.detail,
            }
            for i in issues
        ],
    }


def _selfcheck() -> None:
    # Table values are the documented feature minima.
    assert FEATURE_MINIMA["wall"].threshold == 1.2
    assert FEATURE_MINIMA["wall"].recommended == 2.0
    assert FEATURE_MINIMA["rib"].threshold == 1.2
    assert FEATURE_MINIMA["rib"].recommended == 1.6
    assert FEATURE_MINIMA["through_hole"].threshold == 2.0
    assert FEATURE_MINIMA["through_hole"].recommended == 3.0
    assert FEATURE_MINIMA["blind_hole"].threshold == 3.0
    assert FEATURE_MINIMA["blind_hole"].recommended == 4.0
    assert FEATURE_MINIMA["boss"].threshold == 3.0
    assert FEATURE_MINIMA["boss"].recommended == 5.0
    assert FEATURE_MINIMA["text"].threshold == 0.5
    assert FEATURE_MINIMA["text"].recommended == 1.0
    assert FEATURE_MINIMA["clearance_gap"].threshold == 0.2
    assert FEATURE_MINIMA["clearance_gap"].recommended == 0.4
    assert FEATURE_MINIMA["bridge"].threshold == 20.0
    assert FEATURE_MINIMA["bridge"].kind == KIND_MAXIMUM
    assert FEATURE_MINIMA["overhang_angle"].threshold == 45.0
    assert FEATURE_MINIMA["overhang_angle"].units == "deg"
    assert FEATURE_MINIMA["merge_overlap"].threshold == 0.2
    assert FEATURE_MINIMA["cut_extension"].threshold == 0.5
    for rule in FEATURE_MINIMA.values():
        assert rule.kind in (KIND_MINIMUM, KIND_MAXIMUM)
        assert rule.threshold > 0.0
        assert rule.rule

    # Checker severities: violation below minimum, warning between minimum
    # and recommended, ok at or above recommended.
    assert check_feature(MeasuredFeature("wall", 1.0)).severity == "violation"
    assert check_feature(MeasuredFeature("wall", 1.5)).severity == "warning"
    assert check_feature(MeasuredFeature("wall", 2.5)).severity == "ok"
    assert check_feature(MeasuredFeature("wall", 1.2)).severity == "warning"
    assert check_feature(MeasuredFeature("wall", 2.0)).severity == "ok"
    assert check_feature(MeasuredFeature("through_hole", 1.5)).severity == "violation"
    assert check_feature(MeasuredFeature("through_hole", 2.5)).severity == "warning"
    assert check_feature(MeasuredFeature("blind_hole", 3.5)).severity == "warning"
    assert check_feature(MeasuredFeature("boss", 2.9)).severity == "violation"
    assert check_feature(MeasuredFeature("text", 0.4)).severity == "violation"
    assert check_feature(MeasuredFeature("text", 0.5)).severity == "warning"
    assert check_feature(MeasuredFeature("clearance_gap", 0.3)).severity == "warning"
    # Maximum-kind rules: fail above the ceiling.
    assert check_feature(MeasuredFeature("bridge", 25.0)).severity == "violation"
    assert check_feature(MeasuredFeature("bridge", 15.0)).severity == "ok"
    assert check_feature(MeasuredFeature("overhang_angle", 50.0)).severity == "violation"
    assert check_feature(MeasuredFeature("overhang_angle", 45.0)).severity == "ok"
    assert check_feature(MeasuredFeature("merge_overlap", 0.1)).severity == "violation"
    assert check_feature(MeasuredFeature("cut_extension", 0.5)).severity == "ok"

    # Batch checker preserves input order.
    batch = check_features(
        [MeasuredFeature("wall", 1.0, "shell"), MeasuredFeature("boss", 6.0)]
    )
    assert [f.severity for f in batch] == ["violation", "ok"]
    assert "shell" in batch[0].message

    # Conversion to printability_verdict issues.
    issues = findings_to_issues(batch)
    assert len(issues) == 1  # ok finding dropped
    assert issues[0].code == "FEATURE_WALL"
    assert issues[0].severity == "error"
    assert issues[0].detail["value"] == 1.0
    assert issues[0].detail["threshold"] == 1.2
    issues_all = findings_to_issues(batch, include_ok=True)
    assert len(issues_all) == 2
    assert issues_all[1].severity == "info"

    # Composition with printability_verdict's flow.
    m = Measurements(size_mm=(50.0, 40.0, 30.0), is_valid_solid=True, is_watertight=True)
    verdict = feature_verdict(
        [MeasuredFeature("wall", 1.0), MeasuredFeature("through_hole", 2.5)],
        measurements=m,
    )
    assert verdict["printable"] is False  # wall violation is an error
    assert verdict["score"] == 100 - 35 - 12
    codes = [i["code"] for i in verdict["issues"]]  # type: ignore[index]
    assert "FEATURE_WALL" in codes
    assert "FEATURE_THROUGH_HOLE" in codes
    assert "PRINT_OK" not in codes

    # Clean features + clean measurements -> printable, PRINT_OK sentinel.
    clean = feature_verdict([MeasuredFeature("wall", 2.5)], measurements=m)
    assert clean["printable"] is True
    assert clean["score"] == 100
    assert [i["code"] for i in clean["issues"]] == ["PRINT_OK"]  # type: ignore[index]

    # Features-only path (no measurements) also yields the sentinel.
    only = feature_verdict([MeasuredFeature("boss", 5.0)])
    assert only["printable"] is True
    assert [i["code"] for i in only["issues"]] == ["PRINT_OK"]  # type: ignore[index]

    print("feature_minima selfcheck OK:", len(FEATURE_MINIMA), "rules")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="feature_minima",
        description="Feature-typed FDM printability minima (AgentSCAD, MIT).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="assert table values and checker behaviour, then exit 0",
    )
    args = parser.parse_args(argv)
    if args.selfcheck:
        _selfcheck()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
