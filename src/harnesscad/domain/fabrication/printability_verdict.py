"""Deterministic printability verdict, issue-code taxonomy and fit check.

A printability analysis measures a built solid against a printer profile and
returns a single machine-actionable contract: a ``verdict`` (printable + 0-100
score), an ``issues[]`` list keyed by ``code``, and per-check metrics. The
*geometry measurement* itself (ray-cast wall thickness, facet-normal overhang,
mass properties) belongs to a solid-modelling kernel. The transferable,
kernel-free core is the *judgement layer*: given already-measured metrics,
classify them into issue codes, roll them into a severity-penalty score, and
decide printability -- plus the build-volume fit test with axis-permutation
rotation.

This is distinct from :mod:`harnesscad.domain.fabrication.overhang`, which
*detects* overhang from face normals. This module *judges* a measured metric
bundle: it turns numbers (min wall, overhang area ratio, solid validity, bbox
size, small-feature counts) into issue codes and a verdict, so a
printability check has one deterministic scoring definition independent of which
backend produced the measurements.

The contract:

* **Issue codes**: ``DOES_NOT_FIT``, ``NOT_SOLID``, ``NOT_WATERTIGHT``,
  ``THIN_WALL``, ``OVERHANG``, ``SMALL_FEATURE``, ``PRINT_OK``.
* **Severity penalties**: error 35, warning 12, info 0; score = ``100 - sum``,
  clamped to [0, 100]; ``printable`` iff no error-severity issue.
* **Fit**: usable = build volume minus 2*margin per axis; ``fits`` when every
  model axis is within usable; ``rotatedFits`` when the sorted dims fit the
  sorted usable caps (any axis permutation).
* **Thin wall**: warning below the minimum, escalated to error below 60% of it.

stdlib-only (``math``, ``dataclasses``), deterministic, absolute imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "SEVERITY_PENALTY",
    "PrinterProfile",
    "Measurements",
    "Issue",
    "check_fit",
    "classify_issues",
    "score_issues",
    "printability_verdict",
]

# Severity -> score penalty.
SEVERITY_PENALTY: Dict[str, int] = {"error": 35, "warning": 12, "info": 0}


@dataclass(frozen=True)
class PrinterProfile:
    """Printer constraints (the load-bearing subset of a default profile)."""

    build_volume_mm: Tuple[float, float, float] = (256.0, 256.0, 256.0)
    margin_mm: float = 2.0
    min_wall_mm: float = 0.8
    min_feature_mm: float = 0.4
    support_free_angle_deg: float = 45.0


@dataclass(frozen=True)
class Measurements:
    """Already-measured geometry metrics (whatever backend produced them).

    ``size_mm`` is the model bounding-box size. ``min_wall_mm`` /
    ``overhang_area_ratio`` are ``None`` when unmeasured (e.g. tessellation
    failed) -- an unmeasured metric raises no issue rather than a false pass.
    """

    size_mm: Tuple[float, float, float]
    is_valid_solid: Optional[bool] = None
    is_watertight: Optional[bool] = None
    min_wall_mm: Optional[float] = None
    overhang_area_ratio: Optional[float] = None
    short_edges: int = 0
    tiny_faces: int = 0


@dataclass(frozen=True)
class Issue:
    """A printability issue (forgent3d's issue record)."""

    code: str
    severity: str
    message: str
    detail: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FitResult:
    """Build-volume fit outcome, including the rotated (permuted) fit."""

    fits: bool
    rotated_fits: bool
    over_mm: Tuple[float, float, float]


def check_fit(size_mm: Sequence[float], profile: PrinterProfile) -> FitResult:
    """Check whether the model fits the usable build volume (forgent3d ``_fit``).

    Usable per axis = build volume minus twice the margin. ``fits`` requires the
    model to fit axis-aligned; ``rotated_fits`` allows any axis permutation
    (sorted model dims vs sorted usable caps).
    """
    limits = list(profile.build_volume_mm)
    usable = [max(0.0, lim - 2.0 * profile.margin_mm) for lim in limits]
    over = tuple(max(0.0, size_mm[i] - usable[i]) for i in range(3))
    fits = all(size_mm[i] <= usable[i] + 1e-6 for i in range(3))
    rotated = sorted(size_mm, reverse=True)
    caps = sorted(usable, reverse=True)
    rotated_fits = all(rotated[i] <= caps[i] + 1e-6 for i in range(3))
    return FitResult(fits=fits, rotated_fits=rotated_fits, over_mm=over)  # type: ignore[arg-type]


def classify_issues(m: Measurements, profile: PrinterProfile) -> List[Issue]:
    """Turn measured metrics into forgent3d's issue codes.

    Returns issues worst-first is not guaranteed; callers score with
    :func:`score_issues`. When nothing is wrong, a single ``PRINT_OK`` info issue
    is returned (matching forgent3d).
    """
    issues: List[Issue] = []
    axes = ("x", "y", "z")

    fit = check_fit(m.size_mm, profile)
    if not fit.fits:
        for i, over in enumerate(fit.over_mm):
            if over > 1e-6:
                issues.append(
                    Issue(
                        "DOES_NOT_FIT",
                        "error",
                        f"{axes[i].upper()} exceeds the usable build volume by "
                        f"{round(over, 3)} mm"
                        + (" (rotatable)" if fit.rotated_fits else ""),
                        {"axis": float(i), "over_mm": round(over, 3)},
                    )
                )

    if m.is_valid_solid is False:
        issues.append(
            Issue("NOT_SOLID", "error", "BREP is not a valid closed solid.")
        )
    elif m.is_watertight is False:
        issues.append(
            Issue("NOT_WATERTIGHT", "error", "Model is not a closed watertight solid.")
        )

    if m.min_wall_mm is not None:
        thr = profile.min_wall_mm
        if m.min_wall_mm < thr:
            severity = "error" if m.min_wall_mm < thr * 0.6 else "warning"
            issues.append(
                Issue(
                    "THIN_WALL",
                    severity,
                    f"Thinnest wall is {round(m.min_wall_mm, 3)} mm "
                    f"(min printable {round(thr, 3)} mm).",
                    {"value_mm": round(m.min_wall_mm, 3), "threshold_mm": round(thr, 3)},
                )
            )

    if m.overhang_area_ratio is not None and m.overhang_area_ratio > 0.01:
        issues.append(
            Issue(
                "OVERHANG",
                "warning",
                f"{round(m.overhang_area_ratio * 100, 1)}% of surface area overhangs "
                f"beyond the {round(profile.support_free_angle_deg, 1)} deg limit and "
                f"needs support.",
                {"area_ratio": round(m.overhang_area_ratio, 4)},
            )
        )

    if m.short_edges or m.tiny_faces:
        issues.append(
            Issue(
                "SMALL_FEATURE",
                "warning",
                f"{m.short_edges} edge(s) and {m.tiny_faces} face(s) are below the "
                f"{round(profile.min_feature_mm, 3)} mm printable minimum.",
                {"short_edges": float(m.short_edges), "tiny_faces": float(m.tiny_faces)},
            )
        )

    if not issues:
        issues.append(
            Issue("PRINT_OK", "info", "No printability issues found for the selected printer.")
        )
    return issues


def score_issues(issues: Sequence[Issue]) -> Tuple[bool, int]:
    """forgent3d's verdict: ``(printable, score)``.

    ``score = 100 - sum(penalty)`` clamped to [0, 100]; ``printable`` iff no
    error-severity issue is present.
    """
    penalty = sum(SEVERITY_PENALTY.get(i.severity, 0) for i in issues)
    printable = all(i.severity != "error" for i in issues)
    score = max(0, min(100, 100 - penalty))
    return printable, score


def printability_verdict(m: Measurements, profile: PrinterProfile = PrinterProfile()) -> Dict[str, object]:
    """End-to-end: classify issues, score, and return forgent3d's contract dict."""
    issues = classify_issues(m, profile)
    printable, score = score_issues(issues)
    return {
        "printable": printable,
        "score": score,
        "issues": [
            {"code": i.code, "severity": i.severity, "message": i.message, "detail": i.detail}
            for i in issues
        ],
    }
