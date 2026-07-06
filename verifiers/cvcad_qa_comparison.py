"""cvcad_qa_comparison — quality-assurance comparison of measured vs nominal dims.

Deterministic QA layer of Bhandari & Manandhar (Machines 2023, 11, 1083),
Sec. 6: the vision-measured (or regenerated) dimensions of a part are compared
against the nominal CAD/design values to flag manufacturing defects (shrinkage,
calibration error, warping) in 3D-printed objects. The paper's per-dimension
metric is the percentage error

    percentage error = (nominal - measured) / nominal * 100                    (III)

reported for the cube face as ``(29.9 - 29.8) / 29.9 * 100 == 0.3 %``, and it
accepts discrepancies "less than 1%".

This module generalises that into a tolerance-checked deviation report with
aggregate accuracy metrics (MAE, RMSE, max |error|, mean absolute percentage
error). All external steps (measurement acquisition) are upstream; this consumes
scalar dimensions.

Stdlib-only, deterministic, no wall clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


def percentage_error(nominal: float, measured: float) -> float:
    """Eq. (III): signed percentage error relative to the nominal value."""
    if nominal == 0.0:
        raise ValueError("nominal must be non-zero for percentage error")
    return (nominal - measured) / nominal * 100.0


@dataclass(frozen=True)
class DimensionCheck:
    """One measured-vs-nominal comparison with a tolerance verdict."""

    name: str
    nominal: float
    measured: float
    tolerance: float                 # absolute (+/-) tolerance band
    deviation: float                 # measured - nominal (signed)
    abs_deviation: float
    percent_error: float             # signed, per Eq. (III)
    within_tolerance: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "nominal": self.nominal,
            "measured": self.measured,
            "tolerance": self.tolerance,
            "deviation": self.deviation,
            "abs_deviation": self.abs_deviation,
            "percent_error": self.percent_error,
            "within_tolerance": self.within_tolerance,
        }


def check_dimension(name: str, nominal: float, measured: float,
                    tolerance: float) -> DimensionCheck:
    """Compare one measured dimension against its nominal within ``tolerance``.

    ``tolerance`` is an absolute half-band: pass iff |measured - nominal| <= tol.
    """
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    deviation = measured - nominal
    abs_dev = abs(deviation)
    pct = 0.0 if nominal == 0.0 else percentage_error(nominal, measured)
    # Use a tiny epsilon so exact-boundary deviations count as within tolerance.
    within = abs_dev <= tolerance + 1e-12
    return DimensionCheck(
        name=name,
        nominal=nominal,
        measured=measured,
        tolerance=tolerance,
        deviation=deviation,
        abs_deviation=abs_dev,
        percent_error=pct,
        within_tolerance=within,
    )


def check_by_percent(name: str, nominal: float, measured: float,
                     percent_tolerance: float) -> DimensionCheck:
    """Tolerance expressed as a percentage of nominal (e.g. the paper's 1%)."""
    if percent_tolerance < 0.0:
        raise ValueError("percent_tolerance must be non-negative")
    tol = abs(nominal) * percent_tolerance / 100.0
    return check_dimension(name, nominal, measured, tol)


@dataclass(frozen=True)
class QAReport:
    """Aggregate QA verdict over a set of dimension checks."""

    checks: List[DimensionCheck]
    num_checks: int
    num_pass: int
    num_fail: int
    all_pass: bool
    max_abs_deviation: float
    mean_abs_error: float            # MAE of (measured - nominal)
    rms_error: float                 # RMSE
    mean_abs_percent_error: float    # MAPE (per-dimension |% error|)

    def to_dict(self) -> dict:
        return {
            "num_checks": self.num_checks,
            "num_pass": self.num_pass,
            "num_fail": self.num_fail,
            "all_pass": self.all_pass,
            "max_abs_deviation": self.max_abs_deviation,
            "mean_abs_error": self.mean_abs_error,
            "rms_error": self.rms_error,
            "mean_abs_percent_error": self.mean_abs_percent_error,
            "checks": [c.to_dict() for c in self.checks],
        }

    def failures(self) -> List[DimensionCheck]:
        return [c for c in self.checks if not c.within_tolerance]


def qa_report(checks: Sequence[DimensionCheck]) -> QAReport:
    """Aggregate a list of :class:`DimensionCheck` into a QA report."""
    checks = list(checks)
    n = len(checks)
    if n == 0:
        return QAReport([], 0, 0, 0, True, 0.0, 0.0, 0.0, 0.0)

    num_pass = sum(1 for c in checks if c.within_tolerance)
    abs_devs = [c.abs_deviation for c in checks]
    sq_devs = [c.deviation ** 2 for c in checks]
    abs_pcts = [abs(c.percent_error) for c in checks]

    return QAReport(
        checks=checks,
        num_checks=n,
        num_pass=num_pass,
        num_fail=n - num_pass,
        all_pass=num_pass == n,
        max_abs_deviation=max(abs_devs),
        mean_abs_error=sum(abs_devs) / n,
        rms_error=math.sqrt(sum(sq_devs) / n),
        mean_abs_percent_error=sum(abs_pcts) / n,
    )


def compare_dimensions(nominal: dict, measured: dict,
                       tolerance: float) -> QAReport:
    """Convenience: compare two ``name -> value`` dicts on their shared keys.

    Keys are processed in sorted order for determinism. Missing keys on either
    side are ignored (only the intersection is checked).
    """
    names = sorted(set(nominal) & set(measured))
    checks = [check_dimension(name, nominal[name], measured[name], tolerance)
              for name in names]
    return qa_report(checks)
