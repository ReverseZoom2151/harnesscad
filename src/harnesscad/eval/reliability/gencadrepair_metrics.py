"""Feasibility-rate and repair-success metrics for CAD sequence generation.

Tsuji, Flores Medina, Gupta & Alam, *GenCAD-Self-Repairing* (MIT), Sec. 4.2.

The paper's primary evaluation metric is the **feasibility rate** (Eq. 7)::

    F = V / (V + I)

where ``V`` is the number of generated sequences that yield a valid B-rep and
``I`` the number that do not. Their headline result is a *repair-success* figure:
the self-repair pipeline "successfully fixed 65.84% (532/808) of the baseline
infeasible images", lifting the feasibility rate from 0.931 (baseline) to 0.970.

This module makes those two quantities computable deterministically over boolean
feasibility flags — the exact figures the paper reports — plus the paired
baseline-vs-repaired benchmark (rate improvement, count fixed, and any
*regressions* where a feasible sample became infeasible). The feasibility flags
can come from the real OCCT kernel or, locally, from
:func:`reliability.gencadrepair_taxonomy.is_feasible`; a convenience bridge
(:func:`evaluate_sequences`) wires that predicate up directly.

Accuracy metrics (MMD / FID / JSD) the paper also cites already live in
``bench/generative_brep_metrics.py`` and ``bench/gencad_fid.py`` and are not
duplicated here. Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence


def _as_bools(flags: Iterable) -> List[bool]:
    return [bool(f) for f in flags]


def feasibility_rate(flags: Iterable) -> float:
    """``F = V / (V + I)`` over per-sample feasibility flags (Eq. 7).

    Empty input yields ``0.0`` (no valid generations).
    """
    bits = _as_bools(flags)
    if not bits:
        return 0.0
    valid = sum(1 for b in bits if b)
    return valid / len(bits)


@dataclass(frozen=True)
class FeasibilityReport:
    """Counts and rate for one population of generated sequences."""

    valid: int
    invalid: int

    @property
    def total(self) -> int:
        return self.valid + self.invalid

    @property
    def rate(self) -> float:
        return self.valid / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "invalid": self.invalid,
            "total": self.total,
            "rate": self.rate,
        }


def feasibility_report(flags: Iterable) -> FeasibilityReport:
    """Tally valid/invalid counts into a :class:`FeasibilityReport`."""
    bits = _as_bools(flags)
    valid = sum(1 for b in bits if b)
    return FeasibilityReport(valid=valid, invalid=len(bits) - valid)


def repair_success_rate(before: Sequence, after: Sequence) -> float:
    """Fraction of *baseline-infeasible* samples made feasible by repair.

    ``before[i]`` / ``after[i]`` are feasibility flags for the same sample before
    and after repair. Matches the paper's "fixed 532/808" statistic:
    ``fixed / infeasible_before``. Returns ``0.0`` when nothing was infeasible.
    """
    b = _as_bools(before)
    a = _as_bools(after)
    if len(b) != len(a):
        raise ValueError("before/after must have equal length")
    infeasible_before = [i for i, flag in enumerate(b) if not flag]
    if not infeasible_before:
        return 0.0
    fixed = sum(1 for i in infeasible_before if a[i])
    return fixed / len(infeasible_before)


@dataclass(frozen=True)
class RepairBenchmark:
    """Paired baseline-vs-repaired feasibility benchmark (paper Table 3)."""

    total: int
    infeasible_before: int
    fixed: int
    regressions: int
    baseline_rate: float
    repaired_rate: float

    @property
    def repair_success_rate(self) -> float:
        return self.fixed / self.infeasible_before if self.infeasible_before else 0.0

    @property
    def rate_improvement(self) -> float:
        return self.repaired_rate - self.baseline_rate

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "infeasible_before": self.infeasible_before,
            "fixed": self.fixed,
            "regressions": self.regressions,
            "baseline_rate": self.baseline_rate,
            "repaired_rate": self.repaired_rate,
            "repair_success_rate": self.repair_success_rate,
            "rate_improvement": self.rate_improvement,
        }


def benchmark_repair(before: Sequence, after: Sequence) -> RepairBenchmark:
    """Compare baseline and post-repair feasibility flags sample-by-sample.

    Counts how many baseline-infeasible samples became feasible (``fixed``) and
    how many baseline-feasible samples became infeasible (``regressions``), and
    reports both feasibility rates. Deterministic.
    """
    b = _as_bools(before)
    a = _as_bools(after)
    if len(b) != len(a):
        raise ValueError("before/after must have equal length")
    infeasible_before = sum(1 for flag in b if not flag)
    fixed = sum(1 for i in range(len(b)) if not b[i] and a[i])
    regressions = sum(1 for i in range(len(b)) if b[i] and not a[i])
    return RepairBenchmark(
        total=len(b),
        infeasible_before=infeasible_before,
        fixed=fixed,
        regressions=regressions,
        baseline_rate=feasibility_rate(b),
        repaired_rate=feasibility_rate(a),
    )


def evaluate_sequences(sequences: Iterable[Sequence],
                       is_feasible: Callable[[Sequence], bool] | None = None,
                       ) -> FeasibilityReport:
    """Feasibility report over raw command sequences via a feasibility predicate.

    ``is_feasible`` defaults to
    :func:`reliability.gencadrepair_taxonomy.is_feasible` (the deterministic,
    kernel-free structural check). Pass the real OCCT-backed predicate to score
    against the geometry kernel instead.
    """
    if is_feasible is None:
        from harnesscad.eval.reliability.gencadrepair_taxonomy import is_feasible as _default
        is_feasible = _default
    return feasibility_report(is_feasible(seq) for seq in sequences)
