"""Performance-metrics protocol for iterative text-to-CAD runs (Kumar et al.,
"Generative AI for CAD Automation", 2025, sec. 3.3 & Table 1).

Sec. 3.3 records four metrics per test case: (1) first-attempt success or
failure, (2) number of iterative refinements, (3) total execution time
including re-prompting overhead, and (4) the error type encountered.  Table 1
also assigns each case an *outcome*: success on first attempt, converged after N
refinements, or did-not-converge (hit the max-retry limit).  This module turns a
list of run records into that table plus the aggregate rates the paper reports.

No LLM; a run record is just observed data (iterations, whether it finished
error-free, the max-retry cap, and per-iteration times).  Deterministic.

Stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Outcome labels (Table 1 "Outcome" column).
FIRST_ATTEMPT = "success_first_attempt"
CONVERGED = "converged_after_refinement"
DID_NOT_CONVERGE = "did_not_converge"


@dataclass
class RunRecord:
    """One test-case run.

    ``iterations`` counts total generations (1 == only the initial attempt, so
    refinements == iterations - 1).  ``converged`` is True iff the run finished
    error-free.  ``max_retries`` is the retry cap T; a run that did not converge
    and used all retries is a graceful failure.  ``iter_seconds`` are the
    per-iteration times (generation + execution + re-prompt overhead).
    """
    name: str
    iterations: int
    converged: bool
    max_retries: int
    iter_seconds: List[float] = field(default_factory=list)
    error_type: Optional[str] = None

    def __post_init__(self):
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")


def refinements(rec: RunRecord) -> int:
    """Number of iterative refinements = iterations - 1 (metric 2)."""
    return rec.iterations - 1


def total_time(rec: RunRecord) -> float:
    """Total execution time incl. re-prompt overhead (metric 3)."""
    return float(sum(rec.iter_seconds))


def outcome(rec: RunRecord) -> str:
    """Classify a run into the Table 1 outcome column."""
    if rec.converged:
        return FIRST_ATTEMPT if rec.iterations == 1 else CONVERGED
    return DID_NOT_CONVERGE


def summarize(records: List[RunRecord]) -> Dict[str, object]:
    """Aggregate the sec. 3.3 metrics over a batch of runs.

    Returns first-attempt-success rate, convergence rate, mean refinements
    (over converged runs), total/mean time, error-type distribution, and the
    per-run outcome table.
    """
    n = len(records)
    if n == 0:
        raise ValueError("need at least one run record")
    first = [r for r in records if outcome(r) == FIRST_ATTEMPT]
    converged = [r for r in records if r.converged]
    failed = [r for r in records if not r.converged]
    ref_counts = [refinements(r) for r in converged]
    err_dist: Dict[str, int] = {}
    for r in records:
        if r.error_type:
            err_dist[r.error_type] = err_dist.get(r.error_type, 0) + 1
    times = [total_time(r) for r in records]
    return {
        "n": n,
        "first_attempt_rate": len(first) / n,
        "convergence_rate": len(converged) / n,
        "failure_count": len(failed),
        "mean_refinements": (sum(ref_counts) / len(ref_counts))
        if ref_counts else 0.0,
        "max_refinements": max(ref_counts) if ref_counts else 0,
        "total_time": sum(times),
        "mean_time": sum(times) / n,
        "error_distribution": err_dist,
        "outcomes": {r.name: outcome(r) for r in records},
    }
