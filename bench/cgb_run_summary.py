"""Run-level aggregation and leaderboard rows for a CADGenBench-style benchmark.

Rolls per-sample results into one run summary. Three conventions are load-bearing
and are what make this different from the harness's existing report builders
(``bench.runner.SuiteReport``, ``bench.graphcad_cadbench_report``,
``bench.t2cadbench_scorecard``), none of which model the
valid / invalid / **missing** trichotomy or force non-scorable samples to zero:

1. **Zeros are included.** Invalid *and* missing samples contribute ``0.0`` to
   the aggregate. Averaging only over the successes would let a run that
   produced two good parts and skipped the other 98 top the leaderboard.
2. **A missing candidate is a distinct status.** "The agent never wrote an
   output" is not the same failure as "it wrote something invalid", and the
   summary reports both counts, even though both score zero.
3. **Task types are open.** ``generation`` and ``editing`` are the known
   buckets and lead the headline in that order; any new task type gets its own
   bucket automatically, sorted after them, with no schema change.

The summary is a plain dict, JSON-serializable, with every float rounded to 4
decimals so two runs of the same numbers produce byte-identical output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from bench.cgb_cad_score import STATUS_INVALID, STATUS_MISSING, STATUS_VALID

STATUSES = (STATUS_VALID, STATUS_INVALID, STATUS_MISSING)
KNOWN_TASK_TYPES = ("generation", "editing")
DEFAULT_TASK_TYPE = "generation"
ROUND_DIGITS = 4


@dataclass(frozen=True)
class SampleResult:
    """One sample's contribution to the run."""

    name: str
    status: str = STATUS_MISSING
    cad_score: float = 0.0
    task_type: str = DEFAULT_TASK_TYPE

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status: {self.status!r}")

    @property
    def effective_score(self) -> float:
        """A non-valid sample scores zero regardless of what it reported."""
        if self.status != STATUS_VALID:
            return 0.0
        return float(self.cad_score)


def parse_result(name: str, payload: Optional[dict], task_type: Optional[str] = None) -> SampleResult:
    """Read one per-sample ``result.json``-shaped dict, tolerantly.

    A ``None`` payload (no result file) and an absent status both read as
    ``missing``; an absent or null ``cad_score`` reads as ``0.0``. An unknown
    status is an authoring error and raises rather than being silently coerced.
    """
    payload = payload or {}
    status = payload.get("status") or STATUS_MISSING
    score = payload.get("cad_score")
    return SampleResult(
        name=name,
        status=str(status),
        cad_score=0.0 if score is None else float(score),
        task_type=str(task_type or payload.get("task_type") or DEFAULT_TASK_TYPE),
    )


def _bucket(results: Sequence[SampleResult]) -> Dict[str, Any]:
    n = len(results)
    scores = [r.effective_score for r in results]
    counts = {s: sum(1 for r in results if r.status == s) for s in STATUSES}
    return {
        "score": round(sum(scores) / n, ROUND_DIGITS) if n else 0.0,
        "validity_rate": (
            round(counts[STATUS_VALID] / n, ROUND_DIGITS) if n else 0.0
        ),
        "n_samples": n,
        "n_valid": counts[STATUS_VALID],
        "n_invalid": counts[STATUS_INVALID],
        "n_missing": counts[STATUS_MISSING],
    }


def _ordered_task_types(task_types: Iterable[str]) -> List[str]:
    """Known types first (declaration order), then any unknown type, sorted."""
    present = set(task_types)
    ordered = [t for t in KNOWN_TASK_TYPES if t in present]
    ordered.extend(sorted(t for t in present if t not in KNOWN_TASK_TYPES))
    return ordered


def build_run_summary(results: Iterable[SampleResult]) -> Dict[str, Any]:
    """Aggregate per-sample results into the run summary dict."""
    results = list(results)
    overall = _bucket(results)

    by_task: Dict[str, List[SampleResult]] = {}
    for r in results:
        by_task.setdefault(r.task_type, []).append(r)

    order = _ordered_task_types(by_task)
    per_task_scores = {t: _bucket(by_task[t]) for t in order}
    score_by_task_type = {t: per_task_scores[t]["score"] for t in order}

    per_sample_scores = {
        r.name: {
            "status": r.status,
            "cad_score": round(r.effective_score, ROUND_DIGITS),
            "task_type": r.task_type,
        }
        for r in sorted(results, key=lambda r: r.name)
    }

    return {
        "aggregate_score": overall["score"],
        "validity_rate": overall["validity_rate"],
        "n_samples": overall["n_samples"],
        "n_valid": overall["n_valid"],
        "n_invalid": overall["n_invalid"],
        "n_missing": overall["n_missing"],
        "score_by_task_type": score_by_task_type,
        "per_task_scores": per_task_scores,
        "per_sample_scores": per_sample_scores,
    }


@dataclass(frozen=True)
class LeaderboardRow:
    submission: str
    aggregate_score: float
    validity_rate: float
    n_samples: int
    validated: bool = False

    def to_dict(self) -> dict:
        return {
            "submission": self.submission,
            "aggregate_score": self.aggregate_score,
            "validity_rate": self.validity_rate,
            "n_samples": self.n_samples,
            "validated": self.validated,
        }


def leaderboard_row(
    submission: str, summary: Dict[str, Any], *, validated: bool = False
) -> LeaderboardRow:
    """Project a run summary onto its leaderboard row.

    Rows publish as **unvalidated**: promotion to the validated tier is a
    separate methodology review, never something a submission can assert about
    itself.
    """
    return LeaderboardRow(
        submission=submission,
        aggregate_score=float(summary.get("aggregate_score", 0.0)),
        validity_rate=float(summary.get("validity_rate", 0.0)),
        n_samples=int(summary.get("n_samples", 0)),
        validated=validated,
    )


def rank_leaderboard(rows: Iterable[LeaderboardRow]) -> List[LeaderboardRow]:
    """Rank rows: aggregate score desc, then validity rate desc, then name asc.

    The deterministic name tie-break keeps two identically-scoring submissions
    from swapping places between renders of the same board.
    """
    return sorted(
        rows,
        key=lambda r: (-r.aggregate_score, -r.validity_rate, r.submission),
    )
