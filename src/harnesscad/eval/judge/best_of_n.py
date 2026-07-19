"""Best-of-N aggregation and the invalidity ratio.

This module samples N candidate CAD programs per ground-truth part, evaluates
them, and reports three deterministic aggregates over the run:

* **best-of-N selection** -- per sample, keep the candidate with the smallest
  Chamfer Distance (and, separately, the largest IoU); a sample with no valid
  candidate contributes nothing to the metric means but *does* count against the
  invalidity ratio,
* **invalidity ratio (IR)** -- the fraction of samples for which no candidate
  produced a valid mesh (the model's reliability, orthogonal to accuracy on the
  ones that did compile),
* **skip-worst curve** -- IR and mean CD recomputed after dropping the k
  worst-CD samples, exposing how the tail dominates the mean.

The evaluator fixes a canonical pose before comparison: centre the mesh at the
origin, scale so its largest extent is 1, then translate to the unit box
``[0.5, 0.5, 0.5]`` (:func:`normalize_to_unit_box`).

This module is the pure, deterministic aggregation over already-computed
per-candidate metric records.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "Candidate",
    "SampleResult",
    "RunReport",
    "normalize_to_unit_box",
    "select_best",
    "aggregate_run",
]


@dataclass(frozen=True)
class Candidate:
    """One generated candidate's deterministic metrics against its GT.

    ``valid`` is False when the candidate failed to compile / mesh; its ``cd``
    and ``iou`` are then ignored. ``cd`` (Chamfer Distance) is lower-is-better,
    ``iou`` is higher-is-better.
    """

    valid: bool
    cd: Optional[float] = None
    iou: Optional[float] = None


@dataclass(frozen=True)
class SampleResult:
    """Best-of-N outcome for one ground-truth part."""

    n_candidates: int
    n_valid: int
    best_cd: Optional[float]
    best_iou: Optional[float]

    @property
    def any_valid(self) -> bool:
        return self.n_valid > 0


@dataclass(frozen=True)
class RunReport:
    """Deterministic aggregates over a whole evaluation run."""

    n_samples: int
    invalidity_ratio: float
    mean_iou: float
    mean_cd: float
    median_cd: float
    #: (skip_k, invalidity_ratio_after_skip, mean_cd_after_skip) rows.
    skip_curve: Tuple[Tuple[int, float, float], ...] = field(default=())


def normalize_to_unit_box(points: Sequence[Sequence[float]]) -> List[Tuple[float, float, float]]:
    """Centre, unit-scale and translate a point set into ``[0.5, 0.5, 0.5]``.

    Mirrors cadrille's canonical pose: subtract the bbox centre, divide by the
    largest extent (unless it is ~0), then add 0.5 to every coordinate. Pure
    Python; deterministic. Raises ``ValueError`` on an empty set.
    """
    pts = [tuple(float(c) for c in p) for p in points]
    if not pts:
        raise ValueError("cannot normalise an empty point set")
    dims = len(pts[0])
    mins = [min(p[d] for p in pts) for d in range(dims)]
    maxs = [max(p[d] for p in pts) for d in range(dims)]
    centre = [(mins[d] + maxs[d]) / 2.0 for d in range(dims)]
    extent = max(maxs[d] - mins[d] for d in range(dims))
    scale = 1.0 / extent if extent > 1e-7 else 1.0
    out = []
    for p in pts:
        out.append(tuple((p[d] - centre[d]) * scale + 0.5 for d in range(dims)))
    return out


def select_best(candidates: Sequence[Candidate]) -> SampleResult:
    """Best-of-N: min-CD and max-IoU over the valid candidates of one sample."""
    valid = [c for c in candidates if c.valid]
    cds = [c.cd for c in valid if c.cd is not None]
    ious = [c.iou for c in valid if c.iou is not None]
    return SampleResult(
        n_candidates=len(candidates),
        n_valid=len(valid),
        best_cd=min(cds) if cds else None,
        best_iou=max(ious) if ious else None,
    )


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return math.nan
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def aggregate_run(
    samples: Sequence[SampleResult],
    *,
    skip_max: int = 4,
) -> RunReport:
    """Aggregate per-sample best-of-N results into a :class:`RunReport`.

    ``invalidity_ratio`` is the fraction of samples with no valid candidate.
    Means/median are taken over the samples that *do* have a best CD/IoU. The
    skip-worst curve drops the k largest CDs and folds them into the IR (the
    cadrille ``ir + k`` convention), for k in ``0..skip_max``.
    """
    n = len(samples)
    if n == 0:
        return RunReport(0, 0.0, math.nan, math.nan, math.nan, ())
    cds = sorted(s.best_cd for s in samples if s.best_cd is not None)
    ious = [s.best_iou for s in samples if s.best_iou is not None]
    n_invalid_cd = sum(1 for s in samples if s.best_cd is None)

    mean_iou = sum(ious) / len(ious) if ious else math.nan
    mean_cd = sum(cds) / len(cds) if cds else math.nan
    median_cd = _median(cds)

    skip_rows: List[Tuple[int, float, float]] = []
    for k in range(0, skip_max + 1):
        ir_k = (n_invalid_cd + k) / n
        kept = cds[: len(cds) - k] if k <= len(cds) else []
        mean_k = sum(kept) / len(kept) if kept else math.nan
        skip_rows.append((k, ir_k, mean_k))

    return RunReport(
        n_samples=n,
        invalidity_ratio=n_invalid_cd / n,
        mean_iou=mean_iou,
        mean_cd=mean_cd,
        median_cd=median_cd,
        skip_curve=tuple(skip_rows),
    )
