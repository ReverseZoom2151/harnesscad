"""Pluggable quantitative objective for the exploration / strategies loops.

Where :mod:`exploration.tournament` and :mod:`strategies.best_of_n` rank
candidates by a *verifier* comparator (ok? fewer diagnostics? more ops?), this
module adds a *quantitative* objective on top of the cost/measurement layer in
:mod:`estimate`: minimise mass, minimise cost, meet target dimensions, and
penalise constraint violations. An :class:`Objective` turns a backend (or a
pre-built :class:`estimate.PartEstimate`) into a single scalar ``score`` — higher
is better — so it drops straight into those loops as their scorer.

Design points:
  * ``score(...) -> float`` is higher-is-better, matching
    ``strategies.best_of_n.default_scorer`` and ``EloTournament`` (both prefer
    the larger value), so a caller can pass ``objective.score`` verbatim.
  * ``vector(...) -> tuple`` gives a Pareto-style multi-objective tuple in a
    canonical *minimise* orientation (smaller is better on every axis), with
    :func:`dominates` for Pareto comparisons.
  * Deterministic, stdlib-only, no wall clock. Unmeasurable / unbuildable
    candidates receive a large penalty so they always lose to buildable ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from estimate import MaterialTable, PartEstimate, estimate_part
from verify import Severity, VerifyReport


# A large finite penalty for an unmeasurable / unbuildable candidate: big enough
# to always lose to a real part, finite so scores stay sortable and printable.
PENALTY = 1.0e9


# --------------------------------------------------------------------------- #
# Objective terms
# --------------------------------------------------------------------------- #
# Metrics an objective term can read off a PartEstimate (+ a verify report).
_METRICS = {
    "mass": lambda e: e.mass,
    "volume": lambda e: e.volume,
    "cost": lambda e: e.total_cost,
    "material_cost": lambda e: e.material_cost,
    "machining_cost": lambda e: e.rough_machining_cost,
    "surface_area": lambda e: e.surface_area,
    "bbox_x": lambda e: e.bbox[0] if e.bbox else None,
    "bbox_y": lambda e: e.bbox[1] if e.bbox else None,
    "bbox_z": lambda e: e.bbox[2] if e.bbox else None,
}


@dataclass
class Term:
    """One weighted objective term.

    ``metric`` is a key of :data:`_METRICS` (mass, cost, bbox_x, ...) or the
    special ``"violations"`` (count of ERROR diagnostics in a verify report).
    ``goal`` is ``"min"``, ``"max"`` or ``"target"`` (needs ``target``).
    ``weight`` scales the term; ``scale`` normalises the metric to O(1) so terms
    with different units combine sensibly (default 1.0 = raw units).
    """

    metric: str
    goal: str = "min"
    weight: float = 1.0
    target: Optional[float] = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.goal not in ("min", "max", "target"):
            raise ValueError(f"unknown goal {self.goal!r}")
        if self.goal == "target" and self.target is None:
            raise ValueError("goal='target' requires a target value")
        if self.scale == 0:
            raise ValueError("scale must be non-zero")

    # -- raw metric extraction --------------------------------------------- #
    def raw_value(self, est: PartEstimate,
                  report: Optional[VerifyReport]) -> Optional[float]:
        if self.metric == "violations":
            return float(_count_violations(report))
        getter = _METRICS.get(self.metric)
        if getter is None:
            raise ValueError(f"unknown metric {self.metric!r}")
        return getter(est)

    # -- minimise-oriented cost (smaller is better) ------------------------ #
    def minimise_cost(self, est: PartEstimate,
                      report: Optional[VerifyReport]) -> float:
        """Return this term's contribution oriented so smaller == better.

        Missing/unmeasurable metrics return :data:`PENALTY` so a candidate that
        cannot even be measured is always worse than one that can.
        """
        v = self.raw_value(est, report)
        if v is None:
            return PENALTY
        v = v / self.scale
        if self.goal == "min":
            return v
        if self.goal == "max":
            return -v
        return abs(v - (self.target / self.scale))  # target

    def weighted(self, est: PartEstimate,
                 report: Optional[VerifyReport]) -> float:
        return self.weight * self.minimise_cost(est, report)


def _count_violations(report: Optional[VerifyReport]) -> int:
    if report is None:
        return 0
    return sum(1 for d in report.diagnostics
               if d.severity is Severity.ERROR)


# --------------------------------------------------------------------------- #
# The objective
# --------------------------------------------------------------------------- #
@dataclass
class Objective:
    """A weighted, multi-term quantitative objective.

    ``score(source, verify_report=None)`` returns a **higher-is-better** float
    (the negated weighted sum of the per-term minimise-costs), ready to be a
    scorer for ``best_of_n`` / ``EloTournament``. ``vector(...)`` returns the
    per-term minimise-oriented values as a Pareto tuple.
    """

    terms: List[Term] = field(default_factory=list)
    material: str = "aluminium"
    table: Optional[MaterialTable] = None

    # -- estimate resolution ----------------------------------------------- #
    def _estimate(self, source: Any) -> PartEstimate:
        if isinstance(source, PartEstimate):
            return source
        return estimate_part(source, material=self.material, table=self.table)

    # -- scalar score (higher is better) ----------------------------------- #
    def score(self, source: Any,
              verify_report: Optional[VerifyReport] = None) -> float:
        """Higher-is-better scalar. Also usable as ``best_of_n`` scorer."""
        # A candidate/result that is explicitly not-ok is unbuildable.
        if _is_not_ok(source):
            return -PENALTY
        est = self._estimate(source)
        total = sum(t.weighted(est, verify_report) for t in self.terms)
        return -total  # minimise-cost -> higher-is-better

    __call__ = score

    # -- Pareto vector (smaller is better on every axis) ------------------- #
    def vector(self, source: Any,
               verify_report: Optional[VerifyReport] = None) -> Tuple[float, ...]:
        """Per-term minimise-oriented values (weight applied) as a tuple."""
        if _is_not_ok(source):
            return tuple(PENALTY for _ in self.terms) or (PENALTY,)
        est = self._estimate(source)
        return tuple(t.weighted(est, verify_report) for t in self.terms)

    def as_scorer(self,
                  verify_report: Optional[VerifyReport] = None
                  ) -> Callable[[Any], float]:
        """Return a one-arg ``scorer(source) -> float`` closing over a report."""
        return lambda source: self.score(source, verify_report)


def _is_not_ok(source: Any) -> bool:
    """True only when ``source`` carries an explicit falsey ``ok`` flag.

    Backends/estimates/dicts have no ``ok`` and are treated as buildable; a
    Variant/Candidate/ApplyOpsResult with ``ok == False`` is unbuildable.
    """
    if isinstance(source, (PartEstimate, dict)):
        return False
    ok = getattr(source, "ok", None)
    if isinstance(ok, bool):
        return not ok
    return False


# --------------------------------------------------------------------------- #
# Pareto helper
# --------------------------------------------------------------------------- #
def dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    """True iff Pareto-vector ``a`` dominates ``b`` (minimise on every axis).

    ``a`` dominates ``b`` when it is no worse on every axis and strictly better
    on at least one. Vectors must be the same length.
    """
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    no_worse = all(x <= y for x, y in zip(a, b))
    strictly = any(x < y for x, y in zip(a, b))
    return no_worse and strictly


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
def mass_objective(weight: float = 1.0, material: str = "aluminium",
                   table: Optional[MaterialTable] = None) -> Objective:
    """Minimise part mass. Lower mass -> strictly higher score."""
    return Objective([Term("mass", "min", weight)],
                     material=material, table=table)


def cost_objective(weight: float = 1.0, material: str = "aluminium",
                   table: Optional[MaterialTable] = None) -> Objective:
    """Minimise total (material + machining) cost."""
    return Objective([Term("cost", "min", weight)],
                     material=material, table=table)


def multi_objective(mass_weight: float = 1.0, cost_weight: float = 1.0,
                    violation_weight: float = 100.0,
                    *, material: str = "aluminium",
                    table: Optional[MaterialTable] = None,
                    mass_scale: float = 1.0,
                    cost_scale: float = 1.0) -> Objective:
    """Minimise mass + cost while penalising constraint violations.

    ``violation_weight`` scales the count of ERROR-severity diagnostics from a
    verify report, so a valid-but-heavier part can still beat a lighter but
    broken one. ``*_scale`` normalise mass (g) and cost so their weights are
    comparable when both are 1.0.
    """
    terms = [
        Term("mass", "min", mass_weight, scale=mass_scale),
        Term("cost", "min", cost_weight, scale=cost_scale),
        Term("violations", "min", violation_weight),
    ]
    return Objective(terms, material=material, table=table)


def target_dims_objective(target: Tuple[float, float, float],
                          weight: float = 1.0,
                          *, material: str = "aluminium",
                          table: Optional[MaterialTable] = None) -> Objective:
    """Meet a target bounding box (x, y, z) in mm; deviation is penalised."""
    tx, ty, tz = target
    terms = [
        Term("bbox_x", "target", weight, target=tx),
        Term("bbox_y", "target", weight, target=ty),
        Term("bbox_z", "target", weight, target=tz),
    ]
    return Objective(terms, material=material, table=table)
