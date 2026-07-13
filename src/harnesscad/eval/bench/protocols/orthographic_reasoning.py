"""TriView2CAD orthographic-projection reasoning scorer (CReFT-CAD, NeurIPS 2025).

The TriView2CAD benchmark defines a 15-dimensional parameter space over three
evaluation task families and one overall accuracy metric:

  * **Dimension recognition & pairing** (6 recognition params) — read an annotated
    dimension and map it to its geometric feature.
  * **Primitive counting** (3 counting params) — count instances of a CAD element
    (e.g. number of pier columns).
  * **Composite parameter computation** (6 composite params) — compute an
    engineering-critical derived quantity from a formula over factor parameters
    (e.g. "Cross-Bridge Pier Spacing" = "Pier Column Cross-Bridge Dimension" +
    "Pile Spacing").

Overall accuracy (Sec. 3.2) is::

    accuracy = (# correctly predicted parameters) / (total parameters across all
               test samples)

each of the 15 parameters treated as an independent prediction target. This
module implements the deterministic scoring: exact/tolerant matching per
parameter, per-family and overall accuracy, and a small formula evaluator for the
composite parameters. The VLM being scored is out of scope; this is the verifier.

Pure stdlib, deterministic (parallels :mod:`bench.metrics`).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

# Parameter families of the 15-D TriView2CAD space.
RECOGNITION = "recognition"
COUNTING = "counting"
COMPOSITE = "composite"
FAMILIES: Tuple[str, ...] = (RECOGNITION, COUNTING, COMPOSITE)

# The paper's split of the 15 parameters.
FAMILY_SIZES = {RECOGNITION: 6, COUNTING: 3, COMPOSITE: 6}


def _match(predicted: object, truth: object, tol: float) -> bool:
    """Exact match, with a numeric tolerance when both values are numbers."""
    if isinstance(predicted, bool) or isinstance(truth, bool):
        # Keep bool distinct from int (True == 1 in Python otherwise).
        return type(predicted) is type(truth) and predicted == truth
    if isinstance(predicted, (int, float)) and isinstance(truth, (int, float)):
        return abs(float(predicted) - float(truth)) <= tol
    return predicted == truth


def parameter_correct(name: str,
                      predicted: Mapping[str, object],
                      truth: Mapping[str, object],
                      tol: float = 1e-9) -> bool:
    """True iff parameter ``name`` is present in ``predicted`` and matches truth."""
    if name not in truth:
        raise KeyError("ground truth missing parameter %r" % (name,))
    return name in predicted and _match(predicted[name], truth[name], tol)


@dataclass(frozen=True)
class AccuracyReport:
    total: int
    correct: int
    per_family: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total) if self.total else 0.0

    def family_accuracy(self, family: str) -> float:
        c, t = self.per_family.get(family, (0, 0))
        return (c / t) if t else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "per_family": {f: {"correct": c, "total": t,
                               "accuracy": (c / t) if t else 0.0}
                           for f, (c, t) in self.per_family.items()},
        }


def score_sample(predicted: Mapping[str, object],
                 truth: Mapping[str, object],
                 families: Mapping[str, str] = None,
                 tol: float = 1e-9) -> AccuracyReport:
    """Score one sample: count correct parameters overall and per family.

    ``families`` maps a parameter name to one of FAMILIES; unmapped parameters are
    still counted toward the overall total but not attributed to a family.
    """
    families = families or {}
    correct = 0
    per_family: Dict[str, List[int]] = {f: [0, 0] for f in FAMILIES}
    for name in truth:
        ok = parameter_correct(name, predicted, truth, tol)
        correct += 1 if ok else 0
        fam = families.get(name)
        if fam in per_family:
            per_family[fam][1] += 1
            if ok:
                per_family[fam][0] += 1
    return AccuracyReport(
        total=len(truth),
        correct=correct,
        per_family={f: (c, t) for f, (c, t) in per_family.items()},
    )


def overall_accuracy(samples: Iterable[Tuple[Mapping[str, object],
                                             Mapping[str, object]]],
                     families: Mapping[str, str] = None,
                     tol: float = 1e-9) -> AccuracyReport:
    """Aggregate accuracy over many (predicted, truth) samples (Sec. 3.2)."""
    total = 0
    correct = 0
    fam_acc: Dict[str, List[int]] = {f: [0, 0] for f in FAMILIES}
    for predicted, truth in samples:
        rep = score_sample(predicted, truth, families, tol)
        total += rep.total
        correct += rep.correct
        for f, (c, t) in rep.per_family.items():
            fam_acc[f][0] += c
            fam_acc[f][1] += t
    return AccuracyReport(
        total=total,
        correct=correct,
        per_family={f: (c, t) for f, (c, t) in fam_acc.items()},
    )


# --------------------------------------------------------------------------- #
# Composite parameter computation (formula evaluator)
# --------------------------------------------------------------------------- #
_OPS: Dict[str, Callable[[float, float], float]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
}


@dataclass(frozen=True)
class CompositeFormula:
    """A derived quantity = fold of factor parameters under one operator.

    e.g. Cross-Bridge Pier Spacing = "+" over ("Pier Column Cross-Bridge
    Dimension", "Pile Spacing"). Deterministic left-fold; requires >= 1 factor.
    """

    name: str
    op: str
    factors: Tuple[str, ...]

    def compute(self, params: Mapping[str, float]) -> float:
        if self.op not in _OPS:
            raise ValueError("unsupported op %r" % (self.op,))
        if not self.factors:
            raise ValueError("formula needs at least one factor")
        fn = _OPS[self.op]
        acc = float(params[self.factors[0]])
        for f in self.factors[1:]:
            acc = fn(acc, float(params[f]))
        return acc


def compute_composites(formulas: Sequence[CompositeFormula],
                       params: Mapping[str, float]) -> Dict[str, float]:
    """Evaluate every composite formula against the factor parameters."""
    return {f.name: f.compute(params) for f in formulas}


def composite_correct(formula: CompositeFormula,
                      predicted_value: float,
                      params: Mapping[str, float],
                      tol: float = 1e-9) -> bool:
    """True iff a predicted composite value equals the formula's computed value."""
    return abs(float(predicted_value) - formula.compute(params)) <= tol
