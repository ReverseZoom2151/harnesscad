"""Dimension Accuracy (DA) metric for parametric primitive analysis.

DA is a *dimension-based* evaluation metric: it measures how well a
predicted dimensional annotation aligns with its ground-truth annotation, jointly
across three validation functions (a prediction counts only if all three hold):

    T(P, P_hat) = 1[ type(P_hat) == type(P) ]
    V(P, P_hat) = 1[ | V_hat - V | <= tau_v ]
    E(P, P_hat) = 1[ sum_k 1[ |E_hat_k - E_k| <= tau_e ] == N_i ]
    DA          = (1/M) sum_i T * V * E

where ``type`` is the dimension type (length / diameter / radius / angle),
``V`` is the numeric dimension value, and ``E`` are the coordinates of the
geometric elements the dimension attaches to. ``tau_v`` and ``tau_e`` are the
value and positional tolerances. Every element must be within ``tau_e`` for E to
pass. All computation is deterministic and stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Recognised dimension types: length, diameter, radius, or angle.
DIMENSION_TYPES = ("length", "diameter", "radius", "angle")


@dataclass(frozen=True)
class Dimension:
    """A dimensional annotation: type, numeric value, and attached element points."""

    dim_type: str
    value: float
    elements: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.dim_type not in DIMENSION_TYPES:
            raise ValueError(f"unknown dimension type: {self.dim_type!r}")


def type_match(gt: Dimension, pred: Dimension) -> bool:
    """T(P, P_hat): dimension types agree (Eq. 1)."""
    return gt.dim_type == pred.dim_type


def value_match(gt: Dimension, pred: Dimension, tau_v: float) -> bool:
    """V(P, P_hat): value deviates within ``tau_v`` (Eq. 2)."""
    return abs(pred.value - gt.value) <= tau_v


def element_match(gt: Dimension, pred: Dimension, tau_e: float) -> bool:
    """E(P, P_hat): every attached element aligns within ``tau_e`` (Eq. 3).

    Requires equal element counts and each corresponding element pair within the
    positional tolerance (Euclidean, per-coordinate absolute distance).
    """
    if len(gt.elements) != len(pred.elements):
        return False
    if not gt.elements:
        return True
    hits = 0
    for (gx, gy), (px, py) in zip(gt.elements, pred.elements):
        if abs(px - gx) <= tau_e and abs(py - gy) <= tau_e:
            hits += 1
    return hits == len(gt.elements)


def is_correct(gt: Dimension, pred: Dimension,
               tau_v: float, tau_e: float) -> bool:
    """Single-dimension correctness ``T * V * E`` (product of the three checks)."""
    return (type_match(gt, pred)
            and value_match(gt, pred, tau_v)
            and element_match(gt, pred, tau_e))


@dataclass(frozen=True)
class DAResult:
    """Aggregate DA over a set of paired dimensions."""

    correct: int
    total: int
    accuracy: float


def dimension_accuracy(gts: list[Dimension], preds: list[Dimension],
                       tau_v: float, tau_e: float) -> DAResult:
    """DA over paired ground-truth / predicted dimensions (Eq. 4).

    ``gts`` and ``preds`` are index-aligned; a ``None`` prediction (missing) is
    counted as incorrect. DA is the ratio of correct predictions to the number
    of ground-truth dimensions.
    """
    if len(gts) != len(preds):
        raise ValueError("gts and preds must be index-aligned (equal length)")
    total = len(gts)
    if total == 0:
        return DAResult(0, 0, 0.0)
    correct = 0
    for gt, pred in zip(gts, preds):
        if pred is not None and is_correct(gt, pred, tau_v, tau_e):
            correct += 1
    return DAResult(correct, total, correct / total)
