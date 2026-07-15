"""Quantity-alignment metrics for zero-shot object counting (Zhang et al., 2026,
"Boosting Quantitative and Spatial Awareness for Zero-Shot Object Counting" /
QICA).

QICA argues that counting models suffer from "a lack of fine-grained quantity
awareness" and adds a *multi-level quantity alignment loss* to enforce numerical
consistency across the pipeline. The neural framework is out of scope, but the
paper's evaluation quantities are deterministic and standard for the counting
literature (FSC-147, CARPK), and its multi-level idea reframes cleanly as a
checkable metric:

* :func:`counting_errors` -- the canonical Mean Absolute Error and Root Mean
  Squared Error between predicted and ground-truth counts (the FSC-147 protocol).
* :func:`quantity_bin` -- a numerically-conditioned bin (few / several / many)
  for a count, the "numerically conditioned prompt" discretisation.
* :func:`multi_level_alignment` -- the multi-level quantity-alignment score:
  agreement of predicted vs. ground-truth counts at both the *exact* level
  (relative count error) and the *bin* level (fraction landing in the correct
  quantity bin), combined into one 0..1 consistency score.

Deterministic, stdlib-only.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

__all__ = [
    "counting_errors",
    "quantity_bin",
    "multi_level_alignment",
    "DEFAULT_BIN_EDGES",
]

# Bin upper edges (inclusive) -> label. A count greater than the last edge is "many".
DEFAULT_BIN_EDGES: Tuple[Tuple[float, str], ...] = (
    (2.0, "few"),
    (10.0, "several"),
    (50.0, "many"),
)


def counting_errors(
    predicted: Sequence[float], ground_truth: Sequence[float]
) -> Dict[str, float]:
    """Mean Absolute Error and Root Mean Squared Error over paired counts."""
    if len(predicted) != len(ground_truth) or not predicted:
        raise ValueError("predicted and ground_truth must be equal length, non-empty")
    n = len(predicted)
    abs_err = [abs(float(p) - float(g)) for p, g in zip(predicted, ground_truth)]
    mae = sum(abs_err) / n
    rmse = math.sqrt(sum(e * e for e in abs_err) / n)
    return {"mae": mae, "rmse": rmse}


def quantity_bin(count: float, edges: Sequence[Tuple[float, str]] = DEFAULT_BIN_EDGES) -> str:
    """Map a count to a quantity bin label (last bin extends to infinity)."""
    c = float(count)
    if c < 0:
        raise ValueError("count must be non-negative")
    for edge, label in edges:
        if c <= edge:
            return label
    return edges[-1][1]


def multi_level_alignment(
    predicted: Sequence[float],
    ground_truth: Sequence[float],
    edges: Sequence[Tuple[float, str]] = DEFAULT_BIN_EDGES,
    w_exact: float = 0.5,
) -> Dict[str, float]:
    """Combined multi-level quantity-alignment consistency score in ``[0, 1]``.

    * exact level: ``1 - |p - g| / max(g, 1)`` averaged (clamped at 0), i.e. mean
      relative count accuracy;
    * bin level: fraction of pairs whose predicted count lands in the same
      quantity bin as ground truth.

    The final score is ``w_exact * exact + (1 - w_exact) * bin``. Higher is more
    quantity-consistent.
    """
    if not 0.0 <= w_exact <= 1.0:
        raise ValueError("w_exact must be in [0, 1]")
    if len(predicted) != len(ground_truth) or not predicted:
        raise ValueError("predicted and ground_truth must be equal length, non-empty")
    n = len(predicted)
    exact_accs = []
    bin_hits = 0
    for p, g in zip(predicted, ground_truth):
        p, g = float(p), float(g)
        rel = abs(p - g) / max(g, 1.0)
        exact_accs.append(max(0.0, 1.0 - rel))
        if quantity_bin(p, edges) == quantity_bin(g, edges):
            bin_hits += 1
    exact = sum(exact_accs) / n
    bin_acc = bin_hits / n
    return {
        "exact_level": exact,
        "bin_level": bin_acc,
        "score": w_exact * exact + (1.0 - w_exact) * bin_acc,
    }
