"""Small deterministic effect-size and uncertainty helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EffectReport:
    mean_a: float
    mean_b: float
    difference: float
    cohen_d: float
    ci95_low: float
    ci95_high: float


def compare_samples(a: Sequence[float], b: Sequence[float]) -> EffectReport:
    """Report mean difference, pooled Cohen's d and normal 95% CI."""
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each sample needs at least two observations")
    av = [float(value) for value in a]
    bv = [float(value) for value in b]
    mean_a, mean_b = mean(av), mean(bv)
    var_a, var_b = stdev(av) ** 2, stdev(bv) ** 2
    pooled = math.sqrt(
        ((len(av) - 1) * var_a + (len(bv) - 1) * var_b)
        / (len(av) + len(bv) - 2)
    )
    difference = mean_a - mean_b
    d = difference / pooled if pooled else (0.0 if difference == 0 else math.inf)
    standard_error = math.sqrt(var_a / len(av) + var_b / len(bv))
    margin = 1.96 * standard_error
    return EffectReport(
        mean_a, mean_b, difference, d, difference - margin, difference + margin
    )


def effect_magnitude(cohen_d: float) -> str:
    value = abs(cohen_d)
    if value < 0.2:
        return "negligible"
    if value < 0.5:
        return "small"
    if value < 0.8:
        return "medium"
    return "large"
