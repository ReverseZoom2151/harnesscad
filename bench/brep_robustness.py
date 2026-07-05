"""Seeded B-rep element masking and predictor robustness evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class MaskCase:
    fraction: float
    kept: tuple[str, ...]
    masked: tuple[str, ...]


def mask_cases(ids, fractions=(0.0, 0.25, 0.5), *, seed: int = 0):
    values = tuple(sorted(set(ids)))
    out = []
    for index, fraction in enumerate(fractions):
        if not 0 <= fraction <= 1:
            raise ValueError("mask fractions must be in [0,1]")
        shuffled = list(values)
        random.Random(seed + index).shuffle(shuffled)
        count = round(len(values) * fraction)
        masked = tuple(sorted(shuffled[:count]))
        out.append(MaskCase(fraction, tuple(v for v in values if v not in masked), masked))
    return tuple(out)


def evaluate_masking(ids, predictor, *, fractions=(0.0, 0.25, 0.5), seed=0,
                     comparator=lambda a, b: a == b):
    cases = mask_cases(ids, fractions, seed=seed)
    baseline = predictor(cases[0].kept)
    return {
        "baseline": baseline,
        "cases": tuple({
            "fraction": case.fraction, "masked": case.masked,
            "output": (output := predictor(case.kept)),
            "stable": comparator(output, baseline),
            "coverage": len(case.kept) / len(set(ids)) if ids else 1.0,
        } for case in cases),
    }
