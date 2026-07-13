"""Deterministic confusion matrices and Cohen's kappa."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping


@dataclass(frozen=True)
class AgreementReport:
    labels: tuple[Hashable, ...]
    confusion: Mapping[Hashable, Mapping[Hashable, int]]
    observed: float
    expected: float
    kappa: float
    n: int


def cohen_kappa(
    first: Iterable[Hashable], second: Iterable[Hashable]
) -> AgreementReport:
    a, b = tuple(first), tuple(second)
    if len(a) != len(b):
        raise ValueError("raters must label the same number of items")
    labels = tuple(sorted(set(a) | set(b), key=lambda value: repr(value)))
    matrix = {x: {y: 0 for y in labels} for x in labels}
    for x, y in zip(a, b):
        matrix[x][y] += 1
    n = len(a)
    if not n:
        return AgreementReport(labels, matrix, 0.0, 0.0, 0.0, 0)
    observed = sum(matrix[label][label] for label in labels) / n
    row = {x: sum(matrix[x].values()) for x in labels}
    column = {y: sum(matrix[x][y] for x in labels) for y in labels}
    expected = sum(row[label] * column[label] for label in labels) / (n * n)
    denominator = 1.0 - expected
    kappa = ((observed - expected) / denominator
             if denominator else (1.0 if observed == 1.0 else 0.0))
    frozen_matrix = {x: dict(matrix[x]) for x in labels}
    return AgreementReport(labels, frozen_matrix, observed, expected, kappa, n)
