"""Analytic Bezier utilities and honest kernel/fit adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Protocol


def bezier_curve(control_points, t: float):
    if not 0 <= t <= 1 or not control_points:
        raise ValueError("invalid curve input")
    degree = len(control_points) - 1
    return tuple(sum(comb(degree, i) * (1-t)**(degree-i) * t**i * point[d]
                     for i, point in enumerate(control_points))
                 for d in range(len(control_points[0])))


def bezier_triangle(control_points, degree: int, s: float, t: float):
    """Evaluate mapping ``(i,j)->point`` on the barycentric triangle."""
    if s < 0 or t < 0 or s + t > 1:
        raise ValueError("outside triangle")
    result = None
    for (i, j), point in control_points.items():
        k = degree - i - j
        coefficient = math_multinomial(degree, i, j, k) * s**i * t**j * (1-s-t)**k
        result = [0.0] * len(point) if result is None else result
        for axis, value in enumerate(point):
            result[axis] += coefficient * value
    if result is None:
        raise ValueError("control net is empty")
    return tuple(result)


def math_multinomial(n, *parts):
    if sum(parts) != n or min(parts) < 0:
        raise ValueError("invalid multinomial indices")
    result, remaining = 1, n
    for part in parts:
        result *= comb(remaining, part)
        remaining -= part
    return result


@dataclass(frozen=True)
class TrimCell:
    depth: int
    x: int
    y: int
    classification: str  # inside | outside | intersect


class SurfaceExtractor(Protocol):
    def extract(self, shape: object) -> object: ...


class BoundaryFitter(Protocol):
    def fit(self, samples, initial_control_points, regularization: float): ...
