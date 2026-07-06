"""Deterministic aerodynamic drag proxy for text-to-3D design optimisation.

Paper: T. Rios, S. Menzel, B. Sendhoff, "Large Language and Text-to-3D Models
for Engineering Design Optimization" (Honda Research Institute Europe).

The paper runs OpenFOAM CFD to obtain the drag coefficient ``cd`` of each
generated car mesh. That CFD step is *research-heavy / external* and is skipped
here. What IS deterministic and locally buildable are the geometric bookkeeping
steps the paper describes around the simulation:

  * Re-alignment (Sec. III-C): "we re-align the designs assuming that the
    largest overall dimension corresponds to the length of the car (x-axis) and
    the smallest dimension corresponds to the height (z-axis)."  We reproduce
    that canonicalisation of an axis-aligned bounding box.

  * Baseline metrics (Sec. IV-A): length, height, width and *projected frontal
    area* Af of the generated shapes.  The paper reports that cd and Af are
    linearly correlated (R-squared = 0.8409) and assumes "the projected frontal
    area has the largest impact on the drag coefficient".  We therefore provide
    a transparent linear drag proxy  cd = a * Af + b  standing in for the CFD.

  * Normalisation (Eq. 1): performance measures are normalised on the span of
    the baseline set,  x.N = (x - min) / (max - min).

This module is a *fitness proxy* for the optimisation loop, NOT a CFD solver.
It is deterministic: no wall clock, no randomness of its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Bounding-box extraction and re-alignment
# ---------------------------------------------------------------------------

Point3 = Tuple[float, float, float]


def bounding_box(points: Sequence[Point3]) -> Tuple[Point3, Point3]:
    """Axis-aligned bounding box (min-corner, max-corner) of a point set."""
    if not points:
        raise ValueError("bounding_box requires at least one point")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def box_extents(points: Sequence[Point3]) -> Tuple[float, float, float]:
    """Return the raw (x, y, z) side lengths of the AABB."""
    lo, hi = bounding_box(points)
    return (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])


@dataclass(frozen=True)
class CarDimensions:
    """Canonical car dimensions after re-alignment.

    ``length`` is the largest extent (paper: car length, x-axis), ``height``
    the smallest (z-axis) and ``width`` the middle one (y-axis).
    """

    length: float
    width: float
    height: float

    @property
    def frontal_area(self) -> float:
        """Projected frontal area Af = width * height (the y-z plane the car
        pushes through the air travelling along its length/x-axis)."""
        return self.width * self.height


def realign_dimensions(extents: Iterable[float]) -> CarDimensions:
    """Sort three raw extents into (length>=width>=height).

    Reproduces the paper's re-alignment: largest dimension -> length (x),
    smallest -> height (z), the remaining one -> width (y).
    """
    vals = sorted((float(v) for v in extents), reverse=True)
    if len(vals) != 3:
        raise ValueError("realign_dimensions expects exactly three extents")
    length, width, height = vals
    return CarDimensions(length=length, width=width, height=height)


def car_dimensions_from_points(points: Sequence[Point3]) -> CarDimensions:
    """Full pipeline: AABB extents -> re-aligned canonical car dimensions."""
    return realign_dimensions(box_extents(points))


# ---------------------------------------------------------------------------
# Linear drag proxy (stands in for CFD)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinearDragModel:
    """Linear surrogate  cd = slope * Af + intercept.

    The paper found cd linearly correlated with the projected frontal area
    (R-squared 0.8409).  The default coefficients are illustrative positive
    values; ``fit`` recovers them by least squares from paired data.
    """

    slope: float = 1.0
    intercept: float = 0.0

    def cd(self, frontal_area: float) -> float:
        return self.slope * frontal_area + self.intercept

    def cd_of_points(self, points: Sequence[Point3]) -> float:
        return self.cd(car_dimensions_from_points(points).frontal_area)


def fit_linear_drag(areas: Sequence[float], cds: Sequence[float]) -> LinearDragModel:
    """Ordinary least-squares fit of cd = slope*Af + intercept."""
    n = len(areas)
    if n != len(cds):
        raise ValueError("areas and cds must have equal length")
    if n < 2:
        raise ValueError("need at least two samples to fit a line")
    mean_a = sum(areas) / n
    mean_c = sum(cds) / n
    sxx = sum((a - mean_a) ** 2 for a in areas)
    if sxx == 0.0:
        raise ValueError("frontal areas are all identical; slope undefined")
    sxy = sum((a - mean_a) * (c - mean_c) for a, c in zip(areas, cds))
    slope = sxy / sxx
    intercept = mean_c - slope * mean_a
    return LinearDragModel(slope=slope, intercept=intercept)


def r_squared(areas: Sequence[float], cds: Sequence[float],
              model: LinearDragModel) -> float:
    """Coefficient of determination of ``model`` on the given samples."""
    n = len(cds)
    if n == 0:
        raise ValueError("need at least one sample")
    mean_c = sum(cds) / n
    ss_tot = sum((c - mean_c) ** 2 for c in cds)
    ss_res = sum((c - model.cd(a)) ** 2 for a, c in zip(areas, cds))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Baseline normalisation (Eq. 1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaselineNormaliser:
    """Min-max normaliser over a baseline set of measurements (Eq. 1)."""

    lo: float
    hi: float

    @classmethod
    def from_baseline(cls, values: Sequence[float]) -> "BaselineNormaliser":
        if not values:
            raise ValueError("baseline set is empty")
        return cls(lo=min(values), hi=max(values))

    def normalise(self, value: float) -> float:
        span = self.hi - self.lo
        if span == 0.0:
            return 0.0
        return (value - self.lo) / span

    def normalise_all(self, values: Sequence[float]) -> List[float]:
        return [self.normalise(v) for v in values]
