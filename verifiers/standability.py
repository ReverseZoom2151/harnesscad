"""Deterministic, analytic ground-stability checks for rigid geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, pi, sin
from typing import Iterable, Sequence

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


def convex_hull(points: Iterable[Point2]) -> tuple[Point2, ...]:
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 1:
        return tuple(pts)

    def cross(o: Point2, a: Point2, b: Point2) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point2] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point2] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return tuple(lower[:-1] + upper[:-1])


def signed_stability_margin(point: Point2, hull: Sequence[Point2]) -> float | None:
    """Minimum inward edge distance; negative means outside the support hull."""
    if len(hull) < 3:
        return None
    distances = []
    for a, b in zip(hull, (*hull[1:], hull[0])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        distances.append((dx * (point[1] - a[1]) - dy * (point[0] - a[0]))
                         / hypot(dx, dy))
    return min(distances)


@dataclass(frozen=True)
class TiltSample:
    angle: float
    axis: Point2
    potential_delta: float


@dataclass(frozen=True)
class StandabilityReport:
    support_hull: tuple[Point2, ...]
    projected_com: Point2
    margin: float | None
    supported: bool
    tilt_samples: tuple[TiltSample, ...]
    minimum_potential_delta: float | None
    robust: bool
    diagnostics: tuple[str, ...]


def evaluate_standability(
    center_of_mass: Point3,
    contacts: Iterable[Point3],
    *,
    tilt_radians: float = 0.03,
    sample_count: int = 20,
    tolerance: float = 1e-9,
) -> StandabilityReport:
    if sample_count <= 0 or tilt_radians < 0 or tolerance < 0:
        raise ValueError("invalid standability configuration")
    contacts = tuple(contacts)
    hull = convex_hull((p[0], p[1]) for p in contacts)
    projected = (float(center_of_mass[0]), float(center_of_mass[1]))
    margin = signed_stability_margin(projected, hull)
    supported = margin is not None and margin >= -tolerance
    diagnostics: list[str] = []
    if len(hull) < 3:
        diagnostics.append("degenerate_support")
    elif not supported:
        diagnostics.append("com_outside_support")

    samples: list[TiltSample] = []
    if hull:
        ground_z = min((p[2] for p in contacts), default=0.0)
        c, s = cos(tilt_radians), sin(tilt_radians)
        for i in range(sample_count):
            theta = 2 * pi * i / sample_count
            ax, ay = cos(theta), sin(theta)
            # A body tips about the leading support edge, not the support
            # centroid. Pick the hull point that maximises the upward first
            # derivative for this signed rotation direction.
            px, py = min(hull, key=lambda p: ax * p[1] - ay * p[0])
            x, y, z = (center_of_mass[0] - px, center_of_mass[1] - py,
                       center_of_mass[2] - ground_z)
            # Rodrigues z component for rotation about horizontal (ax, ay, 0).
            new_z = z * c + (ax * y - ay * x) * s
            samples.append(TiltSample(theta, (ax, ay), new_z - z))
    minimum = min((s.potential_delta for s in samples), default=None)
    robust = supported and minimum is not None and minimum >= -tolerance
    if supported and not robust:
        diagnostics.append("tilt_lowers_center_of_mass")
    return StandabilityReport(
        hull, projected, margin, supported, tuple(samples), minimum, robust,
        tuple(diagnostics)
    )
