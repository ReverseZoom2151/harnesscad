"""Versioned, deterministic sphere-distributed CAD prompt cameras."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CanonicalView:
    id: str
    direction: tuple[float, float, float]
    up: tuple[float, float, float]
    version: str = "fibonacci-v1"


def canonical_views(count=12):
    if count < 2:
        raise ValueError("at least two views required")
    golden = math.pi * (3 - math.sqrt(5))
    views = []
    for index in range(count):
        y = 1 - 2 * (index + .5) / count
        radius = math.sqrt(max(0, 1-y*y))
        theta = golden * index
        direction = (radius*math.cos(theta), y, radius*math.sin(theta))
        reference = (0., 0., 1.) if abs(direction[2]) < .9 else (0., 1., 0.)
        cross = (reference[1]*direction[2]-reference[2]*direction[1],
                 reference[2]*direction[0]-reference[0]*direction[2],
                 reference[0]*direction[1]-reference[1]*direction[0])
        norm = math.sqrt(sum(value*value for value in cross))
        up = tuple(value/norm for value in cross)
        views.append(CanonicalView(f"view-{index:02d}", direction, up))
    return tuple(views)
