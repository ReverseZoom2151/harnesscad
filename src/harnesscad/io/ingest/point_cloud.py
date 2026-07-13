"""Finite point-cloud validation, seeded sampling and reversible normalization."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class CloudTransform:
    center: tuple[float, float, float]
    scale: float

    def invert(self, point):
        return tuple(value*self.scale + self.center[i] for i, value in enumerate(point))


def canonicalize_cloud(points, *, count=None, seed=0, normalize=False):
    values = [tuple(map(float, point)) for point in points]
    if any(len(point) not in {3, 6} or any(not math.isfinite(v) for v in point)
           for point in values):
        raise ValueError("points must be finite XYZ or XYZ+normal")
    if count is not None:
        if count <= 0 or count > len(values):
            raise ValueError("invalid sample count")
        values = random.Random(seed).sample(values, count)
    transform = CloudTransform((0., 0., 0.), 1.)
    if normalize and values:
        mins = tuple(min(point[i] for point in values) for i in range(3))
        maxs = tuple(max(point[i] for point in values) for i in range(3))
        center = tuple((mins[i]+maxs[i])/2 for i in range(3))
        scale = max(maxs[i]-mins[i] for i in range(3))
        if scale <= 0:
            raise ValueError("point cloud has degenerate bounding box")
        values = [tuple((point[i]-center[i])/scale for i in range(3)) + point[3:]
                  for point in values]
        transform = CloudTransform(center, scale)
    values.sort(key=lambda point: (point[2], point[1], point[0], point[3:]))
    return tuple(values), transform
