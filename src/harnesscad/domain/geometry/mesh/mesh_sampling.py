"""Area-weighted deterministic triangle-mesh surface sampling."""

from __future__ import annotations

import bisect
import math
import random


def triangle_area(triangle):
    a, b, c = triangle
    u = tuple(b[i] - a[i] for i in range(3))
    v = tuple(c[i] - a[i] for i in range(3))
    cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2],
             u[0]*v[1]-u[1]*v[0])
    return .5 * math.sqrt(sum(value * value for value in cross))


def sample_mesh(triangles, count, seed=0, *, tolerance=1e-15):
    if count < 0:
        raise ValueError("count must be non-negative")
    canonical = sorted((tuple(tuple(map(float, p)) for p in triangle)
                        for triangle in triangles), key=repr)
    weighted = [(triangle_area(triangle), triangle) for triangle in canonical]
    weighted = [(area, triangle) for area, triangle in weighted if area > tolerance]
    if count and not weighted:
        raise ValueError("mesh has no non-degenerate triangles")
    cumulative, total = [], 0.0
    for area, _ in weighted:
        total += area
        cumulative.append(total)
    rng, points = random.Random(seed), []
    for _ in range(count):
        index = bisect.bisect_left(cumulative, rng.random() * total)
        triangle = weighted[min(index, len(weighted)-1)][1]
        r1, r2 = rng.random(), rng.random()
        root = math.sqrt(r1)
        weights = (1-root, root*(1-r2), root*r2)
        points.append(tuple(sum(weights[j]*triangle[j][axis] for j in range(3))
                            for axis in range(3)))
    return tuple(points)
