"""Seeded point-cloud corruption manifests and provider robustness curves."""

from __future__ import annotations

import random
import math


def corrupt_cloud(points, *, seed=0, noise=0., dropout=0., outliers=0):
    if noise < 0 or not 0 <= dropout < 1 or outliers < 0:
        raise ValueError("invalid corruption")
    rng = random.Random(seed)
    kept = [tuple(value + rng.gauss(0, noise) for value in point[:3]) + tuple(point[3:])
            for point in points if rng.random() >= dropout]
    if not kept and points:
        kept.append(tuple(points[0]))
    for _ in range(outliers):
        kept.append((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))
    return tuple(kept), {"seed": seed, "noise": noise, "dropout": dropout,
                         "outliers": outliers}


def robustness_curve(cases, provider):
    rows = []
    for case in cases:
        output = provider(case["cloud"])
        rows.append({**case["manifest"], **output})
    return tuple(rows)
