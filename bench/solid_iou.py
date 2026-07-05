"""Inertia-normalized, symmetry-enumerated solid IoU via an injected adapter."""

from __future__ import annotations

import itertools
import math


def proper_axis_alignments():
    matrices = []
    for perm in itertools.permutations(range(3)):
        parity = 1 if perm in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1
        for signs in itertools.product((-1, 1), repeat=3):
            if parity * math.prod(signs) != 1:
                continue
            matrix = tuple(tuple(signs[row] if perm[row] == col else 0
                                 for col in range(3)) for row in range(3))
            matrices.append(matrix)
    return tuple(matrices)


def inertia_scale(volume, inertia_trace):
    if volume <= 0 or inertia_trace <= 0:
        raise ValueError("solid volume and inertia trace must be positive")
    return math.sqrt(inertia_trace / (2 * volume))


def best_solid_iou(generated, target, adapter):
    """Adapter supplies properties(solid), normalize(solid,scale), align, iou."""
    gp, tp = adapter.properties(generated), adapter.properties(target)
    gs, ts = inertia_scale(gp["volume"], gp["inertia_trace"]), \
        inertia_scale(tp["volume"], tp["inertia_trace"])
    left = adapter.normalize(generated, gp["centroid"], gs)
    right = adapter.normalize(target, tp["centroid"], ts)
    scores = []
    for matrix in proper_axis_alignments():
        scores.append((float(adapter.iou(adapter.align(left, matrix), right)), matrix))
    score, transform = max(scores, key=lambda item: (item[0], item[1]))
    return {"iou": score, "alignment": transform, "candidates": tuple(scores),
            "degenerate": bool(gp.get("repeated_eigenvalues")
                               or tp.get("repeated_eigenvalues"))}
