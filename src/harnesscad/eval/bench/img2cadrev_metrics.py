"""Reconstruction / structural metrics for Img2CAD conditional factorization.

Deterministic implementations of the evaluation quantities reported in the Img2CAD
paper (You et al.), together with two metrics specific to the conditional-
factorization decomposition:

Geometric / structural (paper Sec. 6.2, 6.5):

* ``chamfer_distance``      -- bidirectional mean nearest-neighbour distance between
                               two point sets (the paper's CD, symmetrised).
* ``symmetry_chamfer``      -- Chamfer distance between a point set and its mirror
                               reflection over a chosen axis plane through the
                               origin (paper's symmetry Chamfer, X-axis by default).
* ``num_scc``               -- number of strongly connected components: connect two
                               points when their Euclidean distance is below a
                               threshold (0.05 in the paper) and count components.

Factorization-specific:

* ``structure_accuracy``    -- how well a predicted Stage-1 discrete structure
                               matches ground truth (label + per-command type).
* ``attribute_error``       -- mean absolute error between predicted and ground
                               truth Stage-2 attribute vectors.
* ``factorization_fidelity``-- whether factorize/assemble round-trips a model
                               losslessly (the invariant the pipeline relies on).

Point sets are sequences of equal-length numeric tuples. Everything is pure Python,
stdlib-only and deterministic (no randomness, no wall-clock).
"""

from __future__ import annotations

import math

from harnesscad.domain.reconstruction.img2cadrev_factorization import (
    factorize,
    round_trip,
    structure_command_count,
)


def _dist2(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _directed_mean_nn(src, dst) -> float:
    total = 0.0
    for p in src:
        best = min(_dist2(p, q) for q in dst)
        total += math.sqrt(best)
    return total / len(src)


def chamfer_distance(points_a, points_b) -> float:
    """Symmetric mean-nearest-neighbour Chamfer distance between two point sets."""
    if not points_a or not points_b:
        raise ValueError("chamfer_distance requires non-empty point sets")
    return _directed_mean_nn(points_a, points_b) + _directed_mean_nn(points_b, points_a)


def mirror_points(points, axis: str = "x"):
    """Reflect points across the coordinate plane normal to ``axis``.

    ``axis='x'`` negates the x-component (mirror over the plane x=0), matching the
    paper's X-axis symmetry Chamfer. Works for 2D or 3D points.
    """
    comp = {"x": 0, "y": 1, "z": 2}[axis]
    out = []
    for p in points:
        q = list(p)
        q[comp] = -q[comp]
        out.append(tuple(q))
    return out


def symmetry_chamfer(points, axis: str = "x") -> float:
    """Bidirectional Chamfer between a point set and its mirror image.

    Lower is more symmetric; 0 for a perfectly symmetric set about ``axis``.
    """
    return chamfer_distance(points, mirror_points(points, axis))


def num_scc(points, threshold: float = 0.05) -> int:
    """Count connected components; edge iff Euclidean distance < ``threshold``.

    The paper's structural-quality metric (fewer components == fewer floating
    fragments). Union-find over all point pairs.
    """
    n = len(points)
    if n == 0:
        return 0
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    t2 = threshold * threshold
    for i in range(n):
        for j in range(i + 1, n):
            if _dist2(points[i], points[j]) < t2:
                union(i, j)
    return len({find(i) for i in range(n)})


def structure_accuracy(pred_structure, gt_structure) -> float:
    """Per-command accuracy of a predicted discrete structure vs ground truth.

    Parts are aligned by index; for each aligned command a hit requires matching
    BOTH the part label and the command type. Commands in excess of either side
    (extra or missing parts / commands) count as misses. Range [0, 1].
    """
    gt_total = structure_command_count(gt_structure)
    pred_total = structure_command_count(pred_structure)
    denom = max(gt_total, pred_total)
    if denom == 0:
        return 1.0
    hits = 0
    for pi in range(min(len(pred_structure), len(gt_structure))):
        pp, gp = pred_structure[pi], gt_structure[pi]
        label_ok = pp["label"] == gp["label"]
        pcmds, gcmds = pp["command_types"], gp["command_types"]
        for ci in range(min(len(pcmds), len(gcmds))):
            if label_ok and pcmds[ci] == gcmds[ci]:
                hits += 1
    return hits / denom


def attribute_error(pred_attributes, gt_attributes) -> float:
    """Mean absolute error between aligned attribute vectors.

    Requires equal command counts and equal per-command arity (i.e. predictions
    produced against the same discrete structure). Averaged over every scalar.
    """
    if len(pred_attributes) != len(gt_attributes):
        raise ValueError("attribute lists differ in length")
    total = 0.0
    count = 0
    for pv, gv in zip(pred_attributes, gt_attributes):
        if len(pv) != len(gv):
            raise ValueError("attribute vectors differ in arity")
        for p, g in zip(pv, gv):
            total += abs(p - g)
            count += 1
    if count == 0:
        return 0.0
    return total / count


def factorization_fidelity(model) -> float:
    """1.0 iff factorize->assemble reproduces the (normalized) model exactly."""
    from harnesscad.domain.reconstruction.img2cadrev_factorization import normalize_model

    original = normalize_model(model)
    rebuilt = round_trip(model)
    return 1.0 if rebuilt == original else 0.0


def factorization_report(pred_structure, pred_attributes,
                         gt_structure, gt_attributes) -> dict:
    """Combined Stage-1 / Stage-2 factorization quality summary."""
    return {
        "structure_accuracy": structure_accuracy(pred_structure, gt_structure),
        "attribute_error": (
            attribute_error(pred_attributes, gt_attributes)
            if len(pred_attributes) == len(gt_attributes) else float("nan")
        ),
        "command_count_match": (
            structure_command_count(pred_structure)
            == structure_command_count(gt_structure)
        ),
    }
