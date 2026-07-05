"""Orientation-invariant alignment for cadrille CAD-reconstruction eval (2026).

When building weak VLM baselines (e.g. four orthogonal views handed to a general
LLM), cadrille applies iterative-closest-point (ICP) alignment before computing
metrics "so that correct predictions with wrong orientation are not penalised".

A full free-form ICP needs an SVD-based rigid fit, which is impractical in pure
stdlib. Since the ambiguity here is a discrete orientation flip of an
axis-normalised CAD model, we implement a deterministic discrete-ICP: search the
24 proper (det=+1) axis-aligned rotations, centre both clouds on their centroids,
and keep the rotation minimising the symmetric Chamfer distance. This removes the
orientation penalty without any learned or floating-point-fragile machinery.
"""

from __future__ import annotations

from itertools import permutations
from math import dist


def _det3(m):
    a, b, c = m
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def proper_axis_rotations():
    """The 24 signed axis-permutation matrices with determinant +1."""
    mats = []
    for perm in permutations(range(3)):
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    signs = (sx, sy, sz)
                    m = []
                    for row in range(3):
                        vec = [0, 0, 0]
                        vec[perm[row]] = signs[row]
                        m.append(tuple(vec))
                    if _det3(m) == 1:
                        mats.append(tuple(m))
    return mats


def centroid(points):
    pts = [tuple(float(c) for c in p) for p in points]
    if not pts:
        raise ValueError("points must be non-empty")
    dims = len(pts[0])
    return tuple(sum(p[d] for p in pts) / len(pts) for d in range(dims))


def _translate(points, offset):
    return [tuple(p[d] - offset[d] for d in range(len(p))) for p in points]


def apply_rotation(points, rotation):
    """Apply a 3x3 rotation matrix to every point."""
    out = []
    for p in points:
        out.append(tuple(
            sum(rotation[r][c] * p[c] for c in range(3)) for r in range(3)
        ))
    return out


def _symmetric_chamfer(a, b):
    directed = lambda p, q: sum(min(dist(i, j) for j in q) for i in p) / len(p)
    return (directed(a, b) + directed(b, a)) / 2.0


def align_orientation(source, target):
    """Best axis-aligned rotation + centroid translation of ``source`` to ``target``.

    Both clouds are centred on their centroids; the source is then rotated by
    each of the 24 proper axis rotations and the one minimising the symmetric
    Chamfer distance is kept (ties break by rotation-list order). Returns a dict
    with the chosen ``rotation``, the ``aligned`` source points (in the target's
    centred frame), and the resulting ``chamfer`` distance.
    """
    src = [tuple(float(c) for c in p) for p in source]
    tgt = [tuple(float(c) for c in p) for p in target]
    if not src or not tgt:
        raise ValueError("both point sets must be non-empty")
    src_c = _translate(src, centroid(src))
    tgt_c = _translate(tgt, centroid(tgt))
    best = None
    for rotation in proper_axis_rotations():
        rotated = apply_rotation(src_c, rotation)
        cd = _symmetric_chamfer(rotated, tgt_c)
        if best is None or cd < best["chamfer"]:
            best = {"rotation": rotation, "aligned": rotated, "chamfer": cd}
    return best


def aligned_chamfer(source, target) -> float:
    """Chamfer distance after orientation alignment (orientation-invariant CD)."""
    return align_orientation(source, target)["chamfer"]
