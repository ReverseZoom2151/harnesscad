"""Normal Consistency (NC) between two oriented point clouds.

NC is the third term of the RLCAD composite reward (arXiv:2503.18549, "RL
Training Gym for Revolution Involved CAD Command Sequence Generation"), where
the reward is ``R = a*IoU + b*R_MMD + c*NC`` with ``a=0.2, b=0.5, c=0.3``.
RLCAD's ablation is the reason this metric exists at all: IoU alone scored
0.7045 quality, IoU+MMD 0.7436, and IoU+MMD+NC 0.7932.  IoU is a coarse
volumetric overlap -- an agent maximising it alone learns to fill the right
bounding volume and neglects surface detail.  NC is the *orientation* channel:
it is sensitive to a face being tilted, a fillet being missing or a revolve
being faceted even when the occupied volume barely moves.

Definition
----------
Both clouds are *oriented*: each sample is a ``(point, normal)`` pair.  For
every reference sample the metric finds the corresponding candidate sample and
takes the cosine similarity of their normals; NC is the mean over the reference
cloud::

    NC = (1 / |R|) * sum_{i in R}  cos( n_i , n_{phi(i)} )

Two correspondence rules are offered because the paper's phrase ("max cosine
similarity between corresponding surface normals") admits both, and they are
NOT the same number:

* ``correspondence="nearest"`` (default) -- ``phi(i)`` is the candidate sample
  whose *position* is closest to reference point ``i``.  This is the standard
  Chamfer-style pairing and is what "corresponding surface normals" means in
  the surface-reconstruction literature (Occupancy Networks, ConvONet).
  Positional ties -- ubiquitous on a CAD solid, where every edge and corner
  carries one sample per incident face -- are broken by taking the best-matching
  normal among the tied candidates.  Without that rule a cloud scored against
  ITSELF returns well under 1.0 (a box of face samples scores about 0.65),
  which would make the metric useless as a reward.
* ``correspondence="max"`` -- ``phi(i)`` is the candidate sample maximising the
  cosine similarity itself, i.e. a literal reading of "max cosine similarity".
  This is an optimistic upper bound: it is 1.0 for any candidate that contains
  a single sample with the right normal anywhere in space, so it must never be
  used alone as a reward.  It is provided so the two readings are comparable
  rather than silently conflated.

Orientation
-----------
``unoriented=True`` (default) takes ``|cos|``, so NC lands in ``[0, 1]``.  This
is deliberate: a solid built by a CAD kernel and a target sampled from a mesh
need not agree on which side of a face is "outside" (winding order and shell
orientation are backend-dependent), and a globally flipped normal field would
otherwise score -1 for geometry that is exactly right.  ``unoriented=False``
keeps the signed cosine in ``[-1, 1]`` for callers that control both samplers
and genuinely want to penalise inside-out solids.

Empty clouds are an error rather than a silent 1.0: an empty candidate is an
*execution* failure and belongs to the exec gate, not to a geometry metric.

Pure stdlib, deterministic, O(|R| * |C|) brute force -- these clouds are
evaluation-sized (hundreds to a few thousand points), not render-sized.  This
module is plain geometry and lives in ``domain``; it must not import ``eval``.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

Point = Sequence[float]
Normal = Sequence[float]
#: An oriented sample: a position paired with its surface normal.
OrientedPoint = Tuple[Point, Normal]

#: Correspondence rules accepted by :func:`normal_consistency`.
CORRESPONDENCES = ("nearest", "max")

#: Squared-distance slack below which two candidates count as equally near.
TIE_TOL = 1e-12


def _as_vec(v, what: str) -> Tuple[float, ...]:
    out = tuple(float(c) for c in v)
    if len(out) < 2:
        raise ValueError("%s must have at least 2 components" % what)
    return out


def _as_oriented(cloud, what: str):
    samples = []
    for item in cloud:
        try:
            p, n = item
        except (TypeError, ValueError):
            raise ValueError("%s entries must be (point, normal) pairs" % what)
        samples.append((_as_vec(p, what + " point"), _as_vec(n, what + " normal")))
    if not samples:
        raise ValueError("%s must be non-empty" % what)
    dims = len(samples[0][0])
    for p, n in samples:
        if len(p) != dims or len(n) != dims:
            raise ValueError("%s points and normals must share one dimension" % what)
    return samples


def vector_norm(v: Sequence[float]) -> float:
    """Euclidean length of ``v``."""
    return math.sqrt(sum(float(c) * float(c) for c in v))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between two vectors, clamped to ``[-1, 1]``.

    Raises ``ValueError`` on a zero-length vector: a degenerate normal has no
    direction, and returning 0.0 would silently score it as "perpendicular"
    (a middling result) rather than as the sampling bug it is.
    """
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    na = vector_norm(a)
    nb = vector_norm(b)
    if na == 0.0 or nb == 0.0:
        raise ValueError("cannot take the cosine of a zero-length normal")
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    return max(-1.0, min(1.0, dot / (na * nb)))


def _sq_dist(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))


def nearest_index(point: Point, points: Sequence[Point]) -> int:
    """Index of the positionally nearest entry of ``points`` (ties -> lowest index)."""
    if not points:
        raise ValueError("points must be non-empty")
    best = 0
    best_d = _sq_dist(point, points[0])
    for i in range(1, len(points)):
        d = _sq_dist(point, points[i])
        if d < best_d:
            best_d = d
            best = i
    return best


def nearest_indices(point: Point, points: Sequence[Point],
                    *, tie_tol: float = TIE_TOL) -> list:
    """Every index tied for nearest to ``point``, within ``tie_tol``.

    Ties are the normal case on a CAD solid, not an edge case: an edge or a
    corner carries one sample position per incident face, each with a different
    normal.  Picking the lowest-index tie would compare a face's normal against
    an arbitrary neighbouring face's, so even a cloud scored against ITSELF
    would fall well below 1.0.  :func:`normal_consistency` therefore resolves a
    tie by normal agreement, which restores ``NC(x, x) == 1``.
    """
    if not points:
        raise ValueError("points must be non-empty")
    dists = [_sq_dist(point, q) for q in points]
    best_d = min(dists)
    return [i for i, d in enumerate(dists) if d <= best_d + tie_tol]


def normal_consistency(
    reference,
    candidate,
    *,
    unoriented: bool = True,
    correspondence: str = "nearest",
    tie_tol: float = TIE_TOL,
) -> float:
    """Mean cosine similarity of corresponding normals, reference -> candidate.

    ``reference`` and ``candidate`` are iterables of ``(point, normal)`` pairs.
    See the module docstring for the two ``correspondence`` rules and for why
    ``unoriented`` defaults to True (result in ``[0, 1]``; signed result in
    ``[-1, 1]`` when False).

    This is directed: it asks "is every part of the target reproduced with the
    right orientation", and is blind to extra candidate surface that the
    reference does not cover.  Use :func:`symmetric_normal_consistency` when
    spurious extra geometry must also be penalised.
    """
    if correspondence not in CORRESPONDENCES:
        raise ValueError("correspondence must be one of %r" % (CORRESPONDENCES,))
    ref = _as_oriented(reference, "reference")
    cand = _as_oriented(candidate, "candidate")
    if len(ref[0][0]) != len(cand[0][0]):
        raise ValueError("reference and candidate must share one dimension")

    cand_points = [p for p, _n in cand]
    total = 0.0
    for p, n in ref:
        if correspondence == "nearest":
            pool = (cand[i][1] for i in nearest_indices(
                p, cand_points, tie_tol=tie_tol))
        else:
            pool = (cn for _cp, cn in cand)
        best = None
        for cn in pool:
            cos = cosine_similarity(n, cn)
            value = abs(cos) if unoriented else cos
            if best is None or value > best:
                best = value
        total += best
    return total / len(ref)


def symmetric_normal_consistency(
    a,
    b,
    *,
    unoriented: bool = True,
    correspondence: str = "nearest",
) -> float:
    """Mean of the two directed :func:`normal_consistency` scores.

    Symmetrising costs one extra pass but closes the obvious exploit of the
    directed score: a candidate may not bolt on unreferenced surface for free.
    """
    fwd = normal_consistency(
        a, b, unoriented=unoriented, correspondence=correspondence)
    bwd = normal_consistency(
        b, a, unoriented=unoriented, correspondence=correspondence)
    return (fwd + bwd) / 2.0
