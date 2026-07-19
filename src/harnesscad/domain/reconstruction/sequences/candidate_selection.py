"""Single-step selection: choosing among candidate CAD modelling steps.

Implements the deterministic single-step selection strategies.  The single-step
reconstruction module produces multiple candidate CAD modelling steps
``(o_i^t, l_i^t)`` -- one per planar prompt -- and a selection module picks the
best.  The learned selection network is external, but three of the four compared
strategies are fully deterministic and geometric:

  * ``"geo"``  -- greedy selection by the lowest Chamfer distance between the
    candidate's executed point cloud and the target ``p_full``;
  * ``"heur"`` -- select the candidate extrusion cylinder with the largest
    bounding-box volume (main structure first);
  * ``"rand"`` -- seeded random selection (baseline).

The learned module is supervised with the geometric fitness score
``sc_gt = IoU(bbox(o_i^t), bbox(o_gt^t))`` when the Boolean flags match, else 0.
That fitness function *is* deterministic given a reference step, so we also
provide a ``"bbox_iou"`` strategy that ranks candidates by their bounding-box
IoU agreement with a reference bounding box -- a closed-form proxy for the
learned selector's supervision target.

Everything is stdlib-only.  Candidates are described by a lightweight
:class:`StepCandidate` carrying the executed point cloud, an axis-aligned
bounding box, and the Boolean operation flag ``l`` (1 union / 0 subtraction).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import dist


def axis_aligned_bbox(points):
    """Return ``(lo, hi)`` axis-aligned bounding box of a point cloud."""
    pts = [tuple(p) for p in points]
    if not pts:
        raise ValueError("cannot bound an empty point cloud")
    dims = len(pts[0])
    lo = tuple(min(p[k] for p in pts) for k in range(dims))
    hi = tuple(max(p[k] for p in pts) for k in range(dims))
    return lo, hi


def bbox_volume(box):
    lo, hi = box
    vol = 1.0
    for a, b in zip(lo, hi):
        vol *= max(0.0, b - a)
    return vol


def bbox_iou(box_a, box_b):
    """3D (or n-D) axis-aligned bounding-box intersection-over-union.

    This is the ``IoU(bbox(.), bbox(.))`` quantity used as the fitness target.
    Returns 0.0 for disjoint boxes.
    """
    lo_a, hi_a = box_a
    lo_b, hi_b = box_b
    inter = 1.0
    for la, ha, lb, hb in zip(lo_a, hi_a, lo_b, hi_b):
        overlap = min(ha, hb) - max(la, lb)
        if overlap <= 0:
            return 0.0
        inter *= overlap
    union = bbox_volume(box_a) + bbox_volume(box_b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _symmetric_chamfer(a, b):
    if not a or not b:
        return None
    directed = lambda p, q: sum(min(dist(i, j) for j in q) for i in p) / len(p)
    return (directed(a, b) + directed(b, a)) / 2.0


@dataclass(frozen=True)
class StepCandidate:
    """One candidate CAD modelling step ``(o_i^t, l_i^t)``.

    ``cloud`` is the point cloud of the candidate's executed shape, ``bool_op``
    is ``l`` (1 = union, 0 = subtraction), and ``prompt_index`` records which
    planar prompt produced it (for deterministic tie-breaking).
    """

    prompt_index: int
    cloud: tuple
    bool_op: int

    @property
    def bbox(self):
        return axis_aligned_bbox(self.cloud)

    @property
    def volume(self):
        return bbox_volume(self.bbox)


def fitness_score(candidate_box, reference_box, candidate_bool, reference_bool):
    """Ground-truth fitness ``sc_gt``.

    Bounding-box IoU between the candidate and reference executed shapes, gated
    to 0 when the Boolean operations disagree.
    """
    if candidate_bool != reference_bool:
        return 0.0
    return bbox_iou(candidate_box, reference_box)


@dataclass(frozen=True)
class SelectionResult:
    strategy: str
    winner: StepCandidate
    winner_index: int
    scores: tuple  # (candidate_index, score) pairs in candidate order


def _rank(candidates, key, *, reverse):
    """Deterministic argmax/argmin with prompt-index tie-breaking."""
    scored = [(i, key(c)) for i, c in enumerate(candidates)]
    # None scores sink to the bottom regardless of direction.
    def sort_key(item):
        i, s = item
        if s is None:
            return (1, 0.0, candidates[i].prompt_index)
        primary = -s if reverse else s
        return (0, primary, candidates[i].prompt_index)
    scored_sorted = sorted(scored, key=sort_key)
    return scored_sorted[0][0], tuple((i, s) for i, s in scored)


def select_candidate(candidates, *, strategy="bbox_iou", target_cloud=None,
                     reference_box=None, reference_bool=None, seed=0):
    """Select one candidate step per the named strategy.

    * ``"geo"``     needs ``target_cloud`` (``p_full``); minimises Chamfer
      distance to it.
    * ``"heur"``    picks the largest bounding-box volume.
    * ``"rand"``    seeded random choice.
    * ``"bbox_iou"`` needs ``reference_box`` (and optionally ``reference_bool``);
      maximises the bounding-box fitness against the reference.

    Returns a :class:`SelectionResult`.
    """
    cands = list(candidates)
    if not cands:
        raise ValueError("no candidates to select from")

    if strategy == "geo":
        if target_cloud is None:
            raise ValueError("strategy 'geo' requires target_cloud")
        tgt = [tuple(p) for p in target_cloud]
        idx, scores = _rank(cands,
                             lambda c: _symmetric_chamfer(list(c.cloud), tgt),
                             reverse=False)
    elif strategy == "heur":
        idx, scores = _rank(cands, lambda c: c.volume, reverse=True)
    elif strategy == "rand":
        rng = random.Random(seed)
        idx = rng.randrange(len(cands))
        scores = tuple((i, 1.0 if i == idx else 0.0) for i in range(len(cands)))
    elif strategy == "bbox_iou":
        if reference_box is None:
            raise ValueError("strategy 'bbox_iou' requires reference_box")
        idx, scores = _rank(
            cands,
            lambda c: fitness_score(c.bbox, reference_box, c.bool_op,
                                    c.bool_op if reference_bool is None
                                    else reference_bool),
            reverse=True)
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    return SelectionResult(strategy, cands[idx], idx, scores)
