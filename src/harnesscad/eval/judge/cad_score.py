"""CAD Score: the CADGenBench composite metric, as pure deterministic math.

Mined from CADGenBench (docs/metrics.md and docs/metrics/*). CADGenBench scores
one generated part against a ground truth on four orthogonal axes and combines
them with a validity gate:

    cad_score = 0                                                if not valid
              = 0.4*shape + 0.4*interface + 0.2*topology         (generation)
              = 0.6*shape_renorm + 0.3*interface + 0.1*topology  (editing)

The upstream scoring engine needs pythonOCC/manifold3d to *derive* the axis
values from STEP files, but the aggregation and the topology-match scoring are
plain arithmetic on already-computed inputs. This module extracts exactly those
deterministic pieces so the harness can:

* aggregate component scores into a CAD Score with the correct gate + weights
  (:func:`cad_score`),
* score topology agreement from two Betti triples with the paper's fuzzy
  log-ratio (:func:`topology_match`),
* combine the two shape sub-metrics (:func:`shape_similarity`),
* renormalise an editing task's shape axis against its no-op baseline
  (:func:`renormalize_edit_shape`).

Everything here is stdlib-only and returns values in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

__all__ = [
    "GEN_WEIGHTS",
    "EDIT_WEIGHTS",
    "BETTI_SHARPNESS",
    "topology_match",
    "shape_similarity",
    "renormalize_edit_shape",
    "cad_score",
    "CadScoreBreakdown",
]

#: Generation-task axis weights (shape, interface, topology). Sum to 1.
GEN_WEIGHTS: Dict[str, float] = {"shape": 0.4, "interface": 0.4, "topology": 0.2}

#: Editing-task axis weights, shape-dominant so a no-op cannot win.
EDIT_WEIGHTS: Dict[str, float] = {"shape": 0.6, "interface": 0.3, "topology": 0.1}

#: Sharpness exponent alpha for the Betti log-ratio (paper uses 2).
BETTI_SHARPNESS: float = 2.0


def _betti_axis_score(cand: int, gt: int, alpha: float) -> float:
    """Fuzzy log-ratio score for one Betti axis, in ``[0, 1]``.

    ``s_i = ((min+1)/(max+1)) ** alpha`` -- 1 when the counts match, decaying
    smoothly otherwise; the ``+1`` keeps it finite at zero counts.
    """
    if cand < 0 or gt < 0:
        raise ValueError("Betti numbers must be non-negative")
    lo, hi = (cand, gt) if cand <= gt else (gt, cand)
    return ((lo + 1) / (hi + 1)) ** alpha


def topology_match(
    candidate: Sequence[int],
    ground_truth: Sequence[int],
    *,
    alpha: float = BETTI_SHARPNESS,
) -> float:
    """Topology-match score from two Betti triples ``(b0, b1, b2)``.

    Each axis gets the fuzzy log-ratio :func:`_betti_axis_score`; the aggregate
    is their **product** (not mean), so one wrong count collapses the score --
    topology is discrete, two-of-three right is not a partial match.

    Reproduces the worked examples exactly: GT ``(1,0,0)`` vs ``(2,0,0)`` gives
    ``(2/3)**2 = 0.444``; GT ``(1,2,0)`` vs ``(1,4,0)`` gives ``(3/5)**2 = 0.36``.
    """
    if len(candidate) != 3 or len(ground_truth) != 3:
        raise ValueError("Betti triples must have exactly 3 entries (b0, b1, b2)")
    product = 1.0
    for c, g in zip(candidate, ground_truth):
        product *= _betti_axis_score(int(c), int(g), alpha)
    return product


def shape_similarity(surface_f1: float, volume_iou: float) -> float:
    """Shape similarity = mean of surface-distance F1 and volume IoU.

    The two are complementary (placement vs occupied volume); a candidate must
    satisfy both to score well.
    """
    for name, v in (("surface_f1", surface_f1), ("volume_iou", volume_iou)):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1]; got {v}")
    return 0.5 * (surface_f1 + volume_iou)


def renormalize_edit_shape(shape: float, noop_baseline: float) -> float:
    """Renormalise an editing task's raw shape score against its no-op baseline.

    ``s_renorm = max(0, (shape - b_shape) / (1 - b_shape))`` -- maps the no-op
    (submitting the input unchanged) to 0 and a perfect edit to 1, so an editor
    cannot bank credit for the global similarity that doing nothing already has.
    """
    if not (0.0 <= noop_baseline < 1.0):
        raise ValueError("noop_baseline must be in [0, 1)")
    return max(0.0, (shape - noop_baseline) / (1.0 - noop_baseline))


@dataclass(frozen=True)
class CadScoreBreakdown:
    """A CAD Score with its gated component contributions, for auditability."""

    cad_score: float
    is_valid: bool
    weights: Dict[str, float]
    components: Dict[str, float]


def cad_score(
    *,
    is_valid: bool,
    shape: float,
    interface: float,
    topology: float,
    editing: bool = False,
) -> CadScoreBreakdown:
    """Compose the CADGenBench CAD Score from its axis scores.

    Validity is a hard gate: an invalid part scores 0 regardless of its axes,
    so an invalid solid can never beat a worse-but-valid one. Otherwise the
    score is the weighted mean of the three ``[0, 1]`` axes, using
    :data:`GEN_WEIGHTS` for generation and :data:`EDIT_WEIGHTS` for editing
    (pass ``shape`` already renormalised via :func:`renormalize_edit_shape`).
    """
    for name, v in (("shape", shape), ("interface", interface), ("topology", topology)):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1]; got {v}")
    weights = EDIT_WEIGHTS if editing else GEN_WEIGHTS
    components = {"shape": shape, "interface": interface, "topology": topology}
    if not is_valid:
        return CadScoreBreakdown(0.0, False, weights, components)
    total = (
        weights["shape"] * shape
        + weights["interface"] * interface
        + weights["topology"] * topology
    )
    return CadScoreBreakdown(total, True, weights, components)
