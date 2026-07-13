"""STEP-LLM evaluation metrics and the Scaled Chamfer Distance reward.

The paper (Shi et al., DATE 2026, Sec. 4.1 / 3.3) evaluates generated STEP files
with a small set of metrics and trains with a geometric RL reward. The learned
model, OpenCASCADE-based renderability, and the FPFH+RANSAC/ICP registration are
external / non-deterministic and are *not* reimplemented here. The deterministic
pieces are:

  * **Completion Rate (CR)** - whether a file terminates with the standardized
    ``END-ISO-10303-21;`` line (Sec. 4.1).
  * **Entity count / Average Entity Count (AEC)** - a proxy for design
    complexity; a generated file's count should match the ground-truth
    distribution (Tables 1-2).
  * **Chamfer Distance (CD)** - Eq. (1), the bidirectional mean squared nearest
    neighbour distance between two point clouds.
  * **Scaled Chamfer Distance (SCD)** - Eq. (2), CD after centroid alignment and
    normalization by the ground-truth RMS scale factor. (The optional global
    registration and ICP refinement stages are noted but omitted as external.)
  * **Geometric reward** - Eq. (3), the CAD-Coder-style piecewise-linear reward
    with a lower/upper SCD threshold.

Point clouds are plain lists of ``(x, y, z)`` tuples; everything is stdlib-only
and deterministic.
"""

from __future__ import annotations

from math import sqrt

from harnesscad.io.formats.stepllm_parser import StepFile


# --- Completion Rate ---------------------------------------------------------

_TERMINATOR = "END-ISO-10303-21;"


def completes(text: str) -> bool:
    """True iff the file terminates correctly with ``END-ISO-10303-21;``."""

    stripped = text.rstrip()
    return stripped.endswith(_TERMINATOR)


def completion_rate(texts) -> float:
    """Fraction of raw STEP texts that terminate correctly (Sec. 4.1)."""

    texts = list(texts)
    if not texts:
        return 0.0
    return sum(1 for t in texts if completes(t)) / len(texts)


# --- Entity counts -----------------------------------------------------------

def entity_count(step: StepFile) -> int:
    return len(step.entities)


def average_entity_count(steps) -> float:
    steps = list(steps)
    if not steps:
        return 0.0
    return sum(entity_count(s) for s in steps) / len(steps)


def aec_gap(generated, ground_truth) -> float:
    """Absolute difference between the two average entity counts (AEC).

    A smaller gap indicates the generated files better match the expected
    difficulty distribution of the ground-truth set (Table 2).
    """

    return abs(average_entity_count(generated) - average_entity_count(ground_truth))


# --- point-cloud geometry ----------------------------------------------------

def centroid(points):
    n = len(points)
    if n == 0:
        raise ValueError("empty point cloud")
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sz = sum(p[2] for p in points)
    return (sx / n, sy / n, sz / n)


def center_align(points):
    """Shift a cloud so its centroid coincides with the origin (Sec. 3.3)."""

    cx, cy, cz = centroid(points)
    return [(p[0] - cx, p[1] - cy, p[2] - cz) for p in points]


def rms_scale(points):
    """Root-mean-square distance of points from their centroid (scale factor)."""

    cx, cy, cz = centroid(points)
    n = len(points)
    total = 0.0
    for x, y, z in points:
        total += (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return sqrt(total / n)


def _sq_dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def _nearest_sq(p, cloud):
    best = None
    for q in cloud:
        d = _sq_dist(p, q)
        if best is None or d < best:
            best = d
    return best


def chamfer_distance(pred, gt) -> float:
    """Bidirectional Chamfer Distance (Eq. 1) between two point clouds."""

    if not pred or not gt:
        raise ValueError("both point clouds must be non-empty")
    forward = sum(_nearest_sq(p, gt) for p in pred) / len(pred)
    backward = sum(_nearest_sq(q, pred) for q in gt) / len(gt)
    return forward + backward


def scaled_chamfer_distance(pred, gt) -> float:
    """SCD (Eq. 2): centroid-aligned CD normalized by the GT squared scale.

    The global-registration and ICP refinement stages of the paper's alignment
    pipeline are external (kernel/optimization dependent) and are intentionally
    omitted; centroid alignment and scale normalization are applied.
    """

    scale = rms_scale(gt)
    if scale == 0:
        raise ValueError("ground-truth scale factor is zero")
    pa = center_align(pred)
    ga = center_align(gt)
    return chamfer_distance(pa, ga) / (scale ** 2)


def median_scaled_chamfer_distance(pairs) -> float:
    """Median SCD across ``(pred, gt)`` pairs (the paper's MSCD metric)."""

    scores = sorted(scaled_chamfer_distance(p, g) for p, g in pairs)
    n = len(scores)
    if n == 0:
        raise ValueError("no pairs")
    mid = n // 2
    if n % 2 == 1:
        return scores[mid]
    return (scores[mid - 1] + scores[mid]) / 2


# --- geometric reward (Eq. 3) ------------------------------------------------

def geometric_reward(scd: float, delta_low: float = 0.01,
                     delta_high: float = 0.5) -> float:
    """Piecewise-linear reward on SCD (Eq. 3).

    ``1`` below ``delta_low``, ``0`` above ``delta_high``, linearly interpolated
    in between so the reward is dense. Defaults are the paper's RL thresholds
    (Sec. 4.4): lower bound ``0.01``, upper bound ``0.5``.
    """

    if delta_high <= delta_low:
        raise ValueError("delta_high must exceed delta_low")
    if scd <= delta_low:
        return 1.0
    if scd >= delta_high:
        return 0.0
    return (delta_high - scd) / (delta_high - delta_low)


def geometric_reward_for(pred, gt, delta_low: float = 0.01,
                         delta_high: float = 0.5) -> float:
    """Convenience: SCD of the pair, then the piecewise-linear reward."""

    return geometric_reward(scaled_chamfer_distance(pred, gt),
                            delta_low, delta_high)
