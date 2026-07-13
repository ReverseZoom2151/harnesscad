"""The exact Text-to-CadQuery evaluation protocol (CD / F1 / volumetric IoU).

Reference implementation of paper 171 -- *Text-to-CadQuery* (repo
``Text-to-CadQuery``), inference ``step5_compute_metrics/compute_CD.ipynb`` and
``step4_gemini_eval``. The paper reports Chamfer Distance, F1 and volumetric IoU,
but only the repo pins down the *protocol* -- the normalisation, the exact
formulas, the thresholds, the candidate gate and the reported statistics. Those
choices change the numbers by orders of magnitude, so they are reproduced here
verbatim:

  * **Normalisation for point metrics** (``sample_mesh_normalized``): translate by
    ``-centroid``, then scale by ``1 / max(bounding-box extents)``. Note this is a
    *bounding-box* scale about the centroid -- not the unit-sphere normalisation of
    :mod:`bench.cad_geometry_protocol` -- and it is applied to prediction and
    ground truth *independently*, making the metrics scale-invariant.

  * **Chamfer Distance**: ``mean(d(gt->pred)^2) + mean(d(pred->gt)^2)`` -- the sum
    of the two *mean squared* nearest-neighbour distances (not halved, not
    square-rooted) -- and it is reported **multiplied by 1000**.

  * **F1**: precision = fraction of predicted points within a **raw** (unsquared)
    distance ``threshold = 0.02`` of the ground truth, recall = the symmetric
    fraction, ``F1 = 2PR/(P+R)``, defined as ``0.0`` when ``P + R == 0``.

  * **Volumetric IoU**: each shape is normalised *differently* here -- translate by
    ``-min_bound`` and scale by ``1 / max(extent)`` so it fits ``[0, 1]^3`` -- then
    voxelised at pitch ``0.02`` and compared as occupancy grids; empty-union scores
    ``1.0``. (Grid padding to a common shape in the notebook is equivalent to the
    index-set intersection/union used here.)

  * **Candidate gate** (``step4``): the geometric metrics are computed **only over
    samples whose render the Gemini judge marked ``Match: Yes``**;
    :func:`parse_match_results` parses that judge log (``"<uid>: Match: Yes"``), so
    the population the metrics are averaged over is reproducible even though the
    judge itself (a VLM) is external.

  * **Reported statistics**: both **mean and median** of every metric.

Pure stdlib (``math`` / ``statistics``), deterministic, brute-force nearest
neighbour instead of a KD-tree (same result, no SciPy) and occupancy index sets
instead of a dense voxel array (same result, no NumPy). Mesh sampling and
rendering are external and out of scope: this module consumes point sets that a
caller sampled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

Point = tuple[float, float, float]

DEFAULT_F1_THRESHOLD = 0.02
DEFAULT_VOXEL_SIZE = 0.02
CD_SCALE = 1000.0
MATCH_YES_SUFFIX = "Match: Yes"


def _extents(points: list[Point]) -> tuple[float, float, float]:
    return tuple(
        max(p[i] for p in points) - min(p[i] for p in points) for i in range(3)
    )


def normalize_points(points) -> list[Point]:
    """Centroid-centre, then scale by 1 / max bounding-box extent."""
    pts = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
    if not pts:
        return []
    n = len(pts)
    centroid = tuple(sum(p[i] for p in pts) / n for i in range(3))
    shifted = [tuple(p[i] - centroid[i] for i in range(3)) for p in pts]
    scale = max(_extents(shifted))
    if scale <= 0.0:
        return [tuple(p) for p in shifted]
    return [tuple(v / scale for v in p) for p in shifted]


def normalize_unit_cube(points) -> list[Point]:
    """Translate by -min bound and scale by 1 / max extent, so points fit [0,1]^3."""
    pts = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
    if not pts:
        return []
    low = tuple(min(p[i] for p in pts) for i in range(3))
    shifted = [tuple(p[i] - low[i] for i in range(3)) for p in pts]
    scale = max(_extents(shifted))
    if scale <= 0.0:
        return [tuple(p) for p in shifted]
    return [tuple(v / scale for v in p) for p in shifted]


def _nearest_distance(point: Point, cloud: list[Point]) -> float:
    return min(math.dist(point, q) for q in cloud)


def chamfer_distance(points_a, points_b) -> float:
    """mean(d(b->a)^2) + mean(d(a->b)^2), the repo's (unhalved) squared Chamfer."""
    a = [tuple(map(float, p)) for p in points_a]
    b = [tuple(map(float, p)) for p in points_b]
    if not a or not b:
        raise ValueError("chamfer_distance requires two non-empty point sets")
    d_b = sum(_nearest_distance(p, a) ** 2 for p in b) / len(b)
    d_a = sum(_nearest_distance(p, b) ** 2 for p in a) / len(a)
    return d_b + d_a


def f1_score(
    points_pred, points_gt, threshold: float = DEFAULT_F1_THRESHOLD
) -> float:
    """F1 of the two point sets under a raw nearest-neighbour distance threshold."""
    pred = [tuple(map(float, p)) for p in points_pred]
    gt = [tuple(map(float, p)) for p in points_gt]
    if not pred or not gt:
        raise ValueError("f1_score requires two non-empty point sets")
    precision = sum(
        1 for p in pred if _nearest_distance(p, gt) < threshold
    ) / len(pred)
    recall = sum(1 for p in gt if _nearest_distance(p, pred) < threshold) / len(gt)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def voxelize(points, voxel_size: float = DEFAULT_VOXEL_SIZE) -> frozenset:
    """Occupancy index set of already-normalised points at the given pitch."""
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    return frozenset(
        tuple(int(math.floor(float(p[i]) / voxel_size)) for i in range(3))
        for p in points
    )


def volumetric_iou(
    points_a, points_b, voxel_size: float = DEFAULT_VOXEL_SIZE
) -> float:
    """IoU of the two shapes' occupancy grids after unit-cube normalisation."""
    vox_a = voxelize(normalize_unit_cube(points_a), voxel_size)
    vox_b = voxelize(normalize_unit_cube(points_b), voxel_size)
    union = len(vox_a | vox_b)
    if union == 0:
        return 1.0
    return len(vox_a & vox_b) / union


def parse_match_results(lines) -> tuple:
    """UIDs the judge marked ``Match: Yes`` in a step-4 evaluation log."""
    kept = []
    for raw in lines:
        line = raw.strip()
        if line.endswith(MATCH_YES_SUFFIX):
            kept.append(line.split(":")[0].strip())
    return tuple(kept)


@dataclass(frozen=True)
class SampleMetrics:
    """The three geometric metrics for one prediction / ground-truth pair."""

    uid: str
    chamfer: float
    f1: float
    iou: float


def evaluate_sample(
    uid: str,
    points_pred,
    points_gt,
    *,
    threshold: float = DEFAULT_F1_THRESHOLD,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
) -> SampleMetrics:
    """Full per-sample protocol: normalise, then CD / F1 / IoU."""
    pred_n = normalize_points(points_pred)
    gt_n = normalize_points(points_gt)
    return SampleMetrics(
        uid=uid,
        chamfer=chamfer_distance(gt_n, pred_n),
        f1=f1_score(pred_n, gt_n, threshold),
        iou=volumetric_iou(points_pred, points_gt, voxel_size),
    )


def evaluate_corpus(
    samples,
    *,
    candidates=None,
    threshold: float = DEFAULT_F1_THRESHOLD,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
) -> list[SampleMetrics]:
    """Evaluate ``(uid, pred_points, gt_points)`` triples, gated by ``candidates``.

    ``candidates`` is the judge-approved UID set from :func:`parse_match_results`;
    ``None`` evaluates every sample.
    """
    allowed = None if candidates is None else set(candidates)
    out = []
    for uid, pred, gt in samples:
        if allowed is not None and uid not in allowed:
            continue
        out.append(
            evaluate_sample(
                uid, pred, gt, threshold=threshold, voxel_size=voxel_size
            )
        )
    return out


def aggregate(metrics) -> dict:
    """Mean/median of each metric; Chamfer is reported scaled by 1000."""
    items = list(metrics)
    if not items:
        return {
            "n": 0,
            "cd_mean": None,
            "cd_median": None,
            "f1_mean": None,
            "f1_median": None,
            "iou_mean": None,
            "iou_median": None,
        }
    cds = [m.chamfer * CD_SCALE for m in items]
    f1s = [m.f1 for m in items]
    ious = [m.iou for m in items]
    return {
        "n": len(items),
        "cd_mean": sum(cds) / len(cds),
        "cd_median": median(cds),
        "f1_mean": sum(f1s) / len(f1s),
        "f1_median": median(f1s),
        "iou_mean": sum(ious) / len(ious),
        "iou_median": median(ious),
    }
