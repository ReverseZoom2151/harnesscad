"""Evaluation metrics for B-Rep chain complexes (ComplexGen, SIGGRAPH 2022).

ComplexGen evaluates a reconstructed complex against ground truth by (a) matching
predicted primitives to ground-truth ones with an optimal (Hungarian) assignment
under a Chamfer-style geometric cost, and (b) reporting per-order precision /
recall / F1 over corners, curves and patches, plus the Chamfer distance of the
matched pairs (``matcher_corner.py`` / ``matcher_curve.py`` / ``matcher_patch.py``
and the ``eval_matched`` / ``eval_res_cov`` paths of the backbone).

Everything here is deterministic and stdlib-only:

  * :func:`chamfer_distance`        -- brute-force one-sided / bidirectional Chamfer.
  * :func:`curve_distance`          -- ordered point-to-point mean distance between
    two equally-sampled open curves, minimised over the two orientations.
  * :func:`closed_curve_distance`   -- the same, minimised over every cyclic shift
    and both orientations (``closed_curve_distance`` of the reference).
  * :func:`corner_cost_matrix` / :func:`curve_cost_matrix` / :func:`patch_cost_matrix`
    -- the pairwise costs the Hungarian matcher consumes.
  * :func:`match` / :func:`structure_prf` -- optimal assignment thresholded at a
    distance tolerance, giving precision / recall / F1 per primitive order.
  * :func:`topology_prf`            -- precision / recall / F1 of the incidence
    entries themselves, after the primitives have been matched.
  * :func:`complex_chamfer`         -- Chamfer distance between the two complexes'
    full sample clouds.
  * :func:`evaluate_complex`        -- the whole report in one call.

The Hungarian solver is reused from :mod:`bench.davinci_inference_metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass

from harnesscad.eval.bench.sketch.set_prediction_f1 import hungarian
from harnesscad.domain.reconstruction.brep.chain_complex import ChainComplex
# The geometric distances themselves are plain geometry, not scoring, and the
# reconstruction pipeline needs them too (domain/reconstruction/brep/
# chain_complex_nms).  They therefore live in the geometry layer and are
# re-exported here, so `complex_matching.chamfer_distance` and friends keep
# working unchanged.
from harnesscad.domain.geometry.pointcloud.distance_metrics import (  # noqa: F401
    BIG,
    _dist,
    chamfer_distance,
    closed_curve_distance,
    curve_distance,
    sampled_curve_distance,
)

Point = tuple[float, float, float]


# --------------------------------------------------------------------------- #
# cost matrices
# --------------------------------------------------------------------------- #
def corner_cost_matrix(pred: list, gt: list) -> list[list[float]]:
    return [[_dist(p, g) for g in gt] for p in pred]


def curve_cost_matrix(pred: list, gt: list) -> list[list[float]]:
    return [[sampled_curve_distance(p, g) for g in gt] for p in pred]


def patch_cost_matrix(pred: list, gt: list) -> list[list[float]]:
    return [[chamfer_distance(p.points, g.points) for g in gt] for p in pred]


# --------------------------------------------------------------------------- #
# matching + structure metrics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    matched: int
    n_pred: int
    n_gt: int


def match(cost: list[list[float]], threshold: float) -> list[tuple[int, int, float]]:
    """Optimal one-to-one assignment kept only where cost <= ``threshold``.

    Returns ``[(pred_index, gt_index, cost), ...]`` sorted by ``pred_index``.  The
    cost matrix is padded so the Hungarian solver always sees ``rows <= cols``.
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if m == 0:
        return []
    if n > m:
        transposed = [[cost[i][j] for i in range(n)] for j in range(m)]
        flipped = match(transposed, threshold)
        return sorted((p, g, c) for (g, p, c) in flipped)
    assignment = hungarian(cost)
    out = []
    for i, j in enumerate(assignment):
        if j < 0 or j >= m:
            continue
        if cost[i][j] <= threshold:
            out.append((i, j, cost[i][j]))
    return out


def prf_from_matches(n_matched: int, n_pred: int, n_gt: int) -> PRF:
    precision = n_matched / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
    recall = n_matched / n_gt if n_gt else (1.0 if n_pred == 0 else 0.0)
    denom = precision + recall
    f1 = 2.0 * precision * recall / denom if denom > 0.0 else 0.0
    return PRF(precision, recall, f1, n_matched, n_pred, n_gt)


def structure_prf(cost: list[list[float]], n_pred: int, n_gt: int,
                  threshold: float) -> tuple[PRF, list[tuple[int, int, float]]]:
    matches = match(cost, threshold) if (n_pred and n_gt) else []
    return prf_from_matches(len(matches), n_pred, n_gt), matches


def matched_chamfer(matches: list[tuple[int, int, float]]) -> float:
    """Mean cost of the accepted matches (0.0 when nothing matched)."""
    if not matches:
        return 0.0
    return sum(c for (_, _, c) in matches) / len(matches)


# --------------------------------------------------------------------------- #
# topology metrics
# --------------------------------------------------------------------------- #
def _incidence_pairs(matrix, rows_map: dict, cols_map: dict) -> set:
    """Incidence entries re-indexed into ground-truth ids (unmatched cells dropped)."""
    out = set()
    for i, row in enumerate(matrix):
        if i not in rows_map:
            continue
        for j, v in enumerate(row):
            if v and j in cols_map:
                out.add((rows_map[i], cols_map[j]))
    return out


def topology_prf(pred: ChainComplex, gt: ChainComplex,
                 corner_matches, curve_matches, patch_matches) -> dict[str, PRF]:
    """Precision / recall / F1 of the ``EV`` and ``FE`` incidence entries.

    Predicted incidences are re-indexed through the primitive matching; entries
    touching an unmatched primitive are counted as false positives.
    """
    corner_map = {p: g for (p, g, _) in corner_matches}
    curve_map = {p: g for (p, g, _) in curve_matches}
    patch_map = {p: g for (p, g, _) in patch_matches}

    identity_corner = {j: j for j in range(gt.n_corners)}
    identity_curve = {i: i for i in range(gt.n_curves)}
    identity_patch = {k: k for k in range(gt.n_patches)}

    pred_ev = _incidence_pairs(pred.curve_corner, curve_map, corner_map)
    gt_ev = _incidence_pairs(gt.curve_corner, identity_curve, identity_corner)
    pred_fe = _incidence_pairs(pred.patch_curve, patch_map, curve_map)
    gt_fe = _incidence_pairs(gt.patch_curve, identity_patch, identity_curve)

    n_pred_ev = sum(sum(row) for row in pred.curve_corner)
    n_pred_fe = sum(sum(row) for row in pred.patch_curve)

    return {
        "curve_corner": prf_from_matches(len(pred_ev & gt_ev), n_pred_ev, len(gt_ev)),
        "patch_curve": prf_from_matches(len(pred_fe & gt_fe), n_pred_fe, len(gt_fe)),
    }


# --------------------------------------------------------------------------- #
# whole-complex metrics
# --------------------------------------------------------------------------- #
def complex_points(cx: ChainComplex) -> list[Point]:
    """Every sample of the complex: corners + curve samples + patch samples."""
    pts = list(cx.corners)
    for curve in cx.curves:
        pts.extend(curve.points)
    for patch in cx.patches:
        pts.extend(patch.points)
    return pts


def complex_chamfer(pred: ChainComplex, gt: ChainComplex, direction: str = "bi") -> float:
    return chamfer_distance(complex_points(pred), complex_points(gt), direction)


def evaluate_complex(pred: ChainComplex, gt: ChainComplex,
                     corner_tol: float = 0.1,
                     curve_tol: float = 0.1,
                     patch_tol: float = 0.1) -> dict:
    """Full ComplexGen-style report for one prediction / ground-truth pair."""
    corner_prf, corner_m = structure_prf(
        corner_cost_matrix(pred.corners, gt.corners),
        pred.n_corners, gt.n_corners, corner_tol)
    curve_prf, curve_m = structure_prf(
        curve_cost_matrix(pred.curves, gt.curves),
        pred.n_curves, gt.n_curves, curve_tol)
    patch_prf, patch_m = structure_prf(
        patch_cost_matrix(pred.patches, gt.patches),
        pred.n_patches, gt.n_patches, patch_tol)
    topo = topology_prf(pred, gt, corner_m, curve_m, patch_m)
    return {
        "corners": corner_prf,
        "curves": curve_prf,
        "patches": patch_prf,
        "curve_corner": topo["curve_corner"],
        "patch_curve": topo["patch_curve"],
        "corner_chamfer": matched_chamfer(corner_m),
        "curve_chamfer": matched_chamfer(curve_m),
        "patch_chamfer": matched_chamfer(patch_m),
        "complex_chamfer": complex_chamfer(pred, gt),
    }
