"""PS-CAD reconstruction metrics.

Implements the deterministic evaluation metrics used in PS-CAD (Yang et al. 2024,
Sec. 7.1 "Evaluation metrics"), following [Guo et al. 2022a].  Metrics are
computed between the input CAD model's point cloud and the point cloud of the
executed predicted CAD modelling sequence:

  * Chamfer distance (CD) -- symmetric mean nearest-neighbour distance;
  * Hausdorff distance (HD) -- symmetric max nearest-neighbour distance;
  * Edge Chamfer distance (ECD) -- Chamfer distance between the *edge* point
    sets of two CAD models, reflecting structural similarity;
  * Normal consistency (NC) -- mean cosine similarity between the normals of
    corresponding (nearest) points, measuring surface-normal smoothness;
  * Invalidity ratio (IR) -- fraction of reconstructions where no step executes
    successfully (a sequence is invalid iff none of ``O_0, O_1, ...`` executes).

Following the paper, clouds are normalised into a unit bounding box (keeping the
aspect ratio) before scale-invariant geometric comparison.  Everything is
stdlib-only and deterministic.  The learned NLL/sequence-fidelity metrics that
require a trained autoregressive model are intentionally out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, sqrt


def _as_points(cloud):
    return [tuple(float(c) for c in p) for p in cloud]


def normalize_unit_box(cloud):
    """Normalise a cloud into a unit bounding box, preserving aspect ratio.

    Centres the cloud and divides by the largest extent across all axes, as the
    paper does for scale-invariant comparison.  A single-point (zero-extent)
    cloud is returned centred at the origin.
    """
    pts = _as_points(cloud)
    if not pts:
        return []
    dims = len(pts[0])
    lo = [min(p[k] for p in pts) for k in range(dims)]
    hi = [max(p[k] for p in pts) for k in range(dims)]
    center = [(lo[k] + hi[k]) / 2.0 for k in range(dims)]
    scale = max(hi[k] - lo[k] for k in range(dims))
    if scale <= 0:
        return [tuple(0.0 for _ in range(dims)) for _ in pts]
    return [tuple((p[k] - center[k]) / scale for k in range(dims)) for p in pts]


def _directed_nn(a, b, reducer):
    return reducer(min(dist(p, q) for q in b) for p in a)


def chamfer_distance(a, b, *, normalize=True):
    """Symmetric Chamfer distance (CD): mean of directed mean NN distances."""
    x = normalize_unit_box(a) if normalize else _as_points(a)
    y = normalize_unit_box(b) if normalize else _as_points(b)
    if not x or not y:
        return None
    fwd = sum(min(dist(p, q) for q in y) for p in x) / len(x)
    bwd = sum(min(dist(p, q) for q in x) for p in y) / len(y)
    return (fwd + bwd) / 2.0


def hausdorff_distance(a, b, *, normalize=True):
    """Symmetric Hausdorff distance (HD): max of directed max NN distances."""
    x = normalize_unit_box(a) if normalize else _as_points(a)
    y = normalize_unit_box(b) if normalize else _as_points(b)
    if not x or not y:
        return None
    fwd = _directed_nn(x, y, max)
    bwd = _directed_nn(y, x, max)
    return max(fwd, bwd)


def edge_chamfer_distance(edges_a, edges_b, *, normalize=True):
    """Edge Chamfer distance (ECD): Chamfer distance between edge point sets.

    Edge points are the sampled points lying on the sharp edges of each CAD
    model.  Reflects structural (not just surface) similarity, as in the paper.
    """
    return chamfer_distance(edges_a, edges_b, normalize=normalize)


def _cos(a, b):
    na = sqrt(sum(c * c for c in a))
    nb = sqrt(sum(c * c for c in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def normal_consistency(points_a, normals_a, points_b, normals_b):
    """Normal consistency (NC): mean absolute cosine similarity of corresponding normals.

    For each point in A, its nearest point in B (and vice versa) defines the
    correspondence; the metric averages ``|cos(n_a, n_b)|`` over both directions.
    Absolute value makes it invariant to normal orientation flips.  Result lies
    in ``[0, 1]``; higher is smoother/more consistent.
    """
    pa, pb = _as_points(points_a), _as_points(points_b)
    if not pa or not pb:
        return None
    if len(pa) != len(normals_a) or len(pb) != len(normals_b):
        raise ValueError("points and normals must have equal length")

    def directed(src_pts, src_n, dst_pts, dst_n):
        total = 0.0
        for p, n in zip(src_pts, src_n):
            j = min(range(len(dst_pts)), key=lambda k: dist(p, dst_pts[k]))
            total += abs(_cos(n, dst_n[j]))
        return total / len(src_pts)

    fwd = directed(pa, normals_a, pb, normals_b)
    bwd = directed(pb, normals_b, pa, normals_a)
    return (fwd + bwd) / 2.0


def sequence_is_valid(step_outcomes):
    """A reconstruction is valid iff at least one step executes successfully.

    ``step_outcomes`` is an iterable of booleans (one per retained step output
    ``O_0, O_1, ...``).  Matches the paper: "If none of them leads to a
    successful execution, we consider it to be invalid."
    """
    return any(bool(o) for o in step_outcomes)


def invalidity_ratio(sequences):
    """Invalidity ratio (IR): fraction of reconstructions that are invalid.

    ``sequences`` is an iterable where each element is the per-step outcome list
    of one reconstruction.  Returns a float in ``[0, 1]``.
    """
    seqs = list(sequences)
    if not seqs:
        return 0.0
    invalid = sum(0 if sequence_is_valid(s) else 1 for s in seqs)
    return invalid / len(seqs)


@dataclass(frozen=True)
class ReconstructionReport:
    cd: float | None
    hd: float | None
    ecd: float | None
    nc: float | None
    ir: float


def evaluate_reconstruction(target, prediction, *, target_edges=None,
                            prediction_edges=None, target_normals=None,
                            prediction_normals=None, sequences=None,
                            normalize=True):
    """Compute the full PS-CAD geometric report between target and prediction.

    Only the metrics whose inputs are supplied are populated; the rest are
    ``None`` (except IR, which defaults to 0.0 for an empty ``sequences``).
    """
    cd = chamfer_distance(target, prediction, normalize=normalize)
    hd = hausdorff_distance(target, prediction, normalize=normalize)
    ecd = (edge_chamfer_distance(target_edges, prediction_edges, normalize=normalize)
           if target_edges is not None and prediction_edges is not None else None)
    nc = (normal_consistency(target, target_normals, prediction, prediction_normals)
          if target_normals is not None and prediction_normals is not None else None)
    ir = invalidity_ratio(sequences) if sequences is not None else 0.0
    return ReconstructionReport(cd, hd, ecd, nc, ir)
