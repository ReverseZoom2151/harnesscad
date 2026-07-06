"""CADPrompt / CADCodeVerify geometric evaluation protocol.

Deterministic re-implementation of the evaluation metrics defined in
"Generating CAD Code with Vision-Language Models for 3D Designs"
(Alrashedy et al., ICLR 2025), section 5.2.

The paper compares a generated 3D object (as a point cloud sampled from an STL
mesh) against an expert ground-truth object using three metrics:

  * Point Cloud distance -- the symmetric average nearest-neighbour distance
    (Eq. 8), a Chamfer-style distance with the paper's explicit ``1/(2|.|)``
    weighting on each direction.
  * Hausdorff distance -- ``max`` of the two directed suprema (Eq. 9).
  * Intersection over the Ground Truth (IoGT) -- the ratio of the axis-aligned
    bounding-box intersection volume of the generated object to the bounding-box
    volume of the ground truth (Eq. 10).

Point clouds are aligned/normalised to a unit cube before scoring, and a
compile failure is penalised uniformly: distance = sqrt(3) (the largest
possible distance inside a unit cube) and IoGT = 0.

This module is stdlib-only and operates on already-sampled point clouds
(iterables of ``(x, y, z)`` tuples); mesh -> point-cloud sampling and the ICP
rotation search are left to the caller / other modules.  It is distinct from
``bench/geometry_distance.py`` (a bare symmetric Chamfer with no normalisation,
Hausdorff, IoGT, or failure protocol) and ``bench/cadvlm_metrics.py`` (entity
F1 for a different paper).
"""
from __future__ import annotations

from math import dist, sqrt
from statistics import median

# Largest distance between two points inside a unit cube (space diagonal).
UNIT_CUBE_DIAGONAL = sqrt(3.0)


def _as_points(cloud):
    pts = [tuple(float(c) for c in p) for p in cloud]
    for p in pts:
        if len(p) != 3:
            raise ValueError("each point must have exactly 3 coordinates")
    return pts


def bounding_box(cloud):
    """Return ((minx,miny,minz),(maxx,maxy,maxz)) for a non-empty cloud."""
    pts = _as_points(cloud)
    if not pts:
        raise ValueError("cannot bound an empty point cloud")
    lo = tuple(min(p[i] for p in pts) for i in range(3))
    hi = tuple(max(p[i] for p in pts) for i in range(3))
    return lo, hi


def normalize_unit_cube(cloud):
    """Translate + isotropically scale a cloud to fit inside the unit cube.

    Matches the paper's normalisation: each point cloud is normalised to fit
    within a unit cube (a single scale factor over all axes preserves shape).
    A degenerate (zero-extent) cloud is mapped to the origin.
    """
    pts = _as_points(cloud)
    if not pts:
        return []
    lo, hi = bounding_box(pts)
    extent = max(hi[i] - lo[i] for i in range(3))
    if extent <= 0:
        return [(0.0, 0.0, 0.0) for _ in pts]
    return [tuple((p[i] - lo[i]) / extent for i in range(3)) for p in pts]


def point_cloud_distance(gen, gt):
    """Symmetric point-cloud (Chamfer) distance, Eq. 8.

    D(P,Q) = (1/2|P|) sum_p min_q ||p-q|| + (1/2|Q|) sum_q min_p ||q-p||.
    """
    p = _as_points(gen)
    q = _as_points(gt)
    if not p or not q:
        raise ValueError("both clouds must be non-empty")
    forward = sum(min(dist(a, b) for b in q) for a in p) / (2 * len(p))
    backward = sum(min(dist(b, a) for a in p) for b in q) / (2 * len(q))
    return forward + backward


def hausdorff_distance(gen, gt):
    """Symmetric Hausdorff distance, Eq. 9."""
    p = _as_points(gen)
    q = _as_points(gt)
    if not p or not q:
        raise ValueError("both clouds must be non-empty")
    forward = max(min(dist(a, b) for b in q) for a in p)
    backward = max(min(dist(b, a) for a in p) for b in q)
    return max(forward, backward)


def _bbox_volume(lo, hi):
    v = 1.0
    for i in range(3):
        v *= max(0.0, hi[i] - lo[i])
    return v


def iogt(gen, gt):
    """Intersection over Ground Truth of axis-aligned bounding boxes, Eq. 10.

    ratio of the bbox-intersection volume of (gen & gt) to the bbox volume of
    the ground truth.  Returns 0.0 when the ground-truth box is degenerate.
    """
    glo, ghi = bounding_box(gen)
    tlo, thi = bounding_box(gt)
    inter_lo = tuple(max(glo[i], tlo[i]) for i in range(3))
    inter_hi = tuple(min(ghi[i], thi[i]) for i in range(3))
    inter = _bbox_volume(inter_lo, inter_hi)
    gt_vol = _bbox_volume(tlo, thi)
    if gt_vol <= 0:
        return 0.0
    return inter / gt_vol


def evaluate_object(gen, gt, *, compiled=True, normalize=True):
    """Score one generated object against its ground truth.

    When ``compiled`` is False the object is penalised uniformly per the paper:
    both distances become sqrt(3) and IoGT becomes 0, regardless of geometry.
    """
    if not compiled:
        return {
            "compiled": False,
            "point_cloud_distance": UNIT_CUBE_DIAGONAL,
            "hausdorff_distance": UNIT_CUBE_DIAGONAL,
            "iogt": 0.0,
        }
    g, t = gen, gt
    if normalize:
        g = normalize_unit_cube(gen)
        t = normalize_unit_cube(gt)
    return {
        "compiled": True,
        "point_cloud_distance": point_cloud_distance(g, t),
        "hausdorff_distance": hausdorff_distance(g, t),
        "iogt": iogt(g, t),
    }


def _iqr(values):
    """Interquartile range (Q3 - Q1) using midpoint halves, as reported."""
    xs = sorted(values)
    n = len(xs)
    if n < 2:
        return 0.0
    half = n // 2
    lower = xs[:half]
    upper = xs[half + 1:] if n % 2 else xs[half:]
    return median(upper) - median(lower)


def aggregate(results):
    """Median (IQR) aggregation over a list of ``evaluate_object`` dicts.

    Also reports the compile rate (fraction of objects that compiled).  The
    paper reports every distance table as ``median (IQR)`` and a compile rate.
    """
    rows = list(results)
    if not rows:
        raise ValueError("no results to aggregate")
    out = {}
    for key in ("point_cloud_distance", "hausdorff_distance", "iogt"):
        vals = [r[key] for r in rows]
        out[key] = {"median": median(vals), "iqr": _iqr(vals)}
    out["compile_rate"] = sum(1 for r in rows if r.get("compiled")) / len(rows)
    out["n"] = len(rows)
    return out
