"""cadrille CAD-reconstruction evaluation protocol (ICLR 2026).

Following CAD-Recode, cadrille normalises ground-truth models into
``[-0.5, 0.5]^3`` and reports three metrics:

* **Chamfer Distance (CD)** — median over the dataset, computed from 8192 sampled
  points and multiplied by 1e3 (invalid models bias the mean, hence the median).
* **Intersection over Union (IoU)** — reported as a percentage.
* **Invalidity Ratio (IR)** — percentage of predictions that fail to produce a
  valid CAD model.

Invalid predictions are excluded from the CD/IoU aggregates (they have no
geometry to score) and counted only towards IR. Pure stdlib, deterministic.
"""

from __future__ import annotations

from math import dist

CD_SCALE = 1000.0
DEFAULT_POINTS = 8192


def normalize_to_unit_cube(points):
    """Centre + uniformly scale ``points`` into ``[-0.5, 0.5]^3``."""
    pts = [tuple(float(c) for c in p) for p in points]
    if not pts:
        raise ValueError("points must be non-empty")
    dims = len(pts[0])
    lo = [min(p[d] for p in pts) for d in range(dims)]
    hi = [max(p[d] for p in pts) for d in range(dims)]
    center = [(lo[d] + hi[d]) / 2.0 for d in range(dims)]
    extent = max(hi[d] - lo[d] for d in range(dims))
    scale = (1.0 / extent) if extent > 0 else 1.0
    return [tuple((p[d] - center[d]) * scale for d in range(dims)) for p in pts]


def chamfer_distance(a, b, scale: float = CD_SCALE) -> float:
    """Symmetric (mean) Chamfer distance between two point sets, x ``scale``."""
    x, y = list(a), list(b)
    if not x or not y:
        raise ValueError("both point sets must be non-empty")
    directed = lambda p, q: sum(min(dist(i, j) for j in q) for i in p) / len(p)
    return (directed(x, y) + directed(y, x)) / 2.0 * scale


def _median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        raise ValueError("cannot take median of empty sequence")
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def median_cd(cd_values) -> float:
    """Median Chamfer distance across a dataset of per-model CD scores."""
    return _median(list(cd_values))


def invalidity_ratio(valid_flags) -> float:
    """Percentage of predictions that are invalid."""
    flags = list(valid_flags)
    if not flags:
        raise ValueError("valid_flags must be non-empty")
    invalid = sum(1 for v in flags if not v)
    return invalid / len(flags) * 100.0


def iou_percent(iou: float) -> float:
    """Express a [0, 1] IoU as a percentage."""
    value = float(iou)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"iou must be in [0, 1], got {value!r}")
    return value * 100.0


def evaluation_report(records) -> dict:
    """Aggregate the cadrille protocol over per-model prediction records.

    Each record is a mapping with keys ``valid`` (bool) and, for valid ones,
    ``cd`` (already-scaled Chamfer distance) and ``iou`` (in [0, 1]). Returns
    median CD and mean IoU% over valid predictions plus the invalidity ratio.
    """
    recs = list(records)
    if not recs:
        raise ValueError("records must be non-empty")
    valid = [r for r in recs if r.get("valid", True)]
    cds = [float(r["cd"]) for r in valid if "cd" in r]
    ious = [iou_percent(r["iou"]) for r in valid if "iou" in r]
    return {
        "count": len(recs),
        "valid_count": len(valid),
        "median_cd": median_cd(cds) if cds else None,
        "mean_iou": (sum(ious) / len(ious)) if ious else None,
        "invalidity_ratio": invalidity_ratio([r.get("valid", True) for r in recs]),
    }
