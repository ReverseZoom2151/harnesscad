"""RLCAD set-level generative metrics: COV, MMD-CD / MMD-EMD, JSD.

Source
------
RLCAD (arXiv:2503.18549), Tables 1 / 4 / 5 / 6, which report ``IoU, COV, MMD-CD,
JSD, NC``. This module implements the three set-level members of that row; IoU
is already covered by ``bench.geometry.voxel_iou_points`` (sparse-voxel Jaccard)
and is not re-implemented here.

Exact variants and normalisations
---------------------------------
Let ``G`` be the generated shape set and ``R`` the reference shape set, and let
``d(a, b)`` be a shape-to-shape distance (below).

* **COV (coverage)** -- the Achlioptas et al. / RLCAD direction:

      COV = |{ argmin_{r in R} d(g, r)  :  g in G }| / |R|

  i.e. for EACH GENERATED shape find its nearest REFERENCE shape, then report
  the fraction of the reference set that got matched at least once. Fraction in
  ``[0, 1]``, higher is better; it saturates at ``min(|G|, |R|) / |R|``.

  This is deliberately NOT the direction implemented by
  ``bench.generative.brep_set_metrics.coverage_mmd``, which walks the map the
  other way (for each REFERENCE, nearest GENERATED) and counts distinct
  *generated* indices over ``|R|``. The two agree on many inputs and disagree on
  others -- e.g. reference ``{0, 10}`` and generated ``{1, 2}`` gives 0.5 here
  (only the shape at 0 is covered) and 1.0 there (both generated shapes are
  somebody's nearest neighbour). They are registered as rivals (family
  ``set_coverage``) and must never be pooled.

* **MMD (minimum matching distance)** -- one direction only, and it is the same
  direction in both implementations:

      MMD = ( sum_{r in R} min_{g in G} d(r, g) ) / |R|

  the mean over REFERENCE shapes of the distance to the nearest generated
  shape. Lower is better. The value carries the units of ``d``; no scaling,
  no x1000, no normalisation of the clouds is applied here -- the caller
  chooses the distance and therefore the scale.

* **MMD-CD** uses ``bench.geometry.chamfer.symmetric_chamfer``: the mean-form
  symmetric Chamfer distance ``(mean_{p in A} min_{q in B} ||p-q|| +
  mean_{q in B} min_{p in A} ||q-p||) / 2`` on RAW coordinates -- Euclidean
  (not squared), halved, and with NO centroid/unit-sphere/unit-cube
  normalisation. That is one of the repo's six Chamfer variants; picking a
  different one changes MMD-CD by orders of magnitude, which is why the variant
  is named here rather than left to the reader.

* **MMD-EMD** uses ``domain.reconstruction.evaluate.pointcloud_emd.mean_emd``:
  the exact Hungarian min-cost bijection between two EQUAL-cardinality clouds,
  divided by the number of points (per-point mean, not the summed cost). Clouds
  of unequal size raise -- EMD as a bijection is undefined for them, and
  padding or subsampling to force a match would be an invented protocol.

* **JSD** -- Jensen-Shannon divergence (base 2, so in ``[0, 1]``) between the
  POOLED voxel-occupancy distributions of the generated and the reference point
  sets. Every point of every cloud of a set is dropped into one shared
  ``grid^3`` lattice over ``bounds`` and counted; the two count histograms are
  normalised to probability distributions and compared. 0 means the two sets
  occupy space identically. This delegates to
  ``bench.generative.one_nna.voxel_jsd`` (LION's protocol: grid 28 over
  ``[-1, 1]``) rather than re-deriving it, so the two cannot drift apart.

Stdlib only, deterministic (ties in the nearest-neighbour scan break to the
lowest index).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

__all__ = [
    "chamfer_distance",
    "emd_distance",
    "coverage",
    "mmd",
    "voxel_jsd",
    "set_report",
]

Cloud = Sequence[Sequence[float]]
Distance = Callable[[object, object], float]


def chamfer_distance(a: Cloud, b: Cloud) -> float:
    """Mean-form symmetric Chamfer on raw coordinates (the MMD-CD distance)."""
    from harnesscad.eval.bench.geometry.chamfer import symmetric_chamfer
    value = symmetric_chamfer(a, b)
    if value is None:
        raise ValueError("chamfer distance is undefined for an empty cloud")
    return float(value)


def emd_distance(a: Cloud, b: Cloud) -> float:
    """Per-point Hungarian EMD (the MMD-EMD distance); equal cardinality only."""
    from harnesscad.domain.reconstruction.evaluate.pointcloud_emd import mean_emd
    return float(mean_emd(a, b))


def _nearest(query: object, pool: Sequence[object], distance: Distance) -> int:
    """Index of the closest element of ``pool`` (ties: lowest index)."""
    best_i, best_d = -1, float("inf")
    for i, item in enumerate(pool):
        d = float(distance(query, item))
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def coverage(generated: Sequence[object], reference: Sequence[object],
             distance: Distance) -> float:
    """RLCAD COV: fraction of the reference set matched by some generated shape.

    For each generated shape, its nearest reference shape is marked as covered;
    COV is the number of distinct covered reference shapes over ``|R|``.
    """
    if not generated or not reference:
        raise ValueError("both the generated and the reference set must be non-empty")
    covered = {_nearest(g, reference, distance) for g in generated}
    return len(covered) / len(reference)


def mmd(generated: Sequence[object], reference: Sequence[object],
        distance: Distance) -> float:
    """RLCAD MMD: mean over reference shapes of the distance to the nearest
    generated shape."""
    if not generated or not reference:
        raise ValueError("both the generated and the reference set must be non-empty")
    total = 0.0
    for r in reference:
        total += min(float(distance(r, g)) for g in generated)
    return total / len(reference)


def voxel_jsd(generated: Sequence[Cloud], reference: Sequence[Cloud],
              grid: int = 28, bounds=(-1.0, 1.0)) -> float:
    """Base-2 JSD between the pooled voxel occupancies of the two sets."""
    from harnesscad.eval.bench.generative.one_nna import voxel_jsd as _jsd
    return float(_jsd(generated, reference, grid, bounds))


def set_report(generated: Sequence[Cloud], reference: Sequence[Cloud],
               *, grid: int = 28, bounds=(-1.0, 1.0),
               with_emd: bool = False) -> Dict[str, float]:
    """COV / MMD-CD / JSD (and optionally MMD-EMD) for two point-cloud sets.

    ``with_emd`` is off by default because the Hungarian solver is cubic in the
    number of points and requires equal-cardinality clouds; turning it on when
    the clouds differ in size raises rather than silently resampling.
    """
    gen: List[Cloud] = [list(c) for c in generated]
    ref: List[Cloud] = [list(c) for c in reference]
    out: Dict[str, float] = {
        "cov": coverage(gen, ref, chamfer_distance),
        "mmd_cd": mmd(gen, ref, chamfer_distance),
        "jsd": voxel_jsd(gen, ref, grid, bounds),
    }
    if with_emd:
        out["mmd_emd"] = mmd(gen, ref, emd_distance)
    return out
