"""Topology metrics: Euler-characteristic-based 3D similarity.

This introduces two topology metrics that capture a form of *semantic*
similarity for texture-less CAD objects, where the
Euler characteristic ``chi = V - E + F`` is a topological invariant for
2-manifold (watertight) polyhedra and, for a genus-``g`` surface,
``chi = 2 - 2g`` (intuitively "number of holes").

  * Topology error   (Eq. 6):  ``T_err  = |chi(O) - chi(O_hat)|``
  * Topology correct (Eq. 7):  ``T_corr = 1[chi(O) == chi(O_hat)]``

and dataset-level aggregates: mean ``T_err`` and the percentage of topologically
correct samples, both over the *watertight joint subset* (samples where a valid
chi exists for both ground truth and prediction; non-watertight objects have no
valid chi and are excluded).

This complements ``bench/cadmium_mesh_metrics.py`` -- which already provides
``euler_characteristic`` and the single-sample Exact-Euler-Characteristic-Match
(EECM, equivalent to ``T_corr``). New here: the ``T_err`` *magnitude*, the
genus/holes interpretation, watertight-subset filtering, and the dataset-level
aggregation (``T_corr`` percentage + mean ``T_err``) used for benchmark tables.

Pure stdlib, no wall clock. Meshes are ``(vertices, faces)`` with faces as index
tuples; a ``None`` chi denotes a non-watertight / invalid object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

Face = Sequence[int]


def euler_characteristic(vertices: Sequence, faces: Sequence[Face]) -> int:
    """``chi = V - E + F`` counting only referenced vertices and unique edges."""
    used = set()
    edges = set()
    f = 0
    for face in faces:
        f += 1
        n = len(face)
        for i in range(n):
            a = face[i]
            b = face[(i + 1) % n]
            used.add(a)
            used.add(b)
            edges.add((a, b) if a < b else (b, a))
    return len(used) - len(edges) + f


def genus_from_euler(chi: int) -> float:
    """Genus ``g = (2 - chi) / 2`` -- the "number of holes" (Sec. IV-B)."""
    return (2 - chi) / 2.0


def topology_error(chi_gt: Optional[int], chi_pred: Optional[int]) -> Optional[int]:
    """T_err (Eq. 6): ``|chi(O) - chi(O_hat)|``; ``None`` if either chi invalid."""
    if chi_gt is None or chi_pred is None:
        return None
    return abs(chi_gt - chi_pred)


def topology_correctness(chi_gt: Optional[int], chi_pred: Optional[int]) -> Optional[int]:
    """T_corr (Eq. 7): ``1`` iff chi values match; ``None`` if either invalid."""
    if chi_gt is None or chi_pred is None:
        return None
    return 1 if chi_gt == chi_pred else 0


@dataclass(frozen=True)
class TopologyReport:
    """Dataset-level aggregate over the watertight joint subset."""

    total_samples: int
    watertight_samples: int
    coverage: float          # watertight_samples / total_samples
    topology_correctness_pct: Optional[float]  # % of watertight subset with matching chi
    mean_topology_error: Optional[float]       # mean T_err over watertight subset


def watertight_subset(
    pairs: Iterable[Tuple[Optional[int], Optional[int]]]
) -> List[Tuple[int, int]]:
    """Keep only pairs where both ground-truth and predicted chi are valid.

    ``chi is None`` marks a non-watertight object (no closed volume, no valid
    chi), which the paper excludes before computing spatial/topology metrics.
    """
    return [(g, p) for g, p in pairs if g is not None and p is not None]


def topology_dataset_report(
    pairs: Sequence[Tuple[Optional[int], Optional[int]]]
) -> TopologyReport:
    """Aggregate ``T_corr`` (%) and mean ``T_err`` over the watertight subset.

    ``pairs`` are ``(chi_ground_truth, chi_prediction)``; either may be ``None``
    for a non-watertight object. Matches Sec. IV-B, where metrics are averaged
    over the watertight joint subset (~80% of CADPrompt in the paper).
    """
    total = len(pairs)
    subset = watertight_subset(pairs)
    n = len(subset)
    coverage = n / total if total else 0.0
    if n == 0:
        return TopologyReport(total, 0, coverage, None, None)
    corr = sum(1 for g, p in subset if g == p) / n * 100.0
    err = sum(abs(g - p) for g, p in subset) / n
    return TopologyReport(total, n, coverage, corr, err)
