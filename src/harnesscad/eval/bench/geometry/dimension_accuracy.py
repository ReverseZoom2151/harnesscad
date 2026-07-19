"""Bounding-box dimensional-accuracy metric for generated CAD solids.

Deterministic re-implementation of a bounding-box dimensional-accuracy reward
used by a text-to-CAD reward system. This measures a solid's realised **bounding-box**
extents (width / depth / height / bbox-volume) and scores them against a
target-dimensions dict via bounded relative error.

This is distinct from the harness's ``reconstruction/pht_dimension_accuracy.py``,
which matches *dimensional annotations* (type / value / attached-element
coordinates) between a prediction and a ground truth. Here the "dimensions" are
the physical extents of the produced solid, and the score is a continuous
per-key accuracy = ``max(0, 1 - relative_error)`` rather than a hard type/value
match. It is also distinct from ``quality/cad_reward.py`` (a chamfer-distance +
format reward). All computation is stdlib-only and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Keys the metric knows how to measure from a bounding box.
MEASURABLE_KEYS = ("width", "depth", "height", "volume")


def measure_bbox_dimensions(
    min_x: float, max_x: float,
    min_y: float, max_y: float,
    min_z: float, max_z: float,
) -> Dict[str, float]:
    """Realised extents from an axis-aligned bounding box.

    Convention: ``width`` spans x, ``depth`` spans y,
    ``height`` spans z, ``volume`` is the bbox product.
    """
    width = max_x - min_x
    depth = max_y - min_y
    height = max_z - min_z
    return {
        "width": width,
        "depth": depth,
        "height": height,
        "volume": width * depth * height,
    }


@dataclass(frozen=True)
class DimensionComparison:
    """Per-key comparison of a measured extent against its target."""

    expected: float
    actual: float
    difference: float
    relative_error: float
    within_tolerance: bool
    accuracy: float


def _relative_error(expected: float, actual: float) -> float:
    """|actual - expected| / |expected|; +inf when expected is zero."""
    diff = abs(actual - expected)
    if expected == 0:
        return 0.0 if diff == 0 else float("inf")
    return diff / abs(expected)


@dataclass(frozen=True)
class DimensionAccuracyResult:
    """Aggregate dimensional-accuracy report."""

    actual_dimensions: Dict[str, float]
    comparisons: Dict[str, DimensionComparison]
    average_accuracy: float
    all_within_tolerance: bool


def dimension_accuracy(
    actual_dims: Dict[str, float],
    expected_dims: Dict[str, float],
    tolerance: float = 0.05,
) -> DimensionAccuracyResult:
    """Score realised extents against target extents.

    For every key present in *both* ``expected_dims`` and ``actual_dims`` the
    relative error is computed; ``accuracy = max(0, 1 - relative_error)`` (so a
    perfect match scores 1.0 and a >=100%-off value scores 0.0), and the key is
    ``within_tolerance`` when its relative error is ``<= tolerance``.

    ``average_accuracy`` is the mean per-key accuracy (0.0 when no keys overlap);
    ``all_within_tolerance`` is True when every compared key is within tolerance
    (vacuously True when nothing overlaps).
    """
    comparisons: Dict[str, DimensionComparison] = {}
    accuracies = []

    for key, expected in expected_dims.items():
        if key not in actual_dims:
            continue
        actual = actual_dims[key]
        rel = _relative_error(expected, actual)
        acc = 0.0 if rel == float("inf") else max(0.0, 1.0 - rel)
        comparisons[key] = DimensionComparison(
            expected=expected,
            actual=actual,
            difference=abs(actual - expected),
            relative_error=rel,
            within_tolerance=rel <= tolerance,
            accuracy=acc,
        )
        accuracies.append(acc)

    avg = sum(accuracies) / len(accuracies) if accuracies else 0.0
    all_ok = all(c.within_tolerance for c in comparisons.values())
    return DimensionAccuracyResult(
        actual_dimensions=dict(actual_dims),
        comparisons=comparisons,
        average_accuracy=avg,
        all_within_tolerance=all_ok,
    )


def measure_and_score(
    bbox6: tuple,
    expected_dims: Dict[str, float],
    tolerance: float = 0.05,
) -> DimensionAccuracyResult:
    """Convenience: measure a 6-tuple bbox then score it against targets.

    ``bbox6`` is ``(min_x, max_x, min_y, max_y, min_z, max_z)``.
    """
    if len(bbox6) != 6:
        raise ValueError("bbox6 must be (min_x, max_x, min_y, max_y, min_z, max_z)")
    actual = measure_bbox_dimensions(*bbox6)
    return dimension_accuracy(actual, expected_dims, tolerance)
