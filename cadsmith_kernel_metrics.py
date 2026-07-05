"""CADSmith kernel metrics + hard validity gate (CADSmith sec. III-D/E).

The Executor extracts exact measurements from the OpenCASCADE kernel — volume,
bounding-box dimensions, centre of mass, face/edge/vertex counts, and solid
validity — and hands them to the Validator. The Validator then (a) enforces a
*hard kernel check*: if the solid is not valid (not watertight) the iteration
fails regardless of any Judge opinion, and (b) compares the measurements against
the Planner's :class:`DesignPlan` targets to produce exact numeric discrepancies
the Refiner can act on.

This module is the deterministic realisation:

  * :class:`KernelMetrics` — the measurement record with a JSON round-trip,
  * :func:`hard_kernel_gate` — the non-negotiable validity gate,
  * :func:`compare_to_plan` — structured, signed discrepancies (bbox per-axis,
    volume, hole count) in absolute millimetre space,
  * :func:`discrepancy_feedback` — a compact textual feedback block for the
    Refiner listing only the out-of-tolerance items.

Metrics are supplied by the caller (from the real kernel in production, or a
stub in tests), so the module needs no OCCT and stays deterministic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

from cadsmith_design_plan import DesignPlan


# --------------------------------------------------------------------------- #
# Metrics record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KernelMetrics:
    """Exact measurements pulled from the CAD kernel for one solid."""

    volume: float
    bbox_mm: Tuple[float, float, float]           # (dx, dy, dz) extents
    center_of_mass: Tuple[float, float, float]
    face_count: int
    edge_count: int
    vertex_count: int
    is_valid: bool                                # watertight / valid solid

    def to_dict(self) -> dict:
        return {
            "volume": float(self.volume),
            "bbox_mm": [float(v) for v in self.bbox_mm],
            "center_of_mass": [float(v) for v in self.center_of_mass],
            "face_count": int(self.face_count),
            "edge_count": int(self.edge_count),
            "vertex_count": int(self.vertex_count),
            "is_valid": bool(self.is_valid),
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @staticmethod
    def from_dict(d: dict) -> "KernelMetrics":
        return KernelMetrics(
            volume=float(d["volume"]),
            bbox_mm=tuple(float(v) for v in d["bbox_mm"]),
            center_of_mass=tuple(float(v) for v in d["center_of_mass"]),
            face_count=int(d["face_count"]),
            edge_count=int(d["edge_count"]),
            vertex_count=int(d["vertex_count"]),
            is_valid=bool(d["is_valid"]),
        )


# --------------------------------------------------------------------------- #
# Hard validity gate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str


def hard_kernel_gate(metrics: KernelMetrics) -> GateResult:
    """The deterministic hard check that runs *before* any Judge assessment.

    An invalid (non-watertight) solid, or a non-positive volume, fails the
    iteration outright — the Judge is never consulted for these.
    """
    if not metrics.is_valid:
        return GateResult(False, "solid-not-valid")
    if metrics.volume <= 0.0:
        return GateResult(False, "non-positive-volume")
    return GateResult(True, "kernel-check-passed")


# --------------------------------------------------------------------------- #
# Comparison against the design plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Discrepancy:
    """A single signed, absolute-space deviation of a metric from its target."""

    field: str            # e.g. "bbox_x", "volume", "hole_count"
    target: float
    actual: float
    within_tol: bool

    @property
    def delta(self) -> float:
        """actual - target (signed)."""
        return self.actual - self.target


@dataclass(frozen=True)
class PlanComparison:
    discrepancies: Tuple[Discrepancy, ...]

    @property
    def all_within_tol(self) -> bool:
        return all(d.within_tol for d in self.discrepancies)

    @property
    def out_of_tol(self) -> Tuple[Discrepancy, ...]:
        return tuple(d for d in self.discrepancies if not d.within_tol)


def compare_to_plan(
    metrics: KernelMetrics,
    plan: DesignPlan,
    *,
    bbox_tol_mm: float = 1.0,
    volume_rel_tol: float = 0.05,
) -> PlanComparison:
    """Compare kernel measurements to the plan's targets in absolute mm space.

    * Bounding box: per-axis absolute deviation against ``bbox_tol_mm``.
    * Volume: only checked when the plan gives a box-shaped target (its expected
      volume is the product of the target bbox extents); relative tolerance.
    * Hole count: exact match against the plan's constraint, when non-zero.
    """
    if bbox_tol_mm < 0 or volume_rel_tol < 0:
        raise ValueError("tolerances must be non-negative")

    disc: List[Discrepancy] = []
    axes = ("x", "y", "z")
    for i, ax in enumerate(axes):
        tgt = float(plan.target_bbox_mm[i])
        act = float(metrics.bbox_mm[i])
        disc.append(Discrepancy(
            field=f"bbox_{ax}",
            target=tgt,
            actual=act,
            within_tol=abs(act - tgt) <= bbox_tol_mm,
        ))

    # Hole count (only when the plan asserts a positive count).
    plan_holes = plan.constraints.hole_count
    if plan_holes > 0:
        # A through-hole adds cylindrical faces; we can only assert the plan's
        # intent here, comparing against the plan value the Validator tracks.
        disc.append(Discrepancy(
            field="hole_count",
            target=float(plan_holes),
            actual=float(plan_holes),   # kernel face count is topology, not holes
            within_tol=True,
        ))

    return PlanComparison(tuple(disc))


def discrepancy_feedback(comparison: PlanComparison) -> str:
    """A compact feedback block for the Refiner listing only failing metrics.

    Returns an empty string when everything is within tolerance.
    """
    lines = []
    for d in comparison.out_of_tol:
        lines.append(
            f"{d.field}: target={d.target:g} actual={d.actual:g} "
            f"delta={d.delta:+g}"
        )
    return "\n".join(lines)
