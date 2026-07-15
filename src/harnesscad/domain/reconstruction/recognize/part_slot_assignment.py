"""PartCrafter deterministic part-slot assignment and part-overlap validity.

Deterministic re-encoding of the part bookkeeping PartCrafter relies on when it
turns an object into a fixed set of part *slots* (its multi-part 3D generator gives
each part its own token block, tagged by a per-part embedding indexed
``arange(num_parts)`` -- ``src/models/transformers/partcrafter_transformer.py``).
Two data-side rules are weight-free and re-implemented here:

* **Part -> slot assignment.** Each part is placed in an ordered slot ``0..N-1``.
  PartCrafter shuffles parts during training, but for a *canonical* (reproducible)
  assignment we order parts deterministically -- descending bounding-box volume,
  ties broken by min-corner then point count -- and pad/reject against a
  ``max_num_parts`` capacity. The dataset's own fallback is honoured: *"if parts is
  empty, the object is the only part"* (``src/datasets/objaverse_part.py``).

* **Part-overlap validity.** PartCrafter filters decompositions by how much parts
  overlap, using a voxel-set IoU (``datasets/preprocess/calculate_iou.py`` ->
  ``compute_IoU_for_scene``): voxelise each part on a shared grid and compare the
  occupied-cell sets, ``IoU = |A & B| / |A | B|``. A decomposition is kept when its
  mean and max pairwise IoU stay under thresholds (``max_iou_mean`` / ``max_iou_max``).

This is DISTINCT from ``reconstruction.recognize.part_classifier`` (labels a part)
and ``geometry.mesh.segmentation`` (splits a mesh into parts): here parts already
exist and we assign them to ordered slots and score their mutual overlap.

Stdlib only, deterministic. No learned weights, no randomness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "PartBBox",
    "part_bbox",
    "canonical_part_order",
    "assign_slots",
    "voxel_set",
    "voxel_iou",
    "pairwise_iou",
    "iou_summary",
    "is_valid_decomposition",
]

Point = Sequence[float]


@dataclass(frozen=True)
class PartBBox:
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    @property
    def volume(self) -> float:
        return max(0.0, self.hi[0] - self.lo[0]) * max(0.0, self.hi[1] - self.lo[1]) * max(
            0.0, self.hi[2] - self.lo[2]
        )


def part_bbox(points: Sequence[Point]) -> PartBBox:
    """Axis-aligned bounding box of a part's point set."""
    if not points:
        raise ValueError("empty part has no bounding box")
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    zs = [float(p[2]) for p in points]
    return PartBBox(lo=(min(xs), min(ys), min(zs)), hi=(max(xs), max(ys), max(zs)))


def canonical_part_order(parts: Sequence[Sequence[Point]]) -> list[int]:
    """Deterministic slot order over parts: descending bbox volume, stable tie-breaks.

    Returns a permutation of part indices. Ties in volume are broken by ascending
    min-corner (lexicographic), then by descending point count, then by original
    index -- so the order never depends on input order for distinct parts.
    """
    keyed = []
    for i, pts in enumerate(parts):
        bb = part_bbox(pts)
        keyed.append((-bb.volume, bb.lo, -len(pts), i))
    keyed.sort()
    return [k[-1] for k in keyed]


def assign_slots(
    parts: Sequence[Sequence[Point]],
    max_num_parts: int,
    object_points: Sequence[Point] | None = None,
) -> list[tuple[int, int]]:
    """Assign parts to ordered slots ``0..N-1`` in canonical order.

    Returns a list of ``(slot, part_index)``. Rules mirroring PartCrafter:

    * If ``parts`` is empty, the whole object becomes the single part (slot 0).
      ``object_points`` must then be supplied.
    * More than ``max_num_parts`` parts is rejected (``ValueError``) -- PartCrafter
      drops such configs rather than truncating silently.
    """
    if max_num_parts < 1:
        raise ValueError("max_num_parts must be >= 1")
    if not parts:
        if object_points is None:
            raise ValueError("empty parts require object_points for the fallback part")
        return [(0, 0)]
    if len(parts) > max_num_parts:
        raise ValueError(
            f"{len(parts)} parts exceed max_num_parts={max_num_parts}"
        )
    order = canonical_part_order(parts)
    return [(slot, part_idx) for slot, part_idx in enumerate(order)]


def voxel_set(points: Sequence[Point], num_grids: int = 64, scale: float = 2.0) -> set:
    """Occupied voxel cells of a point set (PartCrafter ``get_voxel_set``).

    ``pitch = scale / num_grids``; a point maps to ``round(point / pitch)``. The
    returned set of integer 3-tuples is the part's voxel occupancy.
    """
    if num_grids < 1:
        raise ValueError("num_grids must be >= 1")
    pitch = scale / num_grids
    cells = set()
    for p in points:
        cells.add(
            (
                int(round(float(p[0]) / pitch)),
                int(round(float(p[1]) / pitch)),
                int(round(float(p[2]) / pitch)),
            )
        )
    return cells


def voxel_iou(a: Sequence[Point], b: Sequence[Point], num_grids: int = 64, scale: float = 2.0) -> float:
    """Voxel-set IoU of two parts (PartCrafter ``compute_IoU``)."""
    va = voxel_set(a, num_grids, scale)
    vb = voxel_set(b, num_grids, scale)
    union = va | vb
    if not union:
        return 0.0
    return len(va & vb) / len(union)


def pairwise_iou(
    parts: Sequence[Sequence[Point]], num_grids: int = 64, scale: float = 2.0
) -> list[float]:
    """IoU over every unordered pair of parts (PartCrafter ``iou_list``)."""
    voxels = [voxel_set(p, num_grids, scale) for p in parts]
    out = []
    for i in range(len(voxels)):
        for j in range(i + 1, len(voxels)):
            union = voxels[i] | voxels[j]
            out.append(0.0 if not union else len(voxels[i] & voxels[j]) / len(union))
    return out


def iou_summary(
    parts: Sequence[Sequence[Point]], num_grids: int = 64, scale: float = 2.0
) -> dict:
    """Mean and max pairwise IoU (PartCrafter ``iou_mean`` / ``iou_max``)."""
    ious = pairwise_iou(parts, num_grids, scale)
    if not ious:
        return {"iou_mean": 0.0, "iou_max": 0.0, "iou_list": []}
    return {"iou_mean": sum(ious) / len(ious), "iou_max": max(ious), "iou_list": ious}


def is_valid_decomposition(
    parts: Sequence[Sequence[Point]],
    max_iou_mean: float,
    max_iou_max: float,
    num_grids: int = 64,
    scale: float = 2.0,
) -> bool:
    """True when part overlap stays under both thresholds (PartCrafter data filter)."""
    s = iou_summary(parts, num_grids, scale)
    return s["iou_mean"] <= max_iou_mean and s["iou_max"] <= max_iou_max
