"""Three-way geometric diff of two solid models (the diff-viewer algorithm).

Ported from Zoo's ``diff-viewer-extension`` browser extension, which shows a CAD
model diff on GitHub PRs.  Its visual algorithm (``src/components/viewer/
CombinedModel.tsx``) partitions the union of the *before* solid ``A`` and the
*after* solid ``B`` into three CSG regions and colours them:

*   **Unchanged** = ``A ∩ B``           (rendered neutral / muted)
*   **Additions** = ``B \\ A``           (rendered green)
*   **Deletions** = ``A \\ B``           (rendered red)

i.e. the same ``Intersection`` / ``Subtraction`` CSG operations three.js-csg
performs on the two meshes.  That is the transferable idea: a solid-model diff
is exactly the intersection plus the two set differences of the occupied space.

This module implements that partition deterministically on an occupancy
representation instead of a full mesh-boolean kernel.  A :class:`VoxelSolid` is a
set of occupied integer cells (a rasterised solid); :func:`model_diff` returns a
:class:`ModelDiff` holding the three cell sets plus volume statistics (an
addition/deletion/unchanged count and a normalised *change ratio* -- the
symmetric difference over the union, i.e. one minus the Jaccard similarity).
:func:`voxelize_boxes` rasterises a list of axis-aligned boxes into cells so the
diff can be driven end-to-end from geometry.

Everything is a pure function of the input cell sets: deterministic, stdlib-only,
no mesh library, no rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Sequence, Tuple

__all__ = [
    "Cell",
    "VoxelSolid",
    "ModelDiff",
    "model_diff",
    "voxelize_boxes",
]

Cell = Tuple[int, int, int]
Box = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


class VoxelSolid:
    """A solid represented as a set of occupied integer cells."""

    __slots__ = ("cells",)

    def __init__(self, cells: Iterable[Cell] = ()):
        self.cells: FrozenSet[Cell] = frozenset(cells)

    def __len__(self) -> int:
        return len(self.cells)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VoxelSolid) and self.cells == other.cells

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"VoxelSolid({sorted(self.cells)!r})"

    def intersection(self, other: "VoxelSolid") -> "VoxelSolid":
        return VoxelSolid(self.cells & other.cells)

    def difference(self, other: "VoxelSolid") -> "VoxelSolid":
        return VoxelSolid(self.cells - other.cells)

    def union(self, other: "VoxelSolid") -> "VoxelSolid":
        return VoxelSolid(self.cells | other.cells)


@dataclass(frozen=True)
class ModelDiff:
    """The three CSG regions of a before/after solid diff, plus statistics."""

    unchanged: VoxelSolid
    additions: VoxelSolid
    deletions: VoxelSolid

    @property
    def unchanged_count(self) -> int:
        return len(self.unchanged)

    @property
    def additions_count(self) -> int:
        return len(self.additions)

    @property
    def deletions_count(self) -> int:
        return len(self.deletions)

    @property
    def union_count(self) -> int:
        return self.unchanged_count + self.additions_count + self.deletions_count

    @property
    def is_identical(self) -> bool:
        return self.additions_count == 0 and self.deletions_count == 0

    @property
    def change_ratio(self) -> float:
        """Symmetric difference over union (= 1 - Jaccard similarity).

        0.0 when the two solids are identical, 1.0 when they are disjoint.
        Defined as 0.0 for two empty solids (nothing changed).
        """
        union = self.union_count
        if union == 0:
            return 0.0
        return (self.additions_count + self.deletions_count) / union


def model_diff(before: VoxelSolid, after: VoxelSolid) -> ModelDiff:
    """Partition ``before`` and ``after`` into unchanged / additions / deletions.

    Mirror of ``CombinedModel``:
        unchanged = before ∩ after
        additions = after \\ before
        deletions = before \\ after
    """
    return ModelDiff(
        unchanged=before.intersection(after),
        additions=after.difference(before),
        deletions=before.difference(after),
    )


def voxelize_boxes(boxes: Sequence[Box], resolution: float) -> VoxelSolid:
    """Rasterise axis-aligned boxes into occupied cells of side ``resolution``.

    A cell ``(i, j, k)`` is occupied iff its centre lies inside any box.  The
    grid is anchored at the origin so two separately voxelised models share a
    common lattice and are directly comparable.  Deterministic.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    import math

    cells = set()
    for (lo, hi) in boxes:
        # cell index range whose centres can fall within [lo, hi]
        i0 = int(math.floor(lo[0] / resolution))
        i1 = int(math.floor(hi[0] / resolution))
        j0 = int(math.floor(lo[1] / resolution))
        j1 = int(math.floor(hi[1] / resolution))
        k0 = int(math.floor(lo[2] / resolution))
        k1 = int(math.floor(hi[2] / resolution))
        for i in range(i0, i1 + 1):
            cx = (i + 0.5) * resolution
            if cx < lo[0] or cx > hi[0]:
                continue
            for j in range(j0, j1 + 1):
                cy = (j + 0.5) * resolution
                if cy < lo[1] or cy > hi[1]:
                    continue
                for k in range(k0, k1 + 1):
                    cz = (k + 0.5) * resolution
                    if cz < lo[2] or cz > hi[2]:
                        continue
                    cells.add((i, j, k))
    return VoxelSolid(cells)
