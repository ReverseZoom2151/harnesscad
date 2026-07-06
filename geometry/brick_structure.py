"""Brick-assembly structure representation from BRICKGPT (Pun et al., CMU).

Paper: "Generating Physically Stable and Buildable Brick Structures from Text"
(BRICKGPT). Section 3 defines the brick representation used throughout the
method: a structure ``B = [b1, ..., bN]`` of ``N`` bricks on a fixed baseplate,
each brick ``bi = [hi, wi, xi, yi, zi]`` where

* ``hi`` -- brick length along X,
* ``wi`` -- brick length along Y,
* ``(xi, yi, zi)`` -- integer grid position of the stud closest to the origin,
  with ``xi in [0, H-1]``, ``yi in [0, W-1]``, ``zi in [0, D-1]``.

All bricks are 1-unit tall, axis-aligned cuboids. The *order* of ``h`` and ``w``
encodes the brick's orientation about the vertical axis (a ``1x2`` and a ``2x1``
are the same piece rotated 90 degrees). The paper's plain-text serialisation is
one line per brick, ``"{h}x{w} ({x},{y},{z})"``, ordered raster-scan bottom to
top (Section 4.1, Appendix A).

This module implements that representation deterministically (stdlib only):
brick geometry, the standard brick library (Appendix A), the custom text format
parser/serialiser, in-bounds checks, and the collision / voxel-overlap check
used by the inference-time validity check (Section 4.2: ``Vt intersect Vi =
empty``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

Cell = tuple[int, int]
Voxel = tuple[int, int, int]

# Standard brick library, Appendix A ("Allowed brick dimensions"). Eight base
# shapes plus their 90-degree orientation swaps. All are 1 unit tall.
STANDARD_BRICKS: frozenset[tuple[int, int]] = frozenset(
    {
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 4),
        (4, 1),
        (1, 6),
        (6, 1),
        (1, 8),
        (8, 1),
        (2, 2),
        (2, 4),
        (4, 2),
        (2, 6),
        (6, 2),
    }
)


@dataclass(frozen=True)
class Brick:
    """A single placed brick ``bi = [h, w, x, y, z]`` (Section 3)."""

    h: int
    w: int
    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        for name in ("h", "w", "x", "y", "z"):
            value = getattr(self, name)
            if not isinstance(value, int):
                raise TypeError(f"{name} must be int, got {type(value).__name__}")
        if self.h < 1 or self.w < 1:
            raise ValueError("brick dimensions must be >= 1")
        if self.x < 0 or self.y < 0 or self.z < 0:
            raise ValueError("brick position must be non-negative")

    @property
    def dims(self) -> tuple[int, int]:
        return (self.h, self.w)

    @property
    def orientation(self) -> int:
        """0 if the brick is placed as-is, 1 if it is the rotated variant.

        A ``1x2`` (h < w) and a ``2x1`` (h > w) are the same physical piece in
        two orientations; square/uniform bricks are canonically orientation 0.
        """
        return 1 if self.h > self.w else 0

    @property
    def stud_count(self) -> int:
        """Number of studs = footprint area (proxy for the brick's mass)."""
        return self.h * self.w

    @property
    def center(self) -> tuple[float, float, float]:
        """Centroid of the brick (its centre of mass), assuming uniform mass."""
        return (self.x + self.h / 2.0, self.y + self.w / 2.0, self.z + 0.5)

    def cells(self) -> Iterator[Cell]:
        """Footprint cells ``(cx, cy)`` occupied at this brick's layer."""
        for i in range(self.h):
            for j in range(self.w):
                yield (self.x + i, self.y + j)

    def voxels(self) -> Iterator[Voxel]:
        """Voxels ``(cx, cy, z)`` occupied by this brick (1 unit tall)."""
        for cx, cy in self.cells():
            yield (cx, cy, self.z)

    def cell_set(self) -> frozenset[Cell]:
        return frozenset(self.cells())

    def voxel_set(self) -> frozenset[Voxel]:
        return frozenset(self.voxels())

    def in_library(self, library: Iterable[tuple[int, int]] = STANDARD_BRICKS) -> bool:
        return (self.h, self.w) in set(library)

    def in_bounds(self, grid_h: int, grid_w: int, grid_d: int) -> bool:
        return (
            0 <= self.x
            and self.x + self.h <= grid_h
            and 0 <= self.y
            and self.y + self.w <= grid_w
            and 0 <= self.z < grid_d
        )

    def to_text(self) -> str:
        """Serialise to the paper's plain-text format ``"{h}x{w} ({x},{y},{z})"``."""
        return f"{self.h}x{self.w} ({self.x},{self.y},{self.z})"

    @classmethod
    def from_text(cls, line: str) -> "Brick":
        """Parse one line of the paper's plain-text brick format.

        Accepts either ``x`` or the unicode ``×`` as the dimension
        separator, e.g. ``"2x4 (1,3,0)"`` or ``"2×4 (1,3,0)"``.
        """
        text = line.strip().replace("×", "x")
        if "(" not in text or ")" not in text:
            raise ValueError(f"malformed brick line: {line!r}")
        dims_part, rest = text.split("(", 1)
        coords_part = rest.split(")", 1)[0]
        dims = dims_part.strip().split("x")
        if len(dims) != 2:
            raise ValueError(f"malformed brick dimensions: {line!r}")
        coords = coords_part.split(",")
        if len(coords) != 3:
            raise ValueError(f"malformed brick coordinates: {line!r}")
        h, w = int(dims[0]), int(dims[1])
        x, y, z = (int(c.strip()) for c in coords)
        return cls(h=h, w=w, x=x, y=y, z=z)


def bricks_overlap(a: Brick, b: Brick) -> bool:
    """True if two bricks share any voxel (occupy the same cell at the same z)."""
    if a.z != b.z:
        return False
    # Axis-aligned rectangle intersection test in the shared layer.
    ax0, ax1 = a.x, a.x + a.h
    ay0, ay1 = a.y, a.y + a.w
    bx0, bx1 = b.x, b.x + b.h
    by0, by1 = b.y, b.y + b.w
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


@dataclass(frozen=True)
class BrickStructure:
    """An ordered brick structure ``B = [b1, ..., bN]`` on a baseplate."""

    bricks: tuple[Brick, ...]
    grid_h: int = 20
    grid_w: int = 20
    grid_d: int = 20

    @classmethod
    def from_bricks(
        cls,
        bricks: Sequence[Brick],
        grid_h: int = 20,
        grid_w: int = 20,
        grid_d: int = 20,
    ) -> "BrickStructure":
        return cls(tuple(bricks), grid_h, grid_w, grid_d)

    @classmethod
    def from_text(
        cls,
        text: str,
        grid_h: int = 20,
        grid_w: int = 20,
        grid_d: int = 20,
    ) -> "BrickStructure":
        bricks = [
            Brick.from_text(line)
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return cls(tuple(bricks), grid_h, grid_w, grid_d)

    def to_text(self) -> str:
        return "\n".join(b.to_text() for b in self.bricks)

    def __len__(self) -> int:
        return len(self.bricks)

    def prefix(self, n: int) -> "BrickStructure":
        """Partial structure ``[b1, ..., bn]`` (used by rollback / assembly)."""
        return BrickStructure(self.bricks[:n], self.grid_h, self.grid_w, self.grid_d)

    def voxel_set(self) -> frozenset[Voxel]:
        out: set[Voxel] = set()
        for b in self.bricks:
            out.update(b.voxels())
        return frozenset(out)

    def colliding_pairs(self) -> list[tuple[int, int]]:
        """Indices ``(i, j)``, ``i < j``, of bricks whose voxels overlap.

        Implements the paper's collision constraint ``Vt intersect Vi = empty``.
        Uses per-layer bucketing so it is efficient on realistic structures.
        """
        by_layer: dict[int, list[int]] = {}
        for idx, b in enumerate(self.bricks):
            by_layer.setdefault(b.z, []).append(idx)
        pairs: list[tuple[int, int]] = []
        for indices in by_layer.values():
            for a_pos in range(len(indices)):
                for b_pos in range(a_pos + 1, len(indices)):
                    i, j = indices[a_pos], indices[b_pos]
                    if bricks_overlap(self.bricks[i], self.bricks[j]):
                        pairs.append((i, j) if i < j else (j, i))
        pairs.sort()
        return pairs

    def has_collision(self) -> bool:
        return bool(self.colliding_pairs())

    def collides_with_existing(self, brick: Brick) -> bool:
        """True if ``brick`` overlaps any brick already in the structure."""
        return any(bricks_overlap(brick, other) for other in self.bricks)

    def all_in_bounds(self) -> bool:
        return all(
            b.in_bounds(self.grid_h, self.grid_w, self.grid_d) for b in self.bricks
        )

    def all_in_library(
        self, library: Iterable[tuple[int, int]] = STANDARD_BRICKS
    ) -> bool:
        lib = set(library)
        return all((b.h, b.w) in lib for b in self.bricks)
