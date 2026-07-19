"""Discrete brick-assembly representation with deterministic validity checks.

The representation places toy-brick models on an integer voxel lattice and rejects any
structure that is not *buildable*: bricks must stay in bounds, not overlap, rest
on support, and connect down to the ground. Those are exactly the deterministic,
kernel-free checks a verifier-first harness wants -- so this module provides the
brick representation and its four structural predicates, with the physics-solver
stability model deliberately left out (that needs a licensed LP solver;
see "What is skipped" below).

A :class:`Brick` is a 1-unit-tall axis-aligned box on the lattice, addressed by
its footprint ``(h, w)`` and its minimum corner ``(x, y, z)``. A
:class:`BrickStructure` maintains a voxel-occupancy grid and answers:

* :meth:`~BrickStructure.has_out_of_bounds_bricks` -- every brick fits in the
  ``world_dim`` cube,
* :meth:`~BrickStructure.has_collisions` -- no two bricks share a voxel,
* :meth:`~BrickStructure.has_floating_bricks` -- every off-ground brick touches
  another brick directly below or above,
* :meth:`~BrickStructure.is_connected` -- every brick reaches the ground through
  a chain of stud connections (union-find, no external graph library).

The line-based brick text format (``"HxW (x,y,z)"`` per line) round-trips through
:func:`parse_text` / :meth:`~BrickStructure.to_text`, so a model's raw output
can be parsed, validated, and turned into a re-promptable diagnostic
deterministically.

What is skipped: the physics-based stability score (needs a commercial LP
solver) and the rendering path (needs an external parts library). The
connectivity check is a licence-free fallback for stability.

Stdlib-only. No numpy: the occupancy grid is a ``set`` of occupied voxels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

__all__ = [
    "Brick",
    "BrickStructure",
    "parse_text",
    "AssemblyReport",
    "validate",
]

_TXT_RE = re.compile(r"^\s*(\d+)x(\d+)\s*\((\d+),(\d+),(\d+)\)\s*$")


@dataclass(frozen=True)
class Brick:
    """A 1-unit-tall rectangular brick with footprint ``h*w`` at corner ``(x,y,z)``."""

    h: int
    w: int
    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        if self.h <= 0 or self.w <= 0:
            raise ValueError("brick footprint must be positive")

    @property
    def area(self) -> int:
        return self.h * self.w

    def voxels(self) -> List[Tuple[int, int, int]]:
        """The ``(x, y, z)`` lattice cells this brick occupies."""
        return [
            (self.x + dx, self.y + dy, self.z)
            for dx in range(self.h)
            for dy in range(self.w)
        ]

    def footprint(self) -> List[Tuple[int, int]]:
        """The ``(x, y)`` cells of the brick's footprint (its z-slice)."""
        return [(self.x + dx, self.y + dy) for dx in range(self.h) for dy in range(self.w)]

    def overlaps_xy(self, other: "Brick") -> bool:
        """True iff the two footprints overlap in the x-y plane (any z)."""
        return (
            self.x < other.x + other.h
            and self.x + self.h > other.x
            and self.y < other.y + other.w
            and self.y + self.w > other.y
        )

    def to_text(self) -> str:
        return f"{self.h}x{self.w} ({self.x},{self.y},{self.z})"

    @classmethod
    def from_text(cls, line: str) -> "Brick":
        m = _TXT_RE.match(line)
        if m is None:
            raise ValueError(f"ill-formatted brick line: {line!r}")
        h, w, x, y, z = (int(g) for g in m.groups())
        return cls(h=h, w=w, x=x, y=y, z=z)


class _UnionFind:
    def __init__(self) -> None:
        self._parent: Dict[int, int] = {}

    def find(self, a: int) -> int:
        self._parent.setdefault(a, a)
        while self._parent[a] != a:
            self._parent[a] = self._parent[self._parent[a]]
            a = self._parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


class BrickStructure:
    """A list of bricks on a ``world_dim`` cube lattice, with validity checks."""

    def __init__(self, bricks: Sequence[Brick], world_dim: int = 20):
        self.world_dim = world_dim
        self.bricks: List[Brick] = list(bricks)
        self._occupancy: Dict[Tuple[int, int, int], int] = {}
        for b in self.bricks:
            for v in b.voxels():
                self._occupancy[v] = self._occupancy.get(v, 0) + 1

    def __len__(self) -> int:
        return len(self.bricks)

    # --- structural predicates -------------------------------------------
    def brick_in_bounds(self, b: Brick) -> bool:
        return (
            0 <= b.x
            and b.x + b.h <= self.world_dim
            and 0 <= b.y
            and b.y + b.w <= self.world_dim
            and 0 <= b.z < self.world_dim
        )

    def has_out_of_bounds_bricks(self) -> bool:
        return any(not self.brick_in_bounds(b) for b in self.bricks)

    def has_collisions(self) -> bool:
        return any(count > 1 for count in self._occupancy.values())

    def brick_floats(self, b: Brick) -> bool:
        """True iff *b* is unsupported: off-ground and touching no brick above/below."""
        if b.z == 0:
            return False  # rests on the ground
        for (fx, fy) in b.footprint():
            if self._occupancy.get((fx, fy, b.z - 1), 0):
                return False  # supported from below
            if b.z + 1 < self.world_dim and self._occupancy.get((fx, fy, b.z + 1), 0):
                return False  # held from above
        return True

    def has_floating_bricks(self) -> bool:
        return any(self.brick_floats(b) for b in self.bricks)

    def is_connected(self) -> bool:
        """True iff every brick reaches the ground through stud connections."""
        if not self.bricks:
            return True
        return len(self.disconnected_bricks()) == 0

    def disconnected_bricks(self) -> List[Brick]:
        """Bricks not connected to the ground via a chain of adjacencies.

        Two bricks connect when one sits directly on the other (``|dz| == 1``)
        with overlapping footprints; a brick connects to the ground at ``z==0``.
        Union-find over the brick indices plus a synthetic ground node.
        """
        uf = _UnionFind()
        ground = -1
        uf.find(ground)
        for i, b in enumerate(self.bricks):
            uf.find(i)
            if b.z == 0:
                uf.union(i, ground)
        for i in range(len(self.bricks)):
            for j in range(i + 1, len(self.bricks)):
                bi, bj = self.bricks[i], self.bricks[j]
                if abs(bi.z - bj.z) == 1 and bi.overlaps_xy(bj):
                    uf.union(i, j)
        gr = uf.find(ground)
        return [b for i, b in enumerate(self.bricks) if uf.find(i) != gr]

    # --- serialisation ----------------------------------------------------
    def to_text(self) -> str:
        return "\n".join(b.to_text() for b in self.bricks)


def parse_text(text: str, world_dim: int = 20) -> BrickStructure:
    """Parse brick text-format lines into a :class:`BrickStructure`.

    Blank lines are ignored; a malformed line raises ``ValueError`` (the
    ``format`` failure mode a caller would feed back to the generator).
    """
    bricks = [Brick.from_text(ln) for ln in text.splitlines() if ln.strip()]
    return BrickStructure(bricks, world_dim=world_dim)


@dataclass(frozen=True)
class AssemblyReport:
    """Deterministic buildability verdict for a brick structure."""

    buildable: bool
    out_of_bounds: bool
    collisions: bool
    floating: bool
    disconnected: bool
    reasons: Tuple[str, ...]


def validate(structure: BrickStructure) -> AssemblyReport:
    """Run every deterministic buildability check and summarise the verdict."""
    oob = structure.has_out_of_bounds_bricks()
    col = structure.has_collisions()
    flt = structure.has_floating_bricks()
    disc = not structure.is_connected()
    reasons: List[str] = []
    if oob:
        reasons.append("one or more bricks lie outside the world bounds")
    if col:
        reasons.append("two or more bricks overlap")
    if flt:
        reasons.append("one or more bricks are unsupported (floating)")
    if disc:
        reasons.append("one or more bricks are not connected to the ground")
    buildable = not (oob or col or flt or disc)
    return AssemblyReport(
        buildable=buildable,
        out_of_bounds=oob,
        collisions=col,
        floating=flt,
        disconnected=disc,
        reasons=tuple(reasons),
    )
