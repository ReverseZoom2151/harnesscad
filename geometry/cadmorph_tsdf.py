"""Voxelised truncated signed-distance fields with Boolean algebra.

From CADMorph (Ma et al., NeurIPS 2025), Appendix B ("Why CADMorph Uses a
Voxel-Based Truncated SDF"). CADMorph represents every shape as a voxelised
*truncated* signed-distance field (tSDF): a regular grid of signed distances
clamped to ``[-tau, +tau]``. The paper motivates this choice with three
properties, two of which are pure, deterministic geometry we can build here:

  * **Boundary fidelity** — a tSDF stores signed distance (not just
    inside/outside occupancy), furnishing sub-voxel detail.
  * **Boolean algebra** — CAD solids arise from Boolean combinations of
    sketch/extrusion primitives, and SDFs support these analytically (paper
    Eqs. 5-7)::

        f_{A u B}(x) = min( f_A(x),  f_B(x) )      # union
        f_{A \\ B}(x) = max( f_A(x), -f_B(x) )      # difference (A minus B)
        f_{A n B}(x) = max( f_A(x),  f_B(x) )      # intersection

    Because these are grid-aligned and computed cell-by-cell, a sequence of
    primitives can be composed/extended by operating directly on their tSDFs.

The third property in the paper — feeding the grid to a 3-D convolutional
diffusion model (P2S) — is the learned part and lives outside this module.

We also expose the *geometric-dissimilarity proxies* CADMorph's verification
stage relies on: the Euclidean distance between two shape grids (the paper
selects the candidate whose shape latent is closest to the target in L2, Eq. 4)
and a voxel IoU (its volume-consistency metric, Table 1). These let the
verifier score a candidate shape against a target with no learned embedding.

Everything is stdlib-only and deterministic (no wall clock, no randomness).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple


Dims = Tuple[int, int, int]


@dataclass(frozen=True)
class TSDFGrid:
    """An immutable regular grid of *truncated* signed distances.

    ``dims`` is ``(nx, ny, nz)`` and ``values`` is the flattened grid with x
    varying fastest (index ``x + nx*(y + ny*z)``). Every stored value lies in
    ``[-truncation, +truncation]``. A cell is *inside* the solid when its value
    is negative (the surface is the zero level set), matching the usual SDF
    convention.
    """

    dims: Dims
    values: Tuple[float, ...]
    truncation: float

    def __post_init__(self) -> None:
        nx, ny, nz = self.dims
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError(f"grid dims must be positive, got {self.dims}")
        if self.truncation <= 0.0:
            raise ValueError(f"truncation must be positive, got {self.truncation}")
        if len(self.values) != nx * ny * nz:
            raise ValueError(
                f"values length {len(self.values)} != product of dims "
                f"{nx * ny * nz}")

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_sdf(cls, dims: Dims, func: Callable[[float, float, float], float],
                 truncation: float = 0.2,
                 *, origin: Sequence[float] = (0.0, 0.0, 0.0),
                 spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> "TSDFGrid":
        """Sample a signed-distance ``func(x, y, z)`` onto a grid and truncate.

        Cell ``(i, j, k)`` is sampled at world coordinate
        ``origin + (i, j, k) * spacing``. The sampled distance is clamped to
        ``[-truncation, +truncation]``.
        """
        nx, ny, nz = dims
        ox, oy, oz = (float(o) for o in origin)
        sx, sy, sz = (float(s) for s in spacing)
        tau = float(truncation)
        vals: List[float] = []
        for k in range(nz):
            z = oz + k * sz
            for j in range(ny):
                y = oy + j * sy
                for i in range(nx):
                    x = ox + i * sx
                    vals.append(_clamp(float(func(x, y, z)), tau))
        return cls(dims, tuple(vals), tau)

    @classmethod
    def sphere(cls, dims: Dims, center: Sequence[float], radius: float,
               truncation: float = 0.2,
               *, spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> "TSDFGrid":
        """A tSDF of a sphere (distance = |x - center| - radius)."""
        cx, cy, cz = (float(c) for c in center)
        r = float(radius)

        def f(x: float, y: float, z: float) -> float:
            return math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - r

        return cls.from_sdf(dims, f, truncation, spacing=spacing)

    @classmethod
    def box(cls, dims: Dims, lo: Sequence[float], hi: Sequence[float],
            truncation: float = 0.2,
            *, spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> "TSDFGrid":
        """A tSDF of an axis-aligned box spanning ``lo``..``hi`` (exact SDF)."""
        lx, ly, lz = (float(v) for v in lo)
        hx, hy, hz = (float(v) for v in hi)

        def f(x: float, y: float, z: float) -> float:
            return _box_sdf(x, y, z, lx, ly, lz, hx, hy, hz)

        return cls.from_sdf(dims, f, truncation, spacing=spacing)

    # -- Boolean algebra (paper Eqs. 5-7) ----------------------------------- #
    def union(self, other: "TSDFGrid") -> "TSDFGrid":
        """A u B: ``min(f_A, f_B)`` (paper Eq. 5)."""
        return self._combine(other, min)

    def difference(self, other: "TSDFGrid") -> "TSDFGrid":
        """A \\ B: ``max(f_A, -f_B)`` (paper Eq. 6) — carve ``other`` out."""
        self._assert_compatible(other)
        vals = tuple(
            _clamp(max(a, -b), self.truncation)
            for a, b in zip(self.values, other.values))
        return TSDFGrid(self.dims, vals, self.truncation)

    def intersection(self, other: "TSDFGrid") -> "TSDFGrid":
        """A n B: ``max(f_A, f_B)`` (paper Eq. 7)."""
        return self._combine(other, max)

    def _combine(self, other: "TSDFGrid",
                 op: Callable[[float, float], float]) -> "TSDFGrid":
        self._assert_compatible(other)
        vals = tuple(
            _clamp(op(a, b), self.truncation)
            for a, b in zip(self.values, other.values))
        return TSDFGrid(self.dims, vals, self.truncation)

    def _assert_compatible(self, other: "TSDFGrid") -> None:
        if self.dims != other.dims:
            raise ValueError(
                f"grid dims mismatch: {self.dims} vs {other.dims}")
        if not math.isclose(self.truncation, other.truncation):
            raise ValueError(
                f"truncation mismatch: {self.truncation} vs {other.truncation}")

    # -- occupancy / measurement -------------------------------------------- #
    def is_inside(self, index: int) -> bool:
        """True when cell ``index`` is inside the solid (value < 0)."""
        return self.values[index] < 0.0

    def occupancy(self) -> Tuple[bool, ...]:
        """The inside/outside mask (value < 0) as a tuple of bools."""
        return tuple(v < 0.0 for v in self.values)

    def occupied_count(self) -> int:
        return sum(1 for v in self.values if v < 0.0)

    def occupancy_fraction(self) -> float:
        return self.occupied_count() / len(self.values)

    def to_dict(self) -> dict:
        return {"dims": list(self.dims), "truncation": self.truncation,
                "values": list(self.values)}


# --------------------------------------------------------------------------- #
# Geometric-dissimilarity proxies (CADMorph verification signal)
# --------------------------------------------------------------------------- #
def l2_distance(a: TSDFGrid, b: TSDFGrid) -> float:
    """Euclidean distance between two shape grids.

    CADMorph's verification stage embeds a candidate sequence and the target
    shape into a shared latent space and selects the candidate of minimal L2
    distance (paper Eq. 4). With voxelised tSDFs as the shared representation
    this is exactly the Euclidean distance between the two flattened grids — a
    learned-embedding-free stand-in for that selection signal.
    """
    a._assert_compatible(b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a.values, b.values)))


def voxel_iou(a: TSDFGrid, b: TSDFGrid) -> float:
    """Intersection-over-union of the two occupancy masks (value < 0).

    CADMorph reports IoU as its volume-consistency metric (Table 1). Two empty
    shapes are defined to have IoU 1.0 (identical, nothing to disagree on).
    """
    a._assert_compatible(b)
    inter = union = 0
    for x, y in zip(a.values, b.values):
        ai, bi = x < 0.0, y < 0.0
        if ai and bi:
            inter += 1
        if ai or bi:
            union += 1
    if union == 0:
        return 1.0
    return inter / union


def occupancy_hamming(a: TSDFGrid, b: TSDFGrid) -> int:
    """Count of cells whose inside/outside classification disagrees."""
    a._assert_compatible(b)
    return sum(1 for x, y in zip(a.values, b.values) if (x < 0.0) != (y < 0.0))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clamp(v: float, tau: float) -> float:
    if v > tau:
        return tau
    if v < -tau:
        return -tau
    return v


def _box_sdf(x: float, y: float, z: float,
             lx: float, ly: float, lz: float,
             hx: float, hy: float, hz: float) -> float:
    """Exact signed distance to an axis-aligned box [lo, hi]."""
    cx, cy, cz = (lx + hx) / 2, (ly + hy) / 2, (lz + hz) / 2
    ex, ey, ez = (hx - lx) / 2, (hy - ly) / 2, (hz - lz) / 2
    dx, dy, dz = abs(x - cx) - ex, abs(y - cy) - ey, abs(z - cz) - ez
    ox, oy, oz = max(dx, 0.0), max(dy, 0.0), max(dz, 0.0)
    outside = math.sqrt(ox * ox + oy * oy + oz * oz)
    inside = min(max(dx, dy, dz), 0.0)
    return outside + inside
