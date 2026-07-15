"""region_group — a deterministic screen-region grouper for the opaque viewport.

ShowUI's contribution to GUI grounding is *UI-guided visual token selection*: a
screenshot has vast redundancy (a flat toolbar is thousands of near-identical
patches), and rather than feed every 14x14 patch to the model as an independent
visual token, ShowUI first groups **adjacent patches whose appearance is nearly
identical** into connected components with a union-find, then keeps one token per
component. The grouping criterion is purely local and deterministic: two
neighbouring patches are merged iff the L2 distance between their pixel vectors is
below a threshold (``image_processing_showui.py::_build_uigraph``). It needs no
model and no labels.

That same primitive is exactly what a CAD *viewport* wants, and for a reason the
rest of :mod:`harnesscad.io.cua` keeps running into: the 3D view is one opaque
``QOpenGLWidget`` with no accessibility tree (see :mod:`harnesscad.io.cua.viewport`).
:mod:`~harnesscad.io.cua.picks` grounds it *analytically* — we own the B-rep, so we
project. This module is the complementary *appearance-only* grouper: given nothing
but the rendered pixels it partitions the frame into flat, uniformly-shaded
regions (a face reads as one region, the background as another, a toolbar row as a
strip of button-sized regions), which is a cheap, model-free pre-segmentation that
a downstream grounder can enumerate — and, unlike ShowUI's, it is here reproduced
in pure stdlib, no numpy, so it round-trips deterministically in a unit test.

Two things this is NOT, on purpose:

* It is not a substitute for the app-adjudicated pick. A region is an appearance
  blob; whether a click on it selects ``Face7`` is still the picker's call. This
  narrows *where to look*, it does not label.
* It is not a re-run of :mod:`harnesscad.eval.grounding.corpus`. That harvests
  verified geometry labels; this partitions raw pixels with zero geometry.

Everything is import-safe and dependency-free: the grid is plain Python lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

Color = Tuple[float, float, float]
Grid = Sequence[Sequence[Color]]


class UnionFind:
    """Disjoint-set with path compression + union by size. Deterministic.

    A faithful port of ShowUI's ``UnionFind`` (which used a numpy parent array);
    here it is plain Python so the whole grouper is stdlib-only and its output is
    a pure function of its input, which is what makes it unit-testable without an
    image in the room.
    """

    def __init__(self, size: int) -> None:
        self.parent: List[int] = list(range(size))
        self.size: List[int] = [1] * size

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # iterative path compression (no recursion depth)
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Union by size, tie-broken toward the SMALLER index so the chosen root is
        # deterministic regardless of merge order — the relabelling below depends
        # on it (ShowUI leaned on numpy + LabelEncoder for the same stability).
        if self.size[ra] < self.size[rb] or (self.size[ra] == self.size[rb] and rb < ra):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def patch_distance(a: Color, b: Color) -> float:
    """L2 distance between two patch colours. ShowUI's exact merge criterion."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


@dataclass(frozen=True)
class Region:
    """One connected, uniformly-shaded region of the frame.

    Coordinates are PATCH indices (row, col) unless converted; ``bbox`` is
    ``(min_col, min_row, max_col, max_row)`` inclusive, in patch space. ``mean``
    is the average colour over the region's patches — a stable referent a caller
    can compare against ("the large dark region", "the light strip at the top").
    """

    label: int
    patches: Tuple[Tuple[int, int], ...]      # (row, col) members
    bbox: Tuple[int, int, int, int]
    mean: Color

    @property
    def area(self) -> int:
        return len(self.patches)

    def center(self) -> Tuple[float, float]:
        """Region centroid in PATCH space, as ``(col, row)`` (x, y order)."""
        cols = sum(c for _r, c in self.patches) / len(self.patches)
        rows = sum(r for r, _c in self.patches) / len(self.patches)
        return (cols, rows)

    def to_dict(self) -> dict:
        return {"label": self.label, "area": self.area, "bbox": list(self.bbox),
                "mean": [round(c, 4) for c in self.mean],
                "center": [round(v, 4) for v in self.center()]}


@dataclass
class RegionMap:
    """The grouping of a patch grid: per-patch labels + the region objects."""

    labels: List[List[int]]                   # labels[row][col], relabelled 0..count-1
    regions: List[Region]
    rows: int
    cols: int

    @property
    def count(self) -> int:
        return len(self.regions)

    def label_at(self, row: int, col: int) -> int:
        return self.labels[row][col]

    def region_at(self, row: int, col: int) -> Region:
        return self.regions[self.labels[row][col]]

    def compression(self) -> float:
        """Regions / patches — ShowUI's whole point: how many tokens we saved.

        1.0 means every patch is its own region (nothing merged); a small value
        means the frame collapsed to a few flat blobs, which is the common case
        for a CAD viewport dominated by background and a couple of shaded faces.
        """
        total = self.rows * self.cols
        return (self.count / total) if total else 1.0

    def to_dict(self) -> dict:
        return {"rows": self.rows, "cols": self.cols, "count": self.count,
                "compression": round(self.compression(), 4),
                "regions": [r.to_dict() for r in self.regions]}


def build_regions(grid: Grid, threshold: float = 12.0) -> RegionMap:
    """Group adjacent, colour-similar patches into connected regions.

    ``grid[row][col]`` is a patch colour ``(r, g, b)`` in 0..255. Two 4-neighbours
    (right and below — the same two edges ShowUI compares) are unioned iff their
    colour distance is ``< threshold``. A LARGER threshold yields sparser, larger
    regions; a smaller one yields denser, finer regions — precisely ShowUI's
    ``uigraph_diff`` knob. Labels are then compacted to ``0..count-1`` in raster
    order of first appearance, the deterministic stand-in for ShowUI's
    ``LabelEncoder``.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows == 0 or cols == 0:
        return RegionMap(labels=[], regions=[], rows=0, cols=0)
    for r, line in enumerate(grid):
        if len(line) != cols:
            raise ValueError("grid is ragged: row %d has %d cols, expected %d"
                             % (r, len(line), cols))

    def idx(r: int, c: int) -> int:
        return r * cols + c

    uf = UnionFind(rows * cols)
    for r in range(rows):
        for c in range(cols):
            here = grid[r][c]
            if c + 1 < cols and patch_distance(here, grid[r][c + 1]) < threshold:
                uf.union(idx(r, c), idx(r, c + 1))
            if r + 1 < rows and patch_distance(here, grid[r + 1][c]) < threshold:
                uf.union(idx(r, c), idx(r + 1, c))

    # Compact roots to consecutive labels in raster order (LabelEncoder analogue).
    relabel: Dict[int, int] = {}
    labels = [[0] * cols for _ in range(rows)]
    members: Dict[int, List[Tuple[int, int]]] = {}
    sums: Dict[int, List[float]] = {}
    for r in range(rows):
        for c in range(cols):
            root = uf.find(idx(r, c))
            if root not in relabel:
                relabel[root] = len(relabel)
                members[root] = []
                sums[root] = [0.0, 0.0, 0.0]
            lab = relabel[root]
            labels[r][c] = lab
            members[root].append((r, c))
            col = grid[r][c]
            sums[root][0] += col[0]
            sums[root][1] += col[1]
            sums[root][2] += col[2]

    regions: List[Region] = [None] * len(relabel)  # type: ignore[list-item]
    for root, lab in relabel.items():
        pts = members[root]
        n = len(pts)
        rr = [p[0] for p in pts]
        cc = [p[1] for p in pts]
        regions[lab] = Region(
            label=lab, patches=tuple(pts),
            bbox=(min(cc), min(rr), max(cc), max(rr)),
            mean=(sums[root][0] / n, sums[root][1] / n, sums[root][2] / n))
    return RegionMap(labels=labels, regions=regions, rows=rows, cols=cols)


def patchify(pixels: Sequence[Sequence[Color]], patch: int = 14) -> List[List[Color]]:
    """Downsample a pixel image into a grid of patch-mean colours.

    ``pixels[row][col] = (r, g, b)``. Each ``patch x patch`` block becomes one
    grid cell holding the block's mean colour — the same "one vector per patch"
    reduction ShowUI feeds its union-find, done here with stdlib arithmetic so the
    grouper can run on a real screenshot (loaded to nested lists) with no numpy.
    A trailing partial block is included and averaged over its actual pixels.
    """
    if patch < 1:
        raise ValueError("patch must be >= 1")
    h = len(pixels)
    w = len(pixels[0]) if h else 0
    out: List[List[Color]] = []
    for r0 in range(0, h, patch):
        row_cells: List[Color] = []
        for c0 in range(0, w, patch):
            acc = [0.0, 0.0, 0.0]
            n = 0
            for r in range(r0, min(r0 + patch, h)):
                for c in range(c0, min(c0 + patch, w)):
                    px = pixels[r][c]
                    acc[0] += px[0]
                    acc[1] += px[1]
                    acc[2] += px[2]
                    n += 1
            row_cells.append((acc[0] / n, acc[1] / n, acc[2] / n))
        out.append(row_cells)
    return out


def largest_regions(rmap: RegionMap, k: int = 5,
                    min_area: int = 1) -> List[Region]:
    """The ``k`` biggest regions (ties broken by raster-order label).

    A viewport's largest region is almost always the background; the next few are
    the shaded faces of the solid. Returning them ordered by area gives a caller a
    ready "candidate surfaces, biggest first" list to aim probes at — the same use
    ShowUI's kept tokens serve, one representative per flat area.
    """
    pool = [r for r in rmap.regions if r.area >= min_area]
    pool.sort(key=lambda r: (-r.area, r.label))
    return pool[:k]
