"""Split-and-remerge legolization: voxel grid -> standard-brick layout.

Distilled from Pun, Deng, Liu et al., *Generating Physically Stable and
Buildable LEGO Designs from Text* (LEGOGPT), Section 3 ("Mesh-to-LEGO") and
appendix A ("StableText2Lego Details").

The companion generic-brick modules ("...Buildable Brick Structures from Text",
``brick_*.py``) generate bricks *directly* from text and never convert a
voxel occupancy grid into a brick layout.  This module fills that gap with the
paper's *legolization* step: given a solid voxelization (occupied cells on a
grid), tile every layer with parts drawn from the standard LEGO library.

The paper's variant of split-and-remerge does **not** initialize with 1x1
bricks and randomly merge; instead it "directly places bricks to fill all the
voxels, prioritizing 1) larger bricks and 2) bricks that connect multiple other
bricks" (appendix A).  We reproduce that priority exactly:

* bricks are 1 unit tall, so every z-layer is tiled independently;
* within a layer, cells are covered greedily, preferring the largest-area
  library footprint that fits, breaking ties toward the footprint that spans
  the most distinct bricks on the layer below (maximizing interlock);
* seeded permutation of equal-priority candidates yields multiple distinct but
  shape-preserving layouts for the same object (the paper's data augmentation).

Everything is stdlib-only and deterministic given a seed.  No stability or
collision analysis lives here -- the output is a plain list of library bricks.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from fabrication.lego_brick_library import STANDARD_FOOTPRINTS, Brick

Cell = Tuple[int, int, int]

# Oriented footprints (both rotations of every non-square part), largest first.
_ORIENTED: Tuple[Tuple[int, int], ...] = tuple(
    sorted(
        {(a, b) for a, b in STANDARD_FOOTPRINTS}
        | {(b, a) for a, b in STANDARD_FOOTPRINTS},
        key=lambda hw: (-hw[0] * hw[1], hw[0], hw[1]),
    )
)


def _layer_cells(voxels: Iterable[Cell]) -> Dict[int, Set[Tuple[int, int]]]:
    """Group occupied cells by z into per-layer (x, y) masks."""
    layers: Dict[int, Set[Tuple[int, int]]] = {}
    for x, y, z in voxels:
        layers.setdefault(z, set()).add((x, y))
    return layers


def _fits(x: int, y: int, h: int, w: int,
          mask: Set[Tuple[int, int]], used: Set[Tuple[int, int]]) -> bool:
    for dx in range(h):
        for dy in range(w):
            c = (x + dx, y + dy)
            if c not in mask or c in used:
                return False
    return True


def _connectivity(x: int, y: int, h: int, w: int,
                  below: Dict[Tuple[int, int], int]) -> int:
    """Number of distinct lower-layer brick ids the footprint straddles."""
    ids: Set[int] = set()
    for dx in range(h):
        for dy in range(w):
            bid = below.get((x + dx, y + dy))
            if bid is not None:
                ids.add(bid)
    return len(ids)


def _legolize_layer(z: int,
                    mask: Set[Tuple[int, int]],
                    below: Dict[Tuple[int, int], int],
                    rng: Optional[random.Random],
                    next_id: int) -> Tuple[List[Brick], Dict[Tuple[int, int], int], int]:
    """Tile one layer; return placed bricks, this layer's cell->id map, next id."""
    used: Set[Tuple[int, int]] = set()
    bricks: List[Brick] = []
    assignment: Dict[Tuple[int, int], int] = {}
    # Raster-scan the anchor cells so placement is deterministic and complete.
    for (x, y) in sorted(mask):
        if (x, y) in used:
            continue
        # Candidate oriented footprints that fit at this anchor.
        cands = [(h, w) for (h, w) in _ORIENTED if _fits(x, y, h, w, mask, used)]
        # A 1x1 always fits (the anchor cell itself), so cands is never empty.
        # Priority: larger area, then more interlock with the layer below.
        def _key(hw: Tuple[int, int]) -> Tuple[int, int]:
            h, w = hw
            return (h * w, _connectivity(x, y, h, w, below))

        best = max(_key(hw) for hw in cands)
        top = [hw for hw in cands if _key(hw) == best]
        if rng is not None and len(top) > 1:
            choice = top[rng.randrange(len(top))]
        else:
            choice = top[0]
        h, w = choice
        for dx in range(h):
            for dy in range(w):
                used.add((x + dx, y + dy))
                assignment[(x + dx, y + dy)] = next_id
        bricks.append(Brick(h=h, w=w, x=x, y=y, z=z))
        next_id += 1
    return bricks, assignment, next_id


def legolize(voxels: Iterable[Cell], seed: Optional[int] = None) -> List[Brick]:
    """Convert an occupancy voxel set into a standard-brick layout.

    Layers are processed bottom-to-top so the interlock (connectivity) tie-break
    can see the layer below.  With ``seed=None`` the result is a fully
    deterministic largest-first tiling; passing an integer seed permutes
    equal-priority candidates to produce a distinct shape-preserving variant.
    """
    layers = _layer_cells(voxels)
    rng = random.Random(seed) if seed is not None else None
    below: Dict[Tuple[int, int], int] = {}
    out: List[Brick] = []
    next_id = 0
    for z in sorted(layers):
        bricks, assignment, next_id = _legolize_layer(
            z, layers[z], below, rng, next_id
        )
        out.extend(bricks)
        below = assignment
    return out


def legolize_variants(voxels: Iterable[Cell],
                      seeds: Sequence[int]) -> List[List[Brick]]:
    """Generate one layout per seed (the paper's structural augmentation)."""
    voxset = list(voxels)
    return [legolize(voxset, seed=s) for s in seeds]


def covers_exactly(bricks: Sequence[Brick], voxels: Iterable[Cell]) -> bool:
    """True iff *bricks* tile exactly *voxels* with no overlap or spill."""
    target = set(voxels)
    covered: Set[Cell] = set()
    for b in bricks:
        for c in b.cells():
            if c in covered:
                return False  # overlap
            covered.add(c)
    return covered == target
