"""ShapeGraMM seeded reproducible massive-model instantiation.

From "ShapeGraMM: On the fly procedural generation of massive models for
real-time visualization" (Santos, Brazil, Raposo, 2023), Sections 4 and 5.

A ShapeGraMM stores *generation rules* rather than an explicit scene: "a small
set of generation rules can fully describe a massive CAD model" (Section 2.1).
The engine derives and interprets the grammar on-the-fly, per frame, using the
camera view -- only the production rules that contribute to the current frame
are expanded, producing batches of objects keyed by geometry id (Section 4).
The reported case study renders 11M+ objects (10 billion if fully expanded).

This module realizes that core idea deterministically, without storing the
model. A ``MassiveModel`` is a compact description of a spatial grid of cells,
each of which lazily instantiates objects. Two guarantees hold:

  * Seeded reproducibility -- ``cell_instances(cell, seed)`` is a pure function
    of the cell index and the base seed (via ``random.Random`` derived per
    cell), so any region can be regenerated identically without storing it.
  * On-the-fly per-view generation -- ``generate_view`` visits only cells whose
    scope is visible (using ``procedural.shapegramm_scope``) and chooses each
    cell's object count / geometry id by its LOD (using
    ``procedural.shapegramm_lod``), returning object batches grouped by
    geometry id ready to hand to a renderer.

Deterministic: all randomness flows through per-cell ``random.Random`` seeded
by ``(base_seed, cell_index)``.
"""

from dataclasses import dataclass
import random

from procedural.shapegramm_scope import (
    classify_scope, is_visible, OUTSIDE,
)
from procedural.shapegramm_lod import projected_size, lod_level


@dataclass(frozen=True)
class Instance:
    """A generated object: a geometry reference and its world transform.

    ``translation`` is the world position; ``geometry_id`` references a shared
    base geometry in the (external) renderer, matching ShapeGraMM's instance
    rule ``I("geometry")``.
    """

    geometry_id: str
    translation: tuple
    scale: float = 1.0


@dataclass(frozen=True)
class MassiveModel:
    """Compact description of a massive gridded model.

    * ``dims`` -- number of cells along (x, y, z).
    * ``cell_size`` -- world size of each cubic cell.
    * ``geometries`` -- ordered geometry ids available, indexed by LOD; the
      last one is reused for all coarser LODs.
    * ``max_objects`` -- object count generated per cell at LOD 0.
    * ``base_seed`` -- seed root for reproducible instantiation.
    """

    dims: tuple
    cell_size: float = 10.0
    geometries: tuple = ("mesh_full", "mesh_coarse", "mesh_coarsest")
    max_objects: int = 8
    base_seed: int = 0

    def cell_count(self):
        return self.dims[0] * self.dims[1] * self.dims[2]

    def cell_box(self, cell):
        """Return ``(box_min, box_max)`` for the integer cell ``(i, j, k)``."""
        i, j, k = cell
        s = self.cell_size
        lo = (i * s, j * s, k * s)
        hi = ((i + 1) * s, (j + 1) * s, (k + 1) * s)
        return lo, hi

    def iter_cells(self):
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                for k in range(self.dims[2]):
                    yield (i, j, k)

    def _cell_seed(self, cell):
        # Combine the base seed with the cell index into a stable 63-bit seed.
        i, j, k = cell
        dj, dk = self.dims[1], self.dims[2]
        linear = (i * dj + j) * dk + k
        # A simple deterministic mix; independent of platform hashing.
        return (self.base_seed * 1000003 + linear * 2654435761) & 0x7FFFFFFFFFFFFFFF

    def _geometry_for_lod(self, lod):
        idx = min(lod, len(self.geometries) - 1)
        return self.geometries[idx]

    def _object_count_for_lod(self, lod):
        # LOD 0 = full detail (max objects); each coarser level roughly halves
        # the count, never below 1 (a single representative object).
        count = self.max_objects >> lod
        return max(1, count)

    def cell_instances(self, cell, lod=0):
        """Instantiate the objects of ``cell`` at ``lod`` -- pure & reproducible.

        Regenerating a cell with the same model, cell index and LOD yields
        identical instances, so the massive model is never stored explicitly.
        """
        rng = random.Random(self._cell_seed(cell))
        lo, hi = self.cell_box(cell)
        geom = self._geometry_for_lod(lod)
        count = self._object_count_for_lod(lod)
        out = []
        for _ in range(count):
            x = rng.uniform(lo[0], hi[0])
            y = rng.uniform(lo[1], hi[1])
            z = rng.uniform(lo[2], hi[2])
            scale = round(rng.uniform(0.5, 1.5), 6)
            out.append(Instance(geom, (round(x, 6), round(y, 6), round(z, 6)), scale))
        return tuple(out)


def generate_view(model, planes, camera_pos, focal_length, thresholds):
    """Generate the visible instances of ``model`` for one camera view.

    Only cells whose scope is visible (``scope != OUTSIDE``) are expanded, and
    each visible cell's LOD is derived from its projected size. Returns
    ``(batches, stats)`` where ``batches`` maps ``geometry_id -> tuple of
    Instance`` (ready to be drawn as one batch) and ``stats`` reports counts.

    The result is a deterministic function of the model and camera parameters:
    the same inputs always produce identical batches (on-the-fly regeneration).
    """
    batches = {}
    stats = {
        "cells_total": model.cell_count(),
        "cells_visible": 0,
        "cells_culled": 0,
        "objects": 0,
        "lod_histogram": {},
    }
    for cell in model.iter_cells():
        lo, hi = model.cell_box(cell)
        scope = classify_scope(lo, hi, planes, camera_pos)
        if not is_visible(scope):
            stats["cells_culled"] += 1
            continue
        stats["cells_visible"] += 1
        size = projected_size(lo, hi, camera_pos, focal_length)
        lod = lod_level(size, thresholds)
        stats["lod_histogram"][lod] = stats["lod_histogram"].get(lod, 0) + 1
        for inst in model.cell_instances(cell, lod=lod):
            batches.setdefault(inst.geometry_id, []).append(inst)
            stats["objects"] += 1
    frozen = {k: tuple(v) for k, v in sorted(batches.items())}
    return frozen, stats


def full_object_count(model):
    """Total objects if the whole model were expanded at LOD 0 (never stored).

    Illustrates the compact representation: this can be astronomically larger
    than any single view's generated set.
    """
    return model.cell_count() * model.max_objects
