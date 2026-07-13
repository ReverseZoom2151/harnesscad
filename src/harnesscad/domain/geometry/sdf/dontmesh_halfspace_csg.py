"""Surface-based Constructive Solid Geometry via half-spaces and cells.

From *Don't Mesh with Me: Generating Constructive Solid Geometry Instead of
Meshes by Fine-Tuning a Code-Generation LLM* (Mews et al., 2024). The paper
represents mechanical parts not as meshes but as **surface-based CSG**: a model
is decomposed (via GEOUNED) into a sequence of *cells* (half-spaces), where each
cell is the intersection of a set of signed surfaces -- *hyperplanes* and
*cylinders* -- and the whole model is the union of its cells. This is the
OpenMC-style constructive geometry the paper's generated Python scripts build
(Sec. 3.1, "Code Generation"). To keep the proof-of-concept tractable the
authors restrict themselves to planes and cylinders that are perpendicular or
parallel to the base coordinate axes (Sec. 4.1); we support that family exactly
but also allow general planes.

This module is the deterministic geometry core the paper's learned model sits on
top of. It provides:

  * ``Plane`` / ``Cylinder`` surfaces with a signed evaluation function
    (negative inside / on the "minus" side, matching OpenMC's half-space sense);
  * a ``HalfSpace`` = (surface, sense) and a ``Cell`` = intersection of
    half-spaces, with exact point-membership;
  * a ``CSGModel`` = union of cells, with point-membership;
  * deterministic grid occupancy sampling, an axis-aligned bounding-box
    estimate, a volume estimate and a voxel **IoU** between two models (the
    paper's plausibility comparison against ground truth);
  * a **cell validity** (finite-volume / bounded) test -- the paper deems a
    cell valid only if "its volume is finite, meaning it is not open on any of
    its sides" (Sec. 3.1);
  * **overlap** detection between two cells (the paper's "No Overlapping Cells"
    metric, Tab. 1).

Learned parts -- the DeepSeek-Coder fine-tuning, GPT-4o annotation and the
GEOUNED BREP->CSG decomposition -- are external and not modelled here.

Pure stdlib; deterministic (seeded sampling only, no wall clock).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Box = Tuple[Vec3, Vec3]  # (min corner, max corner)

_EPS = 1e-9


@dataclass(frozen=True)
class Plane:
    """A hyperplane surface ``a*x + b*y + c*z - d``.

    ``evaluate`` returns the signed value; points with a negative value are on
    the surface's "minus" side (the OpenMC negative half-space).
    """

    a: float
    b: float
    c: float
    d: float
    name: str = ""

    def evaluate(self, p: Vec3) -> float:
        return self.a * p[0] + self.b * p[1] + self.c * p[2] - self.d

    @staticmethod
    def axis_aligned(axis: str, offset: float, name: str = "") -> "Plane":
        """A plane perpendicular to ``axis`` (``'x'``/``'y'``/``'z'``) at ``offset``."""
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        coeff = [0.0, 0.0, 0.0]
        coeff[idx] = 1.0
        return Plane(coeff[0], coeff[1], coeff[2], float(offset), name)

    def kind(self) -> str:
        return "plane"

    def params(self) -> Tuple[float, ...]:
        return (self.a, self.b, self.c, self.d)


@dataclass(frozen=True)
class Cylinder:
    """An axis-aligned infinite cylinder.

    ``axis`` is ``'x'``/``'y'``/``'z'``; ``u``/``v`` are the centre coordinates
    in the two perpendicular axes (in ascending axis order); ``radius`` > 0.
    ``evaluate`` returns ``perp_dist^2 - radius^2`` so points *inside* the
    cylinder are negative -- the same negative-half-space convention as planes.
    """

    axis: str
    u: float
    v: float
    radius: float
    name: str = ""

    def _perp_axes(self) -> Tuple[int, int]:
        return {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[self.axis]

    def evaluate(self, p: Vec3) -> float:
        i, j = self._perp_axes()
        du = p[i] - self.u
        dv = p[j] - self.v
        return du * du + dv * dv - self.radius * self.radius

    def kind(self) -> str:
        return "cylinder"

    def params(self) -> Tuple[float, ...]:
        return (self.u, self.v, self.radius)


Surface = object  # Plane | Cylinder (both expose evaluate/kind/params/name)


@dataclass(frozen=True)
class HalfSpace:
    """A signed surface. ``sense`` is ``-1`` (negative half-space: the side where
    ``surface.evaluate(p) <= 0``, e.g. inside a cylinder) or ``+1`` (positive
    half-space: ``surface.evaluate(p) >= 0``). This matches OpenMC's ``-surface``
    / ``+surface`` region operators.
    """

    surface: Surface
    sense: int  # -1 or +1

    def contains(self, p: Vec3, eps: float = _EPS) -> bool:
        val = self.surface.evaluate(p)
        if self.sense < 0:
            return val <= eps
        return val >= -eps


@dataclass(frozen=True)
class Cell:
    """A cell = intersection of half-spaces (an OpenMC region)."""

    half_spaces: Tuple[HalfSpace, ...]
    name: str = ""

    def contains(self, p: Vec3, eps: float = _EPS) -> bool:
        for hs in self.half_spaces:
            if not hs.contains(p, eps):
                return False
        return True

    def surfaces(self) -> List[Surface]:
        return [hs.surface for hs in self.half_spaces]


@dataclass(frozen=True)
class CSGModel:
    """A model = union of cells."""

    cells: Tuple[Cell, ...]

    def contains(self, p: Vec3, eps: float = _EPS) -> bool:
        for cell in self.cells:
            if cell.contains(p, eps):
                return True
        return False


# --------------------------------------------------------------------------
# Deterministic sampling helpers.
# --------------------------------------------------------------------------

def grid_points(box: Box, res: int) -> List[Vec3]:
    """Cell-centre sample points of a ``res``^3 regular grid over ``box``."""
    if res < 1:
        raise ValueError("res must be >= 1")
    (x0, y0, z0), (x1, y1, z1) = box
    pts: List[Vec3] = []
    for iz in range(res):
        fz = (iz + 0.5) / res
        z = z0 + fz * (z1 - z0)
        for iy in range(res):
            fy = (iy + 0.5) / res
            y = y0 + fy * (y1 - y0)
            for ix in range(res):
                fx = (ix + 0.5) / res
                x = x0 + fx * (x1 - x0)
                pts.append((x, y, z))
    return pts


def occupancy(model: CSGModel, box: Box, res: int) -> List[bool]:
    """Inside/outside mask over a ``res``^3 grid (x fastest)."""
    return [model.contains(p) for p in grid_points(box, res)]


def volume_fraction(model: CSGModel, box: Box, res: int) -> float:
    mask = occupancy(model, box, res)
    return sum(1 for m in mask if m) / len(mask)


def iou(a: CSGModel, b: CSGModel, box: Box, res: int) -> float:
    """Voxel Intersection-over-Union of two models over a shared grid.

    The paper compares a generated model against ground truth by occupancy; IoU
    is the natural symmetric agreement score. Returns ``1.0`` when both models
    are empty over the grid.
    """
    ma = occupancy(a, box, res)
    mb = occupancy(b, box, res)
    inter = sum(1 for x, y in zip(ma, mb) if x and y)
    union = sum(1 for x, y in zip(ma, mb) if x or y)
    if union == 0:
        return 1.0
    return inter / union


def bounding_box(model: CSGModel, probe: Box, res: int, pad: float = 0.0) -> Optional[Box]:
    """Estimate the axis-aligned bounding box of the occupied region by sampling
    ``probe`` at ``res``^3. Returns ``None`` if nothing is occupied.
    """
    pts = grid_points(probe, res)
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for p in pts:
        if model.contains(p):
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
    if not xs:
        return None
    lo = (min(xs) - pad, min(ys) - pad, min(zs) - pad)
    hi = (max(xs) + pad, max(ys) + pad, max(zs) + pad)
    return (lo, hi)


def random_points(box: Box, n: int, seed: int) -> List[Vec3]:
    """``n`` deterministic uniform points in ``box`` from ``random.Random(seed)``."""
    rng = random.Random(seed)
    (x0, y0, z0), (x1, y1, z1) = box
    out: List[Vec3] = []
    for _ in range(n):
        out.append(
            (
                x0 + rng.random() * (x1 - x0),
                y0 + rng.random() * (y1 - y0),
                z0 + rng.random() * (z1 - z0),
            )
        )
    return out


# --------------------------------------------------------------------------
# Cell validity and overlap (Tab. 1 plausibility parameters).
# --------------------------------------------------------------------------

def cell_is_bounded(cell: Cell, probe: Box, res: int) -> bool:
    """Deterministic finite-volume test.

    The paper: "A cell is considered valid if its volume is finite, meaning it
    is not open on any of its sides." We sample the cell over ``probe`` and
    declare it *unbounded* if any interior sample lies on the outer shell of the
    probe box -- i.e. the cell keeps extending past the domain. A cell that is
    empty over the probe is treated as not bounded (degenerate).
    """
    (x0, y0, z0), (x1, y1, z1) = probe
    pts = grid_points(probe, res)
    any_inside = False
    # Shell tolerance: half a voxel.
    tx = 0.5 * (x1 - x0) / res
    ty = 0.5 * (y1 - y0) / res
    tz = 0.5 * (z1 - z0) / res
    for p in pts:
        if cell.contains(p):
            any_inside = True
            on_shell = (
                p[0] <= x0 + tx
                or p[0] >= x1 - tx
                or p[1] <= y0 + ty
                or p[1] >= y1 - ty
                or p[2] <= z0 + tz
                or p[2] >= z1 - tz
            )
            if on_shell:
                return False
    return any_inside


def cells_overlap(a: Cell, b: Cell, box: Box, res: int) -> bool:
    """True if ``a`` and ``b`` share interior volume (a positive-measure overlap)
    detected by grid sampling over ``box``."""
    for p in grid_points(box, res):
        if a.contains(p) and b.contains(p):
            return True
    return False


def any_cells_overlap(model: CSGModel, box: Box, res: int) -> bool:
    """True if any pair of the model's cells overlaps (paper's overlap metric)."""
    cells = model.cells
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            if cells_overlap(cells[i], cells[j], box, res):
                return True
    return False


def all_cells_bounded(model: CSGModel, probe: Box, res: int) -> bool:
    return all(cell_is_bounded(c, probe, res) for c in model.cells)


# --------------------------------------------------------------------------
# Convenience constructors for the axis-aligned family used in the paper.
# --------------------------------------------------------------------------

def axis_box_cell(lo: Vec3, hi: Vec3, name: str = "") -> Cell:
    """An axis-aligned box cell from six planes (the paper's canonical cuboid)."""
    if not (hi[0] > lo[0] and hi[1] > lo[1] and hi[2] > lo[2]):
        raise ValueError("hi must exceed lo on every axis")
    hs: List[HalfSpace] = []
    for axis, l, h in zip("xyz", lo, hi):
        # x >= l  ->  plane (x - l), positive side (sense +1)
        hs.append(HalfSpace(Plane.axis_aligned(axis, l), +1))
        # x <= h  ->  plane (x - h), negative side (sense -1)
        hs.append(HalfSpace(Plane.axis_aligned(axis, h), -1))
    return Cell(tuple(hs), name)


def axis_cylinder_cell(
    axis: str, u: float, v: float, radius: float, lo: float, hi: float, name: str = ""
) -> Cell:
    """A finite axis-aligned cylinder cell: an infinite cylinder capped by two
    perpendicular planes."""
    cyl = Cylinder(axis, u, v, radius)
    hs = [
        HalfSpace(cyl, -1),  # inside the cylinder
        HalfSpace(Plane.axis_aligned(axis, lo), +1),
        HalfSpace(Plane.axis_aligned(axis, hi), -1),
    ]
    return Cell(tuple(hs), name)
