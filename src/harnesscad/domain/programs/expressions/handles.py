"""Parametric handle grids for CAD primitives (paper Section 4 + Appendix Table 4).

To let a user *retrieve* the parametric definition of a point on an object, the
paper decorates every primitive with a grid of named "handles". Each handle's
position is expressed symbolically, relative to the primitive's centre, in terms
of the primitive's size parameters — so that when the size is a variable the
handle position is itself a parametric (linear) expression. From the paper:

* **Cube (27 points)** — a 3x3x3 grid: 1 centre, 8 corners, 6 face-centres,
  12 edge-midpoints.
* **Sphere (27 points)** — a boundary cube of side = diameter, handles placed
  with the cube distribution.
* **Cylinder (27 points)** — a boundary cuboid whose bottom/top square faces are
  sized by the bottom/top diameters (``d1``/``d2``, i.e. ``2*r1``/``2*r2``) and
  whose height is ``h``; handles placed with the cube distribution (so the
  cross-section width varies with z, matching a truncated cone).
* **Square (9 points)** — a 3x3 grid: 1 centre, 4 corners, 4 edge-midpoints.
* **Circle (9 points)** — a boundary square of side = diameter: centre plus 4
  axis extremes.

Every handle offset is a triple of :class:`~programs.paramgeom_linform.LinearForm`
values (dx, dy, dz), so ``size`` parameters may be given as numbers *or* as
variable names. These offsets are exactly the "definition of the position of the
handle relative to the node's centre" that the paper's Position feature adds to
the accumulated translations (see :mod:`programs.paramgeom_position`).

Pure stdlib, deterministic.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Tuple, Union

from harnesscad.domain.programs.expressions.linear_form import LinearForm, Number

# A handle offset is (dx, dy, dz), each an affine form in the size parameters.
Offset = Tuple[LinearForm, LinearForm, LinearForm]

# A size parameter may be a literal number or a variable name.
Dim = Union[Number, str]

_HALF = Fraction(1, 2)


def _half_extent(dim: "Dim | LinearForm") -> LinearForm:
    """Half of a size parameter as a LinearForm (``dim/2``).

    Accepts a variable name, a literal number, or an already-built LinearForm
    (used by the cylinder, whose face diameters are themselves expressions).
    """
    if isinstance(dim, LinearForm):
        return dim.scaled(_HALF)
    if isinstance(dim, str):
        return LinearForm.var(dim, _HALF)
    return LinearForm.const(dim).scaled(_HALF)


def _axis_offsets(dim: "Dim | LinearForm") -> Tuple[LinearForm, LinearForm, LinearForm]:
    """The three grid coordinates along one centred axis: (-dim/2, 0, +dim/2)."""
    half = _half_extent(dim)
    return (-half, LinearForm.const(0), half)


# Symbolic index for the 3 positions along an axis.
_LOW, _MID, _HIGH = 0, 1, 2
_NAME_BY_INDEX = {_LOW: "min", _MID: "mid", _HIGH: "max"}


def _grid3d(sx: Dim, sy: Dim, sz: Dim) -> Dict[str, Offset]:
    """Full 3x3x3 = 27-point centred grid keyed by symbolic names.

    Keys look like ``x{min,mid,max}_y{...}_z{...}`` (the centre is ``center``).
    Corners, face-centres, edge-midpoints and the centre are all produced; each
    can be recognised by how many of its axes are non-central.
    """
    xs = _axis_offsets(sx)
    ys = _axis_offsets(sy)
    zs = _axis_offsets(sz)
    handles: Dict[str, Offset] = {}
    for ix in (_LOW, _MID, _HIGH):
        for iy in (_LOW, _MID, _HIGH):
            for iz in (_LOW, _MID, _HIGH):
                if ix == _MID and iy == _MID and iz == _MID:
                    name = "center"
                else:
                    name = (
                        f"x{_NAME_BY_INDEX[ix]}_"
                        f"y{_NAME_BY_INDEX[iy]}_"
                        f"z{_NAME_BY_INDEX[iz]}"
                    )
                handles[name] = (xs[ix], ys[iy], zs[iz])
    return handles


def _grid2d(sx: Dim, sy: Dim) -> Dict[str, Offset]:
    """3x3 = 9-point centred grid in the z=0 plane."""
    xs = _axis_offsets(sx)
    ys = _axis_offsets(sy)
    zero = LinearForm.const(0)
    handles: Dict[str, Offset] = {}
    for ix in (_LOW, _MID, _HIGH):
        for iy in (_LOW, _MID, _HIGH):
            if ix == _MID and iy == _MID:
                name = "center"
            else:
                name = f"x{_NAME_BY_INDEX[ix]}_y{_NAME_BY_INDEX[iy]}"
            handles[name] = (xs[ix], ys[iy], zero)
    return handles


def cube_handles(size_x: Dim, size_y: Dim, size_z: Dim) -> Dict[str, Offset]:
    """27 handles for a centred cube of the given (parametric) side lengths."""
    return _grid3d(size_x, size_y, size_z)


def sphere_handles(diameter: Dim) -> Dict[str, Offset]:
    """27 handles for a sphere via its boundary cube (side = diameter)."""
    return _grid3d(diameter, diameter, diameter)


def sphere_handles_from_radius(radius: Dim) -> Dict[str, Offset]:
    """27 handles for a sphere given its radius (diameter = 2*radius)."""
    return _grid3d(*(_double(radius),) * 3)


def cylinder_handles(r1: Dim, r2: Dim, h: Dim) -> Dict[str, Offset]:
    """27 handles for a (possibly truncated-cone) cylinder.

    The bounding cuboid's bottom face has side ``2*r1``, the top face ``2*r2``,
    and height ``h``. The mid-plane uses the mean radius ``r1 + r2`` (i.e. the
    average diameter), matching a linear taper. Handles are centred on z.
    """
    d1 = _double(r1)  # bottom diameter
    d2 = _double(r2)  # top diameter
    dmid = _sum(r1, r2)  # (r1 + r2) == mean diameter
    xs_by_z = {_LOW: _axis_offsets(d1), _MID: _axis_offsets(dmid), _HIGH: _axis_offsets(d2)}
    ys_by_z = xs_by_z
    zs = _axis_offsets(h)
    handles: Dict[str, Offset] = {}
    for ix in (_LOW, _MID, _HIGH):
        for iy in (_LOW, _MID, _HIGH):
            for iz in (_LOW, _MID, _HIGH):
                if ix == _MID and iy == _MID and iz == _MID:
                    name = "center"
                else:
                    name = (
                        f"x{_NAME_BY_INDEX[ix]}_"
                        f"y{_NAME_BY_INDEX[iy]}_"
                        f"z{_NAME_BY_INDEX[iz]}"
                    )
                handles[name] = (xs_by_z[iz][ix], ys_by_z[iz][iy], zs[iz])
    return handles


def square_handles(size_x: Dim, size_y: Dim) -> Dict[str, Offset]:
    """9 handles for a centred square/rectangle in the z=0 plane."""
    return _grid2d(size_x, size_y)


def circle_handles(diameter: Dim) -> Dict[str, Offset]:
    """5 handles for a circle: centre plus the 4 axis extremes (paper: 9-pt grid,
    but a circle keeps only centre + 4 axis-aligned boundary points)."""
    xs = _axis_offsets(diameter)
    ys = _axis_offsets(diameter)
    zero = LinearForm.const(0)
    return {
        "center": (LinearForm.const(0), LinearForm.const(0), zero),
        "xmax": (xs[_HIGH], LinearForm.const(0), zero),
        "xmin": (xs[_LOW], LinearForm.const(0), zero),
        "ymax": (LinearForm.const(0), ys[_HIGH], zero),
        "ymin": (LinearForm.const(0), ys[_LOW], zero),
    }


def handle_role(name: str) -> str:
    """Classify a 3D grid handle name as center/face/edge/corner (2D: center/edge/corner)."""
    if name == "center":
        return "center"
    axes = name.split("_")
    non_central = sum(1 for a in axes if not a.endswith("mid"))
    if len(axes) == 3:
        return {0: "center", 1: "face", 2: "edge", 3: "corner"}[non_central]
    if len(axes) == 2:
        return {0: "center", 1: "edge", 2: "corner"}[non_central]
    return "boundary"


# -- small numeric/symbolic helpers -----------------------------------------


def _double(dim: Dim) -> LinearForm:
    if isinstance(dim, str):
        return LinearForm.var(dim, 2)
    return LinearForm.const(dim).scaled(2)


def _sum(a: Dim, b: Dim) -> LinearForm:
    fa = LinearForm.var(a) if isinstance(a, str) else LinearForm.const(a)
    fb = LinearForm.var(b) if isinstance(b, str) else LinearForm.const(b)
    return fa + fb
