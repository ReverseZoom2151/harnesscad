"""Deterministic plate-stack building assembly (Gaudi-backend style).

Gaudi describes a building as an ordered list of plate dictionaries (see
:mod:`spec.gaudi_plate_spec`) and realises it in Blender: for each plate it
builds a 2D outline, extrudes it by ``thickness``, centres the result in X/Y on
its bounding box, rotates it about Z, and offsets it by ``position``
(``template.py:create_shapes_from_list`` + ``center_object_xy``).  All of that
placement logic is pure geometry; only the ``bpy`` mesh construction, the render
and the LLM live outside.

This module reimplements the deterministic realisation:

  * :func:`plate_outline` turns one plate into its base 2D loop --
    ``vertex`` uses the given polygon, ``parametric`` samples the ``x(t)``/``y(t)``
    formulas (via :mod:`geometry.gaudi_parametric_profile`), and ``mixed`` treats
    the control ``vertices`` as a *closed Catmull-Rom* spline (a deterministic
    stand-in for Blender's AUTO-handle Bezier fill);
  * :func:`center_xy` reproduces ``center_object_xy`` (shift so the AABB centre
    sits on the origin in X and Y, leaving Z alone);
  * :func:`rotate_z` rotates a loop about the origin by degrees;
  * :func:`assemble_building` places every plate into 3D: it centres, rotates and
    offsets each outline, assigns a bottom/top Z, and -- when ``auto_stack`` is
    set -- stacks each plate on top of the previous one by cumulative thickness
    (the "position plates on top of one another along Z" guideline the upstream
    prompt asks the model to follow but never enforces).  The result is a
    :class:`Building` of :class:`PlacedPlate` rings plus the overall height and
    axis-aligned bounding box.

Deterministic: fixed sample grids and pure arithmetic; no clock, no randomness,
no I/O.

Public API
----------
``plate_outline(plate, bezier_samples=12) -> list[(x, y)]``
``center_xy(points) -> list``
``rotate_z(points, degrees) -> list``
``catmull_rom_closed(control, samples_per_segment=12) -> list``
``PlacedPlate`` / ``Building``
``assemble_building(plates, auto_stack=True, bezier_samples=12) -> Building``
``BuildingAssemblyError``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from harnesscad.domain.geometry.sketch.gaudi_parametric_profile import (
    dedupe_points,
    is_degenerate,
    sample_curve,
)
from harnesscad.domain.spec.gaudi_plate_spec import normalize_plate, validate_building

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]


class BuildingAssemblyError(ValueError):
    """Raised when a valid-looking plate cannot be realised into geometry."""


def catmull_rom_closed(
    control: Sequence[Point2], samples_per_segment: int = 12
) -> List[Point2]:
    """Tessellate a closed centripetal-uniform Catmull-Rom spline.

    Produces a smooth closed loop passing through every control point -- a
    deterministic replacement for Blender's cyclic AUTO-handle Bezier.
    """
    n = len(control)
    if n < 3:
        return [(p[0], p[1]) for p in control]
    if samples_per_segment < 1:
        raise BuildingAssemblyError("samples_per_segment must be >= 1")
    pts: List[Point2] = []
    for i in range(n):
        p0 = control[(i - 1) % n]
        p1 = control[i]
        p2 = control[(i + 1) % n]
        p3 = control[(i + 2) % n]
        for s in range(samples_per_segment):
            u = s / samples_per_segment
            pts.append(_catmull_rom_point(p0, p1, p2, p3, u))
    return pts


def _catmull_rom_point(
    p0: Point2, p1: Point2, p2: Point2, p3: Point2, u: float
) -> Point2:
    u2 = u * u
    u3 = u2 * u
    # Uniform Catmull-Rom basis (tension 0.5).
    def _axis(a0, a1, a2, a3):
        return 0.5 * (
            (2 * a1)
            + (-a0 + a2) * u
            + (2 * a0 - 5 * a1 + 4 * a2 - a3) * u2
            + (-a0 + 3 * a1 - 3 * a2 + a3) * u3
        )

    return (_axis(p0[0], p1[0], p2[0], p3[0]), _axis(p0[1], p1[1], p2[1], p3[1]))


def plate_outline(plate: dict, bezier_samples: int = 12) -> List[Point2]:
    """Return the base 2D outline for a plate, by category."""
    norm = normalize_plate(plate)
    category = norm["category"]
    if category == "vertex":
        pts = [(float(x), float(y)) for x, y in norm["vertices"]]
    elif category == "parametric":
        formula = norm["formula"]
        start, end = norm["range"]
        pts = sample_curve(
            formula["x"], formula["y"], float(start), float(end), int(norm["steps"])
        )
    elif category == "mixed":
        control = [(float(x), float(y)) for x, y in norm["vertices"]]
        pts = catmull_rom_closed(control, bezier_samples)
    else:  # pragma: no cover - normalize_plate rejects unknown categories
        raise BuildingAssemblyError("unknown category '{0}'".format(category))

    pts = dedupe_points(pts)
    if is_degenerate(pts):
        raise BuildingAssemblyError(
            "plate '{0}' produces a degenerate (zero-area) outline".format(norm["name"])
        )
    return pts


def _bbox_xy(points: Sequence[Point2]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def center_xy(points: Sequence[Point2]) -> List[Point2]:
    """Shift the loop so its AABB centre lies on the origin in X and Y."""
    min_x, min_y, max_x, max_y = _bbox_xy(points)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    return [(p[0] - cx, p[1] - cy) for p in points]


def rotate_z(points: Sequence[Point2], degrees: float) -> List[Point2]:
    """Rotate a loop about the origin by ``degrees`` counter-clockwise."""
    theta = math.radians(degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return [(p[0] * cos_t - p[1] * sin_t, p[0] * sin_t + p[1] * cos_t) for p in points]


@dataclass
class PlacedPlate:
    """A plate realised in 3D as a bottom and a top ring of the extrusion."""

    name: str
    z_bottom: float
    z_top: float
    bottom_ring: List[Point3]
    top_ring: List[Point3]


@dataclass
class Building:
    """An assembled tower of placed plates with overall extent."""

    plates: List[PlacedPlate] = field(default_factory=list)

    @property
    def height(self) -> float:
        if not self.plates:
            return 0.0
        top = max(p.z_top for p in self.plates)
        bottom = min(p.z_bottom for p in self.plates)
        return top - bottom

    def bbox(self) -> Tuple[Point3, Point3]:
        """Axis-aligned bounding box ``((minx,miny,minz),(maxx,maxy,maxz))``."""
        if not self.plates:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        for plate in self.plates:
            for ring in (plate.bottom_ring, plate.top_ring):
                for x, y, z in ring:
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def assemble_building(
    plates: Sequence[dict], auto_stack: bool = True, bezier_samples: int = 12
) -> Building:
    """Realise a list of plate dicts into placed 3D geometry.

    With ``auto_stack`` each plate's bottom Z is the cumulative thickness of the
    plates before it (plus its own ``position.z``); without it, the bottom Z is
    just ``position.z`` (the raw Gaudi behaviour).
    """
    issues = validate_building(plates)
    if issues:
        raise BuildingAssemblyError("; ".join(issues))

    building = Building()
    cursor_z = 0.0
    for plate in plates:
        norm = normalize_plate(plate)
        outline = plate_outline(plate, bezier_samples)
        outline = center_xy(outline)
        outline = rotate_z(outline, norm["rotation"])
        pos = norm["position"]
        dx = float(pos["x"])
        dy = float(pos["y"])
        base_z = cursor_z + float(pos["z"]) if auto_stack else float(pos["z"])
        thickness = float(norm["thickness"])
        z_top = base_z + thickness

        bottom_ring = [(x + dx, y + dy, base_z) for x, y in outline]
        top_ring = [(x + dx, y + dy, z_top) for x, y in outline]
        building.plates.append(
            PlacedPlate(
                name=norm["name"],
                z_bottom=base_z,
                z_top=z_top,
                bottom_ring=bottom_ring,
                top_ring=top_ring,
            )
        )
        if auto_stack:
            cursor_z = z_top
    return building
