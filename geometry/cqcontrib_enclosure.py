"""Parametric enclosure / lid recipe, derived deterministically.

Reimplementation of the geometry *rules* behind the ``cadquery-contrib``
``Parametric_Enclosure.py`` and ``Remote_Enclosure.py`` recipes, with no CAD
kernel: a filleted rounded box, shelled to a wall thickness, screw posts at
an inset rectangle, a lid split off the top with a lip that sits inside the
body, and counterbored / countersunk screw holes through the lid.

What this module gives (all closed-form and exact):

* the fillet ORDERING rule ("weird geometry happens if we make the fillets in
  the wrong order": the larger radius must be filleted first);
* validity predicates for every radius / thickness / inset relation;
* screw-post centres, lid split height, lip footprint;
* exact volumes of the rounded-box shell, the posts and the lid, using the
  rounded-rectangle prism formula ``A = w*l - (4 - pi) r^2``.

Hole feature geometry itself lives in
``geometry.cqcontrib_hole_features``; this module only positions it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

__all__ = [
    "EnclosureError",
    "EnclosureSpec",
    "EnclosurePlan",
    "rounded_rect_area",
    "fillet_order",
    "validate_spec",
    "plan_enclosure",
]


class EnclosureError(ValueError):
    """Raised for a geometrically impossible enclosure specification."""


@dataclass(frozen=True)
class EnclosureSpec:
    """Parameters of the contrib enclosure recipe (millimetres, degrees)."""

    outer_width: float = 100.0
    outer_length: float = 150.0
    outer_height: float = 50.0
    thickness: float = 3.0
    side_radius: float = 10.0
    top_bottom_radius: float = 2.0
    screwpost_inset: float = 12.0
    screwpost_id: float = 4.0
    screwpost_od: float = 10.0
    lip_height: float = 1.0


@dataclass(frozen=True)
class EnclosurePlan:
    """Derived, deterministic geometry of an enclosure."""

    spec: EnclosureSpec
    inner_width: float
    inner_length: float
    inner_height: float
    inner_side_radius: float
    post_centers: Tuple[Tuple[float, float], ...]
    post_height: float
    lid_split_z: float
    lip_width: float
    lip_length: float
    outer_volume: float
    cavity_volume: float
    post_volume: float
    lid_volume: float
    fillet_order: Tuple[float, ...]

    @property
    def body_volume(self) -> float:
        """Material volume of the box body (shell + posts, lid excluded)."""
        return self.outer_volume - self.cavity_volume - self.lid_volume + self.post_volume


def rounded_rect_area(width: float, length: float, radius: float) -> float:
    """Area of a rectangle with four quarter-round corners of ``radius``."""
    if width <= 0.0 or length <= 0.0:
        raise EnclosureError("width and length must be positive")
    if radius < 0.0:
        raise EnclosureError("radius must be non-negative")
    if radius > min(width, length) / 2.0 + 1e-12:
        raise EnclosureError("corner radius exceeds half the smaller side")
    return width * length - (4.0 - math.pi) * radius * radius


def fillet_order(side_radius: float, top_bottom_radius: float) -> Tuple[float, ...]:
    """The contrib fillet-ordering rule: apply the larger radius first.

    Returns the radii in the order they must be applied; equal radii keep the
    vertical-edge (side) fillet first, as in the reference script.
    """
    if side_radius < 0.0 or top_bottom_radius < 0.0:
        raise EnclosureError("fillet radii must be non-negative")
    if side_radius > top_bottom_radius:
        return (side_radius, top_bottom_radius)
    if top_bottom_radius > side_radius:
        return (top_bottom_radius, side_radius)
    return (side_radius, top_bottom_radius)


def validate_spec(spec: EnclosureSpec) -> List[str]:
    """Return a sorted list of human-readable violations (empty == valid)."""
    errs: List[str] = []
    s = spec
    if min(s.outer_width, s.outer_length, s.outer_height) <= 0.0:
        errs.append("outer dimensions must be positive")
    if s.thickness <= 0.0:
        errs.append("thickness must be positive")
    if s.thickness * 2.0 >= min(s.outer_width, s.outer_length):
        errs.append("walls thicker than the box")
    if s.thickness * 2.0 >= s.outer_height:
        errs.append("walls taller than the box height")
    if s.side_radius > min(s.outer_width, s.outer_length) / 2.0:
        errs.append("side radius exceeds half the smaller footprint side")
    if s.side_radius <= s.thickness:
        errs.append("side radius must exceed the wall thickness "
                    "(the inner fillet radius would be non-positive)")
    if s.top_bottom_radius * 2.0 > s.outer_height:
        errs.append("top/bottom radius exceeds half the height")
    if s.lip_height < 0.0:
        errs.append("lip height must be non-negative")
    if s.screwpost_od <= s.screwpost_id:
        errs.append("screw post OD must exceed its ID")
    if s.screwpost_inset <= s.screwpost_od / 2.0:
        errs.append("screw posts stick out of the box footprint")
    if s.screwpost_inset >= min(s.outer_width, s.outer_length) / 2.0:
        errs.append("screw post inset exceeds the box centre")
    if s.thickness + s.lip_height >= s.outer_height:
        errs.append("lid lip consumes the whole box height")
    return sorted(errs)


def plan_enclosure(spec: EnclosureSpec) -> EnclosurePlan:
    """Derive all secondary geometry of the enclosure, or raise."""
    errs = validate_spec(spec)
    if errs:
        raise EnclosureError("; ".join(errs))
    s = spec
    t = s.thickness

    inner_w = s.outer_width - 2.0 * t
    inner_l = s.outer_length - 2.0 * t
    inner_h = s.outer_height - 2.0 * t
    inner_r = s.side_radius - t

    post_w = s.outer_width - 2.0 * s.screwpost_inset
    post_l = s.outer_length - 2.0 * s.screwpost_inset
    centers = tuple(sorted(
        (x, y)
        for x in (-post_w / 2.0, post_w / 2.0)
        for y in (-post_l / 2.0, post_l / 2.0)
    ))
    post_h = s.outer_height + s.lip_height - t

    lid_split_z = s.outer_height + s.lip_height - t - s.lip_height
    lip_w = inner_w
    lip_l = inner_l

    total_h = s.outer_height + s.lip_height
    outer_area = rounded_rect_area(s.outer_width, s.outer_length, s.side_radius)
    outer_vol = outer_area * total_h
    cavity_vol = rounded_rect_area(inner_w, inner_l, inner_r) * inner_h

    ring = math.pi * ((s.screwpost_od / 2.0) ** 2 - (s.screwpost_id / 2.0) ** 2)
    post_vol = 4.0 * ring * post_h

    lid_h = t + s.lip_height
    lid_vol = outer_area * lid_h

    return EnclosurePlan(
        spec=s,
        inner_width=inner_w,
        inner_length=inner_l,
        inner_height=inner_h,
        inner_side_radius=inner_r,
        post_centers=centers,
        post_height=post_h,
        lid_split_z=lid_split_z,
        lip_width=lip_w,
        lip_length=lip_l,
        outer_volume=outer_vol,
        cavity_volume=cavity_vol,
        post_volume=post_vol,
        lid_volume=lid_vol,
        fillet_order=fillet_order(s.side_radius, s.top_bottom_radius),
    )
