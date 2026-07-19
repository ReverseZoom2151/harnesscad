"""Deterministic dimensional-brief parser and parametric OpenSCAD emitter.

The parser turns a natural-language part brief with explicit dimensions into a
typed ``PartSpec``, then emits parametric OpenSCAD with no model call. It
supports mounting plates with corner holes and hollow electronics boxes.

The transferable idea is the *deterministic front half* of a text-to-CAD system:
a brief like

    "mounting plate 80x40x3mm with 4 holes 5mm"
    "electronics enclosure 100x60x40mm wall 2mm"

is parsed to a checkable, typed record (dimensions, hole count/diameter, wall)
that a generator, a validator, and a differential oracle can all agree on --
before any geometry exists. This is distinct from
:mod:`harnesscad.domain.spec.scad_parameters` (which extracts parameters *out of*
existing SCAD): here we go *from a brief to a spec* and synthesize SCAD.

The emitter uses standard templates: a plate is ``cube`` minus corner-hole
cylinders (over-drilled +/-1mm so the difference cleanly punches through), a box
is an outer ``cube`` minus a wall-inset inner ``cube``. Hole placement uses the
same corner-margin rule (``max(2*hole_dia, 0.15*min(width, depth))``).

stdlib-only (``re``, ``dataclasses``), deterministic, absolute imports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

__all__ = [
    "PartSpec",
    "BriefParseError",
    "parse_part_brief",
    "hole_positions",
    "emit_openscad",
    "brief_to_openscad",
]


class BriefParseError(ValueError):
    """Raised when a brief carries no parseable ``WxDxH mm`` dimension triple."""


@dataclass(frozen=True)
class PartSpec:
    """Typed part specification parsed from a dimensional brief.

    ``kind`` is ``"box"`` (a hollow enclosure with a wall) or ``"plate"`` (a
    flat plate with optional corner holes). All lengths are millimetres.
    """

    kind: str
    width: float
    depth: float
    height: float
    holes: int = 0
    hole_diameter: float = 0.0
    wall: float = 0.0


_DIMS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE
)
_HOLES_RE = re.compile(
    r"(\d+)\s+holes?\s+(?:(?:of|dia(?:meter)?|d)\s*)?(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE
)
_WALL_RE = re.compile(r"wall\s+(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)


def parse_part_brief(brief: str) -> PartSpec:
    """Parse a dimensional part brief into a :class:`PartSpec`.

    Requires a ``WxDxH mm`` triple (case-insensitive, whitespace-tolerant).
    Detects a hollow box when the brief mentions ``box`` or ``enclosure``
    (otherwise a plate), and reads an optional ``N holes D mm`` and ``wall W mm``.
    Raises :class:`BriefParseError` when no dimensions are present.
    """
    text = brief.lower()
    dims = _DIMS_RE.search(text)
    if not dims:
        raise BriefParseError("brief must include dimensions like 80x40x3mm")

    width, depth, height = (float(v) for v in dims.groups())
    kind = "box" if ("box" in text or "enclosure" in text) else "plate"

    holes_match = _HOLES_RE.search(text)
    wall_match = _WALL_RE.search(text)

    return PartSpec(
        kind=kind,
        width=width,
        depth=depth,
        height=height,
        holes=int(holes_match.group(1)) if holes_match else 0,
        hole_diameter=float(holes_match.group(2)) if holes_match else 0.0,
        wall=float(wall_match.group(1)) if wall_match else 0.0,
    )


def hole_positions(spec: PartSpec) -> List[Tuple[float, float]]:
    """Corner-hole (x, y) positions for a plate, cad-agent's margin rule.

    Margin = ``max(2*hole_diameter, 0.15*min(width, depth))``. Up to four
    corners are used; ``holes <= 0`` yields none.
    """
    if spec.holes <= 0:
        return []
    margin = max(spec.hole_diameter * 2.0, min(spec.width, spec.depth) * 0.15)
    corners = [
        (margin, margin),
        (spec.width - margin, margin),
        (spec.width - margin, spec.depth - margin),
        (margin, spec.depth - margin),
    ]
    return corners[: spec.holes] if spec.holes <= 4 else corners


def _emit_plate(spec: PartSpec) -> str:
    lines = [
        "// parametric mounting plate (harnesscad)",
        "$fn = 64;",
        "difference() {",
        f"  cube([{spec.width:.1f}, {spec.depth:.1f}, {spec.height:.1f}], center = false);",
    ]
    for x, y in hole_positions(spec):
        lines.append(
            f"  translate([{x:.1f}, {y:.1f}, -1.0]) "
            f"cylinder(h = {spec.height + 2:.1f}, d = {spec.hole_diameter:.1f}, center = false);"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_box(spec: PartSpec) -> str:
    wall = spec.wall or 2.0
    inner_w = max(spec.width - wall * 2.0, 0.0)
    inner_d = max(spec.depth - wall * 2.0, 0.0)
    return "\n".join(
        [
            "// hollow electronics box (harnesscad)",
            "$fn = 64;",
            "difference() {",
            f"  cube([{spec.width:.1f}, {spec.depth:.1f}, {spec.height:.1f}], center = false);",
            f"  translate([{wall:.1f}, {wall:.1f}, {wall:.1f}])",
            f"    cube([{inner_w:.1f}, {inner_d:.1f}, {spec.height:.1f}], center = false);",
            "}",
        ]
    ) + "\n"


def emit_openscad(spec: PartSpec) -> str:
    """Emit parametric OpenSCAD source for a :class:`PartSpec`."""
    return _emit_box(spec) if spec.kind == "box" else _emit_plate(spec)


def brief_to_openscad(brief: str) -> str:
    """Convenience: parse a brief and emit OpenSCAD in one call."""
    return emit_openscad(parse_part_brief(brief))
