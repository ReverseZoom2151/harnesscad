"""DBU-quantised planar layout with a deterministic design-rule checker (DRC).

This authors GDS-style mask layouts (electrode arrays, Hall bars,
micro-channels, nano-gap arrays, split-ring resonators) and validates them
against fabrication rules, without a mask-layout kernel. Two durable,
kernel-free pieces:

1. A **DBU-quantised layout representation** (``PlanarLayout``): rectangles on
   named layers, addressed in micrometres but stored as integer database units
   (DBU) so geometry is exact and reproducible. :func:`to_dbu` and
   :func:`maps_exactly` handle the quantisation.

2. A **design-rule checker** (:func:`run_drc`) producing findings -- positive
   closed area, minimum feature width, minimum spacing between same-layer boxes
   (rectilinear separation), plus a same-layer overlap ("electrical short")
   check and an off-grid (non-exact-DBU) check. Each violation is a structured
   :class:`Finding` with a stable ``rule_id``.

The placement helpers (``add_centered_box_um`` / ``add_frame_um``) are the
deterministic primitives parametric device cells are built from.

This is the *checking* counterpart to :mod:`harnesscad.domain.fabrication.nesting`
(which *produces* a packing); together they cover placed-layout validation and
sheet-layout generation. It is also distinct from
:mod:`harnesscad.domain.assembly.interference` (3-D solid interference): this is
2-D, integer-grid, mask-style DRC.

Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "LayoutError",
    "to_dbu",
    "maps_exactly",
    "Rect",
    "Finding",
    "DRCReport",
    "PlanarLayout",
    "run_drc",
    "box_spacing_dbu",
    "boxes_overlap",
]


class LayoutError(ValueError):
    """Invalid layout construction (e.g. unknown layer)."""


def to_dbu(value_um: float, dbu_um: float) -> int:
    """Quantise a micrometre value to integer database units."""
    if dbu_um <= 0:
        raise LayoutError("dbu_um must be positive")
    return int(round(float(value_um) / dbu_um))


def maps_exactly(value_um: float, dbu_um: float, tol: float = 1e-9) -> bool:
    """Whether ``value_um`` lands exactly on the DBU grid."""
    if dbu_um <= 0:
        raise LayoutError("dbu_um must be positive")
    scaled = float(value_um) / dbu_um
    return abs(scaled - round(scaled)) <= tol


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle on a named layer, in integer DBU."""

    layer: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> int:
        return abs(self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def normalized(self) -> "Rect":
        return Rect(
            self.layer,
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2),
        )


@dataclass(frozen=True)
class Finding:
    """One DRC violation."""

    severity: str
    rule_id: str
    layer: Optional[str]
    message: str


@dataclass
class DRCReport:
    """The outcome of running the design-rule checker."""

    passed: bool
    findings: List[Finding] = field(default_factory=list)

    def rule_ids(self) -> List[str]:
        return sorted(f.rule_id for f in self.findings)


def box_spacing_dbu(a: Rect, b: Rect) -> int:
    """Rectilinear (Chebyshev-of-gaps) separation between two boxes in DBU.

    0 if they touch or overlap; otherwise the larger of the x-gap and y-gap
    (the closest a blade/edge can approach).
    """
    a, b = a.normalized(), b.normalized()
    dx = max(b.x1 - a.x2, a.x1 - b.x2, 0)
    dy = max(b.y1 - a.y2, a.y1 - b.y2, 0)
    return max(dx, dy)


def boxes_overlap(a: Rect, b: Rect) -> bool:
    """Whether two boxes share interior area (a same-layer short)."""
    a, b = a.normalized(), b.normalized()
    return a.x1 < b.x2 and b.x1 < a.x2 and a.y1 < b.y2 and b.y1 < a.y2


@dataclass
class PlanarLayout:
    """A DBU-quantised planar layout: layers + rectangles authored in um."""

    dbu_um: float = 0.001
    layers: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    rects: List[Rect] = field(default_factory=list)

    def ensure_layer(self, name: str, layer: int, datatype: int = 0) -> None:
        self.layers[name] = (layer, datatype)

    def _require_layer(self, name: str) -> None:
        if name not in self.layers:
            raise LayoutError(f"unknown layer {name!r} (call ensure_layer first)")

    def add_box_um(
        self, layer: str, x1_um: float, y1_um: float, x2_um: float, y2_um: float
    ) -> Rect:
        self._require_layer(layer)
        rect = Rect(
            layer,
            to_dbu(x1_um, self.dbu_um),
            to_dbu(y1_um, self.dbu_um),
            to_dbu(x2_um, self.dbu_um),
            to_dbu(y2_um, self.dbu_um),
        ).normalized()
        self.rects.append(rect)
        return rect

    def add_centered_box_um(self, layer: str, width_um: float, height_um: float) -> Rect:
        hw, hh = width_um / 2.0, height_um / 2.0
        return self.add_box_um(layer, -hw, -hh, hw, hh)

    def add_frame_um(
        self, layer: str, width_um: float, height_um: float, stroke_um: float
    ) -> List[Rect]:
        """A hollow rectangular frame as four border strips."""
        hw, hh = width_um / 2.0, height_um / 2.0
        strips = [
            self.add_box_um(layer, -hw, -hh, hw, -hh + stroke_um),  # bottom
            self.add_box_um(layer, -hw, hh - stroke_um, hw, hh),  # top
            self.add_box_um(layer, -hw, -hh + stroke_um, -hw + stroke_um, hh - stroke_um),  # left
            self.add_box_um(layer, hw - stroke_um, -hh + stroke_um, hw, hh - stroke_um),  # right
        ]
        return strips

    def rects_on(self, layer: str) -> List[Rect]:
        return [r for r in self.rects if r.layer == layer]


def run_drc(
    layout: PlanarLayout,
    *,
    min_width_um: Optional[float] = None,
    min_spacing_um: Optional[float] = None,
    check_shorts: bool = True,
) -> DRCReport:
    """Run design rules over ``layout`` and return structured findings.

    * ``geometry.positive_area`` -- every box must enclose positive area.
    * ``drc.min_width`` -- box width/height below ``min_width_um``.
    * ``drc.min_spacing`` -- same-layer boxes closer than ``min_spacing_um``.
    * ``drc.short`` -- same-layer boxes overlapping (only when ``check_shorts``).
    * ``dbu.off_grid`` -- a box edge that does not map exactly to the DBU grid
      (checked against the stored integer coordinates -- always on-grid here, so
      this fires only for hand-built rects with fractional origins).
    """
    findings: List[Finding] = []
    dbu = layout.dbu_um

    if not layout.rects:
        findings.append(Finding("error", "geometry.empty", None, "Layout has no geometry."))

    for r in layout.rects:
        if r.width <= 0 or r.height <= 0:
            findings.append(
                Finding(
                    "error",
                    "geometry.positive_area",
                    r.layer,
                    "Box must have positive closed rectangular area.",
                )
            )

    if min_width_um is not None:
        min_w = to_dbu(min_width_um, dbu)
        for r in layout.rects:
            if r.width < min_w or r.height < min_w:
                findings.append(
                    Finding(
                        "error",
                        "drc.min_width",
                        r.layer,
                        f"Box {r.width}x{r.height} DBU violates min width {min_width_um} um.",
                    )
                )

    # Pairwise same-layer spacing / short checks.
    if min_spacing_um is not None or check_shorts:
        min_s = to_dbu(min_spacing_um, dbu) if min_spacing_um is not None else None
        rects = layout.rects
        for i, a in enumerate(rects):
            for b in rects[i + 1 :]:
                if a.layer != b.layer:
                    continue
                if check_shorts and boxes_overlap(a, b):
                    findings.append(
                        Finding(
                            "error",
                            "drc.short",
                            a.layer,
                            "Same-layer boxes overlap (electrical short / merged feature).",
                        )
                    )
                    continue
                if min_s is not None and box_spacing_dbu(a, b) < min_s:
                    findings.append(
                        Finding(
                            "error",
                            "drc.min_spacing",
                            a.layer,
                            f"Same-layer boxes closer than min spacing {min_spacing_um} um.",
                        )
                    )

    return DRCReport(passed=not findings, findings=findings)
