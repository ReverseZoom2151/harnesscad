"""2-D rectangular part nesting for sheet fabrication (skyline bin-packing).

Ported from **Kerf** (``packages/kerf-cad-core/llm_docs/nesting.md``), whose
``nest_parts`` / ``nest_report`` tools pack rectangular blanks onto stock sheets
for laser cutting, waterjet, plasma, sheet-metal blanking, and flat-pattern
nesting. Kerf ships the capability inside a TypeScript/Python multi-domain CAD
platform; this module reimplements the deterministic geometry core -- a skyline
(bottom-left) packer with a kerf gap, a border margin and optional 90-degree
rotation -- in stdlib Python.

The algorithm is a *skyline* heuristic: each sheet keeps a piecewise-constant
"skyline" of segments ``(x, width, height)`` describing the top profile of what
has already been placed. A part is placed at the candidate x that minimises the
resting height (ties broken by leftmost x), the skyline is raised there, and the
part footprint is inflated by the kerf gap so blades never share material. When
no sheet position fits, a fresh sheet is opened. The result is fully
deterministic: the same parts and sheet always yield the same layout.

This complements -- and does not duplicate -- ``flatpack_panels`` (which *derives*
the panels of a cabinet) and ``legolization`` (voxel -> brick). Nesting takes an
already-flattened list of rectangles and answers the shop-floor questions: how
many sheets, what utilisation, how much cut length.

Distinct from ``planar_layout`` (design-rule *checking* of a placed layout):
this module *produces* a placement that minimises sheet count.

Pure stdlib, no CAD kernel, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

__all__ = [
    "NestingError",
    "Part",
    "Placement",
    "SheetLayout",
    "NestResult",
    "nest_parts",
    "nest_report",
]


class NestingError(ValueError):
    """A part cannot be nested (e.g. it exceeds the usable sheet area)."""


@dataclass(frozen=True)
class Part:
    """A rectangular part to cut, with an optional quantity."""

    name: str
    w: float
    h: float
    qty: int = 1

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise NestingError(f"part {self.name!r} has non-positive size {self.w}x{self.h}")
        if self.qty < 1:
            raise NestingError(f"part {self.name!r} has non-positive quantity {self.qty}")


@dataclass(frozen=True)
class Placement:
    """One placed part instance on a sheet."""

    name: str
    x: float
    y: float
    w: float
    h: float
    rotated: bool
    sheet: int

    @property
    def perimeter(self) -> float:
        return 2.0 * (self.w + self.h)


@dataclass
class SheetLayout:
    """The placements on a single stock sheet."""

    index: int
    placements: List[Placement] = field(default_factory=list)

    def used_area(self) -> float:
        return sum(p.w * p.h for p in self.placements)


@dataclass
class NestResult:
    """The outcome of a nesting run."""

    ok: bool
    sheets: List[SheetLayout] = field(default_factory=list)
    error: Optional[str] = None
    sheet_w: float = 0.0
    sheet_h: float = 0.0

    @property
    def sheets_used(self) -> int:
        return len(self.sheets)

    @property
    def placements(self) -> List[Placement]:
        out: List[Placement] = []
        for sheet in self.sheets:
            out.extend(sheet.placements)
        return out

    @property
    def utilization(self) -> float:
        """Total part area / (sheets_used * sheet area). 0 when no sheets."""
        if not self.sheets:
            return 0.0
        sheet_area = self.sheet_w * self.sheet_h
        if sheet_area <= 0:
            return 0.0
        used = sum(s.used_area() for s in self.sheets)
        return used / (self.sheets_used * sheet_area)

    @property
    def cut_length(self) -> float:
        """Sum of placed part perimeters -- a proxy for laser cut length."""
        return sum(p.perimeter for p in self.placements)


# A skyline segment: a horizontal run [x, x+width) resting at height y.
@dataclass
class _Segment:
    x: float
    width: float
    y: float


class _Skyline:
    """A single sheet's top profile, packed bottom-left."""

    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.segments: List[_Segment] = [_Segment(0.0, width, 0.0)]

    def _fit_at(self, seg_index: int, w: float, h: float) -> Optional[float]:
        """Return the resting y if a w-by-h box fits starting at segment index,
        else ``None``. The box spans possibly several segments; it rests on the
        max of their heights."""
        x = self.segments[seg_index].x
        if x + w > self.width + 1e-9:
            return None
        remaining = w
        y = 0.0
        i = seg_index
        while remaining > 1e-9:
            if i >= len(self.segments):
                return None
            y = max(y, self.segments[i].y)
            remaining -= self.segments[i].width
            i += 1
        if y + h > self.height + 1e-9:
            return None
        return y

    def place(self, w: float, h: float) -> Optional[tuple]:
        """Try to place a w-by-h box. Returns ``(x, y)`` or ``None``.

        Picks the position with the lowest resting y, ties broken by lowest x
        (bottom-left rule) -- deterministic for a given sheet state.
        """
        best: Optional[tuple] = None
        for i, seg in enumerate(self.segments):
            y = self._fit_at(i, w, h)
            if y is None:
                continue
            cand = (y, seg.x, i)
            if best is None or cand < best[0]:
                best = (cand, seg.x, y)
        if best is None:
            return None
        _, x, y = best
        self._raise(x, w, y + h)
        return (x, y)

    def _raise(self, x: float, w: float, new_y: float) -> None:
        """Raise the skyline over [x, x+w) to new_y and re-merge segments."""
        new_segments: List[_Segment] = []
        x_end = x + w
        for seg in self.segments:
            seg_end = seg.x + seg.width
            # Portion entirely outside the raised span stays as-is.
            if seg_end <= x + 1e-9 or seg.x >= x_end - 1e-9:
                new_segments.append(seg)
                continue
            # Left remainder.
            if seg.x < x - 1e-9:
                new_segments.append(_Segment(seg.x, x - seg.x, seg.y))
            # Right remainder.
            if seg_end > x_end + 1e-9:
                new_segments.append(_Segment(x_end, seg_end - x_end, seg.y))
        new_segments.append(_Segment(x, w, new_y))
        new_segments.sort(key=lambda s: s.x)
        # Merge adjacent equal-height segments for a compact profile.
        merged: List[_Segment] = []
        for seg in new_segments:
            if merged and abs(merged[-1].y - seg.y) < 1e-9 and abs(merged[-1].x + merged[-1].width - seg.x) < 1e-9:
                merged[-1] = _Segment(merged[-1].x, merged[-1].width + seg.width, merged[-1].y)
            else:
                merged.append(seg)
        self.segments = merged


def _expand(parts: Sequence[Part]) -> List[tuple]:
    """Expand quantities into one (name, w, h) tuple per instance."""
    out: List[tuple] = []
    for part in parts:
        for _ in range(part.qty):
            out.append((part.name, float(part.w), float(part.h)))
    return out


def nest_parts(
    parts: Sequence[Part],
    sheet_w: float,
    sheet_h: float,
    *,
    kerf: float = 0.0,
    margin: float = 0.0,
    allow_rotate: bool = True,
) -> NestResult:
    """Pack ``parts`` onto ``sheet_w`` x ``sheet_h`` stock using skyline packing.

    ``kerf`` is added to each part footprint so cuts never share material;
    ``margin`` is a keep-out border on every side of the sheet. Parts that
    exceed the usable area are never silently dropped -- the result is
    ``ok=False`` with a descriptive ``error``.
    """
    if sheet_w <= 0 or sheet_h <= 0:
        raise NestingError("sheet dimensions must be positive")
    if kerf < 0 or margin < 0:
        raise NestingError("kerf and margin must be non-negative")

    usable_w = sheet_w - 2.0 * margin
    usable_h = sheet_h - 2.0 * margin
    if usable_w <= 0 or usable_h <= 0:
        raise NestingError("margin consumes the entire sheet")

    instances = _expand(parts)
    # Largest-area first, ties by name then dimensions -> deterministic order.
    instances.sort(key=lambda t: (-t[1] * t[2], t[0], -t[1], -t[2]))

    sheets: List[_Skyline] = []
    layouts: List[SheetLayout] = []

    def _footprints(w: float, h: float) -> List[tuple]:
        """Candidate (packed_w, packed_h, rotated) footprints incl. kerf."""
        pw, ph = w + kerf, h + kerf
        cands = [(pw, ph, False)]
        if allow_rotate and abs(w - h) > 1e-9:
            cands.append((ph, pw, True))
        return cands

    for name, w, h in instances:
        # Reject parts that can never fit on any orientation.
        fits_somewhere = any(
            fw <= usable_w + 1e-9 and fh <= usable_h + 1e-9
            for fw, fh, _ in _footprints(w, h)
        )
        if not fits_somewhere:
            return NestResult(
                ok=False,
                error=(
                    f"part {name!r} ({w}x{h}) exceeds usable sheet area "
                    f"{usable_w}x{usable_h}"
                ),
                sheet_w=sheet_w,
                sheet_h=sheet_h,
            )

        placed = False
        for si, skyline in enumerate(sheets):
            spot = _try_place(skyline, w, h, kerf, allow_rotate, usable_w, usable_h)
            if spot is not None:
                x, y, fw, fh, rotated = spot
                layouts[si].placements.append(
                    Placement(
                        name=name,
                        x=x + margin,
                        y=y + margin,
                        w=fw - kerf if not rotated else fh - kerf,
                        h=fh - kerf if not rotated else fw - kerf,
                        rotated=rotated,
                        sheet=si,
                    )
                )
                placed = True
                break
        if placed:
            continue

        # Open a fresh sheet.
        skyline = _Skyline(usable_w, usable_h)
        spot = _try_place(skyline, w, h, kerf, allow_rotate, usable_w, usable_h)
        if spot is None:  # pragma: no cover - guarded by fits_somewhere
            return NestResult(
                ok=False,
                error=f"part {name!r} failed to place on an empty sheet",
                sheet_w=sheet_w,
                sheet_h=sheet_h,
            )
        sheets.append(skyline)
        si = len(sheets) - 1
        layouts.append(SheetLayout(index=si))
        x, y, fw, fh, rotated = spot
        layouts[si].placements.append(
            Placement(
                name=name,
                x=x + margin,
                y=y + margin,
                w=fw - kerf if not rotated else fh - kerf,
                h=fh - kerf if not rotated else fw - kerf,
                rotated=rotated,
                sheet=si,
            )
        )

    return NestResult(ok=True, sheets=layouts, sheet_w=sheet_w, sheet_h=sheet_h)


def _try_place(
    skyline: "_Skyline",
    w: float,
    h: float,
    kerf: float,
    allow_rotate: bool,
    usable_w: float,
    usable_h: float,
) -> Optional[tuple]:
    """Attempt each orientation; return (x, y, fw, fh, rotated) or None."""
    candidates = [(w + kerf, h + kerf, False)]
    if allow_rotate and abs(w - h) > 1e-9:
        candidates.append((h + kerf, w + kerf, True))
    best = None
    for fw, fh, rotated in candidates:
        if fw > usable_w + 1e-9 or fh > usable_h + 1e-9:
            continue
        # Probe on a copy so the losing orientation does not mutate state.
        probe = _Skyline(skyline.width, skyline.height)
        probe.segments = [
            _Segment(s.x, s.width, s.y) for s in skyline.segments
        ]
        spot = probe.place(fw, fh)
        if spot is None:
            continue
        x, y = spot
        cand = (y, x, fw, fh, rotated)
        if best is None or cand[:2] < best[:2]:
            best = cand
    if best is None:
        return None
    y, x, fw, fh, rotated = best
    skyline.place(fw, fh)  # commit the winning orientation
    return (x, y, fw, fh, rotated)


def nest_report(
    result: NestResult,
    *,
    material: Optional[str] = None,
    kerf: float = 0.0,
) -> str:
    """Format a human-readable cut-optimisation report from a nest result."""
    lines: List[str] = []
    header = "Nesting report"
    if material:
        header += f" -- {material}"
    lines.append(header)
    lines.append("=" * len(header))
    if not result.ok:
        lines.append(f"FAILED: {result.error}")
        return "\n".join(lines)
    lines.append(f"sheet:        {result.sheet_w:g} x {result.sheet_h:g}")
    if kerf:
        lines.append(f"kerf:         {kerf:g}")
    lines.append(f"sheets used:  {result.sheets_used}")
    lines.append(f"utilisation:  {result.utilization * 100:.1f}%")
    lines.append(f"cut length:   {result.cut_length:g}")
    for sheet in result.sheets:
        lines.append(f"  sheet {sheet.index}: {len(sheet.placements)} part(s)")
        for p in sheet.placements:
            rot = " (rot)" if p.rotated else ""
            lines.append(
                f"    {p.name}: {p.w:g}x{p.h:g} @ ({p.x:g},{p.y:g}){rot}"
            )
    return "\n".join(lines)
