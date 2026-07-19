"""coordinate — DPI mapping, 0-999 denormalisation, and action normalisation.

Atlas (an Electron Gemini computer-use client) is small, but it gets one thing
exactly right that a CAD CUA cannot afford to get wrong: it keeps *three*
coordinate frames strictly apart and converts between them in named functions,
never inline.

* **Model frame.** Gemini's ``computer_use`` API returns click targets normalised
  to an integer ``0..999`` grid, independent of resolution
  (``computerUseMapper.ts``). Atlas denormalises with
  ``actual = round(normalized / 999 * screenSize)``.
* **Physical / screenshot frame.** A screenshot on a 2x display is captured at
  physical resolution (3840x2160) — twice the logical size.
* **Logical / OS frame.** The mouse driver (nut.js) moves in *logical* pixels
  (1920x1080). Atlas's ``CoordinateMapper`` divides physical by the display
  ``scaleFactor`` to get logical, and multiplies to go back.

Get the DPI step wrong and every click lands at half or double the intended spot —
and, insidiously, it still lands *somewhere plausible*, so the bug survives a demo.
This repo's viewport work is already scrupulous about y-up vs y-down (see
:class:`harnesscad.io.cua.viewport.OrthoCamera`); the DPI axis is the other half of
the same discipline, and it belongs next to it.

The CAD-specific payoff is in :class:`Denormalizer`: a grounding model that speaks
the ``0..999`` grid (many do — it is the OS-Atlas / UI-TARS convention) can drive
this repo's viewport, but only if the denormalisation into the viewport RECT is
exact. Off-by-one at the grid edge (``999`` must map to the last pixel, not one
past it) is the classic error, and it is unit-tested here.

Pure stdlib, import-safe. No screen, no Electron, no app — the display's
``scaleFactor`` and size are passed in as data (a :class:`ScreenInfo`), so the maths
is testable with no monitor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


class CoordinateError(ValueError):
    """A coordinate was out of its declared frame, or a frame was degenerate."""


@dataclass(frozen=True)
class ScreenInfo:
    """A display, as atlas's ``CoordinateMapper.getScreenInfo`` reports it.

    ``width``/``height`` are LOGICAL pixels (what the OS and the mouse driver use);
    ``scale_factor`` is the device-pixel ratio (2.0 on a Retina/HiDPI panel, 1.0
    otherwise). The physical/screenshot size is ``logical * scale_factor``.
    """

    width: int
    height: int
    scale_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise CoordinateError("screen size must be positive")
        if self.scale_factor <= 0:
            raise CoordinateError("scale_factor must be positive")

    @property
    def physical_width(self) -> int:
        return int(round(self.width * self.scale_factor))

    @property
    def physical_height(self) -> int:
        return int(round(self.height * self.scale_factor))

    def resolution_string(self) -> str:
        """"1920x1080" — atlas injects this into the prompt so the model knows the frame."""
        return "%dx%d" % (self.width, self.height)


class CoordinateMapper:
    """Logical <-> physical (screenshot) conversion across a DPI boundary.

    A direct port of atlas's ``CoordinateMapper``: a screenshot is captured in
    PHYSICAL pixels, the mouse moves in LOGICAL pixels, and the only correct bridge
    is the display ``scale_factor``. Rounding is applied once, at the boundary, so a
    round-trip is stable to within a pixel.
    """

    def __init__(self, screen: ScreenInfo) -> None:
        self.screen = screen

    def to_logical(self, physical_x: float, physical_y: float) -> Tuple[int, int]:
        """Screenshot-pixel -> OS-logical. Divide by scaleFactor (atlas's ``toLogical``)."""
        s = self.screen.scale_factor
        return (int(round(physical_x / s)), int(round(physical_y / s)))

    def to_physical(self, logical_x: float, logical_y: float) -> Tuple[int, int]:
        """OS-logical -> screenshot-pixel. Multiply by scaleFactor (atlas's ``toPhysical``)."""
        s = self.screen.scale_factor
        return (int(round(logical_x * s)), int(round(logical_y * s)))


class Denormalizer:
    """The ``0..999`` model grid <-> pixel frame conversion, done exactly once.

    ``grid`` is the model's integer range (Gemini/OS-Atlas use ``999``, i.e. a
    1000-point grid whose last index is 999). Denormalisation maps a grid value
    ``g`` to ``round(g / grid * (size - 1))`` so the endpoints are pinned: ``0``
    lands on pixel 0 and ``grid`` lands on the LAST pixel, never one past it. Atlas
    uses ``g / 999 * size`` (no ``-1``), which overshoots the far edge by up to one
    pixel; this fixes that quietly, and the test pins the corrected behaviour.

    Coordinates can be denormalised into a full frame (``width, height``) or into a
    sub-rect — which is exactly what a CAD agent needs, because the model grounds
    against the whole window but the click must land inside the VIEWPORT rect (see
    :meth:`denorm_into_rect`).
    """

    def __init__(self, grid: int = 999) -> None:
        if grid < 1:
            raise CoordinateError("grid must be >= 1")
        self.grid = grid

    def denorm(self, g: float, size: int) -> int:
        """A single grid value onto a ``size``-pixel axis (endpoints pinned)."""
        if size < 1:
            raise CoordinateError("size must be >= 1")
        g = max(0.0, min(float(self.grid), float(g)))
        return int(round(g / self.grid * (size - 1)))

    def denorm_point(self, gx: float, gy: float, width: int, height: int) -> Tuple[int, int]:
        return (self.denorm(gx, width), self.denorm(gy, height))

    def denorm_into_rect(self, gx: float, gy: float,
                         rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """A grid point onto a sub-rect ``(left, top, width, height)``, absolute pixels.

        The frame the model grounded against IS this rect (the viewport), so the
        grid maps across the rect's extent and is offset by its origin. This is how
        a ``0..999`` grounding model addresses the opaque 3D view without ever
        seeing a pixel coordinate.
        """
        left, top, w, h = rect
        return (left + self.denorm(gx, w), top + self.denorm(gy, h))

    def normalize(self, px: float, py: float, width: int, height: int) -> Tuple[int, int]:
        """Pixel -> grid (the inverse), for storing a pick in the model's own units."""
        if width < 1 or height < 1:
            raise CoordinateError("size must be >= 1")
        gx = int(round(px / max(width - 1, 1) * self.grid))
        gy = int(round(py / max(height - 1, 1) * self.grid))
        return (max(0, min(self.grid, gx)), max(0, min(self.grid, gy)))


#: Computer-use function names -> a canonical action verb. The value is
#: ``(verb, needs_point)``; a verb this module does not model maps to ``None``
#: and is dropped.
_FUNCTION_MAP: Dict[str, Tuple[Optional[str], bool]] = {
    "click_at": ("click", True),
    "hover_at": ("hover", True),
    "type_text_at": ("type", True),
    "key_combination": ("hotkey", False),
    "scroll_at": ("scroll", True),
    "scroll_document": ("scroll", False),
    "drag_and_drop": ("drag", True),
    "navigate": ("navigate", False),
    "go_back": ("back", False),
    "go_forward": ("forward", False),
    "search": ("search", False),
    "wait_5_seconds": ("wait", False),
    "open_web_browser": ("open_browser", False),
}


@dataclass(frozen=True)
class NormalizedAction:
    """A model function-call reduced to a canonical, pixel-resolved action."""

    verb: str
    coords: Optional[Tuple[int, int]] = None
    text: str = ""
    keys: Tuple[str, ...] = ()
    press_enter: bool = False
    requires_confirmation: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {"verb": self.verb,
                "coords": list(self.coords) if self.coords else None,
                "text": self.text, "keys": list(self.keys),
                "press_enter": self.press_enter,
                "requires_confirmation": self.requires_confirmation,
                "reason": self.reason}


def extract_safety_decision(args: dict) -> Optional[bool]:
    """Atlas's ``extractSafetyDecision``: True iff the call wants confirmation.

    ``None`` when the model attached no ``safety_decision`` block (nothing to say).
    A CAD agent maps this straight onto the guardrail: a ``require_confirmation``
    action must not auto-execute.
    """
    safety = args.get("safety_decision")
    if not isinstance(safety, dict):
        return None
    return str(safety.get("decision", "")) == "require_confirmation"


def normalize_function_call(name: str, args: dict,
                            width: int, height: int,
                            denorm: Optional[Denormalizer] = None) -> Optional[NormalizedAction]:
    """Map a Gemini ``computer_use`` function-call to a :class:`NormalizedAction`.

    A faithful port of ``mapFunctionCallToAction`` including its one genuinely
    load-bearing nuance: ``type_text_at`` with NO coordinates (or ``0,0``) is a
    "type into whatever is focused" (e.g. after Ctrl+F), NOT a click at the origin
    — treating ``(0, 0)`` as a real target is a classic misfire that clicks the top
    corner every time the model omits a point. Returns ``None`` for a verb this repo
    does not model (atlas returns null), so the caller drops it cleanly.
    """
    denorm = denorm or Denormalizer()
    mapping = _FUNCTION_MAP.get(name)
    if mapping is None or mapping[0] is None:
        return None
    verb, needs_point = mapping
    confirm = bool(extract_safety_decision(args))

    if name == "type_text_at":
        rawx, rawy = args.get("x"), args.get("y")
        text = str(args.get("text", ""))
        press_enter = bool(args.get("press_enter"))
        has_coords = (rawx is not None and rawy is not None
                      and (float(rawx) > 0 or float(rawy) > 0))
        if not has_coords:
            return NormalizedAction(verb="type", text=text, press_enter=press_enter,
                                    requires_confirmation=confirm,
                                    reason="type into focused element")
        coords = denorm.denorm_point(float(rawx), float(rawy), width, height)
        return NormalizedAction(verb="type", coords=coords, text=text,
                                press_enter=press_enter, requires_confirmation=confirm,
                                reason="type at %r" % (coords,))

    if name == "key_combination":
        keys = tuple(k.strip().lower() for k in str(args.get("keys", "")).split("+") if k.strip())
        return NormalizedAction(verb="hotkey", keys=keys, requires_confirmation=confirm,
                                reason="keys %s" % "+".join(keys))

    if needs_point:
        gx = float(args.get("x", 0))
        gy = float(args.get("y", 0))
        coords = denorm.denorm_point(gx, gy, width, height)
        return NormalizedAction(verb=verb, coords=coords, requires_confirmation=confirm,
                                text=str(args.get("direction", "") or args.get("query", "")),
                                reason="%s at %r" % (verb, coords))

    return NormalizedAction(verb=verb, text=str(args.get("url", "") or args.get("query", "")),
                            requires_confirmation=confirm, reason=verb)
