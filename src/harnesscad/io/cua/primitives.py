"""primitives — the OS-control action-primitive SPEC + a deterministic template match.

Ported from robotgo (go-vgo/robotgo): a cross-platform system-automation library
whose public surface is the complete set of raw OS-control primitives — mouse
move/click/drag/scroll with SEPARATE down/up, keyboard tap/toggle/type with
modifiers and Unicode, screen capture + per-pixel colour, and window
find/focus/resize. This module ports that surface as DATA (typed, deterministic,
import-safe), not as a live driver, and adds the one algorithm robotgo delegates
to a C/OpenCV sidecar: a **normalized cross-correlation template match** on an
image array.

How this sits next to the rest of :mod:`harnesscad.io.cua`
--------------------------------------------------------
:mod:`uia` is the *semantic* driver: it resolves a control by the accessibility
tree and invokes it coordinate-free — the right tool for FreeCAD's Qt chrome.
This module is the layer BELOW that: the raw coordinate/key primitives every GUI
driver ultimately reduces to, described as a checkable action space, plus a
grounding path that needs neither an a11y tree nor a VLM.

The template match is the high-value piece. A CAD toolbar icon is a FIXED bitmap
at an app-stable position; correlating a stored icon crop against a screenshot
locates it to the pixel, deterministically, for free, with no model and ~100 %
repeatability. That is a third grounding modality alongside the a11y tree
(:mod:`uia`) and the computed viewport pick (:mod:`viewport`/:mod:`picks`): the
match returns an IMAGE pixel, which a :class:`~harnesscad.io.cua.frames.Frame`
then maps to a screen pixel for a real click — no guess anywhere in the chain.

Nothing here launches, moves, or presses anything. It is the SPEC of what a
primitive is and a pure function that finds a bitmap in a bitmap. Building the
live SendInput driver is deliberately left to :mod:`uia` (keyboard/text) and any
future pointer backend; this module is what they would be checked against.

numpy is used ONLY as an optional accelerator for the correlation: the pure
-Python path is the reference implementation and the tests run on it, so the
module imports and works with no numpy installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 1. The action-primitive vocabulary (robotgo's public surface, as an enum)
# ---------------------------------------------------------------------------


class MouseButton(Enum):
    """The pointer buttons robotgo's ``CheckMouse`` accepts, canonicalised."""

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "center"
    WHEEL_UP = "wheelUp"
    WHEEL_DOWN = "wheelDown"
    WHEEL_LEFT = "wheelLeft"
    WHEEL_RIGHT = "wheelRight"


class Modifier(Enum):
    """Keyboard modifiers, side-agnostic. robotgo also exposes ``*l``/``*r``
    variants (``ctrll``/``ctrlr`` ...); the side-agnostic form is what a CAD
    shortcut ever needs, and the sided form is a pure refinement of it."""

    CTRL = "ctrl"
    ALT = "alt"
    SHIFT = "shift"
    CMD = "cmd"  # the Windows key on Windows; Command on macOS


class KeyAction(Enum):
    """The three keyboard verbs. ``TAP`` is down+up; ``DOWN``/``UP`` are the
    halves, which robotgo exposes as ``KeyDown``/``KeyUp`` — the split is what
    lets a modifier be HELD across another key, the exact thing PostMessage
    cannot do (see :mod:`harnesscad.io.cua.hazards`)."""

    TAP = "tap"
    DOWN = "down"
    UP = "up"


class PrimitiveKind(Enum):
    """Every raw primitive robotgo exposes, grouped. This IS the action space a
    pointer/keyboard backend must implement to be complete — the checklist the
    :mod:`uia` semantic driver sits on top of."""

    # -- pointer (each verb separable into down/up, as robotgo's Toggle) ----
    MOUSE_MOVE = "mouse_move"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    MOUSE_CLICK = "mouse_click"
    MOUSE_DOUBLE_CLICK = "mouse_double_click"
    MOUSE_DRAG = "mouse_drag"
    MOUSE_SCROLL = "mouse_scroll"
    # -- keyboard ----------------------------------------------------------
    KEY_TAP = "key_tap"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    TYPE_TEXT = "type_text"  # Unicode, layout-independent (KEYEVENTF_UNICODE)
    # -- screen ------------------------------------------------------------
    SCREEN_CAPTURE = "screen_capture"
    PIXEL_COLOR = "pixel_color"
    SCREEN_SIZE = "screen_size"
    CURSOR_POSITION = "cursor_position"
    # -- window ------------------------------------------------------------
    WINDOW_FIND = "window_find"
    WINDOW_FOCUS = "window_focus"
    WINDOW_BOUNDS = "window_bounds"
    WINDOW_RESIZE = "window_resize"
    WINDOW_MOVE = "window_move"


# The parameter contract of every primitive, as DATA. Mirrors robotgo's typed
# signatures (Move(x,y); Toggle(button, up); KeyTap(key, ...mods); Scroll(x,y);
# GetPixelColor(x,y); CaptureImg(x,y,w,h); ActiveName(name); GetBounds(pid)).
# A backend is complete iff it implements every entry; a call is well-formed iff
# its params are exactly the required set. This is the same reflective contract
# cua-main exposes at /commands (see :mod:`harnesscad.io.cua.wire`).
PRIMITIVE_PARAMS: Dict[PrimitiveKind, Tuple[str, ...]] = {
    PrimitiveKind.MOUSE_MOVE: ("x", "y"),
    PrimitiveKind.MOUSE_DOWN: ("button",),
    PrimitiveKind.MOUSE_UP: ("button",),
    PrimitiveKind.MOUSE_CLICK: ("button",),
    PrimitiveKind.MOUSE_DOUBLE_CLICK: ("button",),
    PrimitiveKind.MOUSE_DRAG: ("x", "y", "button"),
    PrimitiveKind.MOUSE_SCROLL: ("dx", "dy"),
    PrimitiveKind.KEY_TAP: ("key", "modifiers"),
    PrimitiveKind.KEY_DOWN: ("key",),
    PrimitiveKind.KEY_UP: ("key",),
    PrimitiveKind.TYPE_TEXT: ("text",),
    PrimitiveKind.SCREEN_CAPTURE: ("x", "y", "w", "h"),
    PrimitiveKind.PIXEL_COLOR: ("x", "y"),
    PrimitiveKind.SCREEN_SIZE: (),
    PrimitiveKind.CURSOR_POSITION: (),
    PrimitiveKind.WINDOW_FIND: ("name",),
    PrimitiveKind.WINDOW_FOCUS: ("handle",),
    PrimitiveKind.WINDOW_BOUNDS: ("handle",),
    PrimitiveKind.WINDOW_RESIZE: ("handle", "w", "h"),
    PrimitiveKind.WINDOW_MOVE: ("handle", "x", "y"),
}


@dataclass(frozen=True)
class Action:
    """One well-formed primitive invocation: a kind plus its arguments.

    A pure value — constructing it neither dispatches nor validates against a live
    OS. :func:`validate` checks it against :data:`PRIMITIVE_PARAMS`; a backend
    consumes it. Coordinates are IMAGE/screen pixels per the caller's
    :class:`~harnesscad.io.cua.frames.Frame`; this type does not own a frame,
    exactly so it cannot silently mix coordinate spaces.
    """

    kind: PrimitiveKind
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "args": dict(self.args)}


def validate(action: Action) -> List[str]:
    """The problems with ``action``, or ``[]`` if it is well-formed.

    A primitive is well-formed iff every required parameter is present and no
    unknown parameter is passed. Returned as a list (never raised) so a planner
    can surface every mistake at once — the same posture as the rest of the CUA
    surface, where an unverified/ill-formed action is DATA, not an exception.
    """
    problems: List[str] = []
    required = PRIMITIVE_PARAMS.get(action.kind)
    if required is None:
        return ["unknown primitive %r" % (action.kind,)]
    for name in required:
        if name not in action.args:
            problems.append("missing required parameter %r for %s"
                            % (name, action.kind.value))
    for name in action.args:
        if name not in required:
            problems.append("unexpected parameter %r for %s"
                            % (name, action.kind.value))
    # A button parameter, when present, must name a real button.
    if "button" in action.args:
        try:
            MouseButton(action.args["button"])
        except ValueError:
            problems.append("not a MouseButton: %r" % (action.args["button"],))
    if "modifiers" in action.args:
        for m in action.args.get("modifiers") or ():
            try:
                Modifier(m)
            except ValueError:
                problems.append("not a Modifier: %r" % (m,))
    return problems


# -- constructors: the split-verb primitives, spelled out --------------------
# These make the down/up separation first-class. A drag is press -> move -> release
# and a modifier-held tap is mod-down -> key -> mod-up; both are expressed here as
# EXPLICIT sequences of atomic primitives, which is the only honest way to hold a
# modifier across a key (a single fused call cannot).


def click_sequence(x: int, y: int, button: MouseButton = MouseButton.LEFT
                   ) -> List[Action]:
    """move -> button-down -> button-up. The atomic decomposition of a click."""
    return [
        Action(PrimitiveKind.MOUSE_MOVE, {"x": int(x), "y": int(y)}),
        Action(PrimitiveKind.MOUSE_DOWN, {"button": button.value}),
        Action(PrimitiveKind.MOUSE_UP, {"button": button.value}),
    ]


def drag_sequence(x0: int, y0: int, x1: int, y1: int,
                  button: MouseButton = MouseButton.LEFT) -> List[Action]:
    """move(start) -> down -> move(end) -> up. robotgo's DragSmooth, atomised."""
    return [
        Action(PrimitiveKind.MOUSE_MOVE, {"x": int(x0), "y": int(y0)}),
        Action(PrimitiveKind.MOUSE_DOWN, {"button": button.value}),
        Action(PrimitiveKind.MOUSE_MOVE, {"x": int(x1), "y": int(y1)}),
        Action(PrimitiveKind.MOUSE_UP, {"button": button.value}),
    ]


def chord_sequence(key: str, modifiers: Sequence[Modifier]) -> List[Action]:
    """mod-down(s) -> key tap -> mod-up(s), in reverse-nested order.

    This is the primitive robotgo's ``KeyTap(key, mods...)`` compiles to and the
    exact shape a background message API CANNOT emulate: the modifier must be
    physically held (async key state up) while the key is pressed. Ported so the
    contrast is explicit; :mod:`harnesscad.io.cua.hazards` records why the fused
    background path is wrong.
    """
    mods = list(modifiers)
    out: List[Action] = [Action(PrimitiveKind.KEY_DOWN, {"key": m.value})
                         for m in mods]
    out.append(Action(PrimitiveKind.KEY_TAP, {"key": key, "modifiers": []}))
    out.extend(Action(PrimitiveKind.KEY_UP, {"key": m.value})
               for m in reversed(mods))
    return out


# ---------------------------------------------------------------------------
# 2. Colour helpers (robotgo GetPixelColor / RgbToHex / HexToRgb)
# ---------------------------------------------------------------------------


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """(r,g,b) -> ``"rrggbb"``. robotgo returns a padded hex string from a pixel."""
    for c in (r, g, b):
        if not 0 <= int(c) <= 255:
            raise ValueError("channel out of range: %r" % (c,))
    return "%02x%02x%02x" % (int(r), int(g), int(b))


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """``"rrggbb"`` (optionally ``#``-prefixed) -> (r,g,b)."""
    s = hex_str.lstrip("#").strip()
    if len(s) != 6:
        raise ValueError("not a 6-digit hex colour: %r" % (hex_str,))
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


# ---------------------------------------------------------------------------
# 3. The template match: normalized cross-correlation on an image array
# ---------------------------------------------------------------------------
#
# robotgo delegates FindBitmap/gcv.FindImg to a C/OpenCV sidecar; the algorithm
# under it is plain NCC template matching. This is a self-contained reference
# implementation of it. It is grayscale (colour is collapsed to luma before
# matching, which is what an anti-aliased monochrome-ish toolbar icon needs and
# what makes the match robust to the theme's exact accent colour), deterministic,
# and free of any dependency at import.


Image = Sequence[Sequence[float]]  # rows of luma values; also accepts np.ndarray


def _to_rows(img: Any) -> List[List[float]]:
    """Coerce any accepted image (nested sequence, or a numpy 2-D/3-D array) to a
    rectangular list-of-rows of floats. A 3-channel array is reduced to luma."""
    # numpy fast path (optional): reduce colour, hand back rows of floats.
    try:
        import numpy as _np  # noqa: WPS433 - optional accelerator only
        if isinstance(img, _np.ndarray):
            a = img.astype("float64")
            if a.ndim == 3:
                a = a[..., :3] @ _np.array([0.299, 0.587, 0.114])
            if a.ndim != 2:
                raise ValueError("image array must be 2-D (grayscale) or 3-D (RGB)")
            return a.tolist()
    except ImportError:
        pass
    rows: List[List[float]] = []
    for row in img:
        out_row: List[float] = []
        for px in row:
            if isinstance(px, (tuple, list)):
                r, g, b = (float(px[0]), float(px[1]), float(px[2]))
                out_row.append(0.299 * r + 0.587 * g + 0.114 * b)
            else:
                out_row.append(float(px))
        rows.append(out_row)
    if rows and any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("image is not rectangular")
    return rows


def _dims(rows: Sequence[Sequence[float]]) -> Tuple[int, int]:
    """(height, width)."""
    return (len(rows), len(rows[0]) if rows else 0)


@dataclass(frozen=True)
class Match:
    """One template hit: the top-left IMAGE pixel, the centre, and the NCC score.

    ``score`` is in ``[-1, 1]``; 1.0 is a perfect correlation. The centre is the
    pixel a click should target — hand it to
    :meth:`harnesscad.io.cua.frames.Frame.to_screen` to get the screen pixel.
    """

    x: int
    y: int
    score: float
    w: int
    h: int

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "score": round(self.score, 6),
                "w": self.w, "h": self.h, "center": list(self.center)}


def _ncc_at(hay: Sequence[Sequence[float]], tpl: Sequence[Sequence[float]],
            th: int, tw: int, tpl_zero: Sequence[float], tpl_norm: float,
            oy: int, ox: int) -> float:
    """Normalized cross-correlation of ``tpl`` against the haystack window whose
    top-left is ``(ox, oy)``. ``tpl_zero``/``tpl_norm`` are the pre-computed
    mean-subtracted template and its L2 norm (the template is constant across all
    windows, so this is hoisted out of the hot loop)."""
    # window mean
    s = 0.0
    for j in range(th):
        hrow = hay[oy + j]
        for i in range(tw):
            s += hrow[ox + i]
    win_mean = s / (th * tw)
    # dot product of zero-mean window with zero-mean template, and window norm
    dot = 0.0
    win_sq = 0.0
    k = 0
    for j in range(th):
        hrow = hay[oy + j]
        for i in range(tw):
            hv = hrow[ox + i] - win_mean
            dot += hv * tpl_zero[k]
            win_sq += hv * hv
            k += 1
    denom = math.sqrt(win_sq) * tpl_norm
    if denom <= 1e-12:
        # A flat window (or flat template) has no correlation structure. A flat
        # template matching a flat window is degenerate, not a hit: return 0.
        return 0.0
    return dot / denom


def match_template(haystack: Image, template: Image, *,
                   threshold: float = 0.9,
                   step: int = 1,
                   max_results: int = 16) -> List[Match]:
    """Every place ``template`` correlates with ``haystack`` above ``threshold``.

    Slides the template over the image and computes normalized cross-correlation
    at each position; returns the peaks, best score first, after a non-maximum
    suppression that removes overlapping duplicates (so one icon yields one hit,
    not a cluster). ``step`` sub-samples the search (``step=2`` quarters the work
    at the cost of at-most-1px localisation, then refined back to exact) for large
    frames; ``max_results`` caps the returned list.

    Deterministic: identical inputs give identical matches in a fixed order. Never
    raises on a template larger than the image — that just yields ``[]``.
    """
    hay = _to_rows(haystack)
    tpl = _to_rows(template)
    hh, hw = _dims(hay)
    th, tw = _dims(tpl)
    if th == 0 or tw == 0 or th > hh or tw > hw:
        return []

    # Pre-compute the zero-mean template and its norm ONCE.
    tsum = sum(sum(r) for r in tpl)
    tmean = tsum / (th * tw)
    tpl_zero: List[float] = []
    for row in tpl:
        for v in row:
            tpl_zero.append(v - tmean)
    tpl_norm = math.sqrt(sum(v * v for v in tpl_zero))

    step = max(1, int(step))
    raw: List[Match] = []
    for oy in range(0, hh - th + 1, step):
        for ox in range(0, hw - tw + 1, step):
            score = _ncc_at(hay, tpl, th, tw, tpl_zero, tpl_norm, oy, ox)
            if score >= threshold:
                raw.append(Match(x=ox, y=oy, score=score, w=tw, h=th))

    raw.sort(key=lambda m: m.score, reverse=True)
    return _suppress_overlaps(raw, tw, th)[:max_results]


def _suppress_overlaps(matches: Sequence[Match], tw: int, th: int) -> List[Match]:
    """Greedy non-maximum suppression: keep the highest-scoring hit, drop any
    later hit whose top-left is within half a template of a kept one. Determinism
    is preserved because ``matches`` is already sorted by score (ties broken by
    scan order)."""
    kept: List[Match] = []
    for m in matches:
        clash = False
        for k in kept:
            if abs(m.x - k.x) < tw // 2 + 1 and abs(m.y - k.y) < th // 2 + 1:
                clash = True
                break
        if not clash:
            kept.append(m)
    return kept


def best_match(haystack: Image, template: Image, *,
               step: int = 1) -> Optional[Match]:
    """The single highest-NCC location of ``template`` in ``haystack``.

    No threshold: this always reports the best correlation (or ``None`` only when
    the template does not fit). Inspect ``.score`` to decide whether it is good
    enough — the CALLER owns the accept threshold, because "how sure is sure" is a
    grounding-policy question, not a matcher question.
    """
    hits = match_template(haystack, template, threshold=-1.0, step=step,
                          max_results=1)
    return hits[0] if hits else None


def locate_icon(haystack: Image, template: Image, frame: Any = None, *,
                threshold: float = 0.9) -> Optional[Tuple[int, int]]:
    """Find a fixed toolbar ICON and return where to CLICK it.

    Returns the icon centre. With no ``frame`` that centre is an image pixel; with
    a :class:`~harnesscad.io.cua.frames.Frame` it is mapped to a screen pixel via
    ``frame.to_screen`` — the exact hand-off that makes template matching a real
    grounding path: bitmap in, screen click out, no model in the middle. Returns
    ``None`` if nothing clears ``threshold``.
    """
    m = best_match(haystack, template)
    if m is None or m.score < threshold:
        return None
    cx, cy = m.center
    if frame is None:
        return (cx, cy)
    return frame.to_screen(cx, cy)
