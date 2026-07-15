"""windowing — focus/raise discipline, self-occlusion, and monitor targeting.

A DISTINCT extension of :mod:`harnesscad.io.cua.frames`. ``frames`` owns the
coordinate MATHS (DPI awareness, the one immutable letterboxed frame, the
virtual-desktop map). This module owns the WINDOW-STATE discipline that has to
hold before that frame is even allowed to produce a click, and it is built from
three findings the native-desktop repos ship and the a11y repo documents:

**1. The observation must not contain the agent's own UI.** The native Tauri
agent hides its border overlay and its voice orb, waits ~30 ms for the window
server, takes the screenshot EXCLUDING its own top window, then restores the
overlays. If it did not, the model would narrate — and try to click — the
agent's own chrome. For a CAD run our own status HUD, a highlight overlay, or a
"thinking" spinner sitting over the viewport is exactly this hazard.
:class:`OcclusionRegistry` records the rects we own and
:meth:`OcclusionRegistry.contaminates` refuses a frame whose content box a
registered overlay overlaps, so a contaminated screenshot is caught, not clicked.

**2. Focus must be OURS, and stolen focus is a HALT, not a retry.** The a11y repo
keeps a structured ``get_app_state`` — frontmost app, window title, focused UI
element — and its focus-behavior study treats "did the frontmost app change" as
the stable signal. :class:`FocusDiscipline` snapshots the frontmost app before a
step and :meth:`detect_theft` flags a step where a FOREIGN window became
frontmost: keystrokes meant for a CAD dialog would have landed in whatever grabbed
focus, so the run must halt rather than type into the void.

**3. A window lives on ONE monitor; pick it, don't guess.** With a negative-origin
left-hand monitor in play (see ``frames``), :func:`monitor_for_rect` chooses the
monitor a window rect actually belongs to by maximum overlap area, so the frame is
built against the right screen.

Stdlib only. Pure functions / frozen dataclasses over rect-shaped tuples and
snapshot dicts — no live GUI, fully unit-testable. Imports ``frames`` first, per
the DPI-before-anything rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.io.cua import frames  # noqa: F401 - imported FIRST for the DPI one-shot

Rect = Tuple[int, int, int, int]  # (left, top, right, bottom), virtual-desktop px


def _intersection_area(a: Rect, b: Rect) -> int:
    al, at, ar, ab = a
    bl, bt, br, bb = b
    iw = min(ar, br) - max(al, bl)
    ih = min(ab, bb) - max(at, bt)
    if iw <= 0 or ih <= 0:
        return 0
    return iw * ih


def rects_overlap(a: Rect, b: Rect) -> bool:
    return _intersection_area(a, b) > 0


# --- 1: self-occlusion ------------------------------------------------------
@dataclass(frozen=True)
class Overlay:
    """A rect the AGENT owns and that must never appear in the observation."""

    name: str
    rect: Rect
    hideable: bool = True  # can we hide it before capture, or only avoid it?


class OcclusionRegistry:
    """The agent's own on-screen surfaces. A screenshot the model sees must be
    free of them: either hide the hideable ones before capture (and restore
    after), or REFUSE the frame when a non-hideable overlay sits over its content.
    """

    def __init__(self) -> None:
        self._overlays: Dict[str, Overlay] = {}

    def register(self, overlay: Overlay) -> None:
        self._overlays[overlay.name] = overlay

    def unregister(self, name: str) -> None:
        self._overlays.pop(name, None)

    def overlays(self) -> List[Overlay]:
        return list(self._overlays.values())

    def hideable(self) -> List[Overlay]:
        """The overlays to hide before a capture (order is stable by name)."""
        return sorted((o for o in self._overlays.values() if o.hideable),
                      key=lambda o: o.name)

    def contaminates(self, content_rect: Rect) -> List[Overlay]:
        """Overlays that overlap the frame's CONTENT box. Non-empty means the
        screenshot would show the agent's own UI; the capture is not honest."""
        return [o for o in self._overlays.values()
                if rects_overlap(o.rect, content_rect)]

    def blocking(self, content_rect: Rect) -> List[Overlay]:
        """Contaminating overlays that CANNOT be hidden — these make the frame
        unusable rather than merely needing a hide/restore dance."""
        return [o for o in self.contaminates(content_rect) if not o.hideable]


# --- 2: focus / raise discipline -------------------------------------------
@dataclass(frozen=True)
class WindowState:
    """The ``get_app_state`` snapshot: what is frontmost, and what has keyboard
    focus, at one instant. ``owned`` is whether the frontmost window is ours."""

    frontmost_app: str
    window_title: str = ""
    focused_element: str = ""
    owned: bool = False

    def to_dict(self) -> dict:
        return {"frontmost_app": self.frontmost_app,
                "window_title": self.window_title,
                "focused_element": self.focused_element, "owned": self.owned}


@dataclass
class FocusDiscipline:
    """Owns the "is the target window ours, and in front" question over a step.

    ``target_app`` is the process/app name we expect to be driving. ``baseline``
    is the frontmost app recorded just before a step. Nothing here raises; the
    run loop decides what a theft means (it should HALT, per guardrails).
    """

    target_app: str
    baseline: Optional[WindowState] = None

    def snapshot(self, state: WindowState) -> WindowState:
        self.baseline = state
        return state

    def is_foreground(self, state: WindowState) -> bool:
        """Is our target app the frontmost, owned window right now?"""
        return state.owned and state.frontmost_app == self.target_app

    def detect_theft(self, after: WindowState) -> Optional[str]:
        """A description of a focus theft during the step, or None.

        Theft = the frontmost app is no longer our target. It is reported even if
        the baseline was also not us (a step that never had focus is still unsafe
        to have typed into), so the caller gets a truthful signal every time.
        """
        if self.is_foreground(after):
            return None
        return ("focus is on %r (title %r), not the target app %r; input this "
                "step would have landed in the wrong window"
                % (after.frontmost_app, after.window_title, self.target_app))


def raise_plan(current: WindowState, target_app: str) -> List[str]:
    """The minimal, ORDERED steps to bring our target to the foreground.

    Non-intrusive by construction (the a11y repo's principle): if we are already
    foreground we do nothing — we never raise a window that is already raised,
    which is what would yank the user's focus for no reason. Returned as symbolic
    step names so the actual dispatch stays in the driver, testable here.
    """
    if current.owned and current.frontmost_app == target_app:
        return []
    return ["activate:%s" % target_app, "raise", "verify_foreground"]


# --- 3: monitor targeting ---------------------------------------------------
def monitor_for_rect(rect: Rect, monitors: Sequence["frames.Monitor"]
                     ) -> Optional["frames.Monitor"]:
    """The monitor a window rect belongs to: maximum overlap area wins.

    Handles the negative-origin left-hand monitor correctly because it works in
    virtual-desktop coordinates throughout. Returns None if the rect overlaps no
    monitor (fully off-screen), which the caller should treat as an error, never
    as "monitor 0".
    """
    best: Optional[frames.Monitor] = None
    best_area = 0
    for mon in monitors:
        area = _intersection_area(rect, mon.rect)
        if area > best_area:
            best_area, best = area, mon
    return best


def frame_for_window(rect: Rect, monitors: Sequence["frames.Monitor"],
                     max_w: int = 1280, max_h: int = 800,
                     label: str = "window") -> "frames.Frame":
    """A letterboxed :class:`frames.Frame` over a window, tagged with the monitor
    it actually sits on. Refuses a fully off-screen window rather than guessing.
    """
    mon = monitor_for_rect(rect, monitors)
    if mon is None and monitors:
        raise frames.FrameError(
            "window rect %r overlaps no monitor; refusing to guess a screen"
            % (rect,))
    left, top, right, bottom = rect
    return frames.Frame.letterbox(left, top, right - left, bottom - top,
                                  max_w, max_h,
                                  monitor=(mon.index if mon else 0), label=label)
