"""frames — DPI awareness and the ONE immutable coordinate frame.

Five frames are in play whenever a pixel is involved:

    F1 physical pixels -> F2 logical/DIP points -> F3 virtual-desktop screen
    space (a left-hand monitor has NEGATIVE x) -> F4 window-local -> F5 the
    model's downscaled image space.

Two silent bugs live in there, and both are shipped by real CUA repos:

**Process DPI awareness.** An unaware process on a 150%-scaled display is *lied
to* by Windows: ``GetSystemMetrics`` returns virtualised logical pixels while UIA
``BoundingRectangle`` and the screenshot are physical. Clicks then land at
``physical / 1.5`` — systematically off by a third, and it looks like "the model
is a bit inaccurate" rather than a bug. :func:`ensure_dpi_aware` calls
``SetProcessDpiAwareness(2)`` and it MUST run before pyautogui or any UIA object
is touched (the setting is process-wide, one-shot and irreversible), which is why
it runs at *import* of this module and why every other cua module imports this
one first. :func:`assert_frames_agree` then checks ``GetSystemMetrics`` against
the primary monitor's physical rect and **REFUSES to run** if they disagree.

**Aspect-distorting downscale.** Two of the three reference repos resize
screenshots with a non-uniform ``fit: fill`` / ``resize_exact``, so a 16:9 screen
is squashed into 16:10 — circles become ellipses. For an agent whose whole job is
to judge whether a fillet is round, that is indefensible. :func:`letterbox`
computes ONE uniform ``scale = min(w_ratio, h_ratio)`` and pads.

**A frame is never inferred.** TuriX guesses whether the model meant 0-1 or
0-1000 from the *magnitude* of the number it was given. Here a :class:`Frame` is
constructed with the screenshot, carried with it, and an action without a frame
is REJECTED, not guessed.

Stdlib only (``ctypes``). Works on non-Windows as a pure-math module: the frame
maths is portable and unit-tested everywhere; only the DPI/monitor probes are
Windows-gated.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

# --- DPI: THE FIRST EXECUTABLE LINE OF THE CUA STACK ------------------------
# Nothing above this may import pyautogui or uiautomation. This runs at module
# import, and every other cua module imports this module first.
_DPI_STATE = {"aware": False, "detail": "not-windows"}


def _set_dpi_awareness() -> Tuple[bool, str]:
    if not sys.platform.startswith("win"):
        return False, "not-windows"
    import ctypes

    # PROCESS_PER_MONITOR_DPI_AWARE = 2. Prefer the per-monitor-v2 context (Win10
    # 1703+) which also makes non-client areas scale correctly; fall back.
    try:
        ctx = ctypes.c_void_p(-4)  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctx):
            return True, "per-monitor-v2"
    except Exception:  # noqa: BLE001 - older Windows: fall through
        pass
    try:
        hr = ctypes.windll.shcore.SetProcessDpiAwareness(2)
        # S_OK (0) or E_ACCESSDENIED (already set by the host process) both mean
        # the process IS aware from here on.
        if hr in (0, -2147024891):
            return True, "per-monitor"
    except Exception as exc:  # noqa: BLE001
        return False, "SetProcessDpiAwareness failed: %s" % exc
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        return True, "system-aware"
    except Exception as exc:  # noqa: BLE001
        return False, "SetProcessDPIAware failed: %s" % exc


_DPI_STATE["aware"], _DPI_STATE["detail"] = _set_dpi_awareness()


def ensure_dpi_aware() -> bool:
    """True if this process is DPI aware (idempotent; the work happened at import)."""
    return bool(_DPI_STATE["aware"])


def dpi_detail() -> str:
    return str(_DPI_STATE["detail"])


class FrameError(RuntimeError):
    """A coordinate frame is missing, inconsistent, or degenerate. Never guessed."""


# --- monitors ---------------------------------------------------------------
@dataclass(frozen=True)
class Monitor:
    """One physical monitor, in virtual-desktop coordinates (x may be NEGATIVE)."""

    index: int
    x: int
    y: int
    width: int
    height: int
    primary: bool = False

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


def monitors() -> List[Monitor]:
    """Enumerate monitors in physical pixels, virtual-desktop origin. [] off-Windows."""
    if not sys.platform.startswith("win"):
        return []
    import ctypes
    from ctypes import wintypes

    out: List[Monitor] = []

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", RECT),
                    ("rcWork", RECT), ("dwFlags", ctypes.c_ulong)]

    proc_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulonglong,
                                   ctypes.c_ulonglong, ctypes.POINTER(RECT),
                                   ctypes.c_double)

    def _cb(hmon, _hdc, _lprect, _data):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        ctypes.windll.user32.GetMonitorInfoW(ctypes.c_ulonglong(hmon),
                                             ctypes.byref(info))
        r = info.rcMonitor
        out.append(Monitor(index=len(out), x=r.left, y=r.top,
                           width=r.right - r.left, height=r.bottom - r.top,
                           primary=bool(info.dwFlags & 1)))
        return 1

    ctypes.windll.user32.EnumDisplayMonitors(0, 0, proc_type(_cb), 0)
    return out


def primary_monitor() -> Optional[Monitor]:
    for m in monitors():
        if m.primary:
            return m
    mons = monitors()
    return mons[0] if mons else None


def system_metrics() -> Tuple[int, int]:
    """``GetSystemMetrics(SM_CXSCREEN, SM_CYSCREEN)``, i.e. the primary screen."""
    if not sys.platform.startswith("win"):
        return (0, 0)
    import ctypes
    gsm = ctypes.windll.user32.GetSystemMetrics
    return (int(gsm(0)), int(gsm(1)))


def assert_frames_agree() -> Tuple[int, int]:
    """REFUSE to run unless the frames agree. Returns the primary screen size.

    ``GetSystemMetrics`` (which a click path uses) must equal the primary
    monitor's PHYSICAL rect (which UIA and the screenshot use). If a DPI-unaware
    process is on a scaled display these differ, and every click is silently off
    by the scale factor. One assertion kills the whole bug class.
    """
    if not sys.platform.startswith("win"):
        raise FrameError("cua frames require Windows (platform=%s)" % sys.platform)
    if not ensure_dpi_aware():
        raise FrameError("process is not DPI aware (%s); refusing to run"
                         % dpi_detail())
    sw, sh = system_metrics()
    mon = primary_monitor()
    if mon is None:
        raise FrameError("no monitors enumerated; refusing to run")
    if (sw, sh) != (mon.width, mon.height):
        raise FrameError(
            "coordinate frames disagree: GetSystemMetrics=%dx%d but the primary "
            "monitor's physical rect is %dx%d. The process is being lied to by "
            "DPI virtualisation; every click would be off by the scale factor. "
            "Refusing to run." % (sw, sh, mon.width, mon.height))
    if _PYAUTOGUI_SIZE is not None and _PYAUTOGUI_SIZE != (sw, sh):
        raise FrameError(
            "coordinate frames disagree: GetSystemMetrics=%dx%d but "
            "pyautogui.size()=%dx%d. Refusing to run."
            % (sw, sh, _PYAUTOGUI_SIZE[0], _PYAUTOGUI_SIZE[1]))
    return (sw, sh)


#: pyautogui is optional and is NOT imported here (importing it has side effects
#: and it must never be imported before the DPI call above). A caller that has it
#: may register its reported size, and assert_frames_agree() will then check it.
_PYAUTOGUI_SIZE: Optional[Tuple[int, int]] = None


def register_screen_size(width: int, height: int) -> None:
    """Register a third-party library's idea of the screen, to be cross-checked."""
    global _PYAUTOGUI_SIZE
    _PYAUTOGUI_SIZE = (int(width), int(height))


# --- the frame --------------------------------------------------------------
@dataclass(frozen=True)
class Frame:
    """An IMMUTABLE map between an image and the screen. Created WITH a screenshot,
    carried WITH it, never recomputed and never inferred from data.

    The source region ``(origin_x, origin_y, src_w, src_h)`` is in virtual-desktop
    screen pixels (a left-hand monitor gives a negative origin). The image is that
    region scaled UNIFORMLY by ``scale`` and letterboxed into ``(dst_w, dst_h)``
    with ``(pad_x, pad_y)`` of padding. Aspect ratio is therefore preserved by
    construction: a circle on screen is a circle in the image.
    """

    origin_x: int
    origin_y: int
    src_w: int
    src_h: int
    scale: float
    pad_x: int
    pad_y: int
    dst_w: int
    dst_h: int
    monitor: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        if self.src_w <= 0 or self.src_h <= 0:
            raise FrameError("degenerate source region %dx%d" % (self.src_w, self.src_h))
        if self.scale <= 0:
            raise FrameError("non-positive scale %r" % (self.scale,))
        if self.dst_w <= 0 or self.dst_h <= 0:
            raise FrameError("degenerate image %dx%d" % (self.dst_w, self.dst_h))

    # -- construction ---------------------------------------------------
    @classmethod
    def identity(cls, origin_x: int, origin_y: int, width: int, height: int,
                 monitor: int = 0, label: str = "") -> "Frame":
        """A 1:1 frame: the image IS the screen region, no scaling, no padding."""
        return cls(origin_x=int(origin_x), origin_y=int(origin_y),
                   src_w=int(width), src_h=int(height), scale=1.0,
                   pad_x=0, pad_y=0, dst_w=int(width), dst_h=int(height),
                   monitor=monitor, label=label)

    @classmethod
    def letterbox(cls, origin_x: int, origin_y: int, width: int, height: int,
                  max_w: int, max_h: int, monitor: int = 0,
                  label: str = "", allow_upscale: bool = False) -> "Frame":
        """LETTERBOX, never stretch. ONE uniform scale; pad the remainder."""
        if width <= 0 or height <= 0:
            raise FrameError("degenerate source region %dx%d" % (width, height))
        if max_w <= 0 or max_h <= 0:
            raise FrameError("degenerate target %dx%d" % (max_w, max_h))
        scale = min(max_w / float(width), max_h / float(height))
        if not allow_upscale:
            scale = min(scale, 1.0)
        # The scaled content, and the padding that centres it.
        cw = max(1, int(round(width * scale)))
        ch = max(1, int(round(height * scale)))
        dst_w, dst_h = (max_w, max_h) if allow_upscale or scale < 1.0 else (cw, ch)
        return cls(origin_x=int(origin_x), origin_y=int(origin_y),
                   src_w=int(width), src_h=int(height), scale=scale,
                   pad_x=(dst_w - cw) // 2, pad_y=(dst_h - ch) // 2,
                   dst_w=dst_w, dst_h=dst_h, monitor=monitor, label=label)

    # -- the maps -------------------------------------------------------
    def to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Image pixel -> virtual-desktop screen pixel."""
        if not self.contains_image(x, y):
            raise FrameError(
                "image point (%s, %s) is outside the frame's content box "
                "(padding is not part of the screen)" % (x, y))
        sx = (x - self.pad_x) / self.scale + self.origin_x
        sy = (y - self.pad_y) / self.scale + self.origin_y
        return (int(round(sx)), int(round(sy)))

    def to_image(self, x: float, y: float) -> Tuple[int, int]:
        """Virtual-desktop screen pixel -> image pixel."""
        ix = (x - self.origin_x) * self.scale + self.pad_x
        iy = (y - self.origin_y) * self.scale + self.pad_y
        return (int(round(ix)), int(round(iy)))

    def contains_image(self, x: float, y: float) -> bool:
        """Is this image point inside the CONTENT (not the letterbox padding)?"""
        cw = self.src_w * self.scale
        ch = self.src_h * self.scale
        return (self.pad_x <= x <= self.pad_x + cw
                and self.pad_y <= y <= self.pad_y + ch)

    def contains_screen(self, x: float, y: float) -> bool:
        return (self.origin_x <= x <= self.origin_x + self.src_w
                and self.origin_y <= y <= self.origin_y + self.src_h)

    def normalized_to_screen(self, u: float, v: float) -> Tuple[int, int]:
        """Frame-local (0..1, 0..1) -> screen. The ONLY normalised space we accept.

        There is no magnitude heuristic here: 0.5 means half way, always. A value
        outside [0, 1] is an error, not a hint about which space was meant.
        """
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            raise FrameError("normalised point (%s, %s) outside [0,1]" % (u, v))
        return (int(round(self.origin_x + u * self.src_w)),
                int(round(self.origin_y + v * self.src_h)))

    def subframe(self, x: int, y: int, w: int, h: int, label: str = "") -> "Frame":
        """A 1:1 frame over a sub-rect of this frame's SCREEN region (e.g. the
        3D viewport). Used for full-resolution zoom: no downscale at all."""
        if not (self.contains_screen(x, y) and self.contains_screen(x + w, y + h)):
            raise FrameError("subframe (%d,%d,%d,%d) escapes its parent %r"
                             % (x, y, w, h, self.screen_rect))
        return Frame.identity(x, y, w, h, monitor=self.monitor,
                              label=label or (self.label + "/sub"))

    @property
    def screen_rect(self) -> Tuple[int, int, int, int]:
        return (self.origin_x, self.origin_y,
                self.origin_x + self.src_w, self.origin_y + self.src_h)

    @property
    def aspect_preserved(self) -> bool:
        """Always True by construction — kept as an explicit, assertable invariant."""
        return True

    def to_dict(self) -> dict:
        return {"origin": [self.origin_x, self.origin_y],
                "src": [self.src_w, self.src_h],
                "dst": [self.dst_w, self.dst_h],
                "scale": self.scale, "pad": [self.pad_x, self.pad_y],
                "monitor": self.monitor, "label": self.label}


def screen_frame(max_w: int = 1280, max_h: int = 800) -> Frame:
    """A letterboxed frame over the primary monitor. Asserts the frames agree first."""
    assert_frames_agree()
    mon = primary_monitor()
    return Frame.letterbox(mon.x, mon.y, mon.width, mon.height, max_w, max_h,
                           monitor=mon.index, label="primary")


def window_frame(rect: Tuple[int, int, int, int], max_w: int = 1280,
                 max_h: int = 800, label: str = "window") -> Frame:
    """A letterboxed frame over a window/viewport rect (l, t, r, b) in screen px."""
    left, top, right, bottom = rect
    return Frame.letterbox(left, top, right - left, bottom - top,
                           max_w, max_h, label=label)
