"""uia — the Windows UIAutomation driver. Every action returns a VERIFIED outcome.

Grounding, not vision
---------------------
FreeCAD is Qt6, and Qt ships a UIA provider that is on by default. A full tree
walk of ``Gui::MainWindow`` returns ~156 elements in 0.19 s, every toolbar button
carries a stable ``Name`` and a Qt-object-path ``AutomationId``, and every dialog
field is exposed under its Qt ``objectName`` (``boxLength``, ``boxWidth``, ...).
So the chrome is **coordinate-free**: we resolve an element by name/id and
``Invoke()`` it. The 3D viewport is ONE opaque node whose rect the tree hands us,
which is exactly the set of pixels — and the only set — that needs vision.

(``win32gui.EnumChildWindows`` sees almost nothing here, because Qt widgets are
not HWNDs. That API is not the accessibility API. Using it would make us conclude
FreeCAD is as blind as Blender, which is false.)

Three measured traps, designed around
-------------------------------------
1. ``ValuePattern.SetValue()`` **silently no-ops on Qt spinboxes** — ``IsReadOnly``
   says False, nothing raises, the value does not change. READS through
   ``ValuePattern.Value`` are reliable; WRITES are not. The only write path that
   works is ``SetFocus()`` -> Ctrl+A -> type. That is the only one implemented
   here; ``SetValue`` is never called.
2. **SendInput only.** ``PostMessage`` returns TRUE for a disabled control, a
   window behind a modal, or a DirectX viewport, and does nothing; it also never
   updates the async key state, so a modifier-aware app sees Ctrl as up. Every
   keystroke below goes through ``SendInput`` (``KEYEVENTF_UNICODE`` for text).
3. **No return value is evidence.** Every action here captures a before-state,
   dispatches, and then POLLS for a change (a value moved, focus moved, the tree
   grew, a window appeared, the title went dirty). No change = no action, and the
   caller gets ``ok=False``, not a success code.

Import-safe without ``uiautomation``: :func:`available` returns False and every
entry point raises a clear :class:`UiaUnavailable` rather than hanging.
"""

from __future__ import annotations

import ctypes
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.io.cua import frames  # MUST be first: sets process DPI awareness
from harnesscad.io.cua.quantity import (
    Locale, QuantityError, QuantityMismatch, WriteReport, detect_locale,
    parse_quantity, values_match, write_quantity,
)


class UiaError(RuntimeError):
    """A UIA action failed, or could not be PROVEN to have happened."""


class UiaUnavailable(UiaError):
    """The uiautomation package (or Windows) is not present."""


def available() -> bool:
    """True if a UIA driver can run here. Never raises, never hangs."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import uiautomation  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _uia():
    if not sys.platform.startswith("win"):
        raise UiaUnavailable("UIAutomation requires Windows (platform=%s)" % sys.platform)
    try:
        import uiautomation as auto
    except Exception as exc:  # noqa: BLE001
        raise UiaUnavailable(
            "the 'uiautomation' package is not installed. It is an optional extra: "
            "pip install harnesscad[cua]  (or: pip install uiautomation)") from exc
    return auto


# --- SendInput (never PostMessage) ------------------------------------------
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004
_INPUT_KEYBOARD = 1
VK_CONTROL = 0x11
VK_A = 0x41
VK_RETURN = 0x0D
VK_TAB = 0x09
VK_ESCAPE = 0x1B


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulonglong))]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("pad", ctypes.c_byte * 32)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("u", _INPUT_UNION)]


def _send(inputs: Sequence[_INPUT]) -> None:
    n = len(inputs)
    arr = (_INPUT * n)(*inputs)
    sent = ctypes.windll.user32.SendInput(n, arr, ctypes.sizeof(_INPUT))
    if sent != n:
        # SendInput's return value is the ONLY input-API return we check, and even
        # then only as a lower bound: it says the events entered the queue, never
        # that anything happened. The read-back is the real proof.
        raise UiaError("SendInput injected %d of %d events (input blocked?)"
                       % (sent, n))


def _key(vk: int, up: bool = False) -> _INPUT:
    inp = _INPUT()
    inp.type = _INPUT_KEYBOARD
    inp.u.ki = _KEYBDINPUT(vk, 0, _KEYEVENTF_KEYUP if up else 0, 0, None)
    return inp


def _char(ch: str, up: bool = False) -> _INPUT:
    inp = _INPUT()
    inp.type = _INPUT_KEYBOARD
    flags = _KEYEVENTF_UNICODE | (_KEYEVENTF_KEYUP if up else 0)
    inp.u.ki = _KEYBDINPUT(0, ord(ch), flags, 0, None)
    return inp


def send_text(text: str, per_char_delay: float = 0.0) -> None:
    """Type literal text through SendInput/KEYEVENTF_UNICODE.

    Unicode injection means no keyboard-layout translation and no escaping: a
    comma is a comma even on a layout where it is a dead key. ``uiautomation``'s
    own SendKeys treats ``{}()+^%`` as syntax — a decimal string must never go
    through that.
    """
    events: List[_INPUT] = []
    for ch in str(text):
        events.append(_char(ch))
        events.append(_char(ch, up=True))
    if not events:
        return
    if per_char_delay <= 0:
        _send(events)
        return
    for i in range(0, len(events), 2):
        _send(events[i:i + 2])
        time.sleep(per_char_delay)


def send_key(vk: int, ctrl: bool = False) -> None:
    events: List[_INPUT] = []
    if ctrl:
        events.append(_key(VK_CONTROL))
    events.append(_key(vk))
    events.append(_key(vk, up=True))
    if ctrl:
        events.append(_key(VK_CONTROL, up=True))
    _send(events)


def select_all() -> None:
    send_key(VK_A, ctrl=True)


# --- elements ----------------------------------------------------------------
@dataclass(frozen=True)
class Element:
    """One UIA node, flattened. ``control`` is the live uiautomation object."""

    name: str
    automation_id: str
    control_type: str
    class_name: str
    rect: Tuple[int, int, int, int]
    enabled: bool
    depth: int
    control: Any = None

    @property
    def key(self) -> str:
        """A stable identity for the deny-list, the trace, and change detection."""
        return "%s|%s|%s" % (self.control_type, self.automation_id, self.name)

    @property
    def center(self) -> Tuple[int, int]:
        left, top, right, bottom = self.rect
        return ((left + right) // 2, (top + bottom) // 2)

    def to_dict(self) -> dict:
        return {"name": self.name, "automation_id": self.automation_id,
                "control_type": self.control_type, "class_name": self.class_name,
                "rect": list(self.rect), "enabled": self.enabled,
                "depth": self.depth}


@dataclass
class Outcome:
    """The VERIFIED result of one action. ``ok`` without ``evidence`` is impossible."""

    ok: bool
    action: str = ""
    target: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    elapsed: float = 0.0

    def to_dict(self) -> dict:
        return {"ok": self.ok, "action": self.action, "target": self.target,
                "evidence": self.evidence, "error": self.error,
                "elapsed": round(self.elapsed, 3)}


@dataclass(frozen=True)
class Snapshot:
    """The observable state of the app, cheap enough to take before EVERY action."""

    title: str
    element_count: int
    element_keys: Tuple[str, ...]
    top_windows: Tuple[str, ...]
    focused: str = ""

    def diff(self, other: "Snapshot") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.title != other.title:
            out["title"] = [self.title, other.title]
        if self.element_count != other.element_count:
            out["element_count"] = [self.element_count, other.element_count]
        new = [k for k in other.element_keys if k not in set(self.element_keys)]
        gone = [k for k in self.element_keys if k not in set(other.element_keys)]
        if new:
            out["appeared"] = new[:20]
        if gone:
            out["disappeared"] = gone[:20]
        if self.top_windows != other.top_windows:
            out["top_windows"] = [list(self.top_windows), list(other.top_windows)]
        if self.focused != other.focused:
            out["focused"] = [self.focused, other.focused]
        return out

    @property
    def dirty(self) -> bool:
        """FreeCAD prefixes an unsaved document's title with '*'. A free tripwire."""
        return self.title.strip().startswith("*")


class UiaDriver:
    """Drives ONE top-level window by its accessibility tree.

    Construction never launches anything: pass a pid (preferred — it scopes every
    lookup to the process we started) or a window class name.
    """

    def __init__(self, pid: Optional[int] = None, class_name: Optional[str] = None,
                 timeout: float = 20.0, max_depth: int = 12) -> None:
        self._auto = _uia()
        self.pid = pid
        self.class_name = class_name
        self.timeout = float(timeout)
        self.max_depth = int(max_depth)
        self._window = None
        self._locale: Optional[Locale] = None
        frames.assert_frames_agree()   # REFUSE to run on a lied-to coordinate frame

    # -- the window --------------------------------------------------------
    def _root_children(self) -> List[Any]:
        """Top-level windows, tolerant of one dying mid-enumeration.

        Another application closing while we walk the desktop raises a COMError
        out of the middle of GetChildren(). That is an event about someone else's
        window, not a failure of ours, so it is retried rather than propagated.
        """
        for _ in range(5):
            try:
                return list(self._auto.GetRootControl().GetChildren())
            except Exception:  # noqa: BLE001 - a foreign window died mid-walk
                time.sleep(0.2)
        return []

    def window(self, refresh: bool = False):
        if self._window is not None and not refresh:
            return self._window
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            for win in self._root_children():
                try:
                    if self.pid is not None and win.ProcessId != self.pid:
                        continue
                    if self.class_name and win.ClassName != self.class_name:
                        continue
                    if not win.ClassName:
                        continue
                except Exception:  # noqa: BLE001
                    continue
                self._window = win
                return win
            time.sleep(0.25)
        raise UiaError("no top-level window for pid=%s class=%s after %.0fs"
                       % (self.pid, self.class_name, self.timeout))

    def window_rect(self) -> Tuple[int, int, int, int]:
        r = self.window().BoundingRectangle
        return (r.left, r.top, r.right, r.bottom)

    def title(self) -> str:
        try:
            return str(self.window().Name or "")
        except Exception:  # noqa: BLE001 - a window that vanished has no title
            return ""

    def is_foreground(self) -> bool:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return self.pid is None or int(pid.value) == int(self.pid)

    def focus_window(self, timeout: float = 5.0) -> bool:
        """Bring our window to the foreground and PROVE it got there.

        Windows can refuse a foreground change (a just-launched app is still
        settling, or another process holds the foreground lock), and SetActive()
        reports nothing about it. So it is retried and then verified -- if the
        window is not actually in front, no input may be dispatched, because the
        keystrokes would land in whatever is.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_foreground():
                return True
            for method in ("SetActive", "SetTopmost", "SetFocus"):
                try:
                    getattr(self.window(), method)()
                except Exception:  # noqa: BLE001
                    continue
                if self.is_foreground():
                    return True
            time.sleep(0.25)
        return self.is_foreground()

    # -- the tree ----------------------------------------------------------
    def tree(self, root=None, max_depth: Optional[int] = None) -> List[Element]:
        """Flatten the a11y tree. ~0.19 s for FreeCAD's 156 elements — cheap enough
        to run before AND after every action, which is what makes verification real."""
        auto = self._auto
        depth_cap = self.max_depth if max_depth is None else int(max_depth)
        out: List[Element] = []

        def walk(ctrl, depth: int) -> None:
            if depth > depth_cap:
                return
            for child in ctrl.GetChildren():
                try:
                    r = child.BoundingRectangle
                    out.append(Element(
                        name=str(child.Name or ""),
                        automation_id=str(child.AutomationId or ""),
                        control_type=str(child.ControlTypeName or ""),
                        class_name=str(child.ClassName or ""),
                        rect=(r.left, r.top, r.right, r.bottom),
                        enabled=bool(child.IsEnabled),
                        depth=depth, control=child))
                except Exception:  # noqa: BLE001 - a node can die mid-walk
                    continue
                walk(child, depth + 1)

        walk(root if root is not None else self.window(), 0)
        return out

    def top_windows(self) -> List[str]:
        """Top-level windows OWNED BY OUR PID. More than one = a modal is up."""
        out = []
        for win in self._root_children():
            try:
                if self.pid is not None and win.ProcessId != self.pid:
                    continue
                if not win.ClassName:
                    continue
                out.append("%s::%s" % (win.ClassName, win.Name or ""))
            except Exception:  # noqa: BLE001
                continue
        return sorted(out)

    def snapshot(self) -> Snapshot:
        elems = self.tree()
        focused = ""
        try:
            f = self._auto.GetFocusedControl()
            if f is not None:
                focused = "%s|%s|%s" % (f.ControlTypeName, f.AutomationId or "",
                                        f.Name or "")
        except Exception:  # noqa: BLE001
            pass
        return Snapshot(title=self.title(), element_count=len(elems),
                        element_keys=tuple(e.key for e in elems),
                        top_windows=tuple(self.top_windows()), focused=focused)

    # -- resolution (never by coordinate when the tree has it) --------------
    def find(self, name: Optional[str] = None, aid_suffix: Optional[str] = None,
             control_type: Optional[str] = None, class_name: Optional[str] = None,
             enabled_only: bool = False,
             elements: Optional[Sequence[Element]] = None) -> Optional[Element]:
        for e in self.find_all(name=name, aid_suffix=aid_suffix,
                               control_type=control_type, class_name=class_name,
                               enabled_only=enabled_only, elements=elements):
            return e
        return None

    def find_all(self, name: Optional[str] = None, aid_suffix: Optional[str] = None,
                 control_type: Optional[str] = None, class_name: Optional[str] = None,
                 enabled_only: bool = False,
                 elements: Optional[Sequence[Element]] = None) -> List[Element]:
        pool = list(elements) if elements is not None else self.tree()
        out = []
        for e in pool:
            if name is not None and e.name != name:
                continue
            if aid_suffix is not None and not e.automation_id.endswith(aid_suffix):
                continue
            if control_type is not None and e.control_type != control_type:
                continue
            if class_name is not None and e.class_name != class_name:
                continue
            if enabled_only and not e.enabled:
                continue
            out.append(e)
        return out

    def wait_for(self, timeout: float = 10.0, poll: float = 0.25, **kw) -> Optional[Element]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            e = self.find(**kw)
            if e is not None:
                return e
            time.sleep(poll)
        return None

    def resolve(self, **kw) -> Element:
        e = self.find(**kw)
        if e is None:
            raise UiaError("no element matches %r" % (kw,))
        return e

    # -- reads (reliable) ---------------------------------------------------
    def read_value(self, element: Element) -> str:
        """``ValuePattern.Value``. Reads ARE reliable on Qt; writes are not."""
        try:
            return str(element.control.GetValuePattern().Value or "")
        except Exception as exc:  # noqa: BLE001
            raise UiaError("cannot read value of %s: %s" % (element.key, exc)) from exc

    def locale(self, elements: Optional[Sequence[Element]] = None) -> Locale:
        """Detect the app's decimal separator from what IT rendered. Cached."""
        if self._locale is not None:
            return self._locale
        samples: List[str] = []
        for e in (elements if elements is not None else self.tree()):
            if e.control_type in ("SpinnerControl", "EditControl"):
                try:
                    samples.append(self.read_value(e))
                except UiaError:
                    continue
        self._locale = detect_locale(samples)
        return self._locale

    # -- actions (each returns a VERIFIED outcome) --------------------------
    def invoke(self, element: Element, settle: float = 2.0,
               poll: float = 0.15) -> Outcome:
        """``InvokePattern.Invoke()``, then POLL for evidence that it did something.

        Evidence = the tree changed / a window appeared / the title went dirty /
        focus moved. Invoke() itself returning is not evidence of anything.
        """
        t0 = time.time()
        if not element.enabled:
            return Outcome(False, "invoke", element.key,
                           error="element is disabled; refusing to dispatch")
        before = self.snapshot()
        try:
            element.control.GetInvokePattern().Invoke()
        except Exception as exc:  # noqa: BLE001
            try:
                element.control.Click(simulateMove=False)  # SendInput under the hood
            except Exception:  # noqa: BLE001
                return Outcome(False, "invoke", element.key,
                               error="Invoke failed: %s" % exc,
                               elapsed=time.time() - t0)
        deadline = time.time() + settle
        while time.time() < deadline:
            time.sleep(poll)
            after = self.snapshot()
            diff = before.diff(after)
            if diff:
                return Outcome(True, "invoke", element.key,
                               evidence={"diff": diff, "dirty": after.dirty},
                               elapsed=time.time() - t0)
        return Outcome(False, "invoke", element.key,
                       error="Invoke() returned but NOTHING CHANGED in %.1fs "
                             "(no tree change, no new window, no title change). "
                             "An unverified action is not an action." % settle,
                       elapsed=time.time() - t0)

    def focus(self, element: Element, timeout: float = 4.0) -> Outcome:
        """SetFocus, then PROVE focus moved. A field we cannot focus is a field we
        cannot type into -- and typing into an unfocused field sends the keystrokes
        somewhere else entirely."""
        t0 = time.time()
        # Keyboard input follows the FOREGROUND window, so it must be ours first.
        if not self.focus_window():
            return Outcome(False, "focus", element.key,
                           error="the target window is not in the foreground",
                           elapsed=time.time() - t0)
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            try:
                element.control.SetFocus()
            except Exception as exc:  # noqa: BLE001
                last = str(exc)
            try:
                f = self._auto.GetFocusedControl()
            except Exception:  # noqa: BLE001
                f = None
            if f is not None:
                aid = str(f.AutomationId or "")
                # A Qt spinbox delegates focus to its internal line edit, whose
                # AutomationId is the spinbox's path plus a suffix -- so a prefix
                # match either way is the honest test of "focus is in this widget".
                if aid and element.automation_id and (
                        aid == element.automation_id
                        or aid.startswith(element.automation_id)
                        or element.automation_id.startswith(aid)):
                    return Outcome(True, "focus", element.key,
                                   evidence={"focused": aid},
                                   elapsed=time.time() - t0)
                last = "focus is on %r" % aid
            time.sleep(0.1)
        return Outcome(False, "focus", element.key,
                       error="SetFocus() returned but focus did not move (%s)" % last,
                       elapsed=time.time() - t0)

    def set_quantity(self, element: Element, value: float,
                     locale: Optional[Locale] = None,
                     decimals: Optional[int] = None) -> Tuple[Outcome, Optional[WriteReport]]:
        """THE ONLY NUMERIC WRITE PATH: focus -> select-all -> SendInput -> READ BACK.

        ``ValuePattern.SetValue`` is never called: on Qt spinboxes it reports
        success and does nothing. The read-back compare in
        :func:`quantity.write_quantity` is what turns the comma-locale 37.5 -> 375
        catastrophe into a hard, loud failure.
        """
        t0 = time.time()
        foc = self.focus(element)
        if not foc.ok:
            return Outcome(False, "set_quantity", element.key,
                           error="cannot focus the field: " + foc.error,
                           elapsed=time.time() - t0), None
        loc = locale if locale is not None else self.locale()

        def _type(text: str) -> None:
            element.control.SetFocus()
            select_all()
            send_text(text)
            # COMMIT before reading. A Qt quantity spinbox holds the RAW TEXT until
            # it is committed: read straight after typing and '37.5' reads back as
            # '37.5' — and only when the widget commits does it become '375,00 mm'.
            # A read-back taken before the commit would PASS and the part would be
            # ten times too long. Tab commits and moves focus out.
            send_key(VK_TAB)
            time.sleep(0.15)

        def _read() -> str:
            return self.read_value(element)

        try:
            report = write_quantity(element.automation_id.rsplit(".", 1)[-1]
                                    or element.name,
                                    float(value), _type, _read, locale=loc,
                                    decimals=decimals)
        except QuantityMismatch as exc:
            return Outcome(False, "set_quantity", element.key, error=str(exc),
                           evidence={"intended": float(value),
                                     "read_text": exc.read_text,
                                     "read_value": exc.read_value,
                                     "factor": exc.factor},
                           elapsed=time.time() - t0), None
        except Exception as exc:  # noqa: BLE001
            return Outcome(False, "set_quantity", element.key, error=str(exc),
                           elapsed=time.time() - t0), None
        return Outcome(True, "set_quantity", element.key,
                       evidence=report.to_dict(),
                       elapsed=time.time() - t0), report

    def verify_quantity(self, element: Element, value: float,
                        locale: Optional[Locale] = None) -> Outcome:
        """Re-read a field and compare. Run over EVERY field again just before OK:
        a later field's edit can perturb an earlier one (Qt spinboxes clamp, and a
        dialog can re-derive one field from another), so the write-time read-back
        is necessary but not sufficient."""
        t0 = time.time()
        loc = locale if locale is not None else self.locale()
        text = self.read_value(element)
        try:
            parsed, _unit = parse_quantity(text, loc)
        except QuantityError as exc:
            return Outcome(False, "verify_quantity", element.key, error=str(exc),
                           elapsed=time.time() - t0)
        if not values_match(value, parsed):
            return Outcome(False, "verify_quantity", element.key,
                           evidence={"intended": float(value), "read_text": text,
                                     "read_value": parsed},
                           error="field drifted: intended %r, reads %r (%r)"
                                 % (value, parsed, text),
                           elapsed=time.time() - t0)
        return Outcome(True, "verify_quantity", element.key,
                       evidence={"intended": float(value), "read_text": text,
                                 "read_value": parsed},
                       elapsed=time.time() - t0)

    def press(self, vk: int, ctrl: bool = False, settle: float = 0.4) -> Outcome:
        t0 = time.time()
        before = self.snapshot()
        send_key(vk, ctrl=ctrl)
        time.sleep(settle)
        after = self.snapshot()
        diff = before.diff(after)
        return Outcome(bool(diff), "press", "vk=0x%02X" % vk,
                       evidence={"diff": diff},
                       error="" if diff else "key press changed nothing",
                       elapsed=time.time() - t0)

    # -- the one region that needs vision ----------------------------------
    def viewport_frame(self, aid_contains: str = "View3DInventorViewer",
                       max_w: int = 1280, max_h: int = 800) -> frames.Frame:
        """The 3D viewport's rect, as an immutable :class:`frames.Frame`.

        Selected by the AutomationId PATH, never by ClassName: there is a DECOY
        ``QOpenGLWidget`` at (280,127,380,157) — a 100x30 stale widget outside the
        window — and a pipeline that cropped to it would silently feed the model a
        sliver of desktop.
        """
        for e in self.tree():
            if aid_contains in e.automation_id:
                left, top, right, bottom = e.rect
                if right - left < 64 or bottom - top < 64:
                    continue
                return frames.window_frame(e.rect, max_w, max_h, label="viewport")
        raise UiaError("no viewport element whose AutomationId contains %r"
                       % aid_contains)
