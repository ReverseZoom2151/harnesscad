"""guardrails — resolve-before-click, so we can REFUSE.

The architectural advantage of a11y grounding, stated plainly: **a vision agent
does not know what it is about to hit until after it has hit it.** We resolve the
element first, read its ``Name``, and can therefore refuse *before* the click.
That is what makes it defensible to drive a real, licensed CAD application on a
user's real machine with their real, unsaved, possibly irreplaceable files open.

Five guards, all cheap:

1. **Deny-list on Name.** Save / Save As / Save All / Don't Save / Discard /
   Overwrite / Replace / Delete / Exit / Quit / Close Without Saving. Checked on
   the RESOLVED element, not on the model's intent — the prompt is not a security
   boundary.
2. **The agent NEVER saves.** The harness owns all file I/O and exports to a
   scratch path it chose (:class:`Scratch`). There is no code path in this package
   that invokes the application's Save.
3. **Sacrificial copy.** A user file is never opened. It is copied into the
   scratch directory and the copy is what gets driven; if the run dies mid-way the
   original is untouched.
4. **Scope.** A click whose element is outside the target window's rect — or
   whose window is not ours — is refused. When a menu closes unexpectedly the
   agent must not click the user's browser, Slack, or the taskbar.
5. **Modal gate + dirty tripwire.** Top-level windows owned by our PID are
   enumerated every step; an unexpected modal (crash reporter, "unsaved changes",
   license nag) HALTS the loop rather than being dismissed by a model guessing at
   a button. The ``*`` FreeCAD puts in a dirty document's title is a free signal
   that an op really did mutate the document.

Stdlib only. Every check is a pure function over an element-shaped object, so the
whole policy is unit-testable with no GUI running.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


class GuardrailViolation(RuntimeError):
    """An action was REFUSED before it happened. Carries why, for the trace."""

    def __init__(self, rule: str, target: str, detail: str = "") -> None:
        self.rule = rule
        self.target = target
        self.detail = detail
        super().__init__("refused by guardrail '%s': %s%s"
                         % (rule, target, (" -- " + detail) if detail else ""))


#: Exact control names that are never invoked. Compared case-insensitively with
#: the mnemonic ampersand and trailing ellipsis stripped ('&Save As...' -> 'save as').
DENY_NAMES: Set[str] = {
    "save", "save as", "save a copy", "save all", "save selection",
    "don't save", "dont save", "do not save", "discard", "discard changes",
    "close without saving", "overwrite", "replace", "replace file",
    "delete", "delete document", "remove file", "erase",
    "exit", "quit", "restore", "revert",
}

#: Substring patterns on the name (a menu may be 'Save As...' or 'Export as STEP').
DENY_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\bsave\b"),
    re.compile(r"\boverwrite\b"),
    re.compile(r"\bdiscard\b"),
    re.compile(r"\bdelete\b"),
    re.compile(r"\bexit\b"),
    re.compile(r"\bquit\b"),
)

#: A bare 'Yes'/'OK' is only dangerous in a dialog that is ASKING to destroy
#: something. The dialog's own text decides.
DANGEROUS_DIALOG = re.compile(
    r"(unsaved|overwrit|replac|discard|delete|not saved|save changes)", re.I)

#: Windows that may legitimately exist besides the main one. Anything else is an
#: unexpected modal and HALTS the run.
DEFAULT_EXPECTED_WINDOW_CLASSES: Tuple[str, ...] = ("Gui::MainWindow",)


def normalize_name(name: str) -> str:
    """'&Save As...' -> 'save as'. Mnemonics and ellipses are not a disguise."""
    text = str(name or "").replace("&", "").strip()
    text = text.rstrip(".…").strip()
    return " ".join(text.lower().split())


def is_denied(name: str) -> Optional[str]:
    """The deny rule this name trips, or None."""
    norm = normalize_name(name)
    if not norm:
        return None
    if norm in DENY_NAMES:
        return "deny-name"
    for pat in DENY_PATTERNS:
        if pat.search(norm):
            return "deny-pattern"
    return None


def is_confirm_in_dangerous_dialog(name: str, dialog_text: str) -> bool:
    """'Yes'/'OK' inside an overwrite/discard dialog: refused, and the run halts."""
    return (normalize_name(name) in ("yes", "ok", "continue")
            and bool(DANGEROUS_DIALOG.search(str(dialog_text or ""))))


def rect_contains(outer: Sequence[int], inner: Sequence[int]) -> bool:
    ol, ot, orr, ob = outer
    il, it, ir, ib = inner
    return ol <= il and ot <= it and ir <= orr and ib <= ob


@dataclass
class Refusal:
    rule: str
    target: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {"rule": self.rule, "target": self.target, "detail": self.detail}


@dataclass
class Guardrails:
    """The policy. ``check_*`` returns a :class:`Refusal` or None; ``enforce``
    raises. Nothing here needs a live GUI."""

    window_rect: Optional[Tuple[int, int, int, int]] = None
    expected_window_classes: Tuple[str, ...] = DEFAULT_EXPECTED_WINDOW_CLASSES
    allow_names: Tuple[str, ...] = ()
    refusals: List[Refusal] = field(default_factory=list)

    # -- 1 + 2: the deny-list ---------------------------------------------
    def check_element(self, element: Any, dialog_text: str = "") -> Optional[Refusal]:
        name = getattr(element, "name", "") or ""
        target = getattr(element, "key", None) or name
        if normalize_name(name) in {normalize_name(a) for a in self.allow_names}:
            return None
        rule = is_denied(name)
        if rule:
            return self._refuse(rule, target,
                                "the agent never saves, deletes, discards or exits; "
                                "the harness owns all file I/O")
        if is_confirm_in_dangerous_dialog(name, dialog_text):
            return self._refuse("deny-confirm", target,
                                "confirmation inside a destructive dialog (%r)"
                                % dialog_text[:80])
        if not getattr(element, "enabled", True):
            return self._refuse("disabled", target,
                                "control is disabled; a dispatch to it would "
                                "report success and do nothing")
        return None

    # -- 4: scope ----------------------------------------------------------
    def check_scope(self, element: Any) -> Optional[Refusal]:
        if self.window_rect is None:
            return None
        rect = getattr(element, "rect", None)
        if not rect:
            return None
        if not rect_contains(self.window_rect, rect):
            return self._refuse("out-of-scope", getattr(element, "key", "?"),
                                "element rect %r is outside the target window %r"
                                % (tuple(rect), tuple(self.window_rect)))
        return None

    def check_point(self, x: int, y: int) -> Optional[Refusal]:
        if self.window_rect is None:
            return self._refuse("no-frame", "(%d,%d)" % (x, y),
                                "a click without a frame is never guessed")
        left, top, right, bottom = self.window_rect
        if not (left <= x <= right and top <= y <= bottom):
            return self._refuse("out-of-scope", "(%d,%d)" % (x, y),
                                "point is outside the target window %r"
                                % (self.window_rect,))
        return None

    # -- 5: modals + dirty tripwire ---------------------------------------
    def check_modals(self, top_windows: Iterable[str]) -> Optional[Refusal]:
        unexpected = [w for w in top_windows
                      if not any(w.startswith(c) for c in self.expected_window_classes)]
        if unexpected:
            return self._refuse("unexpected-modal", ", ".join(unexpected),
                                "HALT: an unexpected top-level window appeared. It is "
                                "never dismissed by guessing at a button.")
        return None

    # -- composition -------------------------------------------------------
    def check(self, element: Any, top_windows: Iterable[str] = (),
              dialog_text: str = "") -> Optional[Refusal]:
        for refusal in (self.check_modals(top_windows),
                        self.check_element(element, dialog_text),
                        self.check_scope(element)):
            if refusal is not None:
                return refusal
        return None

    def enforce(self, element: Any, top_windows: Iterable[str] = (),
                dialog_text: str = "") -> None:
        refusal = self.check(element, top_windows, dialog_text)
        if refusal is not None:
            raise GuardrailViolation(refusal.rule, refusal.target, refusal.detail)

    def _refuse(self, rule: str, target: str, detail: str = "") -> Refusal:
        r = Refusal(rule, target, detail)
        self.refusals.append(r)
        return r


def guarded_invoke(driver, element, guards: Guardrails, settle: float = 2.0):
    """Resolve -> CHECK -> only then dispatch. The only sanctioned invoke path.

    Also re-asserts that our window is in the foreground immediately before
    dispatch: if focus was stolen, keystrokes meant for a CAD dialog would land in
    whatever grabbed it.
    """
    guards.enforce(element, driver.top_windows())
    if not driver.focus_window():
        raise GuardrailViolation("focus-stolen", getattr(element, "key", "?"),
                                 "the target window is not in the foreground; "
                                 "input would land somewhere else")
    return driver.invoke(element, settle=settle)


# --- 3: the sacrificial copy, and the scratch the harness owns ---------------
class Scratch:
    """A scratch directory the harness owns. The agent is never given a user path.

    :meth:`sacrificial_copy` is the ONLY way a user file may enter a run: it is
    copied in, and the copy is what gets driven. :meth:`export_path` is where the
    harness (never the agent, and never the application's Save) writes exports.
    """

    def __init__(self, root: Optional[str] = None, prefix: str = "harnesscad-cua-") -> None:
        self.root = root or tempfile.mkdtemp(prefix=prefix)
        os.makedirs(self.root, exist_ok=True)
        self._owned = root is None

    def path(self, *parts: str) -> str:
        p = os.path.join(self.root, *parts)
        real = os.path.realpath(p)
        if not real.startswith(os.path.realpath(self.root)):
            raise GuardrailViolation("escape", p,
                                     "path escapes the scratch directory")
        return p

    def export_path(self, name: str) -> str:
        return self.path(name)

    def sacrificial_copy(self, user_file: str) -> str:
        """Copy a user file in. The original is NEVER opened and NEVER written."""
        if not os.path.isfile(user_file):
            raise GuardrailViolation("missing-file", user_file, "no such file")
        dst = self.path(os.path.basename(user_file))
        shutil.copy2(user_file, dst)
        return dst

    def owns(self, path: str) -> bool:
        return os.path.realpath(path).startswith(os.path.realpath(self.root))

    def cleanup(self) -> None:
        if self._owned and os.path.isdir(self.root):
            shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "Scratch":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()


def dirty_tripwire(title: str) -> Dict[str, Any]:
    """FreeCAD's unsaved-changes '*' — free evidence that an op mutated the doc."""
    t = str(title or "")
    return {"title": t, "dirty": t.strip().startswith("*")}
