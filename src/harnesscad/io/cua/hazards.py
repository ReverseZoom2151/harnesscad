"""hazards — the Windows UIA known-hazards checklist, as checkable DATA.

Ported from open-codex-computer-use's hard-won Windows notes: three bugs that
every UIAutomation driver hits, each of which fails SILENTLY — the API returns a
success value and nothing happens. They are the exact traps
:mod:`harnesscad.io.cua.uia` was written to avoid; this module states them as a
checklist plus a *correct-primitive reference* so that our own driver can be
asserted against them (see :func:`audit_driver`) rather than the knowledge living
only in a docstring.

The three documented hazards
----------------------------
1. **PostMessage does not update async key state.** A modifier sent with
   ``PostMessage`` never sets the async key state, so an app that reads
   ``GetAsyncKeyState`` (or holds a modifier across another key) sees Ctrl as UP.
   Correct primitive: **SendInput** (``KEYEVENTF_UNICODE`` for text), which drives
   the real input queue. The modifier can then be physically held across a key.
2. **type_text appends, it does not replace.** Typing into a field that already
   has a value concatenates onto it: setting "37.5" onto a field reading "10"
   yields "1037.5". Correct primitive: **focus -> select-all -> type -> READ BACK**
   and compare; never trust the write, verify the field now reads what you meant.
3. **PostMessage to a disabled control returns TRUE and does nothing.** A control
   behind a modal, disabled, or a DirectX surface accepts the message and no-ops;
   the TRUE return is not evidence. Correct primitive: **verify the OUTCOME** —
   capture a before-state, dispatch, then poll for an actual change (a value moved,
   a window appeared, the tree grew). No change = no action.

Everything here is pure, stdlib-only, import-safe. It touches no OS and drives
nothing; it is the reference a driver is measured against, not a driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class HazardKind(str, Enum):
    """The three documented Windows-UIA silent-failure classes."""

    ASYNC_KEY_STATE = "async_key_state"        # PostMessage never sets async key state
    APPEND_NOT_REPLACE = "append_not_replace"  # type_text concatenates onto old value
    DISABLED_NOOP = "disabled_noop"            # PostMessage to a dead control returns TRUE


@dataclass(frozen=True)
class Hazard:
    """One documented hazard: what breaks, why it is silent, and the fix.

    ``wrong_api`` is the tempting call that fails quietly; ``correct_primitive``
    names the call that actually works; ``verification`` is the evidence that
    proves the action happened (the thing that turns a silent no-op loud).
    """

    kind: HazardKind
    title: str
    symptom: str
    wrong_api: str
    correct_primitive: str
    verification: str

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "title": self.title, "symptom": self.symptom,
                "wrong_api": self.wrong_api, "correct_primitive": self.correct_primitive,
                "verification": self.verification}


#: The checklist, in the order a keystroke travels: press -> type -> confirm.
HAZARDS: Tuple[Hazard, ...] = (
    Hazard(
        kind=HazardKind.ASYNC_KEY_STATE,
        title="PostMessage does not update async key state",
        symptom="A modifier (Ctrl/Alt/Shift) sent via PostMessage is seen as UP by "
                "any app reading GetAsyncKeyState, so Ctrl+A, Ctrl+S, and every "
                "shortcut silently do nothing.",
        wrong_api="PostMessage(WM_KEYDOWN/WM_KEYUP)",
        correct_primitive="SendInput (KEYEVENTF_UNICODE for text); the modifier is "
                          "physically held down across the key.",
        verification="the shortcut's effect (menu opened, text selected) is observed, "
                     "not the send's return value.",
    ),
    Hazard(
        kind=HazardKind.APPEND_NOT_REPLACE,
        title="type_text appends onto the existing value",
        symptom="Typing '37.5' into a field already reading '10' yields '1037.5'; the "
                "old contents are never cleared by typing alone.",
        wrong_api="type_text without first clearing the field",
        correct_primitive="focus -> select-all (Ctrl+A) -> type -> read the value "
                          "back and compare to the intended value.",
        verification="ValuePattern.Value read back equals the intended value (a "
                     "mandatory read-back, per harnesscad.io.cua.quantity).",
    ),
    Hazard(
        kind=HazardKind.DISABLED_NOOP,
        title="PostMessage to a disabled control returns TRUE and does nothing",
        symptom="A control that is disabled, behind a modal, or a DirectX/OpenGL "
                "surface accepts the message and no-ops; the TRUE return is not "
                "evidence the action occurred.",
        wrong_api="PostMessage to a control without checking IsEnabled or the outcome",
        correct_primitive="check IsEnabled first, dispatch via Invoke/SendInput, then "
                          "poll a before/after snapshot for a real change.",
        verification="a state diff (value moved / window appeared / tree grew / title "
                     "went dirty); an empty diff means the action did not happen.",
    ),
)

_BY_KIND: Dict[HazardKind, Hazard] = {h.kind: h for h in HAZARDS}


def by_kind(kind: HazardKind) -> Hazard:
    """The hazard for a kind. Raises KeyError if unknown (the set is closed)."""
    return _BY_KIND[kind]


def checklist() -> List[Dict[str, Any]]:
    """The whole checklist as plain dicts — for a prompt, a report, or a log."""
    return [h.to_dict() for h in HAZARDS]


# ---------------------------------------------------------------------------
# The correct-primitive reference a driver is asserted against.
# ---------------------------------------------------------------------------
#
# A UIA driver is HAZARD-SAFE iff it (a) exposes the SendInput-based keystroke
# primitives, (b) exposes a select-all + read-back path for numeric writes, and
# (c) never exposes a PostMessage-based dispatch. These are structural properties
# a test can check against a live module object without touching Windows.

#: Attribute names a hazard-safe keyboard/text driver MUST provide.
REQUIRED_PRIMITIVES: Tuple[str, ...] = (
    "send_text",    # SendInput / KEYEVENTF_UNICODE text entry (hazard 1)
    "send_key",     # SendInput virtual-key press with optional modifier (hazard 1)
    "select_all",   # the CLEAR step before typing (hazard 2)
    "read_value",   # the READ-BACK that verifies the write (hazard 2 & 3)
)

#: Substrings in a callable/attribute name that betray the forbidden PostMessage
#: dispatch path. A hazard-safe driver names none of these.
FORBIDDEN_PRIMITIVES: Tuple[str, ...] = (
    "post_message",
    "postmessage",
    "set_value_pattern",   # ValuePattern.SetValue: the silent-no-op write (hazard 2)
)


def audit_driver(module_or_obj: Any) -> List[str]:
    """The hazards a driver leaves open, or ``[]`` if it is hazard-safe.

    Checks a live module (e.g. :mod:`harnesscad.io.cua.uia`) or a driver class:
    every required correct-primitive must be present, and no forbidden
    PostMessage-style primitive may be. Attribute lookup spans the object AND, for
    a class, its instance methods — ``UiaDriver.read_value`` and the module-level
    ``send_text`` both count. Returned as a list so a test surfaces every gap at
    once; the same never-raise posture as the rest of the CUA surface.
    """
    names = _visible_names(module_or_obj)
    lower = {n.lower() for n in names}
    problems: List[str] = []
    for req in REQUIRED_PRIMITIVES:
        if req not in names and req.lower() not in lower:
            haz = _hazard_for_primitive(req)
            problems.append("missing correct primitive %r (guards against: %s)"
                            % (req, haz))
    for name in names:
        nl = name.lower()
        for bad in FORBIDDEN_PRIMITIVES:
            if bad in nl:
                problems.append("exposes forbidden primitive %r (a documented "
                                "silent-no-op path)" % name)
                break
    return problems


def _visible_names(obj: Any) -> set:
    """Public attribute names on a module or class (methods included, dunders out)."""
    out = set()
    for name in dir(obj):
        if name.startswith("__"):
            continue
        out.add(name)
    return out


def _hazard_for_primitive(primitive: str) -> str:
    mapping = {
        "send_text": HazardKind.ASYNC_KEY_STATE.value,
        "send_key": HazardKind.ASYNC_KEY_STATE.value,
        "select_all": HazardKind.APPEND_NOT_REPLACE.value,
        "read_value": HazardKind.APPEND_NOT_REPLACE.value + "/"
                      + HazardKind.DISABLED_NOOP.value,
    }
    return mapping.get(primitive, "unknown")


def audit_plan(plan: Dict[str, Any]) -> List[str]:
    """Flag a PROPOSED interaction plan against the checklist.

    ``plan`` is a small dict of choices a caller is about to make::

        {"key_dispatch": "sendinput" | "postmessage",
         "clears_before_type": bool,
         "reads_back": bool,
         "verifies_outcome": bool}

    Returns the hazards the plan walks into. Every unspecified key is treated as
    the UNSAFE default (absence is not safety), so a caller must OPT IN to each
    guard — the same reason the real driver refuses to assume success.
    """
    problems: List[str] = []
    if str(plan.get("key_dispatch", "postmessage")).lower() != "sendinput":
        problems.append("hazard %s: keystrokes not routed through SendInput; %s"
                        % (HazardKind.ASYNC_KEY_STATE.value,
                           by_kind(HazardKind.ASYNC_KEY_STATE).correct_primitive))
    if not plan.get("clears_before_type", False):
        problems.append("hazard %s: field not cleared before typing (append bug); %s"
                        % (HazardKind.APPEND_NOT_REPLACE.value,
                           by_kind(HazardKind.APPEND_NOT_REPLACE).correct_primitive))
    if not plan.get("reads_back", False):
        problems.append("hazard %s: no read-back after write; %s"
                        % (HazardKind.APPEND_NOT_REPLACE.value,
                           by_kind(HazardKind.APPEND_NOT_REPLACE).verification))
    if not plan.get("verifies_outcome", False):
        problems.append("hazard %s: outcome not verified after dispatch; %s"
                        % (HazardKind.DISABLED_NOOP.value,
                           by_kind(HazardKind.DISABLED_NOOP).correct_primitive))
    return problems


def is_hazard_safe_plan(plan: Dict[str, Any]) -> bool:
    """True iff ``plan`` trips none of the three hazards."""
    return not audit_plan(plan)


#: A plan that opts into every guard — the shape a caller should aim to produce,
#: and what :mod:`harnesscad.io.cua.uia` actually does at runtime.
SAFE_PLAN: Dict[str, Any] = {
    "key_dispatch": "sendinput",
    "clears_before_type": True,
    "reads_back": True,
    "verifies_outcome": True,
}
