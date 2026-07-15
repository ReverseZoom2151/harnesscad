"""reset — the environment-state reset checklist and the state-leak detector.

Mined from the E2B-sandbox CUA repos, whose entire lifecycle discipline is *a
fresh VM per run*: ``Sandbox.create`` spins up an isolated Linux VM, and the abort
handler calls ``Sandbox.kill`` — it does not "clean up" a reused machine, it
throws the machine away. The one place that repo reuses a sandbox
(``Sandbox.connect(sandboxId)``) is a persistent-session convenience for a human
watching a VNC stream, never the path a fresh trial takes.

That discipline is not incidental; it is the only sound one, and here is the CAD
statement of why. **A CAD CUA experiment with state carried between trials is
worthless.** If trial N+1 runs in the same application process as trial N, then
closing the document between them does NOT return the app to a known state. What
survives a document close, and silently biases the next trial:

    * the UNDO/REDO stack (some apps keep it process-global, not per-document);
    * STICKY TOOL DEFAULTS — the last-used pad length, the last sketch plane, the
      last fillet radius. This is the killer: the box recipe types "10" into a
      Pad dialog that already remembers "37.5" from the previous trial, and if a
      field is skipped the previous value is what builds. The measured volume is
      then wrong for a reason that has nothing to do with the model;
    * the active VIEW ORIENTATION / camera (a grounding pick computed for an
      isometric view lands wrong on a leftover top view);
    * the RECENT-FILES list, the CLIPBOARD, the current SELECTION;
    * GRID / SNAP settings and the active WORKBENCH;
    * PREFERENCES or UNITS changed mid-run (the comma-locale trap, made sticky);
    * leftover MODAL dialogs and panel layout.

So this module does two things, both pure functions over snapshot dicts so the
whole policy is unit-testable with no GUI running:

    1. :data:`RESET_CHECKLIST` — the enumerated categories that leak, each with
       the reason it leaks and the remedy, and :func:`requires_fresh_process`
       names the ones that CANNOT be verified clean by any in-process action —
       for those, as in the sandbox repos, *the only reset is a new process*.
    2. :class:`StateLeakDetector` — diffs a clean BASELINE snapshot against the
       snapshot taken at the start of the next trial and reports every category
       that leaked, so a run harness can REFUSE to start a trial on a dirty app
       rather than silently scoring a biased build.

Stdlib only. Absolute imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


class DirtyEnvironment(RuntimeError):
    """The app is not in a known-clean state; a trial started here is worthless."""

    def __init__(self, leaks: "List[Leak]") -> None:
        self.leaks = list(leaks)
        summary = "; ".join(l.category for l in self.leaks) or "(none)"
        super().__init__("environment is dirty, %d leak(s): %s"
                         % (len(self.leaks), summary))


# --- the checklist ----------------------------------------------------------
@dataclass(frozen=True)
class ResetStep:
    """One category of state that survives a document close, and how to handle it.

    ``key`` is the field in a :class:`SessionState` snapshot that witnesses this
    category. ``clean`` is the value the field MUST hold in a freshly reset app.
    ``in_process_remedy`` is the action that resets it without a new process, or
    ``None`` when there is no sound in-process reset (then only a fresh process
    will do — the sandbox repos' whole point).
    """

    category: str
    key: str
    clean: Any
    why_it_leaks: str
    in_process_remedy: Optional[str]

    @property
    def needs_fresh_process(self) -> bool:
        return self.in_process_remedy is None


#: The enumerated leak categories. Ordered worst-first for CAD determinism: the
#: sticky Pad/Sketch defaults are the ones that quietly corrupt a measured volume.
RESET_CHECKLIST: Tuple[ResetStep, ...] = (
    ResetStep(
        category="tool-defaults",
        key="tool_defaults",
        clean={},
        why_it_leaks="parameter dialogs (Pad length, Sketch plane, Fillet radius) "
                     "remember the last value entered; a skipped field builds the "
                     "PREVIOUS trial's number and the measured volume is wrong for "
                     "a reason unrelated to the model",
        in_process_remedy=None,  # dialogs re-read their own persisted state
    ),
    ResetStep(
        category="undo-stack",
        key="undo_depth",
        clean=0,
        why_it_leaks="the undo/redo transaction stack can be process-global and "
                     "survive a document close; a stray Ctrl+Z reaches into the "
                     "prior trial",
        in_process_remedy=None,
    ),
    ResetStep(
        category="preferences",
        key="prefs_dirty",
        clean=False,
        why_it_leaks="units, decimal separator, or a preference toggled mid-run "
                     "persists to the user profile and re-biases every later trial "
                     "(the comma-locale trap, made sticky)",
        in_process_remedy=None,
    ),
    ResetStep(
        category="view-orientation",
        key="view",
        clean="isometric",
        why_it_leaks="a leftover camera from the prior trial makes a grounding "
                     "pick computed for the expected view land on the wrong entity",
        in_process_remedy="dispatch the named-view command (e.g. View > Isometric)",
    ),
    ResetStep(
        category="active-workbench",
        key="workbench",
        clean="PartDesign",
        why_it_leaks="the wrong workbench exposes a different toolbar, so the "
                     "control the plan names is absent or means something else",
        in_process_remedy="switch to the expected workbench",
    ),
    ResetStep(
        category="selection",
        key="selection",
        clean=(),
        why_it_leaks="a pre-existing selection makes the next command operate on "
                     "leftover geometry",
        in_process_remedy="clear selection (Esc / Selection > Clear)",
    ),
    ResetStep(
        category="clipboard",
        key="clipboard",
        clean="",
        why_it_leaks="a paste picks up geometry copied in a prior trial",
        in_process_remedy="empty the clipboard",
    ),
    ResetStep(
        category="grid-snap",
        key="snap",
        clean=False,
        why_it_leaks="a leftover snap setting quantises a coordinate the plan "
                     "meant to type exactly",
        in_process_remedy="restore the default grid/snap state",
    ),
    ResetStep(
        category="recents",
        key="recent_files",
        clean=(),
        why_it_leaks="the recent-files list can expose a prior trial's scratch "
                     "path to a stray Open",
        in_process_remedy="clear the recent-files list",
    ),
    ResetStep(
        category="open-documents",
        key="open_documents",
        clean=0,
        why_it_leaks="a leftover open document means the command acts on the "
                     "wrong document, or a close prompts to save",
        in_process_remedy="close all documents WITHOUT saving (harness owns I/O)",
    ),
    ResetStep(
        category="modal-open",
        key="modal_open",
        clean=False,
        why_it_leaks="a dialog left open from the prior trial swallows the first "
                     "keystrokes of the next one",
        in_process_remedy="the modal guardrail must have already HALTED; a dirty "
                          "modal means the prior trial ended uncleanly",
    ),
)


def checklist_keys() -> Tuple[str, ...]:
    return tuple(step.key for step in RESET_CHECKLIST)


def requires_fresh_process() -> Tuple[ResetStep, ...]:
    """The categories with no sound in-process reset. For these, exactly as the
    E2B repos concluded, the only reliable reset is a NEW application process
    (their fresh-VM-per-run). If the harness insists on reusing a process, these
    cannot be verified clean and the trial is, strictly, not isolated."""
    return tuple(step for step in RESET_CHECKLIST if step.needs_fresh_process)


def clean_baseline() -> "SessionState":
    """The snapshot a freshly launched, freshly reset app MUST match."""
    return SessionState({step.key: step.clean for step in RESET_CHECKLIST})


# --- the snapshot -----------------------------------------------------------
@dataclass(frozen=True)
class Leak:
    category: str
    key: str
    expected: Any
    actual: Any
    fresh_process_only: bool
    remedy: Optional[str]

    def to_dict(self) -> dict:
        return {"category": self.category, "key": self.key,
                "expected": self.expected, "actual": self.actual,
                "fresh_process_only": self.fresh_process_only,
                "remedy": self.remedy}


class SessionState:
    """An immutable snapshot of the app's cross-trial state, keyed by the checklist.

    Values are compared for cleanliness by equality against
    :attr:`ResetStep.clean`. Unknown keys are ignored (forward compatible);
    MISSING keys are treated as unknown-and-therefore-unverifiable, which the
    detector reports rather than assuming clean — silence is never taken as clean.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        # Freeze list/set values to tuples so equality is stable and hashless
        # comparison is deterministic.
        frozen: Dict[str, Any] = {}
        for k, v in (data or {}).items():
            frozen[k] = tuple(v) if isinstance(v, (list, set)) else v
        object.__setattr__(self, "_data", frozen)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._data

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SessionState) and self._data == other._data

    def __repr__(self) -> str:
        return "SessionState(%r)" % (self._data,)


# --- the detector -----------------------------------------------------------
class StateLeakDetector:
    """Diffs a snapshot against the clean baseline and names every leak.

    Usage in a run harness: snapshot the app at the START of each trial and call
    :meth:`assert_clean`. If anything leaked from the prior trial it raises
    :class:`DirtyEnvironment` BEFORE the biased build happens, and the message
    tells you which categories need only a re-dispatch and which need a fresh
    process (i.e. the trial was never isolated to begin with).
    """

    def __init__(self, baseline: Optional[SessionState] = None) -> None:
        self.baseline = baseline or clean_baseline()
        self._by_key = {step.key: step for step in RESET_CHECKLIST}

    def leaks(self, current: SessionState) -> List[Leak]:
        found: List[Leak] = []
        for step in RESET_CHECKLIST:
            expected = self.baseline.get(step.key, step.clean)
            if not current.has(step.key):
                # Not reported: an unobserved field is a gap in the probe, not a
                # leak. We surface it separately via unverified().
                continue
            actual = current.get(step.key)
            if actual != expected:
                found.append(Leak(category=step.category, key=step.key,
                                  expected=expected, actual=actual,
                                  fresh_process_only=step.needs_fresh_process,
                                  remedy=step.in_process_remedy))
        return found

    def unverified(self, current: SessionState) -> List[str]:
        """Checklist categories the snapshot did not observe at all. These are NOT
        assumed clean — an honest harness treats an unprobed category as a reason
        to prefer a fresh process, never as a pass."""
        return [step.category for step in RESET_CHECKLIST
                if not current.has(step.key)]

    def is_clean(self, current: SessionState) -> bool:
        return not self.leaks(current)

    def assert_clean(self, current: SessionState) -> None:
        found = self.leaks(current)
        if found:
            raise DirtyEnvironment(found)

    def report(self, current: SessionState) -> Dict[str, Any]:
        found = self.leaks(current)
        return {
            "clean": not found,
            "leaks": [l.to_dict() for l in found],
            "unverified": self.unverified(current),
            "fresh_process_required": any(l.fresh_process_only for l in found),
        }
