"""Computer-use prompt policy.

The coordinate half — the ``0..999`` grid denormalisation and the DPI
logical/physical bridge — lives in :mod:`harnesscad.io.cua.coordinate`
(``Denormalizer``, ``CoordinateMapper``, ``normalize_function_call``). This module provides the other half:
the *prompt policy* atlas wraps around the ``computer_use`` tool — the system
instruction, the resolution it injects so the model grounds against the right frame,
the declared action space, and the confirmation gate for unsafe actions.

Why the policy is separate
--------------------------
A ``computer_use`` model is only as good as the frame it is told it is looking at and
the rules it is told to obey. Atlas gets three of these right and a CAD agent must
not lose them:

* **State the grid.** The model emits ``0..999`` coordinates; the prompt SAYS so, so
  the numbers are unambiguous (:data:`GRID_POLICY`), and they denormalise through the
  exact same ``Denormalizer`` grid the coordinate module uses — asserted equal in the
  test, so the prompt and the maths can never drift apart.
* **State the resolution.** Atlas injects ``"1920x1080"`` (``ScreenInfo.resolution_string``)
  so the model knows the extent it is grounding against.
* **Gate the unsafe.** Atlas's ``safety_decision``/``require_confirmation`` becomes a
  hard rule: a confirmation-required action is NEVER auto-executed — which lands
  straight on this repo's own never-save/never-destroy guardrail.

Everything here is prompt DATA plus a renderer and a policy check. No model is
called; the display is passed in as a :class:`~harnesscad.io.cua.coordinate.ScreenInfo`.
Pure stdlib, import-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from harnesscad.io.cua.coordinate import Denormalizer, ScreenInfo


#: The default Gemini ``computer_use`` grid. Matches ``Denormalizer``'s default.
DEFAULT_GRID = 999

#: The action verbs atlas's ``computer_use`` policy advertises to the model. Each
#: MUST be a function :func:`harnesscad.io.cua.coordinate.normalize_function_call`
#: knows how to map (the test pins this), so the prompt never promises a tool the
#: pipeline would drop.
ADVERTISED_FUNCTIONS: tuple = (
    "click_at", "hover_at", "type_text_at", "key_combination",
    "scroll_at", "drag_and_drop",
)

#: The grid statement injected into the prompt. Pins the model's coordinate space so
#: the numbers it emits are unambiguous and denormalise exactly.
GRID_POLICY = (
    "All coordinates you output are integers on a %d-point grid: x and y each run "
    "from 0 (left/top) to %d (right/bottom), independent of the screen resolution."
)

#: The verbatim-intent safety policy. Atlas surfaces a ``require_confirmation``
#: safety decision; here it is a HARD rule the agent obeys.
SAFETY_POLICY = (
    "Before any irreversible or destructive action (saving, exporting, deleting, "
    "overwriting, or exiting), you MUST request confirmation and wait; never perform "
    "such an action on your own initiative."
)

#: The CAD-specific spine, consistent with agents/cua/prompts.CAD_SYSTEM: geometry is
#: the only score, and file I/O belongs to the harness.
CAD_COMPUTER_USE_SYSTEM = (
    "You are operating a real CAD application through its GUI using the computer-use "
    "tool. You are graded only on the geometry the application actually builds, "
    "measured through its own kernel. Prefer parameter dialogs (typed numbers) to "
    "dragging in the 3D view. Type every dimension EXACTLY as written."
)


@dataclass(frozen=True)
class ComputerUsePolicy:
    """The assembled prompt policy atlas hands a ``computer_use`` model.

    ``grid`` is the coordinate grid the model is told to use (and the one the
    :class:`Denormalizer` inverts); ``functions`` is the advertised action space;
    ``require_confirmation_for`` is the set of verbs that may NOT auto-execute.
    """

    system: str = CAD_COMPUTER_USE_SYSTEM
    grid: int = DEFAULT_GRID
    functions: Sequence[str] = ADVERTISED_FUNCTIONS
    require_confirmation_for: Sequence[str] = ("save", "export", "delete",
                                               "overwrite", "exit")

    def denormalizer(self) -> Denormalizer:
        """The exact inverse of the grid this policy tells the model to use — so the
        prompt's declared space and the pixel maths are the SAME object's convention."""
        return Denormalizer(grid=self.grid)

    def grid_statement(self) -> str:
        return GRID_POLICY % (self.grid + 1, self.grid)

    def needs_confirmation(self, verb: str) -> bool:
        """Whether an action verb must be confirmed before it executes (the guardrail)."""
        return verb.strip().lower() in {v.lower() for v in self.require_confirmation_for}

    def to_dict(self) -> dict:
        return {"system": self.system, "grid": self.grid,
                "functions": list(self.functions),
                "require_confirmation_for": list(self.require_confirmation_for)}


def render_policy(policy: ComputerUsePolicy, screen: ScreenInfo,
                  objective: str) -> str:
    """Assemble the full system prompt atlas would send for one objective.

    Injects the resolution (``ScreenInfo.resolution_string`` — what atlas does), the
    grid statement, the advertised action space, the safety rule, and the objective.
    Deterministic: the same policy/screen/objective always yields the same string.
    """
    lines: List[str] = [
        policy.system,
        "",
        "Screen resolution: %s pixels." % screen.resolution_string(),
        policy.grid_statement(),
        "",
        "Available actions: %s." % ", ".join(policy.functions),
        SAFETY_POLICY,
        "",
        "OBJECTIVE: %s" % objective.strip(),
    ]
    return "\n".join(lines)
