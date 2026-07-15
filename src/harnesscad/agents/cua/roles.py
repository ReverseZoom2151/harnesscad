"""roles — the three-model split: vision-narrate, text-act, and separate grounding.

The os_computer_use config makes a design choice worth naming: it does NOT use one
model for everything. It runs three:

    * a VISION model that narrates the screenshot into the observation shape
      (``qwen-2.5-vl``);
    * an ACTION model that reads that narration and emits the tool call
      (``llama-3.3`` — text-only, no image);
    * a GROUNDING model, SEPARATE from both, that alone turns "the Save button"
      into a pixel (``OS-Atlas`` / ``ShowUI``).

The split matters because the three jobs have different failure modes and
different best-in-class models. Grounding in particular is a specialist: a
general VLM narrates a scene well but localises a small toolbar icon badly, which
is exactly why a dedicated coordinate model exists. Collapsing grounding into the
action model is the most common regression, so :func:`validate` REFUSES an
assignment where grounding is not its own model.

For HarnessCAD the mapping is deliberately conservative: tier 0/1 (parameter
dialogs) need NO vision and NO grounding at all — the plan is structured output
from the action model and the picks are computed, not seen (see agents/cua/models
and eval/grounding/corpus). This module lets a run DECLARE which roles are even in
play, so a tier-0 run can assert "no grounding model is loaded" and a tier-2 run
can assert one is. Data + validation only; no model is constructed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Role(Enum):
    VISION = "vision"        # narrate a screenshot into the observation shape
    ACTION = "action"        # emit the tool call from the narration
    GROUNDING = "grounding"  # localise a described element to a pixel


#: What each role consumes and produces, and whether it needs the image. Kept as
#: data so a harness can reason about the pipeline without running it.
ROLE_IO: Dict[Role, Dict[str, object]] = {
    Role.VISION: {"needs_image": True, "produces": "observation-text"},
    Role.ACTION: {"needs_image": False, "produces": "tool-call"},
    Role.GROUNDING: {"needs_image": True, "produces": "pixel-point"},
}


class RoleError(ValueError):
    """A role assignment is unsound (e.g. grounding folded into another model)."""


@dataclass(frozen=True)
class RoleAssignment:
    """Which model plays which role. A role mapped to None is DECLARED absent (a
    tier-0 run legitimately has no vision and no grounding model)."""

    vision: Optional[str] = None
    action: Optional[str] = None
    grounding: Optional[str] = None

    def model_for(self, role: Role) -> Optional[str]:
        return {Role.VISION: self.vision, Role.ACTION: self.action,
                Role.GROUNDING: self.grounding}[role]

    def present_roles(self) -> List[Role]:
        return [r for r in Role if self.model_for(r) is not None]

    def to_dict(self) -> dict:
        return {"vision": self.vision, "action": self.action,
                "grounding": self.grounding}


def validate(assignment: RoleAssignment, *, require_grounding: bool = False,
             require_vision: bool = False) -> None:
    """Refuse unsound assignments.

    Always: an ACTION model must be present (something has to decide), and IF a
    grounding model is present it must NOT be the same model as action or vision
    (the specialist-collapse regression). ``require_grounding`` / ``require_vision``
    let a tier-2 run demand the specialists are actually loaded.
    """
    if assignment.action is None:
        raise RoleError("no action model: nothing would decide the next step")
    g = assignment.grounding
    if g is not None and g in (assignment.action, assignment.vision):
        raise RoleError(
            "grounding model %r is also the action/vision model; grounding is a "
            "specialist and must be a SEPARATE model" % (g,))
    if require_grounding and g is None:
        raise RoleError("this run requires a grounding model but none is assigned")
    if require_vision and assignment.vision is None:
        raise RoleError("this run requires a vision model but none is assigned")


#: The reference three-model lineup, for provenance and as a validate() example.
REFERENCE_SPLIT = RoleAssignment(
    vision="qwen-2.5-vl", action="llama-3.3", grounding="os-atlas")

#: The HarnessCAD tier-0/1 lineup: structured-output action model only, picks
#: computed, so no vision and no grounding model are loaded at all.
CAD_TIER0_SPLIT = RoleAssignment(vision=None, action="qwen3.6:35b", grounding=None)
