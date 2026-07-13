"""Caption-based feedback + human-in-the-loop scheme for Query2CAD (sec. 4, 6).

Query2CAD's *model refinement* uses a distinctive feedback-generation scheme,
different from the yes/no verification-question feedback of CADCodeVerify and the
error-string feedback of prompt-evolution loops:

  * A caption model (BLIP2) captions the rendered isometric view -- "the caption
    essentially encompasses what the LLM generated" (sec. 4). The refiner is then
    asked to find the *difference* between the caption (what was drawn) and the
    user query (what was asked), and to correct it.
  * To counter the caption model's false negatives, Query2CAD adds
    human-in-the-loop feedback: the user may override or augment the auto caption
    (sec. 2/6). The paper found that supplying "what the generated model looks
    like AND the steps to correct it" refines far better than a bare caption.

This module owns that deterministic scheme: building the difference-feedback
packet from a caption and a query, resolving the auto caption against an optional
human override, and packaging the two-part "looks-like + correction-steps"
feedback the paper recommends. The caption model, VQA model and refiner LLM are
all external and are NOT invoked here.

Stdlib only, deterministic, no model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Feedback provenance labels.
SOURCE_CAPTION = "caption_model"
SOURCE_HUMAN = "human"


def resolve_caption(auto_caption: str,
                    human_caption: Optional[str] = None) -> dict:
    """Resolve the effective caption, honouring a human-in-the-loop override.

    When a non-empty ``human_caption`` is supplied it replaces the auto caption
    (correcting the caption model's false negatives, sec. 6). Returns the chosen
    caption text, its source, and whether a human intervened.
    """
    auto = str(auto_caption).strip()
    if not auto and not human_caption:
        raise ValueError("no caption available")
    if human_caption is not None and str(human_caption).strip():
        return {"caption": str(human_caption).strip(), "source": SOURCE_HUMAN,
                "human_intervened": True}
    return {"caption": auto, "source": SOURCE_CAPTION, "human_intervened": False}


def build_difference_feedback(user_query: str, caption: str) -> str:
    """Package the difference-finding feedback packet (sec. 4).

    States what the user asked for (query) and what was actually drawn (caption),
    and instructs the refiner to correct the discrepancy. Deterministic text.
    """
    q = str(user_query).strip()
    c = str(caption).strip()
    if not q:
        raise ValueError("user_query must be non-empty")
    if not c:
        raise ValueError("caption must be non-empty")
    return "\n".join([
        "User query (target): " + q,
        "Rendered model shows (caption): " + c,
        "Identify the difference between the caption and the user query,",
        "then modify the macro to correct it.",
    ])


@dataclass
class CorrectiveFeedback:
    """The two-part feedback the paper found most effective (sec. 6).

    ``looks_like`` describes what the current model looks like; ``steps`` are the
    concrete correction steps. The paper: providing both "made refinement much
    better" than a bare caption.
    """

    looks_like: str
    steps: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not str(self.looks_like).strip():
            raise ValueError("looks_like must be non-empty")

    def render(self) -> str:
        lines = ["Current model looks like: " + str(self.looks_like).strip()]
        if self.steps:
            lines.append("Steps to correct it:")
            for i, s in enumerate(self.steps, 1):
                if not str(s).strip():
                    raise ValueError("correction step must be non-empty")
                lines.append("  %d. %s" % (i, str(s).strip()))
        return "\n".join(lines)

    @property
    def is_actionable(self) -> bool:
        """A feedback packet is actionable when it lists correction steps."""
        return len(self.steps) > 0


def assemble_feedback(user_query: str, auto_caption: str,
                      human_caption: Optional[str] = None,
                      correction_steps: Optional[List[str]] = None) -> dict:
    """End-to-end model-refinement feedback assembly for one refinement round.

    Resolves the caption (with optional human override), builds the
    difference-finding packet, and -- when correction steps are supplied -- the
    richer two-part corrective feedback. Returns a structured record the refiner
    can consume, plus flags for whether a human intervened and whether the
    feedback is actionable.
    """
    resolved = resolve_caption(auto_caption, human_caption)
    difference = build_difference_feedback(user_query, resolved["caption"])
    corrective = CorrectiveFeedback(resolved["caption"],
                                    list(correction_steps or ()))
    return {
        "caption": resolved["caption"],
        "source": resolved["source"],
        "human_intervened": resolved["human_intervened"],
        "difference_feedback": difference,
        "corrective_feedback": corrective.render(),
        "actionable": corrective.is_actionable,
    }
