"""Workflow comparison, intent-based ranking and reflection checklists.

The heart of CAMeleon (Feng et al.) is *guided comparison*: put fabrication
workflows side by side so a designer reasons across processes instead of locking
into the first one they know. This module builds the deterministic, buildable
core of that idea:

  * ``compare_workflows``  -- a side-by-side table across the comparison criteria
    participants found most useful (equipment, quality/precision, time, cost,
    material, assembly complexity, durability, keywords) -- Figure 7, Section 8.4.

  * ``rank_by_intent``     -- the "intent-based filtering" future-work feature
    (Section 8.4.4): "If I give one line -- food-safe, transparent, under 3 hours
    -- the system could shortlist two or three workflows with reasons." A purely
    rule-based scorer over the declarative capability tags in the taxonomy, with
    a human-readable reason for every match and miss. No model, no training.

  * ``reflection_checklist`` -- the "My Reflection" structured checklist
    (Figure 17): general considerations (materials, tools, size, time, precision,
    cost, skills) plus workflow-specific questions. This is a design-rationale /
    decision-trace schema, complementary to the exploration trace below.

  * ``ExplorationTrace``   -- a small deterministic log of the workflows a
    designer considered vs. selected, and the reasoning shift, mirroring the
    study's pre/post analysis (Table 2). This captures *design reasoning* as
    structured data, which is the paper's stated goal.

Everything is stdlib-only and deterministic. This is the workflow-level
decision-support layer; it consumes the taxonomy and does not duplicate the
per-part critics (``verifiers/dfm.py``) or cost/BOM math (``quality/estimate.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.fabrication.workflow_taxonomy import (
    MATERIALS,
    WORKFLOWS,
    Workflow,
    available_workflows,
    get_workflow,
)


# --------------------------------------------------------------------------- #
# Side-by-side comparison
# --------------------------------------------------------------------------- #
# The comparison criteria, in the priority order participants reported
# (equipment 9/12, final quality 8/12, difficulty 6/12, time 4/12 -- Sec 8.4.3).
COMPARISON_CRITERIA: Tuple[str, ...] = (
    "category",
    "machines",
    "materials",
    "precision",   # proxy for final quality
    "skill",       # proxy for difficulty
    "time",
    "cost",
    "assembly_steps",
    "keywords",
)

_LEVEL_WORDS = {1: "very low", 2: "low", 3: "medium", 4: "high", 5: "very high"}
_TIME_WORDS = {1: "very fast", 2: "fast", 3: "medium", 4: "slow", 5: "very slow"}


def _cell(wf: Workflow, criterion: str) -> object:
    if criterion == "category":
        return wf.category
    if criterion == "machines":
        return tuple(wf.machines)
    if criterion == "materials":
        return tuple(wf.materials)
    if criterion == "precision":
        return _LEVEL_WORDS[wf.precision]
    if criterion == "skill":
        return _LEVEL_WORDS[wf.skill]
    if criterion == "time":
        return _TIME_WORDS[wf.time]
    if criterion == "cost":
        return _LEVEL_WORDS[wf.cost]
    if criterion == "assembly_steps":
        return wf.assembly_steps
    if criterion == "keywords":
        return tuple(wf.keywords)
    raise KeyError(f"unknown criterion {criterion!r}")


def compare_workflows(
    workflow_ids: Sequence[str],
    criteria: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Build a side-by-side comparison table.

    The paper caps the comparison view at four workflows; we enforce the same
    limit to keep the comparison legible (Figure 7: "up to four windows").
    Returns ``{workflow_id: {criterion: value}}`` preserving input order.
    """
    ids = list(workflow_ids)
    if not ids:
        raise ValueError("compare at least one workflow")
    if len(ids) > 4:
        raise ValueError("side-by-side comparison supports at most 4 workflows")
    crit = tuple(criteria) if criteria else COMPARISON_CRITERIA
    table: Dict[str, Dict[str, object]] = {}
    for wid in ids:
        wf = get_workflow(wid)
        table[wid] = {c: _cell(wf, c) for c in crit}
    return table


def comparison_deltas(workflow_ids: Sequence[str]) -> Dict[str, bool]:
    """Which criteria actually differ across the compared workflows?

    Highlighting differences is the pedagogical point of side-by-side
    comparison ("making similarities and differences explicit", Sec 9.1).
    Returns ``{criterion: differs?}`` over the ordinal/scalar criteria.
    """
    ids = list(workflow_ids)
    scalar = ("category", "precision", "skill", "time", "cost", "assembly_steps")
    out: Dict[str, bool] = {}
    for c in scalar:
        vals = {_cell(get_workflow(w), c) for w in ids}
        out[c] = len(vals) > 1
    return out


# --------------------------------------------------------------------------- #
# Intent-based ranking
# --------------------------------------------------------------------------- #
# Boolean capability requirements map straight onto a workflow ``props`` tag.
_BOOL_REQUIREMENTS: Tuple[str, ...] = (
    "food_safe", "transparent", "lightweight", "durable", "heat_resistant",
    "flexible", "complex_geometry", "large_scale", "low_cost", "batch",
)


@dataclass(frozen=True)
class IntentMatch:
    """A workflow's score against an intent, with per-requirement reasons."""

    workflow_id: str
    score: float
    matched: Tuple[str, ...]
    missed: Tuple[str, ...]
    reasons: Tuple[str, ...]


def _material_has(wf: Workflow, attr: str) -> bool:
    return any(
        m in MATERIALS and getattr(MATERIALS[m], attr, False) for m in wf.materials
    )


def _satisfies_bool(wf: Workflow, req: str) -> bool:
    """Does a workflow satisfy a boolean requirement, via prop or material?"""
    if wf.props.get(req):
        return True
    # Fall back to material capabilities for the material-borne properties.
    if req in ("food_safe", "transparent", "heat_resistant", "flexible"):
        return _material_has(wf, req)
    return False


def rank_by_intent(
    requirements: Dict[str, object],
    candidate_ids: Optional[Sequence[str]] = None,
    machine_ids: Optional[Sequence[str]] = None,
    top_k: int = 3,
) -> List[IntentMatch]:
    """Shortlist workflows that best fit a stated intent, with reasons.

    ``requirements`` accepts:
      * boolean capability keys (``food_safe``, ``transparent``, ``lightweight``,
        ``durable``, ``heat_resistant``, ``flexible``, ``complex_geometry``,
        ``large_scale``, ``low_cost``, ``batch``) -> ``True`` to require.
      * ``max_time`` (1..5)  -- reject/penalize slower workflows.
      * ``max_cost`` (1..5)  -- reject/penalize costlier workflows.
      * ``max_skill`` (1..5) -- penalize workflows above the learner's skill.

    Scoring is deterministic: +1 per satisfied boolean requirement, a graded
    penalty for exceeding an ordinal cap, ties broken by (score desc, then
    fewer misses, then workflow id) so the order is stable. Returns the top-k
    with a human-readable reason per requirement -- exactly the "shortlist two
    or three workflows with reasons" feature.
    """
    if candidate_ids is not None:
        pool = [get_workflow(w) for w in candidate_ids]
    elif machine_ids is not None:
        pool = available_workflows(list(machine_ids))
    else:
        pool = sorted(WORKFLOWS.values(), key=lambda w: w.id)

    bool_reqs = [r for r in _BOOL_REQUIREMENTS if requirements.get(r)]
    results: List[IntentMatch] = []
    for wf in pool:
        score = 0.0
        matched: List[str] = []
        missed: List[str] = []
        reasons: List[str] = []
        for req in bool_reqs:
            if _satisfies_bool(wf, req):
                score += 1.0
                matched.append(req)
                reasons.append(f"+ {req.replace('_', ' ')}: supported")
            else:
                missed.append(req)
                reasons.append(f"- {req.replace('_', ' ')}: not supported")
        # Ordinal caps: a soft penalty proportional to the overshoot.
        for cap_key, attr, word in (
            ("max_time", "time", "time"),
            ("max_cost", "cost", "cost"),
            ("max_skill", "skill", "skill"),
        ):
            cap = requirements.get(cap_key)
            if cap is not None:
                actual = getattr(wf, attr)
                if actual > cap:
                    penalty = 0.5 * (actual - cap)
                    score -= penalty
                    reasons.append(
                        f"- {word} {actual} exceeds max {cap} (penalty {penalty:.1f})"
                    )
                else:
                    reasons.append(f"+ {word} {actual} within max {cap}")
        results.append(IntentMatch(
            wf.id, round(score, 3), tuple(matched), tuple(missed), tuple(reasons)
        ))

    results.sort(key=lambda m: (-m.score, len(m.missed), m.workflow_id))
    if top_k is not None and top_k > 0:
        return results[:top_k]
    return results


# --------------------------------------------------------------------------- #
# Reflection checklist (design-rationale schema)
# --------------------------------------------------------------------------- #
# General considerations shown for every workflow (Figure 17b).
_GENERAL_CONSIDERATIONS: Tuple[str, ...] = (
    "Have you considered the material properties for your use case?",
    "Do you have access to the required tools/machines?",
    "Does the part fit within your machine's working size?",
    "Is the estimated time acceptable for your deadline?",
    "Is the achievable precision adequate for your design?",
    "Is the cost within your budget (including repeat production)?",
    "Do you have the skills, or a plan to learn them, for this workflow?",
)

# Workflow-specific reflection questions (Figure 17: e.g. overhang/post-process
# for 3D printing, kerf/nesting for laser cutting).
_WORKFLOW_QUESTIONS: Dict[str, Tuple[str, ...]] = {
    "fdm_3d_printing": (
        "Is the print duration acceptable at your chosen infill?",
        "Does the part fit the print bed, or must it be split?",
        "What overhang / support strategy will you use?",
        "What post-processing (sanding, priming) do you plan?",
    ),
    "laser_cut_interlocking": (
        "Have you applied kerf compensation to the slot joints?",
        "Have you nested the parts to minimize sheet waste?",
        "Is the sheet thickness one your shop actually stocks?",
    ),
    "stacked_layers": (
        "How will you align and bond the stacked layers?",
        "Is the layer adhesive acceptable for the intended use?",
    ),
    "wire_forming": (
        "Is every segment long enough for the bender to grip?",
        "Are all bend angles within the bender's limit?",
        "Is a springy/flexible frame acceptable for this part?",
    ),
    "hot_wire_foam_cutting": (
        "Is the low load-bearing capacity of foam acceptable?",
        "Will you coat/finish the foam surface?",
    ),
    "mold_making": (
        "Do all faces have enough draft to demold?",
        "How many casts do you need (is a reusable mold worth it)?",
    ),
    "silicone_molding": (
        "Do you need flexibility, heat resistance, or food safety?",
        "Have you planned a clean pour/vent to avoid bubbles?",
    ),
    "epoxy_laminating": (
        "Do you have ventilation and PPE for the resin step?",
        "Have you sequenced cut, assemble, sand, and seal?",
    ),
}


@dataclass(frozen=True)
class ReflectionChecklist:
    workflow_id: str
    general: Tuple[str, ...]
    specific: Tuple[str, ...]


def reflection_checklist(workflow_id: str) -> ReflectionChecklist:
    """The "My Reflection" structured checklist for a workflow (Figure 17)."""
    get_workflow(workflow_id)  # validates
    return ReflectionChecklist(
        workflow_id=workflow_id,
        general=_GENERAL_CONSIDERATIONS,
        specific=_WORKFLOW_QUESTIONS.get(workflow_id, ()),
    )


# --------------------------------------------------------------------------- #
# Exploration / reasoning trace (Table 2 pre/post schema)
# --------------------------------------------------------------------------- #
@dataclass
class ExplorationTrace:
    """A deterministic record of a design-reasoning session.

    Captures the reasoning shift the study measured: which workflows were
    *considered* and *selected* before vs. after guided comparison, and the
    criteria cited. This turns the paper's qualitative pre/post analysis into a
    structured, queryable artifact.
    """

    considered_before: List[str] = field(default_factory=list)
    selected_before: Optional[str] = None
    considered_after: List[str] = field(default_factory=list)
    selected_after: Optional[str] = None
    criteria_cited: List[str] = field(default_factory=list)

    def consider(self, workflow_id: str, *, after: bool = True) -> None:
        get_workflow(workflow_id)
        target = self.considered_after if after else self.considered_before
        if workflow_id not in target:
            target.append(workflow_id)

    def select(self, workflow_id: str, *, after: bool = True) -> None:
        get_workflow(workflow_id)
        if after:
            self.selected_after = workflow_id
        else:
            self.selected_before = workflow_id

    def cite(self, criterion: str) -> None:
        if criterion not in self.criteria_cited:
            self.criteria_cited.append(criterion)

    def breadth_gain(self) -> int:
        """How many more workflows were considered after comparison (>=0)."""
        return max(0, len(self.considered_after) - len(self.considered_before))

    def changed_selection(self) -> bool:
        """Did the final workflow choice change after guided comparison?"""
        return (
            self.selected_before is not None
            and self.selected_after is not None
            and self.selected_before != self.selected_after
        )

    def summary(self) -> Dict[str, object]:
        return {
            "considered_before": list(self.considered_before),
            "selected_before": self.selected_before,
            "considered_after": list(self.considered_after),
            "selected_after": self.selected_after,
            "breadth_gain": self.breadth_gain(),
            "changed_selection": self.changed_selection(),
            "criteria_cited": list(self.criteria_cited),
        }
