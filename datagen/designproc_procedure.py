"""Design-procedure representation for CAD data augmentation.

Chen, Shu, Hong, Taber, Li & Klenk, *Learning From Design Procedure To Generate
CAD Programs for Data Augmentation* (NeurIPS 2025 Workshop).

Sec. 3.1 abstracts the industrial CAD *modeling procedure* into an ordered
sequence of design steps:

  1. start from a chosen (B-Spline) **reference surface**;
  2. create the target object so it **conforms to the curvature** of that surface;
  3. add feature primitives (holes, slots, fillets, ...) that "ripple" from the
     surface curvature;
  4. **remove the reference surface** once the object is created (Sec. 3.1:
     "The reference surface will be removed after the CAD object is created").

The paper feeds this procedure to an LLM via a text-prompt template
(Sec. 3.1) with four slots:

    text prompt := [prefix system prompt, design description,
                    design context, postfix system prompt]

The *learned* LLM that consumes the prompt is external. This module implements
the deterministic representation: a ``DesignStep`` vocabulary, a ``DesignProcedure``
(ordered, validity-checked step list), a canonical procedure builder from a
design description, and the exact four-slot prompt-template assembly with the
paper's prefix / context / postfix text (Sec. 3.1, Appendix A).

A *grammar* validity check enforces the procedure's structural invariants
(reference surface first, conform before features, surface removed last).
Determinism: pure functions; no randomness, no wall clock; stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Design-step vocabulary (the abstracted modeling procedure, Sec. 3.1)
# ---------------------------------------------------------------------------

SELECT_REFERENCE_SURFACE = "select_reference_surface"
CONFORM_TO_SURFACE = "conform_to_surface"
ADD_PRIMITIVE = "add_primitive"
BOOLEAN_OP = "boolean_op"
FILLET = "fillet"
REMOVE_REFERENCE_SURFACE = "remove_reference_surface"
EXPORT = "export"

STEP_KINDS: Tuple[str, ...] = (
    SELECT_REFERENCE_SURFACE,
    CONFORM_TO_SURFACE,
    ADD_PRIMITIVE,
    BOOLEAN_OP,
    FILLET,
    REMOVE_REFERENCE_SURFACE,
    EXPORT,
)

# Steps that introduce / are enabled by organic B-Spline curvature.
_ORGANIC_STEPS = frozenset({CONFORM_TO_SURFACE, FILLET})


@dataclass(frozen=True)
class DesignStep:
    """One step of a modeling procedure: a kind plus free-form attributes."""

    kind: str
    detail: str = ""

    def __post_init__(self):
        if self.kind not in STEP_KINDS:
            raise ValueError(
                "unknown step kind %r (known: %s)"
                % (self.kind, ", ".join(STEP_KINDS)))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail}


# ---------------------------------------------------------------------------
# Design procedure
# ---------------------------------------------------------------------------

@dataclass
class DesignProcedure:
    """An ordered sequence of :class:`DesignStep` with a target category."""

    category: str
    steps: List[DesignStep] = field(default_factory=list)
    surface_kind: str = ""

    def kinds(self) -> List[str]:
        return [s.kind for s in self.steps]

    def uses_reference_surface(self) -> bool:
        return SELECT_REFERENCE_SURFACE in self.kinds()

    def organic_step_count(self) -> int:
        """Number of steps that introduce/exploit organic B-Spline curvature."""
        return sum(1 for s in self.steps if s.kind in _ORGANIC_STEPS)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "surface_kind": self.surface_kind,
            "steps": [s.to_dict() for s in self.steps],
        }


# ---------------------------------------------------------------------------
# Grammar validity of a procedure (Sec. 3.1 structural invariants)
# ---------------------------------------------------------------------------

def validate_procedure(proc: DesignProcedure) -> Tuple[bool, List[str]]:
    """Check the procedure's structural invariants; return ``(ok, errors)``.

    Invariants (Sec. 3.1):
      * a reference-surface procedure starts by *selecting* the surface;
      * ``conform_to_surface`` must follow the select and precede feature steps;
      * at least one object-creating feature step (conform / primitive / boolean);
      * the reference surface, if selected, must be *removed* before export and
        must be removed exactly once;
      * no step may appear after ``export``.
    """
    errors: List[str] = []
    kinds = proc.kinds()
    if not kinds:
        return False, ["empty procedure"]

    has_select = SELECT_REFERENCE_SURFACE in kinds
    if has_select:
        if kinds[0] != SELECT_REFERENCE_SURFACE:
            errors.append("reference surface must be selected first")
        if kinds.count(SELECT_REFERENCE_SURFACE) > 1:
            errors.append("reference surface selected more than once")
        n_remove = kinds.count(REMOVE_REFERENCE_SURFACE)
        if n_remove == 0:
            errors.append("reference surface selected but never removed")
        elif n_remove > 1:
            errors.append("reference surface removed more than once")
        else:
            # conform must appear between select and remove.
            i_sel = kinds.index(SELECT_REFERENCE_SURFACE)
            i_rem = kinds.index(REMOVE_REFERENCE_SURFACE)
            if i_rem < i_sel:
                errors.append("reference surface removed before it is selected")
            if CONFORM_TO_SURFACE in kinds:
                i_conf = kinds.index(CONFORM_TO_SURFACE)
                if not (i_sel < i_conf < i_rem):
                    errors.append(
                        "conform_to_surface must lie between select and remove")
    else:
        if REMOVE_REFERENCE_SURFACE in kinds:
            errors.append("remove_reference_surface without a select")

    feature_steps = {CONFORM_TO_SURFACE, ADD_PRIMITIVE, BOOLEAN_OP, FILLET}
    if not any(k in feature_steps for k in kinds):
        errors.append("procedure creates no object features")

    if EXPORT in kinds:
        i_exp = kinds.index(EXPORT)
        if i_exp != len(kinds) - 1:
            errors.append("export must be the final step")
        if has_select and REMOVE_REFERENCE_SURFACE in kinds:
            if kinds.index(REMOVE_REFERENCE_SURFACE) > i_exp:
                errors.append("reference surface removed after export")

    return (len(errors) == 0), errors


def is_valid_procedure(proc: DesignProcedure) -> bool:
    ok, _ = validate_procedure(proc)
    return ok


# ---------------------------------------------------------------------------
# Canonical procedure builder (from a design description)
# ---------------------------------------------------------------------------

def build_procedure(category: str, surface_kind: str, n_primitives: int = 2,
                    with_fillet: bool = True,
                    with_reference_surface: bool = True) -> DesignProcedure:
    """Construct the canonical design procedure the paper describes.

    With a reference surface, the step order is: select surface -> conform ->
    (n_primitives feature cuts) -> optional fillet -> remove surface -> export.
    Without a surface it reduces to the plain "sketch-and-extrude" baseline
    (the ``ours(-RT)`` ablation): (n_primitives features) -> optional fillet ->
    export. The result always satisfies :func:`validate_procedure`.
    """
    if n_primitives < 0:
        raise ValueError("n_primitives must be >= 0")
    steps: List[DesignStep] = []
    if with_reference_surface:
        steps.append(DesignStep(SELECT_REFERENCE_SURFACE, surface_kind))
        steps.append(DesignStep(CONFORM_TO_SURFACE,
                                "match curvature of %s surface" % surface_kind))
    for i in range(n_primitives):
        steps.append(DesignStep(ADD_PRIMITIVE, "feature %d" % (i + 1)))
    if with_fillet:
        steps.append(DesignStep(FILLET, "smooth organic edges"))
    if with_reference_surface:
        steps.append(DesignStep(REMOVE_REFERENCE_SURFACE, "remove reference surface"))
    steps.append(DesignStep(EXPORT, "output.step"))
    return DesignProcedure(
        category=category,
        steps=steps,
        surface_kind=surface_kind if with_reference_surface else "",
    )


# ---------------------------------------------------------------------------
# Prompt-template assembly (Sec. 3.1 four-slot template, Appendix A/B)
# ---------------------------------------------------------------------------

PREFIX_SYSTEM_PROMPT = (
    "Use Python CadQuery library to write a CAD program of a {category} "
    "that is described as follows."
)

DESIGN_CONTEXT = (
    "The shapes of the {category} look smooth. The {category} should conform "
    "to the curvature of the reference surface in the CAD program below. After "
    "the {category} is created, the reference surface should be removed."
)

# The (-R) ablation replaces the design context with a bare text shape hint.
DESIGN_CONTEXT_TEXT_ONLY = (
    "The shapes of the {category} look smooth and organic."
)

POSTFIX_SYSTEM_PROMPT = (
    "Make sure the generated CAD model is watertight solid. Please export the "
    "generated CAD model to output.stl file and output.step file. Please do "
    "not visualize it. Here is the document of CadQuery for your reference "
    "(https://cadquery.readthedocs.io/en/latest/index.html). Do not output "
    "explanation."
)


def build_prompt(category: str, design_description: str,
                 reference_surface_script: str = "",
                 mode: str = "full") -> dict:
    """Assemble the four-slot design-procedure prompt (Sec. 3.1).

    ``mode`` selects the paper's variants:
      * ``"full"``  -- reference surface + design-procedure context (Ours);
      * ``"text"``  -- text-only shape hint, no surface (ablation ours(-R));
      * ``"none"``  -- no context, no surface (ablation ours(-RT)).

    Returns a dict with the four slots plus the assembled ``text`` string, so a
    downstream (external) LLM can be prompted verbatim.
    """
    if mode not in ("full", "text", "none"):
        raise ValueError("mode must be 'full', 'text' or 'none'")
    prefix = PREFIX_SYSTEM_PROMPT.format(category=category)
    if mode == "full":
        context = DESIGN_CONTEXT.format(category=category)
    elif mode == "text":
        context = DESIGN_CONTEXT_TEXT_ONLY.format(category=category)
    else:
        context = ""
    surface = reference_surface_script if mode == "full" else ""

    slots = {
        "prefix": prefix,
        "design_description": design_description,
        "design_context": context,
        "postfix": POSTFIX_SYSTEM_PROMPT,
        "reference_surface_program": surface,
    }
    parts = [prefix, design_description]
    if context:
        parts.append(context)
    parts.append(POSTFIX_SYSTEM_PROMPT)
    if surface:
        parts.append(surface)
    slots["text"] = "\n".join(parts)
    return slots


def procedure_from_prompt_mode(category: str, surface_kind: str,
                               mode: str, n_primitives: int = 2) -> DesignProcedure:
    """Build the procedure implied by a prompt ``mode`` (couples the two APIs).

    ``"full"`` uses the reference surface; ``"text"`` and ``"none"`` do not
    (the ablations that drop the surface program).
    """
    return build_procedure(
        category, surface_kind, n_primitives=n_primitives,
        with_reference_surface=(mode == "full"))
