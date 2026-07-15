"""Five-part CAD modeling-plan scaffold (UniCAD structured description hierarchy).

Paper: *UniCAD -- A Unified Benchmark and Universal Model for Multi-Modal
Multi-Task CAD* (Chen et al., 2026). To turn a CAD program into a high-fidelity
natural-language modeling specification, UniCAD's data pipeline organizes every
description into a fixed "five-part architectural hierarchy" (Appendix A.2):

  1. **Global Parameters and Constants** -- key variables and their proportions;
  2. **Reference Planes and Primary Geometry** -- sketch constraints and the
     foundational 3D operations (workplane, origin, extrude/revolve);
  3. **Secondary Features and Boolean Operations** -- additive/subtractive
     structures (cut, union, pocket, hole) and their positioning;
  4. **Transformation and Pattern Logic** -- linear/circular arrays, mirroring;
  5. **Topological Refinement** -- target edges + parameters for fillets/chamfers.

This deterministic scaffold is the reusable, model-free core of that pipeline: it
emits the empty five-section template a generator fills, and -- given lines of a
modeling description -- classifies each into its canonical section by keyword
evidence and reports which sections are covered. It is a *plan-structuring*
aid, not a generator: no model is called and the classification is a fixed
keyword rule, so the same input always yields the same structure.

Stdlib-only and deterministic. Distinct from
:mod:`harnesscad.domain.spec.spec_decompose` (which decomposes a requirement
into geometric sub-goals): here the taxonomy is UniCAD's five *description*
sections, ordered by construction stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "SECTIONS",
    "SECTION_TITLES",
    "classify_line",
    "structure_description",
    "coverage",
    "empty_scaffold",
    "render",
]

# canonical section keys, in construction order (Appendix A.2).
SECTIONS = (
    "global_parameters",
    "primary_geometry",
    "secondary_features",
    "pattern_logic",
    "topological_refinement",
)

SECTION_TITLES = {
    "global_parameters": "Global Parameters and Constants",
    "primary_geometry": "Reference Planes and Primary Geometry",
    "secondary_features": "Secondary Features and Boolean Operations",
    "pattern_logic": "Transformation and Pattern Logic",
    "topological_refinement": "Topological Refinement",
}

# keyword evidence per section, checked in section order so an earlier, more
# specific stage wins ties only where sensible. Each key -> substrings (lower).
_KEYWORDS = {
    "topological_refinement": (
        "fillet", "chamfer", "round", "deburr", "edge break", "refine",
    ),
    "pattern_logic": (
        "array", "pattern", "mirror", "mirrored", "linear array",
        "circular array", "polar", "rotational symmetry", "instances",
        "repeat",
    ),
    "secondary_features": (
        "boolean", "cut", "union", "subtract", "pocket", "hole", "bore",
        "shell", "secondary", "additive", "subtractive", "groove", "recess",
    ),
    "primary_geometry": (
        "workplane", "plane", "origin", "sketch", "profile", "extrude",
        "revolve", "loft", "sweep", "primary", "cross-section", "base",
    ),
    "global_parameters": (
        "parameter", "constant", "variable", "radius", "diameter", "length",
        "width", "height", "thickness", "dimension", "proportion", "ratio",
        "= ", "mm", "units",
    ),
}

# order to test sections when classifying: latest construction stage first, so a
# line mentioning a fillet lands in refinement even if it also names a radius.
_TEST_ORDER = (
    "topological_refinement",
    "pattern_logic",
    "secondary_features",
    "primary_geometry",
    "global_parameters",
)


def classify_line(text: str) -> str:
    """The canonical section a single description line belongs to.

    Returns a key from :data:`SECTIONS`, or ``""`` if no keyword matches. Tested
    latest-stage-first so refinement/pattern cues win over a bare dimension.
    """
    low = text.lower()
    for section in _TEST_ORDER:
        for kw in _KEYWORDS[section]:
            if kw in low:
                return section
    return ""


def structure_description(lines) -> dict:
    """Group description lines into the five ordered sections.

    Returns a dict keyed by :data:`SECTIONS` (always all five keys, in order),
    each mapping to the list of lines assigned to it. Unclassifiable lines are
    collected under the ``""`` key. Deterministic.
    """
    out: dict = {s: [] for s in SECTIONS}
    out[""] = []
    for ln in lines:
        if not ln.strip():
            continue
        out[classify_line(ln)].append(ln)
    return out


def coverage(structured: dict) -> dict:
    """Which of the five sections are present (non-empty) in a structured dict."""
    return {s: bool(structured.get(s)) for s in SECTIONS}


@dataclass
class _ScaffoldSection:
    key: str
    title: str
    lines: list = field(default_factory=list)


def empty_scaffold() -> list:
    """The empty five-part template (ordered), ready for a generator to fill."""
    return [_ScaffoldSection(key=s, title=SECTION_TITLES[s]) for s in SECTIONS]


def render(structured: dict) -> str:
    """Render a structured description as the five titled sections, in order.

    Sections with no lines are still emitted (with a placeholder) so the fixed
    architecture is always visible. Unclassified lines are omitted from the
    rendered scaffold.
    """
    parts: list = []
    for i, s in enumerate(SECTIONS, start=1):
        parts.append(f"{i}. {SECTION_TITLES[s]}")
        lines = structured.get(s) or []
        if lines:
            parts.extend(f"   - {ln.strip()}" for ln in lines)
        else:
            parts.append("   - (none)")
    return "\n".join(parts)
