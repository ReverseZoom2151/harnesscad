"""Per-tile exemplar selection + ICL prompt assembly (DST inference stage).

Implements the *code-generation stage* of the DST framework (Sec. 3.4 +
Appendix C.2): selected exemplars and their CAD code are structured into a
three-part in-context-learning prompt --

    1. System prompt      -- establishes the "expert CAD engineer" role.
    2. Instruction        -- output-format constraints (```python``` only).
    3. Demonstration seq. -- k (specification, code) exemplar pairs, ordered
                             by relevance to the query.

then the query specification is appended as the final User Input. The template
text mirrors the paper's Appendix C.2 verbatim in spirit.

This module wires the retrieval side (:mod:`rag.spectiling_greedy`) to the
prompt side, and adds the paper-leaves-to-future-work *per-tile* policy: a
composite spec is decomposed into tiles, DST selection runs per tile against
the shared exemplar database, and the union of picked exemplars (deduplicated,
kept in first-seen order) forms the demonstration sequence. This directly
serves the knowledge-sufficiency principle: every tile contributes the
exemplars that best tile *its* components.

stdlib-only; deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from harnesscad.agents.context.spec_components import ComponentSet, DEFAULT_GRANULARITIES
from harnesscad.agents.rag.exemplar_select import dst_select
from harnesscad.domain.spec.spec_decompose import decompose_spec, ordered_tiles

SYSTEM_PROMPT = (
    "You are an expert CAD engineer proficient in Python and the CadQuery "
    "library. Your task is to generate precise, executable CadQuery scripts "
    "according to given natural language descriptions."
)

INSTRUCTION = (
    "Please create a CadQuery Python code which can generate a model based on "
    "the instruction and description. The final CadQuery code MUST BE put in "
    "```python code``` with ONLY the executable code inside the python box, "
    "nothing else. Relevant examples will be provided in sequence according to "
    "their similarity to the final query, and these examples may be helpful "
    "for answer generation. Please don't use the non-existent '.scale()' "
    "method on Workplane objects."
)


@dataclass(frozen=True)
class Exemplar:
    """A (design specification, CAD code) database entry."""

    spec: str
    code: str


def _exemplar_components(
    exemplars: Sequence[Exemplar], granularities: Sequence[int]
) -> List[ComponentSet]:
    return [ComponentSet.from_text(e.spec, granularities) for e in exemplars]


def select_for_query(
    query_spec: str,
    exemplars: Sequence[Exemplar],
    k: int,
    granularities: Sequence[int] = DEFAULT_GRANULARITIES,
) -> List[int]:
    """Whole-query DST selection -> exemplar indices in greedy pick order."""
    query = ComponentSet.from_text(query_spec, granularities)
    comp = _exemplar_components(exemplars, granularities)
    return dst_select(query, comp, k).ordered_indices()


def select_per_tile(
    query_spec: str,
    exemplars: Sequence[Exemplar],
    k_per_tile: int,
    total_budget: int,
    granularities: Sequence[int] = DEFAULT_GRANULARITIES,
) -> List[int]:
    """Per-tile exemplar selection policy.

    Decompose ``query_spec`` into dependency-ordered tiles; for each tile run
    DST to pick up to ``k_per_tile`` exemplars; concatenate the picks in tile
    order, de-duplicating (first tile to want an exemplar keeps it) and capping
    the total at ``total_budget``. Falls back to whole-query selection when the
    spec yields no tiles.
    """
    comp = _exemplar_components(exemplars, granularities)
    tiles = ordered_tiles(decompose_spec(query_spec, granularities))
    if not tiles:
        return select_for_query(
            query_spec, exemplars, total_budget, granularities
        )
    picked: List[int] = []
    seen = set()
    for tile in tiles:
        if len(picked) >= total_budget:
            break
        sel = dst_select(tile.components, comp, k_per_tile)
        for idx in sel.ordered_indices():
            if idx in seen:
                continue
            seen.add(idx)
            picked.append(idx)
            if len(picked) >= total_budget:
                break
    return picked


def assemble_prompt(
    query_spec: str,
    exemplars: Sequence[Exemplar],
    selected_indices: Sequence[int],
) -> str:
    """Render the full ICL prompt (Appendix C.2 template).

    ``selected_indices`` is the demonstration order (as produced by a selection
    policy above). Exemplars are emitted between ``#Examples Begin`` and
    ``#Examples End`` markers, each as a Description + fenced code block, then
    the query is appended under ``User Input``.
    """
    lines: List[str] = []
    lines.append("System Prompt:")
    lines.append(SYSTEM_PROMPT)
    lines.append("")
    lines.append("Instruction:")
    lines.append(INSTRUCTION)
    lines.append("")
    lines.append("#Examples Begin:")
    for i in selected_indices:
        ex = exemplars[i]
        lines.append(f"Description: {ex.spec}")
        lines.append("```python")
        lines.append(ex.code)
        lines.append("```")
    lines.append("")
    lines.append("#Examples End")
    lines.append("")
    lines.append("User Input:")
    lines.append(f"Description: {query_spec}")
    return "\n".join(lines)


def build_icl_prompt(
    query_spec: str,
    exemplars: Sequence[Exemplar],
    k: int,
    per_tile: bool = False,
    k_per_tile: int = 2,
    granularities: Sequence[int] = DEFAULT_GRANULARITIES,
) -> Tuple[str, List[int]]:
    """End-to-end: select exemplars then assemble the prompt.

    Returns ``(prompt_text, selected_indices)``. When ``per_tile`` is True the
    per-tile policy is used with ``k`` as the total budget; otherwise whole-
    query DST selection picks ``k`` exemplars.
    """
    if per_tile:
        idx = select_per_tile(
            query_spec, exemplars, k_per_tile, k, granularities
        )
    else:
        idx = select_for_query(query_spec, exemplars, k, granularities)
    return assemble_prompt(query_spec, exemplars, idx), idx
