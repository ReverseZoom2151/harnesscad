"""The few-shot exemplar bank — verified brief -> CISP op streams.

`agents/agent/system_prompt.py` was PURE ZERO-SHOT: a strict-format structured-
output task (emit a valid typed op array) with no worked example anywhere in the
prompt. The retrieval machinery to fix that was already in the repository and
orphaned: `agents/rag/exemplar_select.py` (greedy submodular DST selection with a
(1 - 1/e) guarantee) and `agents/context/exemplar_prompt.py`. Nothing called
either, because there was no exemplar bank for CISP ops to select FROM. This is
that bank.

Every stream below is EXECUTED against the F-rep backend by
`tests/agents/context/test_cisp_exemplars.py` and must apply cleanly, so a
"verified exemplar" is a claim the test suite checks rather than a comment. An
exemplar that stops building fails the build — a wrong worked example in a
system prompt is a false diagnostic that fires on every single brief.

Selection is by tiled component overlap with the query brief (DST), not by
embedding: stdlib only, deterministic, no model call, no index to keep warm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from harnesscad.agents.context.spec_components import (
    ComponentSet,
    DEFAULT_GRANULARITIES,
)
from harnesscad.agents.rag.exemplar_select import dst_select

__all__ = ["Exemplar", "EXEMPLARS", "select", "render", "few_shot_block"]

#: How many exemplars the prompt carries by default. Three is the point where
#: the format is unambiguous and the prompt is still short.
DEFAULT_K = 3


@dataclass(frozen=True)
class Exemplar:
    """A verified (brief, ops) pair. `ops` is a list of raw CISP op dicts."""

    name: str
    brief: str
    ops: Tuple[Dict[str, Any], ...]

    def as_json(self) -> str:
        return json.dumps([dict(op) for op in self.ops], indent=2)


# NOTE: no `constrain` ops appear below. `distance` is a BINARY constraint and
# needs a second entity, so a one-entity form does not build; an exemplar that
# does not build is worse than no exemplar. The sketches are therefore
# under-constrained (a WARNING, never fatal) and the prompt's RULES still tell the
# model to pin its sketches. The test below is what keeps this honest.
EXEMPLARS: Tuple[Exemplar, ...] = (
    Exemplar(
        name="plate_four_holes",
        brief=("A rectangular steel plate 70 mm long, 70 mm wide and 8 mm thick, "
               "with four 7 mm through holes, one 10 mm in from each corner."),
        ops=(
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 70, "h": 70},
            {"op": "extrude", "sketch": "sk1", "distance": 8},
            {"op": "hole", "face_or_sketch": "", "x": 10, "y": 10,
             "diameter": 7, "through": True},
            {"op": "hole", "face_or_sketch": "", "x": 60, "y": 10,
             "diameter": 7, "through": True},
            {"op": "hole", "face_or_sketch": "", "x": 10, "y": 60,
             "diameter": 7, "through": True},
            {"op": "hole", "face_or_sketch": "", "x": 60, "y": 60,
             "diameter": 7, "through": True},
        ),
    ),
    Exemplar(
        name="round_flange_bolt_circle",
        brief=("A round flange: an 80 mm diameter disc, 8 mm thick, with a 30 mm "
               "central bore and four 7 mm bolt holes on a 60 mm bolt circle."),
        ops=(
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_circle", "sketch": "sk1", "cx": 0, "cy": 0, "r": 40},
            {"op": "extrude", "sketch": "sk1", "distance": 8},
            {"op": "hole", "face_or_sketch": "", "x": 0, "y": 0,
             "diameter": 30, "through": True},
            {"op": "hole", "face_or_sketch": "", "x": 30, "y": 0,
             "diameter": 7, "through": True},
            {"op": "hole", "face_or_sketch": "", "x": -30, "y": 0,
             "diameter": 7, "through": True},
            {"op": "hole", "face_or_sketch": "", "x": 0, "y": 30,
             "diameter": 7, "through": True},
            {"op": "hole", "face_or_sketch": "", "x": 0, "y": -30,
             "diameter": 7, "through": True},
        ),
    ),
    Exemplar(
        name="l_bracket",
        brief=("An L-shaped bracket: a 60 x 40 mm base plate 6 mm thick with a "
               "60 x 30 mm vertical wall 6 mm thick rising from one long edge."),
        ops=(
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 60, "h": 40},
            {"op": "extrude", "sketch": "sk1", "distance": 6},
            {"op": "new_sketch", "plane": "XZ"},
            {"op": "add_rectangle", "sketch": "sk2", "x": 0, "y": 0, "w": 60, "h": 30},
            {"op": "extrude", "sketch": "sk2", "distance": 6},
            {"op": "boolean", "kind": "union", "target": "f1", "tool": "f2"},
        ),
    ),
    Exemplar(
        name="filleted_plate",
        brief=("A 50 x 30 mm plate, 6 mm thick, with its four vertical corner "
               "edges rounded to a 3 mm radius."),
        ops=(
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 50, "h": 30},
            {"op": "extrude", "sketch": "sk1", "distance": 6},
            {"op": "fillet", "edges": [], "radius": 3},
        ),
    ),
)
# NOTE: `shelled_box` and `revolve` exemplars were written and DELETED rather than
# shipped: the F-rep kernel refuses a 3 mm wall on a 60 mm part (resolution) and
# refuses the revolve axis as written. Both were removed rather than "fixed" into
# something the kernel merely tolerates.
#
# NOTE: a `revolve` exemplar was written and DELETED rather than shipped: the op
# stream did not build (the kernel rejected the axis), and the test below would
# have failed. A worked example that does not build is a false diagnostic that
# fires on every brief. The bank contains only streams the kernel accepts.


def _components(exemplars: Sequence[Exemplar], granularities):
    return [ComponentSet.from_text(e.brief, granularities) for e in exemplars]


def select(brief: str, k: int = DEFAULT_K,
           exemplars: Sequence[Exemplar] = EXEMPLARS,
           granularities=DEFAULT_GRANULARITIES) -> List[Exemplar]:
    """Greedy submodular (DST) selection of the k exemplars that best tile `brief`.

    Ties break on lowest index, so the same brief always yields the same prompt.
    When the brief tiles nothing (an empty or wholly novel spec) the greedy stops
    early and we top up in bank order, because a format demonstration is worth
    more than an exact topical match: the failure few-shot fixes is FORMAT.
    """
    if k <= 0 or not exemplars:
        return []
    query = ComponentSet.from_text(brief, granularities)
    picked = list(dst_select(query, _components(exemplars, granularities), k).indices)
    for i in range(len(exemplars)):
        if len(picked) >= min(k, len(exemplars)):
            break
        if i not in picked:
            picked.append(i)
    return [exemplars[i] for i in picked[:k]]


def render(exemplars: Sequence[Exemplar]) -> str:
    """Render exemplars as the WORKED EXAMPLES block of the system prompt."""
    lines: List[str] = []
    for ex in exemplars:
        lines.append(f"BRIEF: {ex.brief}")
        lines.append("OPS:")
        lines.append(ex.as_json())
        lines.append("")
    return "\n".join(lines).rstrip()


def few_shot_block(brief: str, k: int = DEFAULT_K) -> str:
    """The whole block, or "" when there is nothing to show."""
    chosen = select(brief, k)
    if not chosen:
        return ""
    return ("WORKED EXAMPLES (verified: each op stream below builds on the real "
            "kernel). Follow the SHAPE, not the numbers:\n\n" + render(chosen))
