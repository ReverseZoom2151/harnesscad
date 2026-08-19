"""Curriculum ordering: present a task collection simple-to-complex.

Two orderings, both total and both deterministic.

FLAT ordering (RLCAD, arXiv:2503.18549 sec 6.4) sorts the whole collection by
the structural complexity score of
:mod:`harnesscad.eval.curriculum.complexity`. RLCAD orders its 500-geometry
training set by increasing complexity and initialises the weights for a complex
model from the simple one it already solved; the curriculum arm beats
case-by-case training on every metric they report.

HIERARCHICAL ordering (ReCAD, arXiv:2512.06328) stages the collection over the
primitive hierarchy P = {L, F, S, SE, MSE} and, *within* a stage, orders by the
number of curves. :func:`structural_level` is the missing half of that recipe
for this repo: it reads a CISP op stream and says which level of P the task
lives at. The level ranking itself is imported from
:mod:`harnesscad.data.dataengine.curation.primitive_curriculum`, which already
transcribes ReCAD Eq. 1 -- this module binds it to real op streams instead of
to abstract ``{"primitive": ..., "curves": ...}`` records.

Determinism
-----------
Both sort keys end in the task id (see ``complexity.task_id``), so the order is
a total order on the collection and re-sorting the same collection -- or a
shuffled copy of it -- yields byte-identical output. Nothing here reads the
clock or draws a random number.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from harnesscad.data.dataengine.curation.primitive_curriculum import (
    PRIMITIVE_ORDER,
    primitive_rank,
)
from harnesscad.eval.curriculum import complexity as _cx

__all__ = [
    "PRIMITIVE_ORDER",
    "CLOSED_REGION_ENTITIES",
    "structural_level",
    "level_rank",
    "flat_key",
    "hierarchical_key",
    "order_tasks",
    "stages",
    "batches",
    "order_table",
]


#: Sketch entities that are a closed region on their own. Everything else
#: (line, arc, open spline, point) is boundary geometry that has to be chained
#: into a loop before it bounds anything.
CLOSED_REGION_ENTITIES = frozenset({
    "add_circle", "add_ellipse", "add_rectangle", "add_polygon",
})


def _regions(ops: Sequence[Any]) -> int:
    """Number of self-closing regions the sketch entities declare."""
    count = 0
    for op in ops:
        tag = _cx.op_tag(op)
        if tag in CLOSED_REGION_ENTITIES:
            count += 1
        elif tag == "add_spline" and _op_closed(op):
            count += 1
    return count


def _op_closed(op: Any) -> bool:
    if isinstance(op, dict):
        return bool(op.get("closed", False))
    return bool(getattr(op, "closed", False))


def structural_level(task: Any) -> str:
    """Which level of ReCAD's primitive hierarchy P a task lives at.

    ``P = {L (loop), F (face), S (sketch), SE (sketch-extrude), MSE (multi-SE)}``
    mapped onto the CISP op set:

      * ``MSE`` -- two or more solid GENERATORS (extrude / revolve / loft /
        sweep / primitive). This is the multi-sketch-extrude case: the model has
        to build several bodies and compose them.
      * ``SE``  -- exactly one generator: one sketch turned into one solid, then
        modified. The overwhelming majority of real briefs.
      * ``S``   -- no generator, but more than one sketch or more than one closed
        region: a sketch that groups faces.
      * ``F``   -- no generator, one sketch, at most one closed region: a single
        face.
      * ``L``   -- no sketch at all: loose entities, a bare loop. An EMPTY op
        stream also lands here, which is the right default -- a task with no
        reference solution sorts to the front of the curriculum rather than
        exploding the sort.
    """
    ops = _cx.task_ops(task)
    generators = sum(1 for op in ops if _cx.op_tag(op) in _cx.GENERATOR_OPS)
    if generators >= 2:
        return "MSE"
    if generators == 1:
        return "SE"
    sketches = sum(1 for op in ops if _cx.op_tag(op) == "new_sketch")
    if sketches == 0:
        return "L"
    if sketches > 1 or _regions(ops) > 1:
        return "S"
    return "F"


def level_rank(task: Any) -> int:
    """Curriculum rank of a task's structural level (0 = ``L`` .. 4 = ``MSE``)."""
    return primitive_rank(structural_level(task))


# --------------------------------------------------------------------------- #
# sort keys
# --------------------------------------------------------------------------- #
def flat_key(task: Any) -> Tuple[float, int, int, str]:
    """RLCAD-style key: increasing complexity, total and deterministic.

    ``(score, curve_count, op_count, task_id)``. The trailing id makes the order
    a TOTAL order -- two tasks with identical structure still have one fixed
    relative position, so the same collection always yields the same sequence.
    """
    f = _cx.task_features(task)
    return (f.score, f.curve_count, f.op_count, _cx.task_id(task))


def hierarchical_key(task: Any) -> Tuple[int, int, float, str]:
    """ReCAD-style key: stage by primitive level, then by CURVE COUNT.

    ``(level_rank, curve_count, score, task_id)``. Curve count comes before the
    score because that is exactly what ReCAD specifies as the within-level
    difficulty; the score is a secondary tie-break and the id a final one.
    """
    f = _cx.task_features(task)
    return (level_rank(task), f.curve_count, f.score, _cx.task_id(task))


#: The two supported orderings, by name.
KEYS: Dict[str, Callable[[Any], tuple]] = {
    "flat": flat_key,
    "hierarchical": hierarchical_key,
}


def order_tasks(tasks: Iterable[Any], mode: str = "flat") -> List[Any]:
    """Return ``tasks`` ordered simple-to-complex.

    ``mode`` is ``"flat"`` (RLCAD: one global sort by complexity score) or
    ``"hierarchical"`` (ReCAD: staged over P, ordered by curve count inside a
    stage). Both are stable AND total, so the result is reproducible.
    """
    try:
        key = KEYS[mode]
    except KeyError:
        raise ValueError(
            "unknown curriculum mode %r; expected one of %s"
            % (mode, sorted(KEYS))
        )
    return sorted(tasks, key=key)


def stages(tasks: Iterable[Any]) -> List[Tuple[str, Tuple[Any, ...]]]:
    """Group tasks into ReCAD's per-level stages, in curriculum order.

    Returns ``[(level, (task, ...)), ...]`` walking ``L -> F -> S -> SE -> MSE``,
    each stage internally ordered by curve count. Empty levels are omitted, so a
    corpus that is entirely single-extrude yields a single ``("SE", ...)`` stage
    rather than four empty ones.
    """
    ordered = order_tasks(tasks, mode="hierarchical")
    out: List[Tuple[str, List[Any]]] = []
    for task in ordered:
        level = structural_level(task)
        if not out or out[-1][0] != level:
            out.append((level, []))
        out[-1][1].append(task)
    return [(level, tuple(members)) for level, members in out]


def batches(tasks: Sequence[Any], size: int, mode: str = "flat") -> List[Tuple[Any, ...]]:
    """Split the curriculum into fixed-size batches, easiest batch first."""
    if size < 1:
        raise ValueError("batch size must be >= 1")
    ordered = order_tasks(tasks, mode=mode)
    return [tuple(ordered[i:i + size]) for i in range(0, len(ordered), size)]


def order_table(tasks: Iterable[Any], mode: str = "flat") -> List[Dict[str, Any]]:
    """One row per task, in curriculum order -- the reportable form."""
    rows = []
    for position, task in enumerate(order_tasks(tasks, mode=mode)):
        row = _cx.task_features(task).to_dict()
        row["position"] = position
        row["id"] = _cx.task_id(task)
        row["level"] = structural_level(task)
        rows.append(row)
    return rows
