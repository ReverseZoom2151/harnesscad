"""Grounded answer engine for CAD question answering.

An LLM writes Python code that (a) calls a segmentation module to retrieve the
parts referenced by the question and (b) computes the answer from the parts'
properties -- "the radius of a hole", "the center of a rod", counts, and so on.
The LLM/code-generation step is external (skipped). This module implements the
DETERMINISTIC computation that the generated code performs: given a structured
CAD model (a list of :class:`rag.querycad_segmentation_grounding.Part`) and a
typed :class:`bench.querycad_query_schema.CadQaQuestion`, it grounds the question
via the segmentation module and computes the answer directly from geometry.

Supported question types (11 different properties):

    count        -> number of grounded parts (integer)
    existence    -> whether any grounded part exists (boolean)
    measurement  -> the requested property; single part, or a list, or an
                    aggregate (min/max) when the question aggregates
    position     -> the center / tip of a part (a vector)
    comparison   -> the extreme part (largest/smallest by a property) and its
                    property value ("what is the diameter of the largest bore?")

Every answer is *grounded*: it records the ids of the parts it was computed from
(answers are traceable to CAD geometry, not
hallucinated). Property filters from the query (e.g. "radius of 5 mm")
are applied before answering.

All metrics are computed in world coordinates; local-orientation
and surface-normal questions as future work; those are intentionally out of scope
here. Pure, deterministic, stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from harnesscad.eval.bench.data.qa_query_schema import CadQaQuestion, PropertyFilter
from harnesscad.agents.rag.segmentation_grounding import Part, ground

# Position properties resolve against these attribute keys, in order.
_POSITION_KEYS = {
    "center": ("center", "position", "centroid"),
    "position": ("position", "center", "centroid"),
    "tip": ("tip", "position", "center"),
}


@dataclass(frozen=True)
class Answer:
    """A grounded answer.

    ``value``    the computed answer (int / float / bool / tuple / None).
    ``kind``     the answer kind (see CadQaQuestion.answer_kind()).
    ``part_ids`` ids of the parts the answer was grounded in (evidence).
    ``abstained``True when the question could not be answered from geometry
                 (e.g. a measurement on a part that lacks that property).
    """
    value: Any
    kind: str
    part_ids: Tuple[str, ...] = ()
    abstained: bool = False


def _passes_filters(part, filters):
    for f in filters:
        if not isinstance(f, PropertyFilter):
            raise TypeError("filters must be PropertyFilter instances")
        if not f.matches(part.attrs.get(f.prop)):
            return False
    return True


def _apply_filters(parts, filters):
    if not filters:
        return tuple(parts)
    return tuple(p for p in parts if _passes_filters(p, filters))


def _grounded_parts(model, question):
    """Segment the model for the question, then apply its property filters."""
    if question.targets_whole_model:
        # Whole-model questions ground on every part (no segmentation filter),
        # but still respect view + property constraints.
        parts = tuple(model)
        if question.views:
            from harnesscad.agents.rag.segmentation_grounding import visible_from
            parts = tuple(p for p in parts if visible_from(p, question.views))
    else:
        parts = ground(model, question.part, views=question.views)
    return _apply_filters(parts, question.filters)


def _measurement_value(part, prop):
    return part.attrs.get(prop)


def _position_value(part, prop):
    for key in _POSITION_KEYS.get(prop, (prop,)):
        if key in part.attrs:
            return tuple(part.attrs[key])
    return None


def answer_question(model, question):
    """Answer a :class:`CadQaQuestion` against a structured CAD model.

    ``model``    iterable of :class:`Part`.
    ``question`` a :class:`CadQaQuestion`.

    Returns an :class:`Answer`. Deterministic.
    """
    if not isinstance(question, CadQaQuestion):
        raise TypeError("question must be a CadQaQuestion")

    parts = _grounded_parts(model, question)
    ids = tuple(p.id for p in parts)
    qt = question.question_type

    if qt == "count":
        return Answer(len(parts), "int", ids)

    if qt == "existence":
        return Answer(bool(parts), "bool", ids)

    if qt == "measurement":
        vals = [(p.id, _measurement_value(p, question.prop)) for p in parts]
        vals = [(pid, v) for pid, v in vals if v is not None]
        if not vals:
            return Answer(None, "number", ids, abstained=True)
        if len(vals) == 1:
            pid, v = vals[0]
            return Answer(float(v), "number", (pid,))
        # Multiple matching parts and no aggregation: return the ordered list.
        return Answer(tuple(float(v) for _pid, v in vals), "number",
                      tuple(pid for pid, _v in vals))

    if qt == "position":
        vals = [(p.id, _position_value(p, question.prop)) for p in parts]
        vals = [(pid, v) for pid, v in vals if v is not None]
        if not vals:
            return Answer(None, "vector", ids, abstained=True)
        pid, v = vals[0]
        return Answer(v, "vector", (pid,))

    if qt == "comparison":
        vals = [(p.id, _measurement_value(p, question.prop)) for p in parts]
        vals = [(pid, v) for pid, v in vals if v is not None]
        if not vals:
            return Answer(None, "part_property", ids, abstained=True)
        # Select the extreme value, breaking ties by input order (first wins).
        best_pid, best_v = vals[0]
        for pid, v in vals[1:]:
            if question.aggregation == "max" and float(v) > float(best_v):
                best_pid, best_v = pid, v
            elif question.aggregation == "min" and float(v) < float(best_v):
                best_pid, best_v = pid, v
        return Answer((best_pid, float(best_v)), "part_property", (best_pid,))

    raise ValueError("unhandled question type: %r" % (qt,))  # pragma: no cover
