"""Structural complexity of a HarnessCAD task, measured from its op stream.

Three of the four CAD-RL papers the harness tracks order their training data
simple-to-complex and all three measure a gain:

  * RLCAD (arXiv:2503.18549, sec 6.4) sorts a 500-geometry training set by
    increasing complexity and warm-starts complex models from simple ones. The
    curriculum arm wins on every reported metric (IoU 0.8757 vs 0.8354, COV
    0.8692 vs 0.8544, MMD-CD 0.0139 vs 0.0165, JSD 0.1111 vs 0.1307, NC 0.8812
    vs 0.8638).
  * ReCAD (arXiv:2512.06328) stages training over the primitive hierarchy
    P = {Loop, Face, Sketch, SketchExtrude, MultiSketchExtrude} and, WITHIN each
    level, orders samples by difficulty *defined as the number of curves*.
    Ablating the curriculum raises both reconstruction error and failure rate.

Neither paper can be applied to this repo directly: they order CAD-model
datasets, and a HarnessCAD task is a natural-language brief plus a reference
CISP op stream. This module supplies the missing measurement -- a complexity
score computed from the op stream that the ordering in
:mod:`harnesscad.eval.curriculum.ordering` consumes.

The metric
----------
Every term is a count over the op stream, so the score is a pure function of the
ops. No wall clock, no randomness, no I/O, no kernel evaluation.

  ``op_count``          number of ops in the stream.
  ``distinct_op_types`` number of distinct op tags. A stream that uses five
                        different verbs asks more of a model than one that
                        repeats a single verb five times.
  ``curve_count``       ReCAD's within-level difficulty measure, transcribed to
                        the CISP sketch vocabulary: each sketch entity op
                        contributes the number of curves it puts in the sketch
                        (see :data:`ENTITY_CURVES`). A rectangle is four curves,
                        a circle is one, an n-gon is n.
  ``constraint_count``  number of ``constrain`` ops -- the sketch's dimensional
                        and geometric burden.
  ``sketch_count``      number of ``new_sketch`` ops.
  ``feature_count``     number of solid-affecting ops (generators, combiners and
                        modifiers). This is the length of the feature list.
  ``feature_depth``     the DEPTH of the feature tree, which is not the same
                        number: the op stream is a register machine over a stack
                        of bodies, so a generator pushes a body at depth 1, a
                        modifier deepens the body on top of the stack by one, and
                        a boolean pops two bodies and pushes ``max(a, b) + 1``.
                        The score reports the deepest node ever reached. Two
                        extrusions unioned together is depth 2, not 3, while four
                        holes drilled in one plate is depth 5.
  ``max_op_tier``       the hardest op in the stream, by :data:`OP_TIER` -- a
                        hand-assigned 0..4 scale over the CISP op set (sketch
                        scaffolding 0, straight-line entities and basic
                        extrusion 1, curved entities and local modifiers 2,
                        topology-sensitive features such as shell/draft/revolve
                        3, freeform and assembly features 4).
  ``tier_sum``          sum of the per-op tiers. Reported but NOT scored: it is
                        almost collinear with ``op_count``.

The score is the fixed linear combination in :data:`WEIGHTS`::

    score = 1.00 * op_count
          + 1.00 * distinct_op_types
          + 0.25 * curve_count
          + 0.50 * constraint_count
          + 2.00 * feature_depth
          + 1.00 * max_op_tier

The weights are declared, not fitted. ``feature_depth`` carries the most weight
per unit because each level of the feature tree is a place the geometry can go
wrong downstream of a correct prefix; ``curve_count`` carries the least because
a rectangle contributing four curves must not outweigh a real second feature.
See :mod:`harnesscad.eval.curriculum.validate` for the measured agreement with
the pressure corpus's hand-assigned difficulty column, including the ablation
that shows what each term buys.

Accepted task shapes
--------------------
:func:`task_ops` duck-types rather than importing any corpus, so this module has
no dependency on ``eval.pressure`` / ``eval.corpus`` / ``eval.hardcorpus`` and
can score anything that carries ops: a raw op stream, a
``pressure.briefs.Brief`` (``.reference``), or any object exposing ``.ops`` /
``.op_stream`` / ``.reference``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

__all__ = [
    "ENTITY_CURVES",
    "GENERATOR_OPS",
    "COMBINER_OPS",
    "MODIFIER_OPS",
    "SOLID_OPS",
    "OP_TIER",
    "WEIGHTS",
    "ComplexityFeatures",
    "op_tag",
    "curve_count",
    "feature_depth",
    "features",
    "score",
    "task_ops",
    "task_id",
    "task_features",
    "task_score",
]


#: Curves each CISP sketch-entity op contributes to its sketch. This is ReCAD's
#: within-level difficulty measure ("the number of curves involved"), read off
#: the entity's own geometry: a rectangle is four straight curves, a circle /
#: ellipse / arc / spline is one, a point is none, and a polygon is one curve per
#: vertex (its ``points`` field is a FLAT (x0, y0, x1, y1, ...) tuple, so the
#: vertex count is half its length).
ENTITY_CURVES: Dict[str, int] = {
    "add_point": 0,
    "add_line": 1,
    "add_arc": 1,
    "add_circle": 1,
    "add_ellipse": 1,
    "add_spline": 1,
    "add_rectangle": 4,
    # add_polygon is computed from its point count, not from this table.
}

#: Ops that CREATE a solid body from nothing (or from a sketch).
GENERATOR_OPS = frozenset({
    "extrude", "revolve", "loft", "sweep", "primitive",
})

#: Ops that CONSUME two bodies and leave one.
COMBINER_OPS = frozenset({"boolean", "hull", "minkowski", "mate"})

#: Ops that transform the body already on top of the stack.
MODIFIER_OPS = frozenset({
    "hole", "shell", "fillet", "chamfer", "draft", "split", "thicken",
    "transform", "scale", "mirror",
    "linear_pattern", "circular_pattern", "pattern_transform",
    "add_instance",
})

#: Every op that touches the solid. ``feature_count`` counts these.
SOLID_OPS = GENERATOR_OPS | COMBINER_OPS | MODIFIER_OPS

#: Hand-assigned 0..4 difficulty tier per CISP op tag. The scale is ordinal, not
#: metric: it exists so that "a stream containing a loft" scores above "a stream
#: containing an extrude" even when the two streams are the same length.
#:
#:   0  scaffolding that cannot fail on its own (new_sketch, add_point)
#:   1  straight-line entities, dimensional constraints, plain extrusion
#:   2  curved / freeform sketch entities, local solid modifiers, patterns
#:   3  topology-sensitive features whose validity depends on the whole body
#:      (shell, draft, revolve, split, thicken)
#:   4  freeform surfacing and assembly (loft, sweep, hull, minkowski, mate)
OP_TIER: Dict[str, int] = {
    "new_sketch": 0,
    "add_point": 0,
    "add_line": 1,
    "add_rectangle": 1,
    "add_circle": 1,
    "constrain": 1,
    "set_param": 1,
    "extrude": 1,
    "primitive": 1,
    "add_arc": 2,
    "add_ellipse": 2,
    "add_polygon": 2,
    "add_spline": 2,
    "boolean": 2,
    "hole": 2,
    "fillet": 2,
    "chamfer": 2,
    "transform": 2,
    "scale": 2,
    "mirror": 2,
    "linear_pattern": 2,
    "circular_pattern": 2,
    "pattern_transform": 2,
    "shell": 3,
    "draft": 3,
    "revolve": 3,
    "split": 3,
    "thicken": 3,
    "loft": 4,
    "sweep": 4,
    "hull": 4,
    "minkowski": 4,
    "add_instance": 4,
    "mate": 4,
}

#: Tier assumed for an op tag this module has never heard of. Deliberately
#: mid-scale: an unknown verb is treated as a normal modifier rather than as
#: free or as maximally hard. ``tests/eval/curriculum/`` pins OP_TIER against the
#: live CISP registry so this default never silently absorbs a new op.
DEFAULT_TIER = 2

#: The scored terms and their weights. Declared, not fitted.
WEIGHTS: Dict[str, float] = {
    "op_count": 1.0,
    "distinct_op_types": 1.0,
    "curve_count": 0.25,
    "constraint_count": 0.5,
    "feature_depth": 2.0,
    "max_op_tier": 1.0,
}


@dataclass(frozen=True)
class ComplexityFeatures:
    """The counted terms behind a complexity score."""

    op_count: int = 0
    distinct_op_types: int = 0
    curve_count: int = 0
    constraint_count: int = 0
    sketch_count: int = 0
    feature_count: int = 0
    feature_depth: int = 0
    max_op_tier: int = 0
    tier_sum: int = 0

    @property
    def score(self) -> float:
        """The weighted sum defined by :data:`WEIGHTS`."""
        return round(
            sum(w * getattr(self, name) for name, w in sorted(WEIGHTS.items())),
            6,
        )

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "op_count": self.op_count,
            "distinct_op_types": self.distinct_op_types,
            "curve_count": self.curve_count,
            "constraint_count": self.constraint_count,
            "sketch_count": self.sketch_count,
            "feature_count": self.feature_count,
            "feature_depth": self.feature_depth,
            "max_op_tier": self.max_op_tier,
            "tier_sum": self.tier_sum,
        }
        out["score"] = self.score
        return out


# --------------------------------------------------------------------------- #
# op-stream readers
# --------------------------------------------------------------------------- #
def op_tag(op: Any) -> str:
    """The op tag of ``op``, whichever of the three shapes it arrives in.

    A dict (``{"op": "extrude", ...}``), a ``core.cisp.ops.Op`` instance (whose
    ``OP`` class var carries the tag), or a bare tag string.
    """
    if isinstance(op, str):
        return op
    if isinstance(op, dict):
        return str(op.get("op", ""))
    tag = getattr(op, "OP", None)
    if tag is None:
        tag = getattr(op, "op", None)
    return str(tag) if tag is not None else ""


def _op_field(op: Any, name: str, default: Any = None) -> Any:
    if isinstance(op, dict):
        return op.get(name, default)
    return getattr(op, name, default)


def curve_count(ops: Sequence[Any]) -> int:
    """ReCAD's difficulty measure: the number of curves the sketches contain."""
    total = 0
    for op in ops:
        tag = op_tag(op)
        if tag == "add_polygon":
            points = _op_field(op, "points", ()) or ()
            total += max(0, len(points) // 2)
        else:
            total += ENTITY_CURVES.get(tag, 0)
    return total


def feature_depth(ops: Sequence[Any]) -> int:
    """Depth of the feature tree the op stream builds.

    The stream is a register machine over a stack of bodies:

      * a GENERATOR pushes a fresh body at depth 1;
      * a MODIFIER replaces the body on top with one a level deeper;
      * a COMBINER pops two bodies and pushes ``max(a, b) + 1`` (with one body
        on the stack it degrades to a modifier, which is what an unbalanced
        stream means in practice).

    The answer is the deepest node the stream ever reaches. This distinguishes
    "two extrusions unioned" (depth 2) from "one extrusion with two holes"
    (depth 3) even though both are three feature ops.
    """
    stack = []
    peak = 0
    for op in ops:
        tag = op_tag(op)
        if tag in GENERATOR_OPS:
            stack.append(1)
        elif tag in COMBINER_OPS:
            if len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(max(a, b) + 1)
            elif stack:
                stack[-1] += 1
            else:
                stack.append(1)
        elif tag in MODIFIER_OPS:
            if stack:
                stack[-1] += 1
            else:
                stack.append(1)
        else:
            continue
        if stack and max(stack) > peak:
            peak = max(stack)
    return peak


def features(ops: Sequence[Any]) -> ComplexityFeatures:
    """Count every term of the metric over ``ops``."""
    ops = tuple(ops)
    tags = [op_tag(op) for op in ops]
    tiers = [OP_TIER.get(t, DEFAULT_TIER) for t in tags]
    return ComplexityFeatures(
        op_count=len(ops),
        distinct_op_types=len(set(tags)),
        curve_count=curve_count(ops),
        constraint_count=sum(1 for t in tags if t == "constrain"),
        sketch_count=sum(1 for t in tags if t == "new_sketch"),
        feature_count=sum(1 for t in tags if t in SOLID_OPS),
        feature_depth=feature_depth(ops),
        max_op_tier=max(tiers) if tiers else 0,
        tier_sum=sum(tiers),
    )


def score(ops: Sequence[Any]) -> float:
    """The complexity score of an op stream."""
    return features(ops).score


# --------------------------------------------------------------------------- #
# task adapters (duck-typed: no corpus import, so no layering edge)
# --------------------------------------------------------------------------- #
#: Attribute names, in priority order, under which a task may carry its ops.
OPS_ATTRS: Tuple[str, ...] = ("ops", "op_stream", "reference", "program")


def task_ops(task: Any) -> Tuple[Any, ...]:
    """Extract the op stream from a task, whatever shape the task is.

    Accepts a bare sequence of ops, a mapping carrying one of :data:`OPS_ATTRS`,
    or any object with one of those attributes (``pressure.briefs.Brief`` keeps
    its known-good solution on ``.reference``). Returns an empty tuple when the
    task carries no ops -- a task with no reference solution scores 0 rather
    than raising, so a mixed corpus can still be ordered.
    """
    if task is None:
        return ()
    if isinstance(task, (list, tuple)):
        return tuple(task)
    if isinstance(task, dict):
        for name in OPS_ATTRS:
            if name in task:
                value = task[name]
                if value:
                    return tuple(value)
        return ()
    for name in OPS_ATTRS:
        value = getattr(task, name, None)
        if value:
            return tuple(value)
    return ()


def task_id(task: Any) -> str:
    """A stable identifier for a task, used as the deterministic tie-break.

    Falls back to the repr of the task's op stream when the task has no id, so
    two distinct anonymous tasks still receive distinct, reproducible keys.
    """
    if isinstance(task, dict):
        for name in ("id", "task_id", "name", "brief_id"):
            if task.get(name):
                return str(task[name])
    else:
        for name in ("id", "task_id", "name", "brief_id"):
            value = getattr(task, name, None)
            if value:
                return str(value)
    return repr([op_tag(op) for op in task_ops(task)])


def task_features(task: Any) -> ComplexityFeatures:
    """Complexity features of a task (see :func:`task_ops` for the shapes)."""
    return features(task_ops(task))


def task_score(task: Any) -> float:
    """Complexity score of a task."""
    return task_features(task).score
