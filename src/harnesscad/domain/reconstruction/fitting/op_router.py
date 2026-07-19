"""Sketch-driven operation vocabulary and branch router.

Sequential CAD modelling is decomposed into four sketch-triggered
operations applied *in context* to the current shape.  A tiny classifier reads
(user stroke, context normal map, context depth map) and predicts the operation
type; the predicted type then selects one of four regression sub-networks, each
of which emits exactly the parameter maps its operation needs.  When the five
graphs are merged into one, a single forward pass produces every branch's maps
and the classifier decides which are consumed.

The vocabulary, the per-operation parameter requirements and the routing rule
are deterministic bookkeeping and are implemented here:

  * :data:`OP_SPECS` -- the four operations with the guiding curve each one
    regresses, whether the curve head is a masked regression or a sigmoid
    heat map, whether the operation needs an offset (distance/direction/sign)
    and whether it is additive, subtractive or shape-preserving;
  * :func:`softmax` / :func:`route` -- classifier logits to a :class:`Routing`
    decision (operation, confidence, margin, the branch outputs to read and the
    ones to ignore);
  * :func:`select_branch_outputs` -- pull just the routed branch's maps out of a
    combined-graph output dict, exactly what a caller of the merged network does;
  * :func:`required_parameters` -- what the downstream geometric decoder must
    produce for that operation to be applicable.

Operation ids follow the sub-network enumeration order used by the merged graph
(addSub, extrusion, bevel, sweep) and ``nb_cls = 4``.

Stdlib-only, deterministic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

OP_ADD_SUB = 0
OP_EXTRUSION = 1
OP_BEVEL = 2
OP_SWEEP = 3

NUM_OPERATIONS = 4


class OperationError(ValueError):
    """Raised for an unknown operation id/name or a malformed routing input."""


@dataclass(frozen=True)
class OperationSpec:
    op_id: int
    name: str
    #: label channel the branch's curve head regresses against
    guiding_curve: str
    #: 'regression' (masked L2 against a signed field) or 'heatmap' (sigmoid mask)
    curve_head: str
    #: does the op consume the offset distance/direction/sign channels?
    needs_offset: bool
    #: 'add', 'subtract', 'either' or 'modify' (bevel changes an existing edge)
    volume_effect: str
    #: the maps the branch emits, in merged-graph naming
    outputs: Tuple[str, ...]


OP_SPECS: Tuple[OperationSpec, ...] = (
    OperationSpec(
        op_id=OP_ADD_SUB,
        name="addSub",
        guiding_curve="base_curve",
        curve_head="regression",
        needs_offset=True,
        volume_effect="either",
        outputs=("face_heatmap", "base_curve"),
    ),
    OperationSpec(
        op_id=OP_EXTRUSION,
        name="extrusion",
        guiding_curve="offset_curve",
        curve_head="regression",
        needs_offset=True,
        volume_effect="add",
        outputs=("face_heatmap", "offset_curve"),
    ),
    OperationSpec(
        op_id=OP_BEVEL,
        name="bevel",
        guiding_curve="base_curve",
        curve_head="heatmap",
        needs_offset=False,
        volume_effect="modify",
        outputs=("face_heatmap", "base_curve"),
    ),
    OperationSpec(
        op_id=OP_SWEEP,
        name="sweep",
        guiding_curve="profile_curve",
        curve_head="regression",
        needs_offset=True,
        volume_effect="add",
        outputs=("face_heatmap", "profile_curve"),
    ),
)

_BY_ID: Dict[int, OperationSpec] = {s.op_id: s for s in OP_SPECS}
_BY_NAME: Dict[str, OperationSpec] = {s.name: s for s in OP_SPECS}

#: every map a merged (whole-graph) forward pass can produce
ALL_BRANCH_OUTPUTS: Tuple[str, ...] = tuple(
    sorted({o for s in OP_SPECS for o in s.outputs})
)


def spec_for(op: object) -> OperationSpec:
    """Look an operation up by id (int) or name (str)."""
    if isinstance(op, bool):
        raise OperationError("bool is not an operation id")
    if isinstance(op, int):
        if op not in _BY_ID:
            raise OperationError("unknown operation id: {}".format(op))
        return _BY_ID[op]
    if isinstance(op, str):
        if op not in _BY_NAME:
            raise OperationError("unknown operation name: {}".format(op))
        return _BY_NAME[op]
    raise OperationError("operation must be an int id or a str name")


def required_parameters(op: object) -> Tuple[str, ...]:
    """Geometric parameters the decoder must resolve for this operation."""
    s = spec_for(op)
    params: List[str] = ["stitching_face", s.guiding_curve]
    if s.needs_offset:
        params.extend(["offset_distance", "offset_direction", "offset_sign"])
    return tuple(params)


# ---------------------------------------------------------------------------
# classifier routing
# ---------------------------------------------------------------------------
def softmax(logits: Sequence[float]) -> List[float]:
    """Numerically stable softmax."""
    if not logits:
        raise OperationError("empty logits")
    m = max(logits)
    exps = [math.exp(float(v) - m) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


@dataclass(frozen=True)
class Routing:
    op: OperationSpec
    probs: Tuple[float, ...]
    confidence: float
    #: gap between the top-1 and runner-up probability
    margin: float
    #: True when confidence clears the acceptance threshold
    accepted: bool
    used_outputs: Tuple[str, ...]
    ignored_outputs: Tuple[str, ...]

    @property
    def op_id(self) -> int:
        return self.op.op_id

    @property
    def op_name(self) -> str:
        return self.op.name


def route(logits: Sequence[float], threshold: float = 0.0) -> Routing:
    """Classifier logits -> which regression branch to read.

    Ties are broken toward the lower operation id, keeping the decision stable.
    ``threshold`` (on the top-1 probability) marks low-confidence predictions as
    ``accepted=False`` so callers can fall back to asking the user.
    """
    if len(logits) != NUM_OPERATIONS:
        raise OperationError(
            "expected {} logits, got {}".format(NUM_OPERATIONS, len(logits))
        )
    probs = softmax(logits)
    best = 0
    for i in range(1, NUM_OPERATIONS):
        if probs[i] > probs[best]:
            best = i
    ordered = sorted(probs, reverse=True)
    margin = ordered[0] - ordered[1]
    spec = _BY_ID[best]
    used = spec.outputs
    ignored = tuple(o for o in ALL_BRANCH_OUTPUTS if o not in used)
    return Routing(
        op=spec,
        probs=tuple(probs),
        confidence=probs[best],
        margin=margin,
        accepted=probs[best] >= threshold,
        used_outputs=used,
        ignored_outputs=ignored,
    )


def select_branch_outputs(
    routing: Routing, combined: Mapping[str, object]
) -> Dict[str, object]:
    """Take only the routed branch's maps from a merged-graph output dict."""
    missing = [k for k in routing.used_outputs if k not in combined]
    if missing:
        raise OperationError(
            "merged outputs missing {} for op {}".format(missing, routing.op_name)
        )
    return {k: combined[k] for k in routing.used_outputs}


def confusion_matrix(
    predicted: Sequence[int], truth: Sequence[int]
) -> List[List[int]]:
    """``NUM_OPERATIONS x NUM_OPERATIONS`` matrix, rows = truth, cols = prediction."""
    if len(predicted) != len(truth):
        raise OperationError("prediction/truth length mismatch")
    mat = [[0] * NUM_OPERATIONS for _ in range(NUM_OPERATIONS)]
    for p, t in zip(predicted, truth):
        if p not in _BY_ID or t not in _BY_ID:
            raise OperationError("operation id out of range")
        mat[t][p] += 1
    return mat


def classification_accuracy(predicted: Sequence[int], truth: Sequence[int]) -> float:
    """Top-1 accuracy; an empty batch scores 0.0."""
    if len(predicted) != len(truth):
        raise OperationError("prediction/truth length mismatch")
    if not truth:
        return 0.0
    hits = sum(1 for p, t in zip(predicted, truth) if p == t)
    return hits / float(len(truth))
