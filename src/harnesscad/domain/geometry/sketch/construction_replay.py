"""Construction-sequence -> CAD profile compiler (replay engine).

Replays a :class:`geometry.euclid_dsl.ConstructionSequence` starting from a
"geometric prompt" environment of named geometry, executing each atomic
construction step with its closed-form solver from
:mod:`geometry.euclid_construction`, and assembling the ``CreatedCurve`` outputs
into a final CAD profile of closed loops.

Because every step is closed-form, the replay is exact ("floating point
precision", as the paper puts it) and deterministic; editing a parameter value
and replaying gives a parametric edit of the profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple, Union

from harnesscad.domain.geometry.sketch import ruler_compass as ec
from harnesscad.domain.geometry.sketch.construction_dsl import ConstructionSequence, Step

Geom = Union[ec.Point, ec.Line, ec.Circle, ec.Arc, ec.Segment]
Curve = Union[ec.Segment, ec.Arc, ec.Circle]


class ReplayError(ValueError):
    """Raised when a construction sequence cannot be replayed."""


@dataclass
class Profile:
    """A CAD profile: an ordered list of loops.

    Each loop is either a list of ``Segment``/``Arc`` curves forming a closed
    chain, or a single ``Circle``. Loop order follows the paper's convention of
    outer loop first.
    """

    loops: List[Union[List[Curve], ec.Circle]] = field(default_factory=list)

    def curve_count(self) -> int:
        n = 0
        for loop in self.loops:
            n += 1 if isinstance(loop, ec.Circle) else len(loop)
        return n


# ---------------------------------------------------------------------------
# Step dispatch: op name -> callable(inputs, params) -> outputs (tuple)
# ---------------------------------------------------------------------------
def _dispatch(op: str, ins: Sequence[Geom], params: Sequence[float]) -> Tuple[Geom, ...]:
    if op == "CircleOffsetCircle":
        return (ec.circle_offset_circle(ins[0], params[0]),)
    if op == "LineXLine":
        return (ec.line_x_line(ins[0], ins[1]),)
    if op == "LineOffsetLine":
        return (ec.line_offset_line(ins[0], params[0]),)
    if op == "LineXCircle":
        pts = ec.line_x_circle(ins[0], ins[1])
        if not pts:
            raise ReplayError("LineXCircle: no intersection")
        return (pts[0],)
    if op == "CircleReverseCircle":
        return (ec.circle_reverse_circle(ins[0]),)
    if op == "CirclePointPointArc":
        return (ec.circle_point_point_arc(ins[0], ins[1], ins[2]),)
    if op == "LineDatumParallelLine":
        return (ec.line_datum_parallel_line(ins[0], ins[1]),)
    if op == "LineLineFillet":
        return (ec.line_line_fillet(ins[0], ins[1], params[0]),)
    if op == "LineCircleParallelLine":
        return (ec.line_circle_parallel_line(ins[0], ins[1]),)
    if op == "LineSymLineLine":
        return (ec.line_sym_line_line(ins[0], ins[1]),)
    if op == "PointLineSymPoint":
        return (ec.point_line_sym_point(ins[0], ins[1]),)
    if op == "LineReverseLine":
        return (ec.line_reverse_line(ins[0]),)
    if op == "LineAxisRotatedLine":
        return (ec.line_axis_rotated_line(ins[0], ins[1], params[0]),)
    if op == "PointRadiusCircle":
        return (ec.point_radius_circle(ins[0], params[0]),)
    if op == "SymlineOffsetLineLine":
        return ec.symline_offset_line_line(ins[0], params[0])
    raise ReplayError("unknown op: %r" % (op,))


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def _curve_from_geom(g: Geom) -> Curve:
    if isinstance(g, (ec.Segment, ec.Arc, ec.Circle)):
        return g
    raise ReplayError("CreatedCurve output is not a curve: %r" % (type(g),))


def replay(seq: ConstructionSequence,
           prompt: Dict[str, Geom]) -> Tuple[Dict[str, Geom], List[Curve]]:
    """Execute ``seq`` against a ``prompt`` environment.

    Returns ``(env, created_curves)`` where ``env`` is the final geometry
    environment (prompt geometry plus every constructed output) and
    ``created_curves`` is the ordered list of curves flagged with CreatedCurve.
    """
    env: Dict[str, Geom] = dict(prompt)
    created: List[Curve] = []
    for k, step in enumerate(seq.steps):
        spec = step.spec()
        try:
            ins = [env[i] for i in step.inputs]
        except KeyError as exc:
            raise ReplayError(
                "step %d (%s): undefined input %s" % (k, step.op, exc)) from exc
        params = [seq.parameters.get(pi, 0.0) for pi in step.param_indices]
        outs = _dispatch(step.op, ins, params)
        if len(outs) != len(step.outputs):
            raise ReplayError(
                "step %d (%s): produced %d outputs, expected %d"
                % (k, step.op, len(outs), len(step.outputs)))
        for oid, g in zip(step.outputs, outs):
            env[oid] = g
        if step.creates_curve:
            created.append(_curve_from_geom(env[step.outputs[-1]]))
    return env, created


# ---------------------------------------------------------------------------
# Profile assembly
# ---------------------------------------------------------------------------
def _endpoints(curve: Curve) -> Tuple[ec.Point, ec.Point]:
    if isinstance(curve, ec.Segment):
        return curve.start, curve.end
    if isinstance(curve, ec.Arc):
        return curve.start, curve.end
    raise ReplayError("circle has no endpoints")


def assemble_profile(curves: Sequence[Curve], tol: float = ec.TOL) -> Profile:
    """Chain created curves into closed loops, isolating circles into own loops.

    Non-circle curves are greedily chained end-to-start into loops; a loop
    closes when the running end meets the loop start within ``tol``.
    """
    profile = Profile()
    open_curves = [c for c in curves if not isinstance(c, ec.Circle)]
    for c in curves:
        if isinstance(c, ec.Circle):
            profile.loops.append(c)

    used = [False] * len(open_curves)
    for start_idx in range(len(open_curves)):
        if used[start_idx]:
            continue
        loop: List[Curve] = [open_curves[start_idx]]
        used[start_idx] = True
        _, cur_end = _endpoints(open_curves[start_idx])
        loop_start, _ = _endpoints(open_curves[start_idx])
        closed = False
        while not closed:
            if cur_end.almost_equals(loop_start, tol):
                closed = True
                break
            found = False
            for j in range(len(open_curves)):
                if used[j]:
                    continue
                s, e = _endpoints(open_curves[j])
                if cur_end.almost_equals(s, tol):
                    loop.append(open_curves[j])
                    used[j] = True
                    cur_end = e
                    found = True
                    break
            if not found:
                break
        profile.loops.append(loop)
    return profile


def compile_profile(seq: ConstructionSequence,
                    prompt: Dict[str, Geom],
                    tol: float = ec.TOL) -> Profile:
    """Full pipeline: replay ``seq`` on ``prompt`` and assemble the profile."""
    _, created = replay(seq, prompt)
    return assemble_profile(created, tol)
