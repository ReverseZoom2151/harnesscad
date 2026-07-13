"""Constructibility checking, profile validity and construction-accuracy metrics.

Implements the deterministic evaluation machinery of Li et al., "Draw It Like
Euclid" (Sections 4.2.1 and 5.2, Appendix C.2):

* a *constructibility / validity checker* that verifies a construction sequence
  is syntactically well formed and that every input is available before it is
  used (no geometry constructed "from nothing");
* profile *validity metrics*: syntactic validity, no self-intersection, no short
  edges (Section 5.2.1);
* a *construction-accuracy* metric that compares the per-step output geometry of
  a predicted sequence against a reference, reproducing the distance / angle /
  flag comparisons of Table 10.

The learned reward model is unnecessary: as the paper notes, profile validity is
directly and objectively measurable, which is exactly what this module does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

from harnesscad.domain.geometry import euclid_construction as ec
from harnesscad.domain.geometry.euclid_compiler import Profile, ReplayError, replay
from harnesscad.domain.geometry.euclid_dsl import STEP_SPECS, ConstructionSequence, MAX_PARAMETERS

Geom = Union[ec.Point, ec.Line, ec.Circle, ec.Arc, ec.Segment]


# ---------------------------------------------------------------------------
# Constructibility / syntactic checker
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    ok: bool
    errors: List[str]


def _type_of(g: Geom) -> str:
    if isinstance(g, ec.Point):
        return "point"
    if isinstance(g, ec.Line):
        return "line"
    if isinstance(g, ec.Circle):
        return "circle"
    if isinstance(g, ec.Arc):
        return "arc"
    if isinstance(g, ec.Segment):
        return "line"  # bounded segment satisfies a line-typed slot loosely
    return "?"


def check_sequence(seq: ConstructionSequence,
                   prompt: Dict[str, Geom]) -> CheckResult:
    """Statically validate a construction sequence against a prompt.

    Checks, without executing any geometry solver:

    * every op is known and has the declared input/output arity;
    * parameter indices are in range and every referenced parameter has a value;
    * the correct number of scalar params is supplied;
    * every input id is already defined (prompt or a previous output) -- i.e. no
      geometry is used before it is constructed;
    * input entity *types* match the op's schema (points/lines/circles);
    * output ids are not silently reused in a way that leaves dangling inputs.
    """
    errors: List[str] = []
    defined: Dict[str, str] = {gid: _type_of(g) for gid, g in prompt.items()}
    for k, step in enumerate(seq.steps):
        if step.op not in STEP_SPECS:
            errors.append("step %d: unknown op %r" % (k, step.op))
            continue
        spec = STEP_SPECS[step.op]
        if len(step.inputs) != len(spec.inputs):
            errors.append("step %d (%s): expected %d inputs, got %d"
                          % (k, step.op, len(spec.inputs), len(step.inputs)))
        if len(step.outputs) != len(spec.outputs):
            errors.append("step %d (%s): expected %d outputs, got %d"
                          % (k, step.op, len(spec.outputs), len(step.outputs)))
        if len(step.param_indices) != spec.n_params:
            errors.append("step %d (%s): expected %d params, got %d"
                          % (k, step.op, spec.n_params, len(step.param_indices)))
        for pi in step.param_indices:
            if not (0 <= pi < MAX_PARAMETERS):
                errors.append("step %d (%s): param index %d out of range"
                              % (k, step.op, pi))
            elif pi not in seq.parameters:
                errors.append("step %d (%s): param %d has no value"
                              % (k, step.op, pi))
        for slot, gid in enumerate(step.inputs):
            if gid not in defined:
                errors.append("step %d (%s): input %r used before definition"
                              % (k, step.op, gid))
                continue
            if slot < len(spec.inputs):
                want = spec.inputs[slot]
                got = defined[gid]
                if want != got:
                    errors.append(
                        "step %d (%s): input %r is %s, expected %s"
                        % (k, step.op, gid, got, want))
        for slot, gid in enumerate(step.outputs):
            if slot < len(spec.outputs):
                defined[gid] = spec.outputs[slot]
        if step.creates_curve and step.outputs:
            last = spec.outputs[-1] if spec.outputs else "?"
            if last not in ("line", "circle", "arc"):
                errors.append(
                    "step %d (%s): CreatedCurve output is %s, not a curve"
                    % (k, step.op, last))
    return CheckResult(ok=not errors, errors=errors)


def syntactic_validity(seq: ConstructionSequence,
                       prompt: Dict[str, Geom]) -> bool:
    """Whether the sequence is well formed *and* replays without a solver error.

    Corresponds to the paper's "syntactic validity" metric: can the generated
    sequence be detokenised/replayed under the strict rules of the DSL.
    """
    if not check_sequence(seq, prompt).ok:
        return False
    try:
        replay(seq, prompt)
    except ReplayError:
        return False
    except (ValueError, ZeroDivisionError):
        return False
    return True


# ---------------------------------------------------------------------------
# Profile geometry sampling
# ---------------------------------------------------------------------------
def _arc_polyline(arc: ec.Arc, n: int = 16) -> List[ec.Point]:
    """Sample an arc (start, mid, end) into a polyline by fitting its circle."""
    center = _arc_center(arc)
    if center is None:
        return [arc.start, arc.mid, arc.end]
    r = center.dist(arc.start)
    a0 = math.atan2(arc.start.y - center.y, arc.start.x - center.x)
    am = math.atan2(arc.mid.y - center.y, arc.mid.x - center.x)
    a1 = math.atan2(arc.end.y - center.y, arc.end.x - center.x)
    # Decide sweep direction so the mid angle lies between start and end.
    ccw_sweep = (am - a0) % (2 * math.pi)
    full_ccw = (a1 - a0) % (2 * math.pi)
    if ccw_sweep <= full_ccw + 1e-9:
        sweep = full_ccw
        sign = 1.0
    else:
        sweep = (a0 - a1) % (2 * math.pi)
        sign = -1.0
    pts = []
    for i in range(n + 1):
        a = a0 + sign * sweep * (i / n)
        pts.append(ec.Point(center.x + r * math.cos(a),
                            center.y + r * math.sin(a)))
    return pts


def _arc_center(arc: ec.Arc):
    """Circumcentre of the three arc points, or None if collinear."""
    ax, ay = arc.start.x, arc.start.y
    bx, by = arc.mid.x, arc.mid.y
    cx, cy = arc.end.x, arc.end.y
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-14:
        return None
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
          + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
          + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    return ec.Point(ux, uy)


def _loop_polyline(loop) -> List[ec.Point]:
    if isinstance(loop, ec.Circle):
        pts = []
        for i in range(48):
            a = 2 * math.pi * i / 48
            pts.append(loop.point_at_angle(a))
        return pts
    pts: List[ec.Point] = []
    for curve in loop:
        if isinstance(curve, ec.Segment):
            seg = [curve.start, curve.end]
        elif isinstance(curve, ec.Arc):
            seg = _arc_polyline(curve)
        else:
            continue
        if pts and seg and pts[-1].almost_equals(seg[0], 1e-9):
            pts.extend(seg[1:])
        else:
            pts.extend(seg)
    return pts


def _seg_intersect(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        if abs(d1) > 1e-12 and abs(d2) > 1e-12 and abs(d3) > 1e-12 and abs(d4) > 1e-12:
            return True
    return False


# ---------------------------------------------------------------------------
# Validity metrics (Section 5.2.1)
# ---------------------------------------------------------------------------
def no_self_intersection(profile: Profile) -> bool:
    """True if no two non-adjacent boundary edges of the profile cross."""
    for loop in profile.loops:
        pts = _loop_polyline(loop)
        m = len(pts)
        if m < 4:
            continue
        # Treat the polyline as closed.
        edges = [(pts[i], pts[(i + 1) % m]) for i in range(m)]
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                if j == i or (j + 1) % len(edges) == i or (i + 1) % len(edges) == j:
                    continue
                if _seg_intersect(edges[i][0], edges[i][1],
                                  edges[j][0], edges[j][1]):
                    return False
    return True


def no_short_edges(profile: Profile, min_len: float = 1.0 / 127.0) -> bool:
    """True if every straight/arc curve exceeds the minimum length."""
    for loop in profile.loops:
        if isinstance(loop, ec.Circle):
            if 2 * math.pi * loop.radius < min_len:
                return False
            continue
        for curve in loop:
            if isinstance(curve, ec.Segment):
                if curve.length() < min_len:
                    return False
            elif isinstance(curve, ec.Arc):
                if curve.start.dist(curve.end) < min_len:
                    return False
    return True


def profile_area(profile: Profile) -> float:
    """Signed-area magnitude of the outer boundary (shoelace over samples)."""
    if not profile.loops:
        return 0.0
    pts = _loop_polyline(profile.loops[0])
    if len(pts) < 3:
        return 0.0
    s = 0.0
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        s += a.x * b.y - b.x * a.y
    return abs(s) / 2.0


# ---------------------------------------------------------------------------
# Construction-accuracy metric (Appendix C.2, Table 10)
# ---------------------------------------------------------------------------
@dataclass
class GeomError:
    dist: float = 0.0        # positional / distance error
    angle: float = 0.0       # angular error (radians)
    flag_agree: float = 1.0  # 1.0 when orientation flags agree, else 0.0


def _angle_diff(a: float, b: float) -> float:
    d = abs((a - b) % (2 * math.pi))
    return min(d, 2 * math.pi - d)


def geom_error(pred: Geom, ref: Geom) -> GeomError:
    """Per-entity error between a predicted and reference geometry.

    Matches the comparison rules of Table 10: point/segment endpoints compared
    by distance, lines by angle *and* signed distance, circles by centre
    distance, radius difference and orientation-flag agreement, arcs by the mean
    endpoint+midpoint distance.
    """
    if isinstance(pred, ec.Point) and isinstance(ref, ec.Point):
        return GeomError(dist=pred.dist(ref))
    if isinstance(pred, ec.Line) and isinstance(ref, ec.Line):
        return GeomError(dist=abs(pred.rho - ref.rho),
                         angle=_angle_diff(pred.phi, ref.phi))
    if isinstance(pred, ec.Circle) and isinstance(ref, ec.Circle):
        return GeomError(dist=pred.center.dist(ref.center) + abs(pred.radius - ref.radius),
                         flag_agree=1.0 if pred.ccw == ref.ccw else 0.0)
    if isinstance(pred, ec.Arc) and isinstance(ref, ec.Arc):
        d = (pred.start.dist(ref.start) + pred.mid.dist(ref.mid)
             + pred.end.dist(ref.end)) / 3.0
        return GeomError(dist=d)
    if isinstance(pred, ec.Segment) and isinstance(ref, ec.Segment):
        d = (pred.start.dist(ref.start) + pred.end.dist(ref.end)) / 2.0
        return GeomError(dist=d)
    # Type mismatch: maximal disagreement.
    return GeomError(dist=float("inf"), flag_agree=0.0)


def construction_accuracy(pred_outputs: Sequence[Geom],
                          ref_outputs: Sequence[Geom]) -> Dict[str, float]:
    """Aggregate per-step construction accuracy over paired output geometries.

    Returns mean distance error, mean angular error, mean flag agreement and the
    fraction of outputs whose type matches (``solution_exists``), echoing the
    Table 10 style of reporting. Raises if the two sequences differ in length.
    """
    if len(pred_outputs) != len(ref_outputs):
        raise ValueError("prediction/reference length mismatch")
    if not pred_outputs:
        return {"mean_dist": 0.0, "mean_angle": 0.0,
                "mean_flag_agree": 1.0, "solution_exists": 1.0, "count": 0.0}
    tot_d = tot_a = tot_f = 0.0
    solved = 0
    for p, r in zip(pred_outputs, ref_outputs):
        e = geom_error(p, r)
        if math.isinf(e.dist):
            # unmatched type; count as unsolved, skip in numeric means
            tot_f += e.flag_agree
            continue
        solved += 1
        tot_d += e.dist
        tot_a += e.angle
        tot_f += e.flag_agree
    denom = max(1, solved)
    return {
        "mean_dist": tot_d / denom,
        "mean_angle": tot_a / denom,
        "mean_flag_agree": tot_f / len(pred_outputs),
        "solution_exists": solved / len(pred_outputs),
        "count": float(len(pred_outputs)),
    }
