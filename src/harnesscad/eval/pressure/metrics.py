"""Grading and metrics.

``grade()`` is the referee, and it is deliberately blind to which arm produced
the op stream. It takes the ops, rebuilds them from scratch in a fresh
``verify_level="full"`` F-rep session, and asks the brief's own ground truth
whether the resulting SOLID is right: is there a solid, is it manifold, is its
bounding box the size the brief asked for, is its volume in range, is there
material where there should be material, is there air where the brief asked for
a hole, and did the plan contain the features it was supposed to contain.

None of that is ever shown to a model. A loop can therefore only pass by
producing correct geometry -- not by producing geometry that pleases the
verifier that is talking to it.

The metric that matters most is not on the scoreboard the harness would choose
for itself: ``fleet_missed``. A brief is a fleet MISS when the geometry is
provably wrong by the ground truth and the whole 23-verifier fleet said nothing
actionable about it. Those are holes in the fleet, and they are bugs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.eval.pressure import shape as shape_mod
from harnesscad.eval.pressure.briefs import Brief, Expect, OpSpec
from harnesscad.eval.pressure.prompts import is_actionable, model_facing

# THE BLOCKING axis, and ONLY the blocking axis. `fleet_caught` asks "did the
# fleet claim something was wrong", which is a question about SEVERITY. What the
# MODEL is told is a question about SOUNDNESS and is answered by
# `prompts.model_facing` -- see the block comment there. v1 used this constant
# for both, which is how a rule with unmeasured precision ended up issuing
# instructions to a 14b that follows them perfectly.
#
# Diagnostics that fire on every part regardless of what the model emits (see
# prompts.UNACTIONABLE_CODES) are excluded, or every attempt would score a catch
# and the number would mean nothing.
BLOCKING_SEVERITIES = ("error", "warning")


@dataclass
class Grade:
    """The verdict on one op stream."""

    solved: bool = False                       # ENVELOPE verdict (+ the gate)
    built: bool = False                        # a valid solid exists at all
    reasons: List[str] = field(default_factory=list)   # why it failed
    measure: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[dict] = field(default_factory=list)
    fleet_actionable: List[dict] = field(default_factory=list)
    fleet_model_facing: List[dict] = field(default_factory=list)
    apply_ok: bool = False
    applied: int = 0
    # -- v2 --------------------------------------------------------------- #
    gate_ok: bool = False                      # io/gate.py accepted the artifact
    gate_failures: List[dict] = field(default_factory=list)
    shape: Dict[str, Any] = field(default_factory=dict)   # IoU vs the reference
    solved_shape: bool = False                 # ENVELOPE *and* SHAPE agree

    @property
    def fleet_caught(self) -> bool:
        """Did the fleet raise anything a model could act on?"""
        return any(d.get("severity") in BLOCKING_SEVERITIES
                   for d in self.fleet_actionable)

    @property
    def fleet_missed(self) -> bool:
        """Geometry is wrong AND the fleet said nothing. This is a HOLE."""
        return (not self.solved) and self.built and (not self.fleet_caught)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fleet_caught"] = self.fleet_caught
        d["fleet_missed"] = self.fleet_missed
        return d


# --------------------------------------------------------------------------- #
# op-level assertions
# --------------------------------------------------------------------------- #
def _check_ops(ops: Sequence[dict], specs: Sequence[OpSpec]) -> List[str]:
    reasons: List[str] = []
    for spec in specs:
        matching = [o for o in ops if o.get("op") == spec.tag]
        n = len(matching)
        if n < spec.count_min:
            reasons.append(
                f"expected at least {spec.count_min} '{spec.tag}' op(s), found {n}")
            continue
        if spec.count_max is not None and n > spec.count_max:
            reasons.append(
                f"expected at most {spec.count_max} '{spec.tag}' op(s), found {n}")
        for field_name, (lo, hi) in spec.params.items():
            for o in matching:
                if field_name not in o:
                    reasons.append(
                        f"'{spec.tag}' op is missing field '{field_name}'")
                    continue
                try:
                    v = float(o[field_name])
                except (TypeError, ValueError):
                    reasons.append(
                        f"'{spec.tag}.{field_name}' is not a number: {o[field_name]!r}")
                    continue
                if not (lo <= v <= hi):
                    reasons.append(
                        f"'{spec.tag}.{field_name}' = {v:g}, outside the "
                        f"feasible/required range [{lo:g}, {hi:g}]")
    return reasons


# --------------------------------------------------------------------------- #
# geometric assertions
# --------------------------------------------------------------------------- #
def _check_geometry(backend, expect: Expect) -> Tuple[List[str], Dict[str, Any]]:
    reasons: List[str] = []
    measure: Dict[str, Any] = {}

    try:
        summary = backend.query("summary")
    except Exception as exc:                                   # pragma: no cover
        return [f"backend summary failed: {exc}"], measure
    if not summary.get("solid_present"):
        return ["no solid was produced"], measure

    try:
        validity = backend.query("validity")
        measure["validity"] = validity
    except Exception as exc:                                   # pragma: no cover
        validity = {}
        reasons.append(f"validity query failed: {exc}")
    if validity and not validity.get("is_valid"):
        reasons.append(
            "solid is not valid (manifold=%s watertight=%s)"
            % (validity.get("manifold"), validity.get("watertight")))

    try:
        m = backend.query("measure")
        measure.update(m)
    except Exception as exc:                                   # pragma: no cover
        return reasons + [f"measure failed: {exc}"], measure

    bbox = m.get("bbox")
    volume = m.get("volume")

    if expect.bbox is not None and bbox:
        for axis, got, want in zip("xyz", bbox, expect.bbox):
            tol = max(expect.bbox_tol * want, 0.3)
            if abs(got - want) > tol:
                reasons.append(
                    f"bounding box {axis} = {got:.2f} mm, brief requires "
                    f"{want:g} mm (+/- {tol:.2f})")

    if expect.volume is not None and volume is not None:
        lo, hi = expect.volume
        if not (lo <= volume <= hi):
            reasons.append(
                f"volume = {volume:.0f} mm^3, outside the required "
                f"[{lo:.0f}, {hi:.0f}] mm^3")

    # The probes. These use the EXACT signed-distance field, not the mesh, so
    # they carry no discretisation error and they are the only check that can
    # tell "the hole was cut" from "the hole was cut somewhere else".
    fn = getattr(backend, "field", None)
    f = fn() if callable(fn) else None
    if f is not None:
        tol = expect.probe_tol
        for p in expect.inside:
            try:
                d = f(p)
            except Exception as exc:                           # pragma: no cover
                reasons.append(f"field probe at {p} raised {exc}")
                continue
            if d > -tol:
                reasons.append(
                    f"point {p} should be solid material but the surface is "
                    f"{d:+.2f} mm away (positive = outside the part)")
        for p in expect.outside:
            try:
                d = f(p)
            except Exception as exc:                           # pragma: no cover
                reasons.append(f"field probe at {p} raised {exc}")
                continue
            if d < tol:
                reasons.append(
                    f"point {p} should be empty (a hole / a cavity) but it is "
                    f"{d:+.2f} mm from the surface (negative = inside material)")
    return reasons, measure


# --------------------------------------------------------------------------- #
# the referee
# --------------------------------------------------------------------------- #
def grade(brief: Brief, ops: Sequence[dict], *, shape: bool = True) -> Grade:
    """Rebuild `ops` from scratch and score them against the brief's ground truth.

    Always runs at ``verify_level="full"`` regardless of which arm produced the
    ops, so every arm is measured by exactly the same instrument.

    v2 adds two things v1 did not have, and it reports them SEPARATELY rather
    than folding them into one number:

    THE GATE. v1's ``grade()`` constructed a raw ``CISPServer`` and never routed
    through ``io/gate.py`` -- the one component in the repository that refuses a
    dilated shell, an open surface, a cut that added volume, an extrude of the
    wrong height. An op stream that produced a wrong-sized part could therefore
    be scored ``solved=True`` by the very experiment that existed to catch it.
    The gate now runs on every graded attempt and a refusal is a loss.

    THE SHAPE. bbox + volume + sparse SDF probes are all ENVELOPE families and
    are many-to-one by construction: a hole in the wrong place scores perfectly.
    ``shape.score`` computes volumetric IoU against the brief's own hand-written
    ``reference`` op stream (which every brief already carried, and which v1 used
    only to prove the corpus was solvable). ``solved`` remains the v1-comparable
    ENVELOPE verdict; ``solved_shape`` is the stricter conjunction. BOTH are
    reported. Neither silently replaces the other.
    """
    from harnesscad.io import gate

    from harnesscad.eval.pressure.session import frep_server

    g = Grade()
    if not ops:
        g.reasons.append("no operations were produced")
        return g

    server = frep_server("full")          # PINNED mesher -- see session.py
    try:
        result = server.applyOps([dict(o) for o in ops])
    except Exception as exc:
        g.reasons.append(f"the plan could not be applied: {exc}")
        return g

    g.apply_ok = bool(result.get("ok"))
    g.applied = int(result.get("applied") or 0)
    g.diagnostics = list(result.get("diagnostics") or [])
    g.fleet_actionable = [d for d in g.diagnostics if is_actionable(d)]
    g.fleet_model_facing = model_facing(g.diagnostics)

    if not g.apply_ok:
        g.reasons.append(
            "the plan was rejected by the core verifiers after %d op(s)" % g.applied)

    geom_reasons, measure = _check_geometry(server.backend, brief.expect)
    g.measure = measure
    g.built = bool(measure) and "no solid was produced" not in geom_reasons
    g.reasons.extend(geom_reasons)
    g.reasons.extend(_check_ops(ops, brief.expect.ops))

    # -- the output gate ---------------------------------------------------- #
    bounds = None
    try:
        report = gate.check(server.backend, source=server)
        g.gate_ok = bool(report.ok)
        g.gate_failures = [f.to_dict() for f in report.failures]
        # The gate has already tessellated this solid -- the single most
        # expensive operation here -- so take its bounding box rather than
        # meshing the same part a second time for the shape metric.
        lo = report.measurement.get("bbox_min")
        hi = report.measurement.get("bbox_max")
        if lo and hi and report.measurement.get("triangle_count"):
            bounds = (tuple(float(c) for c in lo), tuple(float(c) for c in hi))
    except Exception as exc:                              # pragma: no cover
        g.gate_ok = False
        g.gate_failures = [{"check": "gate-error", "family": "measured",
                            "detail": f"{type(exc).__name__}: {exc}"}]
    if not g.gate_ok:
        for f in g.gate_failures:
            g.reasons.append("the output gate REFUSED this artifact "
                             f"[{f['check']}]: {f['detail']}")

    g.solved = g.apply_ok and g.gate_ok and not g.reasons

    # -- the shape metric --------------------------------------------------- #
    if shape:
        s = shape_mod.score(brief, server.backend, candidate_bounds=bounds)
        g.shape = s.to_dict()
        g.solved_shape = bool(g.solved and s.matched)
    return g


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass
class AttemptRecord:
    attempt: int
    raw: str
    parse_ok: bool
    parse_error: Optional[str]
    ops: List[dict]
    grade: Optional[dict]
    feedback: Optional[str]          # what this arm handed back to the model
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BriefResult:
    model: str
    loop: str
    brief: str
    category: str
    trap: bool
    seed: int
    solved: bool                         # ENVELOPE verdict (+ gate)
    solved_shape: bool                   # ENVELOPE and SHAPE (IoU vs reference)
    shape_iou: Optional[float]
    model_calls: int                     # THE COMPUTE. Compare arms on this.
    attempts_used: int
    attempts_to_solve: Optional[int]     # 1-based; None when never solved
    invalid_ops: int                     # attempts whose output would not parse
    fleet_caught: int                    # attempts where the fleet said something
    fleet_missed: int                    # attempts with bad geometry + silent fleet
    final_reasons: List[str]
    final_diagnostics: List[dict]
    seconds: float
    records: List[dict]
    #: Only the selection arms (oracle-BoN / self-consistency) fill this in.
    selection: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)
