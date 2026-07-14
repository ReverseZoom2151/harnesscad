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

from harnesscad.eval.pressure.briefs import Brief, Expect, OpSpec
from harnesscad.eval.pressure.prompts import is_actionable

# Diagnostics that fire on every model regardless of what it emits (see
# prompts.UNACTIONABLE_CODES) are excluded from "the fleet caught something",
# otherwise every single attempt would score a catch and the number would mean
# nothing.
BLOCKING_SEVERITIES = ("error", "warning")


@dataclass
class Grade:
    """The verdict on one op stream."""

    solved: bool = False                       # geometry matches the brief
    built: bool = False                        # a valid solid exists at all
    reasons: List[str] = field(default_factory=list)   # why it failed
    measure: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[dict] = field(default_factory=list)
    fleet_actionable: List[dict] = field(default_factory=list)
    apply_ok: bool = False
    applied: int = 0

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
def grade(brief: Brief, ops: Sequence[dict]) -> Grade:
    """Rebuild `ops` from scratch and score them against the brief's ground truth.

    Always runs at ``verify_level="full"`` regardless of which arm produced the
    ops, so both arms are measured by exactly the same instrument.
    """
    from harnesscad.io.surfaces.server import CISPServer

    g = Grade()
    if not ops:
        g.reasons.append("no operations were produced")
        return g

    server = CISPServer(backend="frep", verify_level="full")
    try:
        result = server.applyOps([dict(o) for o in ops])
    except Exception as exc:
        g.reasons.append(f"the plan could not be applied: {exc}")
        return g

    g.apply_ok = bool(result.get("ok"))
    g.applied = int(result.get("applied") or 0)
    g.diagnostics = list(result.get("diagnostics") or [])
    g.fleet_actionable = [d for d in g.diagnostics if is_actionable(d)]

    if not g.apply_ok:
        g.reasons.append(
            "the plan was rejected by the core verifiers after %d op(s)" % g.applied)

    geom_reasons, measure = _check_geometry(server.backend, brief.expect)
    g.measure = measure
    g.built = bool(measure) and "no solid was produced" not in geom_reasons
    g.reasons.extend(geom_reasons)
    g.reasons.extend(_check_ops(ops, brief.expect.ops))

    g.solved = g.apply_ok and not g.reasons
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
    solved: bool
    attempts_used: int
    attempts_to_solve: Optional[int]     # 1-based; None when never solved
    invalid_ops: int                     # attempts whose output would not parse
    fleet_caught: int                    # attempts where the fleet said something
    fleet_missed: int                    # attempts with bad geometry + silent fleet
    final_reasons: List[str]
    final_diagnostics: List[dict]
    seconds: float
    records: List[dict]

    def to_dict(self) -> dict:
        return asdict(self)
