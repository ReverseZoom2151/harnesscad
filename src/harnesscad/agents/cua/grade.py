"""grade — the geometric oracle. The whole reason this agent exists.

The success signal for a computer-use CAD agent is NOT a model's opinion of a
screenshot. It is the geometry the agent built, measured three ways:

1.  **The GUI part, read back through the real kernel.** The harness's macro
    channel measures the exact B-rep of the body the agent drove the GUI to
    build (``env.measure``). This is out-of-band and asynchronous — hence the
    environment's ``synchronous_read=False`` — but it is the REAL kernel's
    answer, not a guess from pixels.

2.  **The differential against the scripted backend.** The SAME op stream the
    agent planned is run through :class:`harnesscad.io.backends.freecad.FreeCADBackend`
    — the scripted kernel that matches ANALYTIC to 4.5e-16 — and the two parts
    must agree. This is the test nobody else in the CUA field can write, because
    nobody else knows what is supposed to be on the screen. It measures whether
    the GUI DRIVE was faithful, independently of whether the plan was any good.

3.  **The output gate** (:mod:`harnesscad.io.gate`) on the scripted side: a
    valid, closed, manifold, non-self-intersecting solid, measured off a
    tolerance-controlled tessellation.

A brief is SOLVED when: the part is a valid solid (gate + GUI validity), the
differential is within tolerance (the GUI built what was planned), and the
GUI-measured geometry satisfies the brief's :class:`~harnesscad.agents.cua.briefs.Target`
(the plan was the right plan). Any one of those failing is not a solve.

Import-safe: no FreeCAD, no GUI and no scripted backend is touched at import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Op

#: Differential tolerances, mirroring tests/io/cua/test_differential_gui.py:
#: volume/area to 1e-6, lengths (bbox, centre of mass) to 1e-9. The GUI has
#: driven the box case to delta 0.00e+00; these are the ceilings, not the norm.
VOL_TOL = 1e-6
LEN_TOL = 1e-9


@dataclass
class Differential:
    """GUI-vs-scripted, field by field. ``max_delta`` is the headline number."""

    deltas: Dict[str, float] = field(default_factory=dict)
    max_delta: float = 0.0
    agree: bool = False
    mismatches: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"deltas": {k: _fmt(v) for k, v in self.deltas.items()},
                "max_delta": _fmt(self.max_delta), "agree": self.agree,
                "mismatches": list(self.mismatches)}


def _fmt(x: float) -> float:
    # Keep it a float, but round away float-noise in the report so 0.0 reads 0.0.
    return float("%.3e" % x) if x else 0.0


def differential(gui: Dict[str, Any], scripted: Dict[str, Any]) -> Differential:
    """Compare two exact B-rep measurements. Scalars by abs-delta, counts exactly.

    The bounding box and centre of mass are compared as SORTED extents, for the
    same reason the target is: a box does not care which axis is called length,
    and the scripted backend and the GUI need not have chosen the same one.
    """
    d = Differential()
    scalar_tol = {"volume": VOL_TOL, "surface_area": VOL_TOL}
    for key, tol in scalar_tol.items():
        if key in gui and key in scripted:
            delta = abs(float(gui[key]) - float(scripted[key]))
            d.deltas[key] = delta
            if delta > tol:
                d.mismatches.append("%s: |%.9f - %.9f| = %.3e > %.1e"
                                    % (key, gui[key], scripted[key], delta, tol))
    for key in ("bbox", "center_of_mass"):
        if key in gui and key in scripted:
            g = sorted(float(v) for v in gui[key])
            s = sorted(float(v) for v in scripted[key])
            delta = max((abs(a - b) for a, b in zip(g, s)), default=0.0)
            d.deltas[key] = delta
            if delta > LEN_TOL:
                d.mismatches.append("%s: max extent delta %.3e > %.1e"
                                    % (key, delta, LEN_TOL))
    for key in ("faces", "edges", "solids"):
        if key in gui and key in scripted:
            delta = abs(int(gui[key]) - int(scripted[key]))
            d.deltas[key] = float(delta)
            if delta != 0:
                d.mismatches.append("%s: %d != %d" % (key, gui[key], scripted[key]))
    d.max_delta = max((v for v in d.deltas.values()), default=0.0)
    d.agree = not d.mismatches
    return d


@dataclass
class GradeResult:
    """The verdict on one built part. ``solved`` is the number that goes on the
    scorecard; every other field is the evidence for it."""

    solved: bool = False
    reason: str = ""
    gui_metrics: Dict[str, Any] = field(default_factory=dict)
    scripted_metrics: Dict[str, Any] = field(default_factory=dict)
    diff: Optional[Differential] = None
    gui_valid: bool = False
    gate_ok: bool = False
    gate_failures: List[str] = field(default_factory=list)
    target_ok: bool = False
    target_misses: List[str] = field(default_factory=list)
    step_bytes: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "solved": self.solved, "reason": self.reason,
            "gui_metrics": self.gui_metrics,
            "scripted_metrics": self.scripted_metrics,
            "diff": None if self.diff is None else self.diff.to_dict(),
            "gui_valid": self.gui_valid, "gate_ok": self.gate_ok,
            "gate_failures": self.gate_failures,
            "target_ok": self.target_ok, "target_misses": self.target_misses,
            "step_bytes": self.step_bytes, "error": self.error,
        }


def scripted_measure(ops: Sequence[Op]) -> Tuple[Dict[str, Any], bool, List[str]]:
    """Run the op stream through the SCRIPTED FreeCAD backend: (metrics, gate_ok,
    gate_failures).

    This is ground truth. It is deliberately independent of the GUI: the whole
    point of the differential is that the two were built by different machinery
    and still agree. Raises ``BackendUnavailable`` (propagated) if freecadcmd is
    not installed — the caller SKIPs, never fails, on absence.
    """
    from harnesscad.core.environment import BackendEnvironment, require
    from harnesscad.io import gate as gate_mod
    from harnesscad.io.backends.freecad import FreeCADBackend

    backend = FreeCADBackend()
    env = BackendEnvironment(backend)
    require(env, "content_digest", "nonmutating_reject", "synchronous_read")
    env.reset()
    for op in ops:
        result = env.step(op)
        if not result.ok:
            msgs = [d.message for d in result.diagnostics]
            raise ValueError("scripted backend rejected %s: %s"
                             % (type(op).__name__, "; ".join(msgs)))
    metrics = backend.query("metrics")
    report = gate_mod.check(backend)
    failures = [f.code + ": " + f.detail for f in report.failures]
    return metrics, report.ok, failures


def grade_ops(env: Any, ops: Sequence[Op], target: Any = None) -> GradeResult:
    """Grade the part the agent built in ``env`` against the scripted oracle.

    ``env`` is a live :class:`~harnesscad.io.cua.environment_freecad.FreeCADGuiEnvironment`
    that has already been driven (its body exists). ``target`` is an optional
    :class:`~harnesscad.agents.cua.briefs.Target`. Never raises: any failure is a
    grade of ``solved=False`` with the reason, because an ungraded part is a part
    we cannot claim.
    """
    res = GradeResult()
    # 1. The GUI part, read back through the real kernel.
    try:
        res.gui_metrics = env.measure("full")
        if res.gui_metrics.get("solid_present") is False or "volume" not in res.gui_metrics:
            res.reason = "no solid in the GUI document"
            res.error = res.gui_metrics.get("error", "no solid")
            return res
        validity = env.measure("validity")
        res.gui_valid = bool(validity.get("is_valid")
                             and validity.get("solids", 0) >= 1)
    except Exception as exc:  # noqa: BLE001 - an unmeasurable part is not solved
        res.error = "GUI measurement failed: %s: %s" % (type(exc).__name__, exc)
        res.reason = "no measurable solid in the GUI"
        return res

    # 2. The scripted ground truth + the differential.
    try:
        res.scripted_metrics, res.gate_ok, res.gate_failures = scripted_measure(ops)
    except Exception as exc:  # noqa: BLE001 - unbuildable-by-script is a real signal
        res.error = "scripted reference failed: %s: %s" % (type(exc).__name__, exc)
        res.reason = "the planned op stream is not buildable by the scripted kernel"
        return res
    res.diff = differential(res.gui_metrics, res.scripted_metrics)

    # 3. The export, through the HARNESS channel (never the app's Save).
    try:
        res.step_bytes = len(env.export("step"))
    except Exception as exc:  # noqa: BLE001
        res.gate_failures.append("export: %s: %s" % (type(exc).__name__, exc))

    # 4. The target (brief satisfaction), if one was given.
    if target is not None:
        res.target_ok, res.target_misses = target.satisfied(res.gui_metrics)
    else:
        res.target_ok = True

    reasons: List[str] = []
    if not res.gui_valid:
        reasons.append("GUI part is not a valid closed solid")
    if not res.diff.agree:
        reasons.append("GUI-vs-scripted differential: " + "; ".join(res.diff.mismatches))
    if not res.gate_ok:
        reasons.append("output gate: " + "; ".join(res.gate_failures))
    if not res.target_ok:
        reasons.append("target: " + "; ".join(res.target_misses))
    res.solved = not reasons
    res.reason = "solved" if res.solved else " | ".join(reasons)
    return res
