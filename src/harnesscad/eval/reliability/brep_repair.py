"""B-rep repair + repair-advisor — the *heal* half of the recycling loop.

The verification layer (verify.py, checks_geometry.py) DETECTS invalidity; this
module is what lets the harness ACT on it. Per the blueprint's recycling loop
("diagnostics AND repair suggestions — loosen dim, add fillet continuity, heal
geometry") it offers two complementary capabilities:

1. GEOMETRIC repair — :func:`repair_solid` runs the real OCCT healing toolkit
   (``ShapeFix`` / ``ShapeUpgrade``) over a backend's B-rep to close small gaps,
   fix face orientation, sew unshared edges and drop degeneracies, then reports
   exactly what changed (before/after validity + a topology diff). It is lazy and
   cadquery-guarded: with no OCCT, no solid, or nothing to fix it returns a clean
   no-op result with a note and NEVER raises. The heal is an escalating TOLERANCE
   LADDER (see below), not a single fixed-precision pass.

2. DIAGNOSTIC-TO-REPAIR ADVISOR — :class:`RepairAdvisor` deterministically maps
   the diagnostic *codes* the verifiers emit (over-constrained, invalid-brep,
   empty-solid, self-intersection, ...) to concrete candidate CISP ops/edits the
   agent can try next (drop a specific Constrain, run a heal, add a small fillet,
   check the profile is closed, ...). This is the deterministic prior handed to
   the agent, and it composes :class:`guardrails.ErrorRecovery` so each suggestion
   also carries its rung on the detect -> handle -> recover ladder.

TOLERANCE LADDER (attribution)
-----------------------------
The escalation ladder, the ShapeFix_Wire/ShapeFix_Face flag recipe and the
sewing rung are REIMPLEMENTED (no copied text) from the facts documented in
``resources/cad_repos/Brepler-main/Brepler-main/network/post/utils.py`` — the
B-repLer post-processing stage that turns network-predicted surfaces into a
valid OCCT solid. That repository ships no LICENSE file, so nothing is vendored
from it; what is reused here are *facts about OCCT behaviour* (numeric
tolerances and which ShapeFix flags must be on/off), expressed in original code.
The specific facts taken:

  * CONNECT tolerance ladder ``2e-3 .. 8e-2`` for connecting/fixing generated
    topology; FIX precision/tolerance ``1e-2``; SEWING tolerance ``1e-1``;
    small-edge removal tolerance ``1e-3``.
  * The ShapeFix_Face recipe: FixOrientation / FixMissingSeam / FixWire ON, and
    — the non-obvious part — FixIntersectingWires, FixLoopWires,
    FixPeriodicDegenerated and FixSmallAreaWire OFF, with
    AutoCorrectPrecisionMode OFF so the requested tolerance is actually used.
  * The ShapeFix_Wire recipe: ModifyTopology + ModifyGeometry ON, FixShifted and
    ClosedWire ON, FixSmall/FixGaps3d enabled at the rung tolerance.

Corroboration: AutoBrep independently reports that the sew tolerance for
*generated* (as opposed to authored) geometry is ``1e-2``, not ``1e-6``. That is
the point of the ladder: a single 1e-6 pass is the wrong precision for exactly
the geometry a text-to-CAD harness produces.

DEFAULT-SAFE: the ladder's first rung is byte-for-byte the historical behaviour
(ShapeFix_Shape at precision 1e-6 / max tolerance 1e-2 + UnifySameDomain), and
escalation happens ONLY when a rung leaves the shape invalid. A shape that
healed (or was already valid) before still stops at rung 0, so the ladder can
only ever widen what heals, never narrow it. Every rung restarts from the
ORIGINAL shape, so a failed rung cannot compound damage into the next one.

Absolute imports; OCCT touched only inside cadquery-guarded paths; degrades
gracefully everywhere.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Constrain, Op
from harnesscad.eval.reliability.guardrails import ErrorRecovery
from harnesscad.eval.verifiers.verify import Diagnostic, Severity


# ======================================================================
# Geometric repair
# ======================================================================
@dataclass
class RepairResult:
    """Outcome of a geometric repair pass.

    ``healed`` is True only when the OCCT healing toolkit actually changed the
    topology (or turned an invalid solid valid). ``actions`` names the fixes that
    fired; ``before_validity`` / ``after_validity`` are the backend's own validity
    reports around the pass; ``diff`` is the topology/volume delta; ``note``
    explains any no-op (missing OCCT, no solid, nothing to fix, kernel error).
    """

    healed: bool
    actions: List[str] = field(default_factory=list)
    before_validity: dict = field(default_factory=dict)
    after_validity: dict = field(default_factory=dict)
    diff: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "healed": self.healed,
            "actions": list(self.actions),
            "before_validity": dict(self.before_validity),
            "after_validity": dict(self.after_validity),
            "diff": dict(self.diff),
            "note": self.note,
        }


def _cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return False


def _safe_validity(backend) -> dict:
    """Read ``backend.query('validity')`` defensively (never raise)."""
    try:
        v = backend.query("validity")
        return dict(v) if isinstance(v, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _describe_shape(shape) -> dict:
    """A small, comparable topology descriptor for a cq Shape (never raises)."""
    d: dict = {}
    for key, meth in (("faces", "Faces"), ("edges", "Edges"),
                      ("vertices", "Vertices"), ("solids", "Solids")):
        try:
            d[key] = len(getattr(shape, meth)())
        except Exception:  # noqa: BLE001
            d[key] = None
    try:
        d["volume"] = round(float(shape.Volume()), 6)
    except Exception:  # noqa: BLE001
        d["volume"] = None
    return d


# ----------------------------------------------------------------------
# Tolerance ladder (facts reimplemented from Brepler network/post/utils.py)
# ----------------------------------------------------------------------
#: The historical single-pass parameters. Rung 0 reproduces these exactly, so
#: the ladder is a superset of the old behaviour.
BASELINE_PRECISION = 1e-6
BASELINE_MAX_TOLERANCE = 1e-2

#: Brepler CONNECT_TOLERANCE: the escalating band over which generated topology
#: is connected/fixed. Authored CAD needs the low end; network- or LLM-generated
#: geometry routinely needs 1e-2 and up.
CONNECT_TOLERANCE: Tuple[float, ...] = (2e-3, 6e-3, 1e-2, 1.5e-2, 2e-2, 2.5e-2,
                                        5e-2, 8e-2)
#: Brepler FIX_PRECISION / FIX_TOLERANCE — the precision at which the explicit
#: ShapeFix_Face/ShapeFix_Wire flag recipe is run.
FIX_PRECISION = 1e-2
FIX_TOLERANCE = 1e-2
#: Brepler SEWING_TOLERANCE — deliberately an order of magnitude looser than the
#: fix tolerance; sewing is the last resort before giving up on a shell.
SEWING_TOLERANCE = 1e-1
#: Brepler REMOVE_EDGE_TOLERANCE — edges shorter than this are dropped, not fixed.
REMOVE_EDGE_TOLERANCE = 1e-3


@dataclass(frozen=True)
class LadderRung:
    """One rung of the escalating repair ladder.

    ``precision`` / ``max_tolerance`` are handed to the OCCT fixers. ``face_recipe``
    runs the explicit per-face ShapeFix_Face + ShapeFix_Wire flag recipe before the
    shape-level fix; ``sew`` additionally sews the faces into a shell (and, when a
    single shell results, into a solid) at ``max_tolerance``.
    """

    name: str
    precision: float
    max_tolerance: float
    face_recipe: bool = False
    sew: bool = False


def default_ladder() -> List[LadderRung]:
    """The default escalation ladder, cheapest and tightest rung first.

    Rung 0 is the historical fixed pass. Rungs 1..8 escalate ShapeFix_Shape over
    Brepler's CONNECT_TOLERANCE band. Rung 9 adds the explicit face/wire flag
    recipe at FIX_TOLERANCE. Rung 10 is the sewing last resort at SEWING_TOLERANCE.
    """
    rungs = [LadderRung("shapefix@1e-06", BASELINE_PRECISION, BASELINE_MAX_TOLERANCE)]
    for tol in CONNECT_TOLERANCE:
        rungs.append(LadderRung("shapefix@%.1e" % tol, tol, tol))
    rungs.append(LadderRung("face-recipe@%.1e" % FIX_TOLERANCE, FIX_PRECISION,
                            FIX_TOLERANCE, face_recipe=True))
    rungs.append(LadderRung("sew@%.1e" % SEWING_TOLERANCE, FIX_PRECISION,
                            SEWING_TOLERANCE, face_recipe=True, sew=True))
    return rungs


def _shape_is_valid(wrapped) -> bool:
    """BRepCheck_Analyzer verdict for a raw OCCT shape (never raises)."""
    try:
        from OCP.BRepCheck import BRepCheck_Analyzer
        return bool(BRepCheck_Analyzer(wrapped).IsValid())
    except Exception:  # noqa: BLE001 - no analyzer -> cannot claim validity
        return False


def _set_tolerance(wrapped, tol: float):
    """ShapeFix_ShapeTolerance.SetTolerance over a whole shape (best-effort).

    Brepler sets an explicit tolerance on the shape before/after every sew so the
    downstream fixers agree with the sewer about what "coincident" means.
    """
    try:
        from OCP.ShapeFix import ShapeFix_ShapeTolerance
        ShapeFix_ShapeTolerance().SetTolerance(wrapped, tol)
    except Exception:  # noqa: BLE001 - tolerance forcing is best-effort
        pass
    return wrapped


def _apply_face_recipe(wrapped, rung: LadderRung) -> Tuple[object, bool]:
    """Run the ShapeFix_Face/ShapeFix_Wire flag recipe over every face.

    Returns ``(shape, fired)``. Faces are replaced in place through a
    BRepTools_ReShape context, so a face the fixer cannot improve is simply left
    alone. Never raises: any OCCT failure yields ``(wrapped, False)``.

    The flag choices (notably DISABLING FixIntersectingWires and
    FixPeriodicDegenerated, and turning AutoCorrectPrecisionMode off so the rung
    tolerance is honoured) are Brepler's; see the module docstring.
    """
    try:
        from OCP.BRepTools import BRepTools_ReShape
        from OCP.ShapeFix import ShapeFix_Face, ShapeFix_Wire
        from OCP.TopAbs import TopAbs_FACE, TopAbs_WIRE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
    except Exception:  # noqa: BLE001 - OCP build without these -> skip the rung
        return wrapped, False

    tol = rung.max_tolerance
    reshape = BRepTools_ReShape()
    fired = False

    explorer = TopExp_Explorer(wrapped, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        explorer.Next()
        try:
            # -- per-wire pass: drop sub-1e-3 edges, close 3d gaps at rung tol --
            wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
            while wire_exp.More():
                wire = TopoDS.Wire_s(wire_exp.Current())
                wire_exp.Next()
                try:
                    wf = ShapeFix_Wire(wire, face, tol)
                    wf.ModifyTopologyMode = True
                    wf.ModifyGeometryMode = True
                    wf.SetPrecision(tol)
                    wf.SetMaxTolerance(tol)
                    wf.FixSmall(False, REMOVE_EDGE_TOLERANCE)
                    wf.FixGaps3d()
                    fixed_wire = wf.Wire()
                    # Brepler's guard: a single-edge wire that gap-fixing still
                    # leaves open is unsalvageable — leave it rather than force it.
                    if fixed_wire.Closed() or fixed_wire.NbChildren() > 1:
                        reshape.Replace(wire, fixed_wire)
                        fired = True
                except Exception:  # noqa: BLE001 - one bad wire must not stop the pass
                    continue

            # -- per-face pass: the flag recipe --
            ff = ShapeFix_Face(face)
            ff.AutoCorrectPrecisionMode = False
            ff.SetPrecision(rung.precision)
            ff.SetMaxTolerance(tol)
            ff.FixOrientationMode = True
            ff.FixMissingSeamMode = True
            ff.FixWireMode = True
            ff.FixLoopWiresMode = False
            ff.FixIntersectingWiresMode = False
            ff.FixPeriodicDegeneratedMode = False
            ff.FixSmallAreaWireMode = False
            wire_tool = ff.FixWireTool()
            wire_tool.ModifyGeometryMode = True
            wire_tool.FixShiftedMode = True
            wire_tool.ClosedWireMode = True
            wire_tool.SetPrecision(rung.precision)
            wire_tool.SetMaxTolerance(tol)
            ff.Perform()
            fixed_face = ff.Face()
            if fixed_face is not None and not fixed_face.IsNull():
                reshape.Replace(face, fixed_face)
                fired = True
        except Exception:  # noqa: BLE001 - one bad face must not stop the pass
            continue

    if not fired:
        return wrapped, False
    try:
        return reshape.Apply(wrapped), True
    except Exception:  # noqa: BLE001
        return wrapped, False


def _sew_shape(wrapped, tol: float) -> Tuple[object, bool]:
    """Sew the shape's faces into a shell (and a solid when exactly one results).

    Returns ``(shape, fired)``. This is Brepler's ``get_solid`` reduced to the
    heal case: sew at ``tol``, force the tolerance onto the sewn shell, fix the
    shell if the analyzer rejects it, then make a solid. A sew that yields a
    compound (i.e. the faces did not close up even at this tolerance) is reported
    as not fired, so the caller keeps the pre-sew shape.
    """
    try:
        from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeSolid,
                                        BRepBuilderAPI_Sewing)
        from OCP.ShapeFix import ShapeFix_Shell
        from OCP.TopAbs import TopAbs_SHELL
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
    except Exception:  # noqa: BLE001
        return wrapped, False

    try:
        sewing = BRepBuilderAPI_Sewing()
        sewing.SetTolerance(tol)
        sewing.Add(wrapped)
        sewing.Perform()
        sewn = sewing.SewedShape()
        if sewn is None or sewn.IsNull():
            return wrapped, False
        _set_tolerance(sewn, tol)

        shells = []
        explorer = TopExp_Explorer(sewn, TopAbs_SHELL)
        while explorer.More():
            shells.append(TopoDS.Shell_s(explorer.Current()))
            explorer.Next()
        if len(shells) != 1:
            # Zero shells (nothing closed) or several (Brepler returns None on a
            # COMPOUND): the sew did not produce the single closed shell we want.
            return wrapped, False

        shell = shells[0]
        if not _shape_is_valid(shell):
            fixer = ShapeFix_Shell(shell)
            fixer.SetPrecision(tol)
            fixer.FixFaceMode = True
            fixer.FixOrientationMode = True
            fixer.Perform()
            shell = fixer.Shell()
            _set_tolerance(shell, tol)

        maker = BRepBuilderAPI_MakeSolid()
        maker.Add(shell)
        maker.Build()
        if not maker.IsDone():
            return wrapped, False
        return maker.Solid(), True
    except Exception:  # noqa: BLE001 - sewing is the last resort; never raise
        return wrapped, False


def _run_rung(wrapped, rung: LadderRung) -> Tuple[object, List[str]]:
    """Apply one ladder rung to a raw OCCT shape. Returns ``(shape, actions)``."""
    from OCP.ShapeFix import ShapeFix_Shape

    actions: List[str] = []
    current = wrapped

    if rung.face_recipe:
        current, fired = _apply_face_recipe(current, rung)
        if fired:
            actions.append("ShapeFix_Face+ShapeFix_Wire recipe @%g" % rung.max_tolerance)

    if rung.sew:
        current = _set_tolerance(current, rung.max_tolerance)
        current, fired = _sew_shape(current, rung.max_tolerance)
        if fired:
            actions.append("BRepBuilderAPI_Sewing @%g" % rung.max_tolerance)

    fixer = ShapeFix_Shape(current)
    try:
        fixer.SetPrecision(rung.precision)
        fixer.SetMaxTolerance(rung.max_tolerance)
    except Exception:  # noqa: BLE001 - precision setters are best-effort
        pass
    fixer.Perform()
    current = fixer.Shape()
    actions.append("ShapeFix_Shape @%g/%g" % (rung.precision, rung.max_tolerance))

    # ShapeUpgrade: unify coplanar faces / co-linear edges left after the fix,
    # removing the unshared-edge slivers that make a solid non-manifold. Guarded
    # independently so an OCP build lacking it still yields the ShapeFix result.
    try:
        from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
        unifier = ShapeUpgrade_UnifySameDomain(current, True, True, True)
        unifier.Build()
        current = unifier.Shape()
        actions.append("ShapeUpgrade_UnifySameDomain")
    except Exception:  # noqa: BLE001 - upgrade is optional
        pass

    return current, actions


def heal_shape_ladder(cq, shape, ladder: Optional[Sequence[LadderRung]] = None):
    """Heal one cq Shape by escalating through ``ladder`` until it is valid.

    Returns ``(healed_shape, actions)``. Rungs are tried in order, each starting
    from the ORIGINAL shape; the first rung whose result passes BRepCheck_Analyzer
    wins. If no rung produces a valid shape, the FIRST rung's result is returned
    — that is the historical behaviour, so a shape the ladder cannot save is left
    exactly as the old single-pass code left it.

    ``actions`` names the passes of the winning rung, prefixed by the rung name.
    Raises only on a hard OCCT failure, which the caller turns into a no-op.
    """
    rungs = list(ladder) if ladder is not None else default_ladder()
    if not rungs:
        rungs = default_ladder()

    original = shape.wrapped
    first_result = None
    first_actions: List[str] = []

    for index, rung in enumerate(rungs):
        try:
            fixed, actions = _run_rung(original, rung)
        except Exception:  # noqa: BLE001 - a rung that throws is simply skipped
            if index == 0:
                raise
            continue
        if index == 0:
            first_result, first_actions = fixed, actions
        if _shape_is_valid(fixed):
            named = ["rung:%s" % rung.name] + actions
            if index > 0:
                named.append("escalated %d rung(s) past the 1e-06 baseline" % index)
            return cq.Shape.cast(fixed), named

    if first_result is None:  # pragma: no cover - rung 0 raises before this
        raise RuntimeError("repair ladder produced no result")
    return cq.Shape.cast(first_result), (["rung:%s" % rungs[0].name] + first_actions
                                         + ["ladder exhausted; shape still invalid"])


def _heal_shape(cq, shape, ladder: Optional[Sequence[LadderRung]] = None):
    """Backwards-compatible alias for :func:`heal_shape_ladder`."""
    return heal_shape_ladder(cq, shape, ladder)


def repair_solid(backend, ladder: Optional[Sequence[LadderRung]] = None) -> RepairResult:
    """Heal the backend's current B-rep with the OCCT ShapeFix tolerance ladder.

    ``ladder`` overrides the rungs (see :func:`default_ladder`); pass a
    single-rung list to pin one precision. The default ladder starts at the
    historical 1e-6/1e-2 pass and only escalates when that leaves the shape
    invalid, so this argument is never needed to preserve old behaviour.

    Contract:
      * No OCCT installed, no solid present, or a backend that exposes no OCCT
        solids -> a clean no-op ``RepairResult`` (``healed=False``) with a note.
      * A valid solid the toolkit leaves unchanged -> ``healed=False`` (the
        topology descriptor and validity are identical before/after).
      * An imperfect solid the toolkit repairs -> ``healed=True`` with the fixes
        listed in ``actions`` and the topology/volume delta in ``diff``.
      * Any kernel exception is swallowed into a no-op result; state is only
        mutated after a successful, safe heal.
    """
    before = _safe_validity(backend)

    if not before.get("solid_present"):
        return RepairResult(
            healed=False, before_validity=before, after_validity=before,
            note="no solid present; nothing to repair")

    if not _cadquery_available():
        return RepairResult(
            healed=False, before_validity=before, after_validity=before,
            note="cadquery/OCCT not available; geometric repair skipped")

    solids = getattr(backend, "_solids", None)
    if not solids:
        return RepairResult(
            healed=False, before_validity=before, after_validity=before,
            note="backend exposes no OCCT solids; geometric repair skipped")

    try:
        import cadquery as cq

        # Snapshot the pre-repair topology so we can (a) decide whether anything
        # actually changed and (b) restore on a no-improvement / failure.
        original = list(solids)
        before_desc = _combined_describe(cq, backend)

        healed_solids: List = []
        actions: List[str] = []
        for wp in original:
            shape = _workplane_shape(cq, wp)
            if shape is None:
                healed_solids.append(wp)
                continue
            fixed, acts = heal_shape_ladder(cq, shape, ladder)
            healed_solids.append(wp.newObject([fixed]))
            for a in acts:
                if a not in actions:
                    actions.append(a)

        # Commit tentatively, re-measure, then decide.
        solids[:] = healed_solids
        after = _safe_validity(backend)
        after_desc = _combined_describe(cq, backend)
    except Exception as exc:  # noqa: BLE001 - never let a kernel hiccup escape
        try:
            solids[:] = original  # type: ignore[name-defined]
        except Exception:  # noqa: BLE001
            pass
        after = _safe_validity(backend)
        return RepairResult(
            healed=False, before_validity=before, after_validity=after,
            note=f"geometric repair skipped (kernel error: {exc})")

    topology_changed = before_desc != after_desc
    became_valid = (not before.get("is_valid")) and bool(after.get("is_valid"))
    regressed = bool(before.get("is_valid")) and not bool(after.get("is_valid"))

    if regressed:
        # The heal made a valid solid worse — roll back and report the no-op.
        solids[:] = original
        after = _safe_validity(backend)
        return RepairResult(
            healed=False, actions=[], before_validity=before,
            after_validity=after, diff={},
            note="repair would have regressed a valid solid; rolled back")

    if not topology_changed and not became_valid:
        # Nothing to fix: keep the (equivalent) healed shape, report a no-op.
        return RepairResult(
            healed=False, actions=[], before_validity=before,
            after_validity=after,
            diff={"before": before_desc, "after": after_desc},
            note="solid already valid; ShapeFix made no topological change")

    return RepairResult(
        healed=True, actions=actions, before_validity=before,
        after_validity=after,
        diff={"before": before_desc, "after": after_desc,
              "became_valid": became_valid},
        note="healed geometry via the OCCT ShapeFix tolerance ladder")


def _workplane_shape(cq, wp):
    """The single cq Shape (Compound of solids) held by a cq Workplane, or None."""
    try:
        shapes = wp.solids().vals()
    except Exception:  # noqa: BLE001
        shapes = []
    if not shapes:
        try:
            v = wp.val()
        except Exception:  # noqa: BLE001
            return None
        return v if v is not None and hasattr(v, "wrapped") else None
    if len(shapes) == 1:
        return shapes[0]
    try:
        return cq.Compound.makeCompound(shapes)
    except Exception:  # noqa: BLE001
        return shapes[0]


def _combined_describe(cq, backend) -> dict:
    """Topology descriptor of the backend's whole combined shape (never raises)."""
    combiner = getattr(backend, "_combined", None)
    shape = None
    if callable(combiner):
        try:
            shape = combiner()
        except Exception:  # noqa: BLE001
            shape = None
    if shape is None:
        return {}
    return _describe_shape(shape)


# ======================================================================
# Diagnostic-to-repair advisor
# ======================================================================
@dataclass
class RepairSuggestion:
    """A concrete, agent-facing repair prior for one diagnostic.

    ``code`` echoes the diagnostic it addresses; ``rationale`` says why; each
    entry of ``candidate_ops`` is a small dict naming a concrete CISP op / edit
    (``{"action": ..., "op": ..., ...}``) the agent can attempt. ``recovery``
    is this suggestion's rung on the guardrails ErrorRecovery ladder
    (``{"detect": ..., "handle": ..., "recover": ...}``).
    """

    code: str
    rationale: str
    candidate_ops: List[dict] = field(default_factory=list)
    recovery: dict = field(default_factory=dict)
    where: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "rationale": self.rationale,
            "candidate_ops": [dict(c) for c in self.candidate_ops],
            "recovery": dict(self.recovery),
            "where": self.where,
        }


def _recovery(detect: str, handle: str, recover: str) -> dict:
    """Compose an ErrorRecovery ladder rung, validating every strategy name.

    Falls back to the first strategy of a stage if a name is not in the ladder,
    so the mapping can never emit a strategy the guardrail layer doesn't define.
    """
    def _pick(stage: str, want: str) -> str:
        strategies = ErrorRecovery.strategies(stage)
        return want if want in strategies else strategies[0]

    return {
        "detect": _pick("detect", detect),
        "handle": _pick("handle", handle),
        "recover": _pick("recover", recover),
    }


class RepairAdvisor:
    """Deterministic map from diagnostic codes to concrete repair suggestions.

    ``suggest`` is pure and order-preserving: identical diagnostics always yield
    identical suggestions. Unknown codes yield a conservative generic suggestion
    (never an empty list for a real diagnostic), so the agent always gets *some*
    prior to try.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[Diagnostic, Optional[object]], RepairSuggestion]] = {
            # --- sketch constraint DOF ---
            "over-constrained": self._over_constrained,
            "under-constrained": self._under_constrained,
            # --- B-rep topology ---
            "invalid-brep": self._non_manifold,
            "non-manifold": self._non_manifold,
            "not-manifold": self._non_manifold,
            "sliver": self._non_manifold,
            "self-intersection": self._self_intersection,
            "self-intersect": self._self_intersection,
            # --- empty / degenerate solids ---
            "empty-solid": self._empty_solid,
            "empty": self._empty_solid,
            "degenerate": self._empty_solid,
            "empty-sketch": self._empty_solid,
            # --- boolean ---
            "boolean-fail": self._boolean,
            "boolean-nulls-body": self._boolean,
            # --- fillet / dimension guardrails ("loosen dim") ---
            "fillet-too-large": self._fillet_too_large,
            "fillet-nonpositive": self._fillet_nonpositive,
            "dim-out-of-range": self._dim_out_of_range,
            "extrude-nonpositive": self._extrude_nonpositive,
        }

    # -- public API --------------------------------------------------------
    def suggest(self, diagnostics, opdag=None) -> List[RepairSuggestion]:
        """Map a list of :class:`verify.Diagnostic` to repair suggestions.

        ``opdag`` (an ``OpDAG`` or anything exposing ``ops()``) is consulted, when
        present, to name the *concrete* op to drop/edit (e.g. the exact Constrain
        over-constraining a sketch). It is optional — every handler degrades to a
        generic-but-actionable suggestion without it.
        """
        out: List[RepairSuggestion] = []
        for diag in diagnostics or []:
            code = getattr(diag, "code", None)
            handler = self._handlers.get(code, self._generic)
            out.append(handler(diag, opdag))
        return out

    # -- op-history helpers ------------------------------------------------
    @staticmethod
    def _ops(opdag) -> List[Op]:
        if opdag is None:
            return []
        try:
            return list(opdag.ops())
        except Exception:  # noqa: BLE001
            return []

    def _constrains_for(self, opdag, where: Optional[str]) -> List[Constrain]:
        """Every Constrain op, most-recent first, that could over-constrain ``where``.

        A Constrain names entities (``a``/``b``), not a sketch id, so without a
        richer index we return all constraints newest-first; the newest is the
        best drop candidate for an over-constrained sketch.
        """
        cons = [op for op in self._ops(opdag) if isinstance(op, Constrain)]
        return list(reversed(cons))

    # -- handlers ----------------------------------------------------------
    def _over_constrained(self, diag: Diagnostic, opdag) -> RepairSuggestion:
        where = getattr(diag, "where", None)
        cons = self._constrains_for(opdag, where)
        candidates: List[dict] = []
        if cons:
            newest = cons[0]
            candidates.append({
                "action": "drop_op", "op": "constrain",
                "kind": newest.kind, "a": newest.a, "b": newest.b,
                "value": newest.value,
                "detail": f"drop the most-recent Constrain(kind={newest.kind}, "
                          f"a={newest.a}) — it pushes DOF negative",
            })
            # Also surface redundant dimensional constraints as drop candidates.
            for op in cons[1:]:
                if op.kind in ("distance", "radius"):
                    candidates.append({
                        "action": "drop_op", "op": "constrain",
                        "kind": op.kind, "a": op.a, "b": op.b, "value": op.value,
                        "detail": f"or drop the redundant {op.kind} constraint on "
                                  f"{op.a}",
                    })
                    break
        else:
            candidates.append({
                "action": "drop_op", "op": "constrain",
                "detail": "drop one constraint from the over-constrained sketch "
                          f"{where or ''}".strip() + " (start with the most "
                          "recently added dimensional one)",
            })
        return RepairSuggestion(
            code=diag.code,
            rationale="Sketch is over-constrained (DOF < 0): a redundant/conflicting "
                      "constraint must be removed before the solver can converge.",
            candidate_ops=candidates,
            recovery=_recovery("over-constrained", "retry-adjusted-params",
                               "reflect-diagnose"),
            where=where)

    def _under_constrained(self, diag: Diagnostic, opdag) -> RepairSuggestion:
        where = getattr(diag, "where", None)
        return RepairSuggestion(
            code=diag.code,
            rationale="Sketch is under-constrained (DOF > 0): add constraints to "
                      "pin the remaining freedoms so the profile is fully defined.",
            candidate_ops=[
                {"action": "add_op", "op": "constrain", "kind": "distance",
                 "detail": "add a dimensional (distance/radius) constraint to "
                           "remove a free DOF"},
                {"action": "add_op", "op": "constrain", "kind": "coincident",
                 "detail": "or add a geometric (coincident/horizontal/vertical) "
                           "constraint"},
            ],
            recovery=_recovery("under-constrained", "retry-adjusted-params",
                               "reflect-diagnose"),
            where=where)

    def _non_manifold(self, diag: Diagnostic, opdag) -> RepairSuggestion:
        return RepairSuggestion(
            code=diag.code,
            rationale="Solid is present but not manifold/watertight/valid (unshared "
                      "edges or sliver faces): heal the B-rep, then optionally add a "
                      "small fillet for edge continuity.",
            candidate_ops=[
                {"action": "repair_solid",
                 "detail": "run repair_solid() — OCCT ShapeFix/ShapeUpgrade sews "
                           "gaps, fixes orientation and unifies sliver faces"},
                {"action": "add_op", "op": "fillet", "radius": 0.5,
                 "detail": "add a small fillet on the affected edges to restore "
                           "continuity if ShapeFix leaves a sharp sliver"},
            ],
            recovery=_recovery("non-manifold", "fallback-simpler-strategy",
                               "reflect-diagnose"),
            where=getattr(diag, "where", None))

    def _self_intersection(self, diag: Diagnostic, opdag) -> RepairSuggestion:
        return RepairSuggestion(
            code=diag.code,
            rationale="Solid self-intersects: heal the geometry, or offset/thicken "
                      "the offending profile so the faces no longer overlap.",
            candidate_ops=[
                {"action": "repair_solid",
                 "detail": "run repair_solid() — ShapeFix resolves small "
                           "self-intersections"},
                {"action": "edit_op", "op": "extrude",
                 "detail": "reduce the offset/extrude magnitude, or offset the "
                           "profile outward, so faces stop overlapping"},
            ],
            recovery=_recovery("non-manifold", "retry-adjusted-params",
                               "rollback-feature-tree"),
            where=getattr(diag, "where", None))

    def _empty_solid(self, diag: Diagnostic, opdag) -> RepairSuggestion:
        return RepairSuggestion(
            code=diag.code,
            rationale="Features ran but no solid resulted (degenerate build): the "
                      "sketch profile is most likely not closed, or the extrude "
                      "distance is zero.",
            candidate_ops=[
                {"action": "check", "op": "new_sketch",
                 "detail": "verify the sketch profile is CLOSED (a rectangle/circle, "
                           "or lines whose endpoints are coincident) before extruding"},
                {"action": "edit_op", "op": "extrude",
                 "detail": "set a non-zero extrude distance"},
            ],
            recovery=_recovery("empty", "fallback-simpler-strategy",
                               "rollback-feature-tree"),
            where=getattr(diag, "where", None))

    def _boolean(self, diag: Diagnostic, opdag) -> RepairSuggestion:
        return RepairSuggestion(
            code=diag.code,
            rationale="Boolean produced no / a null solid: the operands may not "
                      "overlap, or a cut/intersect consumes the whole body.",
            candidate_ops=[
                {"action": "edit_op", "op": "boolean",
                 "detail": "reposition/resize the tool so it overlaps the target, "
                           "or switch union<->cut, so a solid remains"},
                {"action": "repair_solid",
                 "detail": "heal the operands first — a non-manifold input often "
                           "makes the boolean degenerate"},
            ],
            recovery=_recovery("boolean-fail", "fallback-simpler-strategy",
                               "rollback-feature-tree"),
            where=getattr(diag, "where", None))

    def _fillet_too_large(self, diag: Diagnostic, opdag) -> RepairSuggestion:
        return RepairSuggestion(
            code=diag.code,
            rationale="Fillet radius meets/exceeds the shortest adjacent edge, so "
                      "the fillet cannot be built: loosen the radius.",
            candidate_ops=[
                {"action": "edit_op", "op": "fillet",
                 "detail": "reduce the fillet radius below the shortest adjacent "
                           "edge length"},
            ],
            recovery=_recovery("non-manifold", "retry-adjusted-params",
                               "reflect-diagnose"),
            where=getattr(diag, "where", None))

    def _fillet_nonpositive(self, diag: Diagnostic, opdag) -> RepairSuggestion:
        return RepairSuggestion(
            code=diag.code,
            rationale="Fillet radius must be > 0.",
            candidate_ops=[
                {"action": "edit_op", "op": "fillet",
                 "detail": "set the fillet radius to a positive value"},
            ],
            recovery=_recovery("non-manifold", "retry-adjusted-params",
                               "reflect-diagnose"),
            where=getattr(diag, "where", None))

    def _dim_out_of_range(self, diag: Diagnostic, opdag) -> RepairSuggestion:
        return RepairSuggestion(
            code=diag.code,
            rationale="A positive dimension falls outside the manufacturable band: "
                      "loosen/adjust it into range (or widen GuardrailLimits).",
            candidate_ops=[
                {"action": "edit_op",
                 "detail": "adjust the offending dimension into "
                           "[min_dim, max_dim], or relax GuardrailLimits"},
            ],
            recovery=_recovery("non-manifold", "retry-adjusted-params",
                               "reflect-diagnose"),
            where=getattr(diag, "where", None))

    def _extrude_nonpositive(self, diag: Diagnostic, opdag) -> RepairSuggestion:
        return RepairSuggestion(
            code=diag.code,
            rationale="Extrude depth must be positive.",
            candidate_ops=[
                {"action": "edit_op", "op": "extrude",
                 "detail": "set the extrude distance to a positive value"},
            ],
            recovery=_recovery("empty", "retry-adjusted-params",
                               "rollback-feature-tree"),
            where=getattr(diag, "where", None))

    def _generic(self, diag: Diagnostic, opdag) -> RepairSuggestion:
        """Fallback for unmapped codes — still actionable, never empty."""
        return RepairSuggestion(
            code=getattr(diag, "code", "unknown"),
            rationale=f"No specialised repair prior for '{getattr(diag, 'code', '?')}'"
                      f": {getattr(diag, 'message', '')}".strip(),
            candidate_ops=[
                {"action": "reflect",
                 "detail": "inspect the failing op and its inputs, then retry with "
                           "adjusted parameters — never re-emit the same invalid op"},
            ],
            recovery=_recovery("regen-fail", "log", "reflect-diagnose"),
            where=getattr(diag, "where", None))


# ======================================================================
# CLI
# ======================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` proves the ladder's real properties:
    the default-safety invariant, monotone escalation, and (when OCCT is
    installed) that the sew rung actually rebuilds a solid from loose faces."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.eval.reliability.brep_repair",
        description="B-rep repair: OCCT ShapeFix tolerance ladder + repair advisor.",
    )
    parser.add_argument(
        "--selfcheck", action="store_true",
        help="run deterministic checks over the ladder and advisor; exit 0 on success.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    failures: List[str] = []
    checks = 0

    def check(label: str, condition: bool) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(label)

    rungs = default_ladder()

    # 1. DEFAULT-SAFETY: rung 0 is exactly the historical single pass.
    check("rung 0 is the 1e-6/1e-2 baseline",
          rungs[0].precision == BASELINE_PRECISION
          and rungs[0].max_tolerance == BASELINE_MAX_TOLERANCE
          and not rungs[0].face_recipe and not rungs[0].sew)

    # 2. Escalation is monotone in precision across the plain ShapeFix band: each
    #    rung asks the fixer to treat a strictly larger deviation as "needs
    #    fixing" than the rung before it.
    plain = [r for r in rungs if not r.face_recipe and not r.sew]
    precisions = [r.precision for r in plain]
    check("ShapeFix band escalates monotonically in precision",
          all(a < b for a, b in zip(precisions, precisions[1:])))

    # 3. Strategy rungs (face recipe, sewing) come only AFTER the whole plain
    #    band: they are last resorts, not shortcuts.
    strategy_start = min(i for i, r in enumerate(rungs) if r.face_recipe or r.sew)
    check("strategy rungs come after the plain band",
          strategy_start == len(plain))

    # 4. The ladder spans the Brepler CONNECT band and ends at the sewing rung.
    check("ladder covers the CONNECT band",
          all(t in precisions for t in CONNECT_TOLERANCE))
    check("last rung sews at SEWING_TOLERANCE",
          rungs[-1].sew and rungs[-1].max_tolerance == SEWING_TOLERANCE)
    check("only the last rung sews",
          sum(1 for r in rungs if r.sew) == 1)

    # 4. Determinism: the ladder is a pure function of nothing.
    check("default_ladder is deterministic", default_ladder() == rungs)

    # 5. The advisor stays pure and total.
    advisor = RepairAdvisor()
    diags = [Diagnostic(Severity.ERROR, "invalid-brep", "m"),
             Diagnostic(Severity.ERROR, "totally-unknown-code", "m")]
    first = [s.to_dict() for s in advisor.suggest(diags)]
    check("advisor is deterministic",
          first == [s.to_dict() for s in advisor.suggest(diags)])
    check("advisor never returns an empty suggestion",
          all(s["candidate_ops"] for s in first))

    if not _cadquery_available():
        if failures:
            print("SELFCHECK FAILED: %s" % ", ".join(failures), file=sys.stderr)
            return 1
        print("PASS: brep_repair selfcheck (%d checks; OCCT absent, kernel rungs "
              "skipped)" % checks)
        return 0

    import cadquery as cq

    # 6. REAL KERNEL PROPERTY: a valid solid heals at rung 0 and never escalates.
    box = cq.Workplane("XY").box(20, 10, 5).val()
    healed, actions = heal_shape_ladder(cq, box)
    check("valid solid heals at rung 0",
          actions and actions[0] == "rung:%s" % rungs[0].name)
    check("valid solid does not escalate",
          not any("escalated" in a for a in actions))
    check("valid solid keeps its volume",
          abs(healed.Volume() - 1000.0) < 1e-6)

    # 7. REAL KERNEL PROPERTY: the sew rung rebuilds a solid from loose faces.
    #    A bare compound of a box's 6 faces has no solid; sewing at the Brepler
    #    tolerance must close it into one whose volume is the box's.
    faces = box.Faces()
    compound = cq.Compound.makeCompound(faces).wrapped
    sewn, fired = _sew_shape(compound, SEWING_TOLERANCE)
    check("sew rung fires on loose faces", fired)
    if fired:
        sewn_shape = cq.Shape.cast(sewn)
        check("sew rung yields exactly one solid", len(sewn_shape.Solids()) == 1)
        check("sewn solid has the original volume",
              abs(sewn_shape.Volume() - 1000.0) < 1e-6)

    # 8. REAL KERNEL PROPERTY: the face/wire recipe is volume-preserving on a
    #    curved (periodic-surface) solid — the recipe must heal, never mangle.
    cyl = cq.Workplane("XY").circle(4).extrude(10).val()
    recipe_rung = LadderRung("t", FIX_PRECISION, FIX_TOLERANCE, face_recipe=True)
    fixed, fired = _apply_face_recipe(cyl.wrapped, recipe_rung)
    check("face recipe fires on a cylinder", fired)
    if fired:
        fixed_shape = cq.Shape.cast(fixed)
        check("face recipe preserves cylinder volume",
              abs(fixed_shape.Volume() - cyl.Volume()) < 1e-3)
        check("face recipe leaves the cylinder valid", _shape_is_valid(fixed))

    if failures:
        print("SELFCHECK FAILED: %s" % ", ".join(failures), file=sys.stderr)
        return 1
    print("PASS: brep_repair selfcheck (%d checks; ladder default-safety, monotone "
          "escalation, real OCCT sew + face-recipe rungs)" % checks)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
