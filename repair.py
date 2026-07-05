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
   no-op result with a note and NEVER raises.

2. DIAGNOSTIC-TO-REPAIR ADVISOR — :class:`RepairAdvisor` deterministically maps
   the diagnostic *codes* the verifiers emit (over-constrained, invalid-brep,
   empty-solid, self-intersection, ...) to concrete candidate CISP ops/edits the
   agent can try next (drop a specific Constrain, run a heal, add a small fillet,
   check the profile is closed, ...). This is the deterministic prior handed to
   the agent, and it composes :class:`guardrails.ErrorRecovery` so each suggestion
   also carries its rung on the detect -> handle -> recover ladder.

Absolute imports; OCCT touched only inside cadquery-guarded paths; degrades
gracefully everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from cisp.ops import Constrain, Op
from guardrails import ErrorRecovery
from verify import Diagnostic


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


def _heal_shape(cq, shape):
    """Run OCCT ShapeFix (+ ShapeUpgrade) over one cq Shape.

    Returns ``(healed_shape, actions)``. ``healed_shape`` is a new cq Shape;
    ``actions`` names the healing passes that ran. Raises only on a hard OCCT
    failure, which the caller turns into a graceful no-op.
    """
    actions: List[str] = []
    from OCP.ShapeFix import ShapeFix_Shape

    fixer = ShapeFix_Shape(shape.wrapped)
    try:
        fixer.SetPrecision(1e-6)
        fixer.SetMaxTolerance(1e-2)
    except Exception:  # noqa: BLE001 - precision setters are best-effort
        pass
    fixer.Perform()
    fixed_wrapped = fixer.Shape()
    actions.append("ShapeFix_Shape")

    # ShapeUpgrade: unify coplanar faces / co-linear edges left after the fix,
    # removing the unshared-edge slivers that make a solid non-manifold. Guarded
    # independently so an OCP build lacking it still yields the ShapeFix result.
    try:
        from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
        unifier = ShapeUpgrade_UnifySameDomain(fixed_wrapped, True, True, True)
        unifier.Build()
        fixed_wrapped = unifier.Shape()
        actions.append("ShapeUpgrade_UnifySameDomain")
    except Exception:  # noqa: BLE001 - upgrade is optional
        pass

    return cq.Shape.cast(fixed_wrapped), actions


def repair_solid(backend) -> RepairResult:
    """Heal the backend's current B-rep with OCCT ShapeFix / ShapeUpgrade.

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
            fixed, acts = _heal_shape(cq, shape)
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
        note="healed geometry via OCCT ShapeFix/ShapeUpgrade")


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
