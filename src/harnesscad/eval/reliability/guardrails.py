"""Guardrails — the ``before_tool_callback`` hard gate + error-recovery ladder.

Per HARNESS_BLUEPRINT.md sec.10: a hard validation gate runs on every kernel op
BEFORE it is applied, blocking geometrically invalid ops (block-and-correct) so a
bad op never corrupts the model. This is a PRE-apply layer that complements
HarnessSession's post-apply block-and-correct in loop.py — the gate stops obviously
invalid values up front (zero-depth extrude, oversize fillet, out-of-range dims)
without ever touching kernel state, and the session/verifier catch the rest.

``GuardrailGate.check(op, backend=None)`` returns a list of ERROR ``Diagnostic``s
(empty == allowed). It NEVER mutates anything. Measurement-dependent rules (fillet
radius vs. adjacent edge length; a boolean that would null the body) consult the
backend when one is available and DEGRADE GRACEFULLY — skipping cleanly — when no
backend or no measurement is present.

``ErrorRecovery`` enumerates the blueprint's detect -> handle -> recover ladder as
named strategies the loop can consult when the gate (or a downstream verifier) fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from harnesscad.core.cisp.ops import (
    Op, AddCircle, AddRectangle, Boolean, Extrude, Fillet,
)
from harnesscad.eval.verifiers.verify import Diagnostic, Severity


@dataclass
class GuardrailLimits:
    """Configurable manufacturability envelope.

    ``min_dim``/``max_dim`` bound any positive linear dimension (extrude depth,
    fillet radius, circle radius, rectangle side). Zero/negative values are caught
    by their own dedicated rules; this range catches positive-but-unmanufacturable
    values (a 0.0001 mm wall, a 50 m fillet).
    """

    min_dim: float = 0.01
    max_dim: float = 10_000.0


def _err(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.ERROR, code, msg, where)


class GuardrailGate:
    """Hard pre-apply validation gate. Pure: reads the op (and optional backend),
    returns diagnostics, mutates nothing.
    """

    def __init__(self, limits: Optional[GuardrailLimits] = None) -> None:
        self.limits = limits if limits is not None else GuardrailLimits()

    # --- public API -------------------------------------------------------
    def check(self, op: Op, backend=None) -> List[Diagnostic]:
        """Return ERROR diagnostics for every guardrail ``op`` violates.

        Empty list == the op is allowed to proceed. Each diagnostic carries an
        actionable message telling the agent how to correct the op.
        """
        if isinstance(op, Extrude):
            return self._check_extrude(op)
        if isinstance(op, Fillet):
            return self._check_fillet(op, backend)
        if isinstance(op, AddCircle):
            return self._check_circle(op)
        if isinstance(op, AddRectangle):
            return self._check_rectangle(op)
        if isinstance(op, Boolean):
            return self._check_boolean(op, backend)
        return []

    # --- dimension helper -------------------------------------------------
    def _dim_range(self, value: float, where: str, label: str) -> List[Diagnostic]:
        """Flag a positive dimension that falls outside the manufacturable band."""
        lo, hi = self.limits.min_dim, self.limits.max_dim
        if value < lo or value > hi:
            return [_err(
                "dim-out-of-range",
                f"{label} {value} is outside the manufacturable range "
                f"[{lo}, {hi}]; choose a value within limits or adjust "
                f"GuardrailLimits",
                where)]
        return []

    # --- per-op rules -----------------------------------------------------
    def _check_extrude(self, op: Extrude) -> List[Diagnostic]:
        if op.distance <= 0:
            return [_err(
                "extrude-nonpositive",
                f"extrude depth must be positive (got {op.distance}); set "
                f"distance > 0",
                op.sketch)]
        return self._dim_range(op.distance, op.sketch, "extrude depth")

    def _check_fillet(self, op: Fillet, backend) -> List[Diagnostic]:
        if op.radius <= 0:
            return [_err(
                "fillet-nonpositive",
                f"fillet radius must be > 0 (got {op.radius}); set radius > 0")]
        diags = self._dim_range(op.radius, None, "fillet radius")
        # Measurement-dependent: radius must be < the shortest adjacent edge, or the
        # fillet cannot be built. Consult the backend only if it can measure edges;
        # otherwise degrade gracefully and skip this check.
        for eid, length in self._edge_lengths(backend, op.edges).items():
            if op.radius >= length:
                diags.append(_err(
                    "fillet-too-large",
                    f"fillet radius {op.radius} >= adjacent edge '{eid}' length "
                    f"{length}; reduce radius below {length}",
                    eid))
        return diags

    def _check_circle(self, op: AddCircle) -> List[Diagnostic]:
        if op.r <= 0:
            return [_err(
                "circle-nonpositive",
                f"circle radius must be > 0 (got {op.r}); set r > 0")]
        return self._dim_range(op.r, None, "circle radius")

    def _check_rectangle(self, op: AddRectangle) -> List[Diagnostic]:
        diags: List[Diagnostic] = []
        if op.w <= 0 or op.h <= 0:
            diags.append(_err(
                "rect-nonpositive",
                f"rectangle width and height must be > 0 (got w={op.w}, "
                f"h={op.h}); set both > 0"))
            return diags
        diags += self._dim_range(op.w, None, "rectangle width")
        diags += self._dim_range(op.h, None, "rectangle height")
        return diags

    def _check_boolean(self, op: Boolean, backend) -> List[Diagnostic]:
        if op.kind not in ("union", "cut", "intersect"):
            return [_err(
                "boolean-bad-kind",
                f"unknown boolean kind '{op.kind}'; use union | cut | intersect")]
        # Best-effort / consultable: a cut or intersect can leave an empty body.
        # If the backend can preview the result volume, block a null result;
        # otherwise skip (degrade gracefully — the post-apply verifier still guards).
        if op.kind in ("cut", "intersect"):
            vol = self._boolean_result_volume(backend, op)
            if vol is not None and vol <= 0:
                return [_err(
                    "boolean-nulls-body",
                    f"boolean '{op.kind}' of '{op.tool}' into '{op.target}' would "
                    f"null the body (result volume {vol}); adjust geometry so a "
                    f"solid remains",
                    op.target)]
        return []

    # --- backend measurement adapters (all degrade gracefully) ------------
    @staticmethod
    def _edge_lengths(backend, edges) -> dict:
        """Return {edge_id: length} for the op's edges the backend can measure.

        Returns {} when no backend, no measurement support, or none of the op's
        edges are measurable — the caller then skips the length-dependent check.
        """
        if backend is None:
            return {}
        try:
            table = backend.query("edge_length")
        except Exception:
            return {}
        if not isinstance(table, dict) or not table:
            return {}
        return {e: table[e] for e in edges if e in table}

    @staticmethod
    def _boolean_result_volume(backend, op: Boolean) -> Optional[float]:
        """Best-effort predicted result volume for a boolean, or None if unknown."""
        if backend is None:
            return None
        try:
            table = backend.query("boolean_preview")
        except Exception:
            return None
        if not isinstance(table, dict):
            return None
        # Keyed by target id when the backend can preview; absent => unknown.
        val = table.get(op.target)
        return float(val) if isinstance(val, (int, float)) else None


# --- error-recovery ladder ------------------------------------------------
@dataclass(frozen=True)
class RecoveryStage:
    """One rung of the detect -> handle -> recover ladder."""

    name: str
    strategies: tuple


class ErrorRecovery:
    """The blueprint sec.10 error-recovery ladder as named, consultable strategies.

    The loop consults this after a guardrail block or a verifier failure to pick the
    next move: *detect* what went wrong, *handle* it (never re-emit the same invalid
    op unchanged), then *recover* (roll back via the event log, reflect, replan, or
    escalate). This is advisory metadata — it enumerates the ladder; it does not act.
    """

    DETECT = RecoveryStage("detect", (
        "regen-fail",          # kernel regeneration failed
        "non-manifold",        # B-rep not watertight / manifold
        "boolean-fail",        # boolean produced no / degenerate solid
        "over-constrained",    # sketch DOF < 0
        "under-constrained",   # sketch DOF > 0
        "timeout",             # op exceeded time budget
        "empty",               # empty result where a solid was expected
    ))
    HANDLE = RecoveryStage("handle", (
        "log",                       # record the failure + diagnostics
        "retry-adjusted-params",     # retry with adjusted params — never the same invalid op unchanged
        "fallback-simpler-strategy", # switch to a simpler modeling strategy
        "graceful-degradation",      # deliver a valid partial + report the failed feature
    ))
    RECOVER = RecoveryStage("recover", (
        "rollback-feature-tree",   # roll back via the event log to last-good
        "reflect-diagnose",        # synthesize an insight from the failure
        "replan",                  # re-decompose the remaining plan
        "escalate",                # hand off to a human / higher tier
    ))

    LADDER = (DETECT, HANDLE, RECOVER)

    @classmethod
    def stages(cls) -> List[str]:
        """Ordered stage names: ['detect', 'handle', 'recover']."""
        return [s.name for s in cls.LADDER]

    @classmethod
    def strategies(cls, stage: str) -> List[str]:
        """Named strategies for a stage. Raises KeyError for an unknown stage."""
        for s in cls.LADDER:
            if s.name == stage:
                return list(s.strategies)
        raise KeyError(stage)

    @classmethod
    def next_stage(cls, stage: str) -> Optional[str]:
        """The stage after ``stage`` in the ladder, or None past the end."""
        names = cls.stages()
        i = names.index(stage)
        return names[i + 1] if i + 1 < len(names) else None
