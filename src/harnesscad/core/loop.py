"""HarnessSession — the applyOps -> regen -> verify -> checkpoint spine.

This is the Aider-style loop from the blueprint, kernel-agnostic:
  - block-and-correct: an op the backend rejects (bad reference/value) never
    mutates state; the batch stops and returns diagnostics for the agent to fix.
  - transactional verify: after each accepted op we regen + run the plural
    verifier; an ERROR-severity diagnostic rolls that op back (last-good state
    preserved) and returns.
  - checkpoint on success: every accepted+verified op is checkpointed, giving
    deterministic replay and rollback to any point.
  - PER-STEP CREDIT: every op the loop decides about emits a `step_reward` event
    (1.0 applied+verified, 0.0 for the op that broke the trajectory) and lands in
    `session.step_rewards`. The harness was running outcome-only supervision on a
    3-8-op trajectory while carrying a finished process-reward implementation
    (`agents/agent/tool_reward.py`). A per-step reward shows a loop being poisoned
    at op 3 instead of at brief 12. `agents/agent/trace_reward.py` folds this
    vector into the full R = a*R_ORM + b*mean(R_step) + c*R_format.
  - the verifier FLEET: at verify_level="full" the session additionally runs
    every verifier discovered by harnesscad.eval.verifiers.registry (DFM, plan
    preflight, standards, interference, kernel preflight, plausibility, ...).
    Fleet diagnostics are surfaced alongside the core ones; by default they are
    advisory (they never roll an op back), so the transactional semantics above
    are unchanged.
"""

from __future__ import annotations

import hashlib
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional,
                    Sequence, Union)

from harnesscad.io.backends.base import GeometryBackend
from harnesscad.core.cisp.ops import Op, canonical_json
from harnesscad.core.cisp.protocol import ApplyOpsResult
# Read-only reuse of the built provenance structures (OpDelta / Provenance / the
# measurement-diff and the two set differences). This module NEVER redefines them;
# native recording just accumulates the same Provenance the orphan gate consumes,
# so the gate no longer has to rebuild a session per op-prefix via build_provenance.
from harnesscad.core.cisp import provenance as _provenance
from harnesscad.core.state.opdag import OpDAG

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from harnesscad.core.cisp import op_gate as _op_gate
from harnesscad.core.trace import NullTracer, Tracer
from harnesscad.eval.verifiers.verify import (Diagnostic, Severity, VerifyReport,
                                              Verifier, default_verifiers)


#: Verification levels for HarnessSession.
#:   "core" -- only the three checks the loop has always run (sketch DOF, solid
#:             presence, B-rep validity). The default: identical to the old
#:             behaviour, byte for byte.
#:   "full" -- also run the discovered verifier FLEET
#:             (harnesscad.eval.verifiers.registry): plan preflight, DFM,
#:             standards, compliance, interference, kernel preflight, tolerance
#:             stacks, plausibility, standability, ... Their diagnostics are
#:             SURFACED but, unless `fleet_blocking=True`, do not roll an op back:
#:             the core verifiers gate the transaction, the fleet advises.
VERIFY_LEVELS = ("core", "full")

#: The loop lifecycle events a `hook_bus.HookBus` may be registered against.
#: Deliberately NOT the freecad-ai tool-call vocabulary hook_bus ships as its
#: default -- these are the points THIS loop actually has:
#:   "iteration_start" -- before an op is handed to the backend;
#:   "plan_accepted"   -- the op applied and regenerated cleanly;
#:   "verify_verdict"  -- the verifier fleet has ruled on the op;
#:   "iteration_end"   -- the op is fully resolved (checkpointed or rolled back).
#: A bus is an OBSERVER here: hook_bus's `block` veto is not honoured, because
#: an op that has already mutated the backend cannot be un-fired by a listener.
#: Refusal is the op gate's job (see `_gate_preflight`), which runs before
#: anything mutates and can actually refuse.
LOOP_EVENTS = ("iteration_start", "plan_accepted", "verify_verdict",
               "iteration_end")


class HarnessSession:
    def __init__(self, backend: GeometryBackend,
                 verifiers: Optional[List[Verifier]] = None,
                 tracer: Optional[Tracer] = None,
                 verify_level: str = "core",
                 fleet_tiers: Optional[Sequence[str]] = None,
                 fleet_only: Optional[Sequence[str]] = None,
                 fleet_skip: Sequence[str] = (),
                 fleet_blocking: bool = False,
                 record_provenance: bool = False,
                 provenance_measure: Optional[
                     Callable[["HarnessSession"], Optional[Mapping]]] = None,
                 op_catalog: Optional[Union["_op_gate.AllowedOperationsCatalog",
                                            Mapping]] = None,
                 op_gate_context: Optional[Mapping] = None,
                 op_protected_regions: Sequence = (),
                 hook_bus: Optional[Any] = None) -> None:
        self.backend = backend
        self.opdag = OpDAG()
        self.verifiers = verifiers if verifiers is not None else default_verifiers()
        # NullTracer is the default: tracing is opt-in and zero-cost, so a
        # HarnessSession(backend) behaves exactly as before.
        self.tracer = tracer if tracer is not None else NullTracer()
        if verify_level not in VERIFY_LEVELS:
            raise ValueError(
                f"verify_level must be one of {VERIFY_LEVELS!r}, got {verify_level!r}")
        self.verify_level = verify_level
        self.fleet_tiers = tuple(fleet_tiers) if fleet_tiers is not None else None
        self.fleet_only = tuple(fleet_only) if fleet_only is not None else None
        self.fleet_skip = tuple(fleet_skip)
        self.fleet_blocking = fleet_blocking
        #: Per-op reward vector of the LAST batch (see _step_reward). The loop
        #: is a multi-step trajectory and was graded only on its final solid;
        #: this is the process signal that says WHICH op broke it.
        self.step_rewards: List[dict] = []
        self._run_seq = 0
        #: Native op->geometry-delta provenance (traceSDD orphan-op / orphan-feature
        #: check). Opt-in and OFF by default: when record_provenance is False the
        #: whole loop is byte-for-byte the old behaviour and pays nothing. When True
        #: the session accumulates a `provenance.Provenance` incrementally as ops
        #: land -- it measures the state once per accepted+verified op and diffs
        #: successive measurements, so the orphan gate never has to replay op
        #: prefixes through build_provenance. See `provenance()` / `orphan_ops()`.
        self.record_provenance = bool(record_provenance)
        self._provenance_measure = provenance_measure
        self._prov: Optional[_provenance.Provenance] = None
        self._prov_saw_measurement = False
        #: Operation-admissibility catalog (core/cisp/op_gate.py). None -- the
        #: default -- means NO preflight: apply_ops behaves byte for byte as
        #: before. When a catalog IS provided, every batch is judged against it
        #: BEFORE a single op reaches the backend, and an inadmissible batch is
        #: refused whole (nothing mutates) with the gate's typed diagnostics.
        self.op_catalog = op_catalog
        self.op_gate_context = op_gate_context
        self.op_protected_regions = tuple(op_protected_regions)
        #: The GateReport of the last preflighted batch, or None when no catalog
        #: is configured / no batch has run. Carries the patch_proposal
        #: bookkeeping (protected_targets_checked / _avoided, patch_status).
        self.last_gate_report: Optional["_op_gate.GateReport"] = None
        #: An optional `agents/agent/hook_bus.HookBus` observing the loop's
        #: lifecycle (LOOP_EVENTS). None -- the default -- fires nothing, and a
        #: bus with no handlers registered for an event is itself a no-op, so
        #: both the unwired and the empty-bus cases cost a dict build at most.
        self.hook_bus = hook_bus
        self.backend.reset()
        self.opdag.checkpoint("start")
        if self.record_provenance:
            self._prov = _provenance.Provenance()
            baseline = self._measure_state() or {}
            self._prov.measurements.append(baseline)
            self._prov_saw_measurement = bool(baseline)

    def _make_run_id(self, ops: List[Op]) -> str:
        """Deterministic, wall-clock-free run id.

        Derived from the batch contents and a per-session sequence number, so
        replaying the same op stream yields the same ids (no time dependency).
        """
        self._run_seq += 1
        blob = "|".join(canonical_json(op) for op in ops)
        h = hashlib.sha256(f"{self._run_seq}|{blob}".encode()).hexdigest()
        return f"run-{self._run_seq}-{h[:12]}"

    # --- per-step credit assignment ---------------------------------------
    def _step_reward(self, run_id: str, index: int, op: Op,
                     reward: float, reason: str) -> None:
        """Emit the PER-OP reward and record it on the session.

        1.0 for an op that applied and verified; 0.0 for the op that broke the
        trajectory. Ops after the break are never reached and never scored --
        trajectory slicing, not collective punishment.
        """
        record = {"index": index, "op": op.to_dict()["op"],
                  "reward": float(reward), "reason": reason}
        self.step_rewards.append(record)
        self.tracer.event("step_reward", run_id, dict(record))

    def mean_step_reward(self) -> float:
        """Mean per-op reward of the last batch; 0.0 when nothing was scored."""
        if not self.step_rewards:
            return 0.0
        return sum(r["reward"] for r in self.step_rewards) / len(self.step_rewards)

    # --- native provenance (opt-in) ---------------------------------------
    def _default_provenance_measure(self) -> Optional[Mapping]:
        """A backend-agnostic Measurement of the CURRENT state via read-only queries.

        Reuses the queries every GeometryBackend already answers: the `summary`
        counts/flags (sketch/entity/feature counts, solid presence) plus the total
        sketch DOF, so a Constrain that only tightens a sketch still reads as live.
        Returns a plain mapping of comparable quantities -- exactly the contract
        provenance.build_provenance expects -- or None when nothing is measurable
        (which degrades the provenance to a clean `skipped` PASS, no engine case).
        """
        measurement: dict = {}
        try:
            summary = self.backend.query("summary")
        except Exception:  # noqa: BLE001 - a query crash is a non-measurement
            summary = None
        if isinstance(summary, Mapping):
            for key, value in summary.items():
                if isinstance(value, (int, float, bool, str)):
                    measurement[key] = value
        try:
            dof = self.backend.query("sketch_dof")
        except Exception:  # noqa: BLE001
            dof = None
        if isinstance(dof, Mapping) and dof:
            total = 0.0
            for value in dof.values():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    total += float(value)
            measurement["sketch_dof_total"] = total
        return measurement or None

    def _measure_state(self) -> Optional[Mapping]:
        """Measure the current backend state, via the caller's hook or the default.

        A custom `provenance_measure` (backend that can report volume/n_faces/...)
        takes precedence; otherwise the query-based default is used. Never raises:
        a measurement crash is recorded as "no measurement", never propagated into
        the loop (provenance is advisory and must not break the transaction).
        """
        try:
            if self._provenance_measure is not None:
                return self._provenance_measure(self)
            return self._default_provenance_measure()
        except Exception:  # noqa: BLE001
            return None

    def _record_provenance(self, op: Op) -> None:
        """Attribute the just-applied op's measured geometry delta.

        Called once per accepted+verified op (the op is already appended to the
        opdag and the backend already holds its state). Measures now, diffs against
        the previous measurement with the same helper build_provenance uses, and
        appends the resulting OpDelta -- an op that moved nothing measurable lands
        as an orphan. Delta index matches the op's opdag position, so a later
        rollback can trim the tail cleanly (see `_trim_provenance`).
        """
        if self._prov is None:
            return
        prev: Mapping = self._prov.measurements[-1] if self._prov.measurements else {}
        after = self._measure_state() or {}
        if after:
            self._prov_saw_measurement = True
        changed, magnitude = _provenance._feature_delta(prev, after)
        index = len(self.opdag) - 1
        self._prov.deltas.append(_provenance.OpDelta(
            index=index,
            op_id=_provenance._op_id(index, op),
            op_tag=op.OP,
            before=prev,
            after=after,
            changed=changed,
            magnitude=magnitude,
            error="",
        ))
        self._prov.measurements.append(after)

    def _trim_provenance(self) -> None:
        """Keep the recorded provenance aligned with the (possibly truncated) opdag.

        After a rollback the opdag is shorter; drop the deltas/measurements for the
        ops that no longer exist. measurements holds the baseline plus one per op,
        so it keeps len(opdag)+1 entries. Replaying rebuilds identical geometry
        deterministically, so the retained tail measurement stays valid.
        """
        if self._prov is None:
            return
        n = len(self.opdag)
        del self._prov.deltas[n:]
        del self._prov.measurements[n + 1:]

    def provenance(self) -> Optional["_provenance.Provenance"]:
        """The accumulated op->geometry-delta provenance, or None when not recording.

        The returned Provenance is the same structure the orphan gate consumes
        (`provenance.orphan_ops` / `unattributed_features` accept it directly), built
        natively as ops landed -- no per-prefix session rebuild. When no state was
        ever measurable it is flagged `skipped`, degrading downstream checks to a
        clean PASS exactly as build_provenance does for the no-engine case.
        """
        if self._prov is None:
            return None
        if not self._prov_saw_measurement:
            self._prov.skipped = ("measure_state returned no state for any prefix "
                                  "(no engine / empty measurement)")
        else:
            self._prov.skipped = ""
        return self._prov

    def orphan_ops(self) -> List["_provenance.OpDelta"]:
        """Ops that landed but moved nothing measurable (the orphan set difference).

        Convenience over `provenance()`: returns [] when not recording. Delegates to
        provenance.orphan_ops so the definition of "orphan" stays single-sourced.
        """
        prov = self.provenance()
        if prov is None:
            return []
        return _provenance.orphan_ops(prov)

    # --- lifecycle hooks (opt-in) -----------------------------------------
    def _fire(self, event: str, context: Dict[str, Any]) -> None:
        """Fire one LOOP_EVENTS hook. A no-op when no bus is attached.

        Observation only: the returned FireReport (including any `block`) is
        deliberately ignored -- see LOOP_EVENTS. Error-isolated twice over: the
        bus already isolates a raising handler, and this swallows anything the
        bus itself throws (e.g. an event it does not know), because a listener
        must never be able to break the transaction it is watching.
        """
        bus = self.hook_bus
        if bus is None:
            return
        try:
            bus.fire(event, context)
        except Exception:  # noqa: BLE001 - an observer cannot break the loop
            pass

    # --- op-admissibility preflight (opt-in) ------------------------------
    def _gate_preflight(self, ops: List[Op], run_id: str) -> Optional[ApplyOpsResult]:
        """Judge the whole batch against the op catalog BEFORE executing any op.

        Returns None to proceed (no catalog configured, or every op admissible)
        and a refusing ApplyOpsResult when the gate rejects the batch. This is a
        PREFLIGHT, not a per-op gate: it runs against the pending stream, so a
        refused batch never touches the backend and `applied` is 0 -- strictly
        stronger than the loop's block-and-correct, which stops at the first bad
        op after earlier ones already landed.

        No-op unless `op_catalog` was supplied to the constructor.
        """
        if self.op_catalog is None:
            return None
        # Imported lazily so a session without a catalog -- the default -- never
        # pays for the gate module.
        from harnesscad.core.cisp import op_gate

        report = op_gate.gate(ops, self.op_catalog,
                              context=self.op_gate_context,
                              protected_regions=self.op_protected_regions)
        self.last_gate_report = report
        self.tracer.event("op_gate", run_id, {
            "ok": report.ok,
            "patch_status": report.patch_status,
            "protected_targets_checked": list(report.protected_targets_checked),
            "protected_targets_avoided": list(report.protected_targets_avoided),
        })
        if report.ok:
            return None
        # The gate's refusals are TYPED (op_index / op_name / feature /
        # reason_code); carry all of that through to the loop's Diagnostic
        # channel rather than flattening it to a string.
        diags = [Diagnostic(
            severity=Severity.ERROR,
            code=f"op_gate.{r.reason_code}",
            message=r.message,
            where=f"op[{r.op_index}] {r.op_name} on {r.feature}",
        ) for r in report.refusals]
        first = report.refusals[0]
        rejected = ops[first.op_index].to_dict() if 0 <= first.op_index < len(ops) else None
        self.tracer.event("rejected", run_id, {
            "op": rejected,
            "reason": "op-gate-refused",
            "diagnostics": [d.to_dict() for d in diags],
        })
        self.tracer.event("run_end", run_id, {
            "ok": False, "applied": 0, "digest": self.backend.state_digest(),
            "step_rewards": list(self.step_rewards),
            "mean_step_reward": self.mean_step_reward()})
        return ApplyOpsResult(False, 0, self.backend.state_digest(), diags,
                              rejected=rejected)

    # --- core loop --------------------------------------------------------
    def apply_ops(self, ops: List[Op]) -> ApplyOpsResult:
        diags: List[Diagnostic] = []
        applied = 0
        run_id = self._make_run_id(ops)
        # The per-op reward vector of THIS batch. Reset per batch so a caller
        # reading it after apply_ops always sees the trajectory it just ran.
        self.step_rewards: List[dict] = []
        self.tracer.event("run_start", run_id, {"op_count": len(ops)})
        # Preflight: refuse an inadmissible batch before anything mutates.
        # Returns None (and costs nothing) unless an op catalog was configured.
        refusal = self._gate_preflight(ops, run_id)
        if refusal is not None:
            return refusal
        for index, op in enumerate(ops):
            self._fire("iteration_start", {"run_id": run_id, "index": index,
                                           "op": op.to_dict()})
            res = self.backend.apply(op)
            if not res.ok:
                self._fire("iteration_end", {
                    "run_id": run_id, "index": index, "op": op.to_dict(),
                    "ok": False, "reason": "backend-rejected"})
                diags += res.diagnostics
                self.tracer.event("rejected", run_id, {
                    "op": op.to_dict(),
                    "reason": "backend-rejected",
                    "diagnostics": [d.to_dict() for d in res.diagnostics],
                })
                self._step_reward(run_id, index, op, 0.0, "backend-rejected")
                self.tracer.event("run_end", run_id, {
                    "ok": False, "applied": applied,
                    "digest": self.backend.state_digest(),
                    "step_rewards": list(self.step_rewards),
                    "mean_step_reward": self.mean_step_reward()})
                return ApplyOpsResult(False, applied, self.backend.state_digest(),
                                      diags, rejected=op.to_dict())
            self.opdag.append(op)
            diags += self.backend.regenerate()
            self._fire("plan_accepted", {
                "run_id": run_id, "index": index, "op": op.to_dict(),
                "digest": self.backend.state_digest()})
            self.tracer.event("op_applied", run_id, {
                "op": op.to_dict(),
                "index": len(self.opdag) - 1,
                "digest": self.backend.state_digest(),
            })
            report = self._verify()
            if self.fleet_blocking:
                # Opt-in: the fleet gates every op like a core verifier does, so
                # it must run per-op -- an ERROR rolls that op back.
                report = VerifyReport(report.diagnostics + self.run_fleet())
            diags += report.diagnostics
            self._fire("verify_verdict", {
                "run_id": run_id, "index": index, "op": op.to_dict(),
                "ok": report.ok,
                "diagnostics": [d.to_dict() for d in report.diagnostics]})
            self.tracer.event("verify_result", run_id, {
                "ok": report.ok,
                "diagnostics": [d.to_dict() for d in report.diagnostics],
            })
            if not report.ok:
                self.tracer.event("rejected", run_id, {
                    "op": op.to_dict(),
                    "reason": "verify-failed",
                    "diagnostics": [d.to_dict() for d in report.diagnostics],
                })
                self._step_reward(run_id, index, op, 0.0, "verify-failed")
                self._rollback_last()
                # Fired AFTER the rollback: the op is only fully resolved once
                # the state is back to last-good, and a listener reading the
                # digest mid-rollback would read a state that never shipped.
                self._fire("iteration_end", {
                    "run_id": run_id, "index": index, "op": op.to_dict(),
                    "ok": False, "reason": "verify-failed",
                    "digest": self.backend.state_digest()})
                self.tracer.event("run_end", run_id, {
                    "ok": False, "applied": applied,
                    "digest": self.backend.state_digest(),
                    "step_rewards": list(self.step_rewards),
                    "mean_step_reward": self.mean_step_reward()})
                return ApplyOpsResult(False, applied, self.backend.state_digest(),
                                      diags, rejected=op.to_dict())
            self._step_reward(run_id, index, op, 1.0, "applied+verified")
            applied += 1
            # Native provenance: attribute this op's measured geometry delta now,
            # while its state is live. No-op unless record_provenance was requested.
            self._record_provenance(op)
            label = f"auto-{len(self.opdag)}"
            self.opdag.checkpoint(label)
            self.tracer.event("checkpoint", run_id, {
                "label": label, "index": len(self.opdag)})
            self._fire("iteration_end", {
                "run_id": run_id, "index": index, "op": op.to_dict(),
                "ok": True, "reason": "applied+verified", "label": label,
                "digest": self.backend.state_digest()})
        # The advisory fleet runs ONCE per accepted batch, against the final
        # state: it verifies the model, not each intermediate op, and running it
        # per-op would only repeat the same findings N times. (When the fleet is
        # blocking it has already run per-op above, so skip it here.)
        if self.verify_level == "full" and not self.fleet_blocking:
            fleet = self.run_fleet()
            diags += fleet
            self.tracer.event("fleet_result", run_id, {
                "diagnostics": [d.to_dict() for d in fleet]})
        digest = self.backend.state_digest()
        self.tracer.event("run_end", run_id, {
            "ok": True, "applied": applied, "digest": digest,
            "step_rewards": list(self.step_rewards),
            "mean_step_reward": self.mean_step_reward()})
        return ApplyOpsResult(True, applied, digest, diags)

    def _verify(self) -> VerifyReport:
        diags: List[Diagnostic] = []
        for v in self.verifiers:
            diags += v.check(self.backend, self.opdag).diagnostics
        return VerifyReport(diags)

    def run_fleet(self, tiers: Optional[Sequence[str]] = None) -> List[Diagnostic]:
        """Run the discovered verifier fleet against the current state.

        Returns [] at verify_level="core" (the default), so the loop's cost and
        output are unchanged unless the fleet is asked for. Never raises: the
        registry catches a misbehaving verifier and reports it as a diagnostic.
        """
        if self.verify_level != "full" and tiers is None:
            return []
        # Imported lazily: the registry imports ~30 verifier modules on first
        # use, which a core-level session must not pay for.
        from harnesscad.eval.verifiers.registry import model_state, run_all

        state = model_state(self.backend, self.opdag)
        return run_all(state,
                       tiers=tiers if tiers is not None else self.fleet_tiers,
                       only=self.fleet_only,
                       skip=self.fleet_skip)

    # --- history ----------------------------------------------------------
    def checkpoint(self, label: str) -> None:
        self.opdag.checkpoint(label)

    def rollback(self, label: str) -> None:
        self.opdag.rollback(label)
        self._replay()
        self._trim_provenance()

    def _rollback_last(self) -> None:
        self.opdag.truncate(len(self.opdag) - 1)
        self._replay()
        self._trim_provenance()

    def _replay(self) -> None:
        self.backend.reset()
        for op in self.opdag.ops():
            self.backend.apply(op)

    # --- export -----------------------------------------------------------
    def export(self, path: str, **options) -> str:
        """Write the current model to `path`, in whatever format the extension names.

        Dispatches through the format registry (harnesscad.io.formats.registry),
        so every adapted writable codec (STL, OBJ, GLB, AMF, STEP, XCSG, SVG) is
        reachable from a session. Raises UnknownFormatError for an unknown
        extension and ExportError when the backend cannot produce the geometry
        the target format needs.
        """
        from harnesscad.io.formats.registry import export_session

        return export_session(self, str(path), **options)

    def digest(self) -> str:
        return self.backend.state_digest()

    def summary(self) -> dict:
        return self.backend.query("summary")
