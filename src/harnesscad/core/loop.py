"""HarnessSession — the applyOps -> regen -> verify -> checkpoint spine.

This is the Aider-style loop from the blueprint, kernel-agnostic:
  - block-and-correct: an op the backend rejects (bad reference/value) never
    mutates state; the batch stops and returns diagnostics for the agent to fix.
  - transactional verify: after each accepted op we regen + run the plural
    verifier; an ERROR-severity diagnostic rolls that op back (last-good state
    preserved) and returns.
  - checkpoint on success: every accepted+verified op is checkpointed, giving
    deterministic replay and rollback to any point.
  - the verifier FLEET: at verify_level="full" the session additionally runs
    every verifier discovered by harnesscad.eval.verifiers.registry (DFM, plan
    preflight, standards, interference, kernel preflight, plausibility, ...).
    Fleet diagnostics are surfaced alongside the core ones; by default they are
    advisory (they never roll an op back), so the transactional semantics above
    are unchanged.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence

from harnesscad.io.backends.base import GeometryBackend
from harnesscad.core.cisp.ops import Op, canonical_json
from harnesscad.core.cisp.protocol import ApplyOpsResult
from harnesscad.core.state.opdag import OpDAG
from harnesscad.core.trace import NullTracer, Tracer
from harnesscad.eval.verifiers.verify import Diagnostic, VerifyReport, Verifier, default_verifiers


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


class HarnessSession:
    def __init__(self, backend: GeometryBackend,
                 verifiers: Optional[List[Verifier]] = None,
                 tracer: Optional[Tracer] = None,
                 verify_level: str = "core",
                 fleet_tiers: Optional[Sequence[str]] = None,
                 fleet_only: Optional[Sequence[str]] = None,
                 fleet_skip: Sequence[str] = (),
                 fleet_blocking: bool = False) -> None:
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
        self._run_seq = 0
        self.backend.reset()
        self.opdag.checkpoint("start")

    def _make_run_id(self, ops: List[Op]) -> str:
        """Deterministic, wall-clock-free run id.

        Derived from the batch contents and a per-session sequence number, so
        replaying the same op stream yields the same ids (no time dependency).
        """
        self._run_seq += 1
        blob = "|".join(canonical_json(op) for op in ops)
        h = hashlib.sha256(f"{self._run_seq}|{blob}".encode()).hexdigest()
        return f"run-{self._run_seq}-{h[:12]}"

    # --- core loop --------------------------------------------------------
    def apply_ops(self, ops: List[Op]) -> ApplyOpsResult:
        diags: List[Diagnostic] = []
        applied = 0
        run_id = self._make_run_id(ops)
        self.tracer.event("run_start", run_id, {"op_count": len(ops)})
        for op in ops:
            res = self.backend.apply(op)
            if not res.ok:
                diags += res.diagnostics
                self.tracer.event("rejected", run_id, {
                    "op": op.to_dict(),
                    "reason": "backend-rejected",
                    "diagnostics": [d.to_dict() for d in res.diagnostics],
                })
                self.tracer.event("run_end", run_id, {
                    "ok": False, "applied": applied,
                    "digest": self.backend.state_digest()})
                return ApplyOpsResult(False, applied, self.backend.state_digest(),
                                      diags, rejected=op.to_dict())
            self.opdag.append(op)
            diags += self.backend.regenerate()
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
                self._rollback_last()
                self.tracer.event("run_end", run_id, {
                    "ok": False, "applied": applied,
                    "digest": self.backend.state_digest()})
                return ApplyOpsResult(False, applied, self.backend.state_digest(),
                                      diags, rejected=op.to_dict())
            applied += 1
            label = f"auto-{len(self.opdag)}"
            self.opdag.checkpoint(label)
            self.tracer.event("checkpoint", run_id, {
                "label": label, "index": len(self.opdag)})
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
            "ok": True, "applied": applied, "digest": digest})
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

    def _rollback_last(self) -> None:
        self.opdag.truncate(len(self.opdag) - 1)
        self._replay()

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
