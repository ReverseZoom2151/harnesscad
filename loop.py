"""HarnessSession — the applyOps -> regen -> verify -> checkpoint spine.

This is the Aider-style loop from the blueprint, kernel-agnostic:
  - block-and-correct: an op the backend rejects (bad reference/value) never
    mutates state; the batch stops and returns diagnostics for the agent to fix.
  - transactional verify: after each accepted op we regen + run the plural
    verifier; an ERROR-severity diagnostic rolls that op back (last-good state
    preserved) and returns.
  - checkpoint on success: every accepted+verified op is checkpointed, giving
    deterministic replay and rollback to any point.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

from backends.base import GeometryBackend
from cisp.ops import Op, canonical_json
from cisp.protocol import ApplyOpsResult
from state.opdag import OpDAG
from trace import NullTracer, Tracer
from verifiers.verify import Diagnostic, VerifyReport, Verifier, default_verifiers


class HarnessSession:
    def __init__(self, backend: GeometryBackend,
                 verifiers: Optional[List[Verifier]] = None,
                 tracer: Optional[Tracer] = None) -> None:
        self.backend = backend
        self.opdag = OpDAG()
        self.verifiers = verifiers if verifiers is not None else default_verifiers()
        # NullTracer is the default: tracing is opt-in and zero-cost, so a
        # HarnessSession(backend) behaves exactly as before.
        self.tracer = tracer if tracer is not None else NullTracer()
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
        digest = self.backend.state_digest()
        self.tracer.event("run_end", run_id, {
            "ok": True, "applied": applied, "digest": digest})
        return ApplyOpsResult(True, applied, digest, diags)

    def _verify(self) -> VerifyReport:
        diags: List[Diagnostic] = []
        for v in self.verifiers:
            diags += v.check(self.backend, self.opdag).diagnostics
        return VerifyReport(diags)

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

    def digest(self) -> str:
        return self.backend.state_digest()

    def summary(self) -> dict:
        return self.backend.query("summary")
