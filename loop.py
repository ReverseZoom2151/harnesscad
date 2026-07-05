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

from typing import List, Optional

from backends.base import GeometryBackend
from cisp.ops import Op
from cisp.protocol import ApplyOpsResult
from state.opdag import OpDAG
from verify import Diagnostic, VerifyReport, Verifier, default_verifiers


class HarnessSession:
    def __init__(self, backend: GeometryBackend,
                 verifiers: Optional[List[Verifier]] = None) -> None:
        self.backend = backend
        self.opdag = OpDAG()
        self.verifiers = verifiers if verifiers is not None else default_verifiers()
        self.backend.reset()
        self.opdag.checkpoint("start")

    # --- core loop --------------------------------------------------------
    def apply_ops(self, ops: List[Op]) -> ApplyOpsResult:
        diags: List[Diagnostic] = []
        applied = 0
        for op in ops:
            res = self.backend.apply(op)
            if not res.ok:
                diags += res.diagnostics
                return ApplyOpsResult(False, applied, self.backend.state_digest(),
                                      diags, rejected=op.to_dict())
            self.opdag.append(op)
            diags += self.backend.regenerate()
            report = self._verify()
            diags += report.diagnostics
            if not report.ok:
                self._rollback_last()
                return ApplyOpsResult(False, applied, self.backend.state_digest(),
                                      diags, rejected=op.to_dict())
            applied += 1
            self.opdag.checkpoint(f"auto-{len(self.opdag)}")
        return ApplyOpsResult(True, applied, self.backend.state_digest(), diags)

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
