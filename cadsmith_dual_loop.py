"""CADSmith dual nested correction loops (CADSmith sec. III).

CADSmith's central architectural contribution is a pair of *nested* correction
loops around code generation — distinct from a single flat refinement loop:

  * Inner loop (execution errors): when the Executor fails to run the code, an
    Error Refiner receives the traceback (+ KB2 context) and produces corrected
    code, retrying up to three times before declaring a code-execution failure.
  * Outer loop (geometric errors): when the code executes but the Validator
    judges the geometry wrong, a Refiner produces corrected code (targeting the
    exact discrepancies), restarting the outer loop, up to five iterations.

The inner loop is fully nested inside each outer iteration: every candidate code
is first driven to a clean execution (or gives up) before its geometry is
validated. The outer loop's hard kernel gate can also short-circuit validation.

This module orchestrates that control flow deterministically. All agent
behaviours are *injected callables* (executor, error-refiner, validator,
refiner), so the loop runs in tests with no LLM and no CAD kernel, exercising
exactly the branching the paper describes. It returns a structured, replayable
:class:`DualLoopResult`.

Stdlib only, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Injected agent contracts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExecResult:
    """Outcome of running one code candidate in the sandboxed Executor."""

    ok: bool
    traceback: str = ""            # populated when ok is False
    metrics: object = None         # KernelMetrics-like, populated when ok


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of the Validator on an executed candidate."""

    passed: bool
    feedback: str = ""
    issue_code: str = ""           # normalised code for escalation tracking


# Executor: code -> ExecResult.
Executor = Callable[[str], ExecResult]
# Error Refiner: (code, traceback, attempt) -> corrected code.
ErrorRefiner = Callable[[str, str, int], str]
# Validator: (code, ExecResult) -> ValidationResult.
Validator = Callable[[str, ExecResult], ValidationResult]
# Refiner: (code, ValidationResult, iteration) -> corrected code.
Refiner = Callable[[str, ValidationResult, int], str]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
class Stop(Enum):
    VALIDATED = "validated"
    EXEC_FAILURE = "code-execution-failure"
    MAX_OUTER = "max-outer-iterations"


@dataclass(frozen=True)
class InnerAttempt:
    code: str
    exec_result: ExecResult


@dataclass(frozen=True)
class InnerLoopRecord:
    attempts: Tuple[InnerAttempt, ...]
    resolved: bool                 # did the code eventually execute?

    @property
    def final(self) -> InnerAttempt:
        return self.attempts[-1]

    @property
    def retries(self) -> int:
        return len(self.attempts) - 1


@dataclass(frozen=True)
class OuterIteration:
    index: int
    inner: InnerLoopRecord
    validation: Optional[ValidationResult]


@dataclass
class DualLoopResult:
    stop: Stop
    iterations: List[OuterIteration] = field(default_factory=list)
    final_code: str = ""

    @property
    def passed(self) -> bool:
        return self.stop is Stop.VALIDATED

    @property
    def outer_count(self) -> int:
        return len(self.iterations)


# --------------------------------------------------------------------------- #
# Inner loop
# --------------------------------------------------------------------------- #
def run_inner_loop(code: str, executor: Executor, error_refiner: ErrorRefiner,
                   *, max_retries: int = 3) -> InnerLoopRecord:
    """Drive one code candidate to a clean execution (or give up).

    Runs the code; on failure, invokes the Error Refiner with the traceback and
    retries, up to ``max_retries`` corrective attempts after the first run.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    attempts: List[InnerAttempt] = []
    current = code
    # First execution + up to max_retries corrections => max_retries+1 runs.
    for attempt in range(max_retries + 1):
        res = executor(current)
        attempts.append(InnerAttempt(current, res))
        if res.ok:
            return InnerLoopRecord(tuple(attempts), resolved=True)
        if attempt < max_retries:
            current = error_refiner(current, res.traceback, attempt)
    return InnerLoopRecord(tuple(attempts), resolved=False)


# --------------------------------------------------------------------------- #
# Outer loop (with inner nested inside)
# --------------------------------------------------------------------------- #
def run_dual_loop(code: str, executor: Executor, error_refiner: ErrorRefiner,
                  validator: Validator, refiner: Refiner,
                  *, max_outer: int = 5, max_inner_retries: int = 3) -> DualLoopResult:
    """Run the full nested control flow.

    For each outer iteration: drive the current code through the inner loop to a
    clean execution; if it never executes, stop with a code-execution failure.
    Otherwise validate the geometry — pass => done; fail => Refiner produces new
    code and the outer loop repeats. Stops after ``max_outer`` iterations.
    """
    if max_outer < 1:
        raise ValueError("max_outer must be >= 1")

    result = DualLoopResult(stop=Stop.MAX_OUTER)
    current = code
    for i in range(max_outer):
        inner = run_inner_loop(current, executor, error_refiner,
                               max_retries=max_inner_retries)
        if not inner.resolved:
            result.iterations.append(OuterIteration(i, inner, None))
            result.stop = Stop.EXEC_FAILURE
            result.final_code = inner.final.code
            return result

        exec_res = inner.final.exec_result
        validation = validator(inner.final.code, exec_res)
        result.iterations.append(OuterIteration(i, inner, validation))

        if validation.passed:
            result.stop = Stop.VALIDATED
            result.final_code = inner.final.code
            return result

        # Geometric refinement: produce a corrected candidate for the next outer
        # iteration (unless this was the last allowed iteration).
        result.final_code = inner.final.code
        if i < max_outer - 1:
            current = refiner(inner.final.code, validation, i)

    return result
