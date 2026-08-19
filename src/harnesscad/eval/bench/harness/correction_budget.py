"""SUC / Pass@1 / AVG Re -- the correction-budget stability triple.

Source
------
Memory-Augmented RL for text-to-CAD (arXiv:2605.19748), Table 1. The paper
reports three numbers per configuration and they are only interpretable
together::

    configuration      SUC       Pass@1    AVG Re
    wo-memory          0.9494    0.7528    0.3018
    both-memory        0.9950    0.8300    0.3467

Definitions implemented here
----------------------------
A *task* is one prompt run through a generate -> check -> correct loop. Each
attempt is recorded as ``{"executed": bool, "valid": bool}``: ``executed`` means
the generated program ran without raising, ``valid`` means the resulting model
passed the basic topological validity check. An attempt **succeeds** iff BOTH
are true; neither alone is enough, and this module never treats "it ran" as
"it is a model".

* **SUC** -- success rate: the fraction of tasks with at least one succeeding
  attempt *within the correction budget*. Fraction in ``[0, 1]``, matching the
  paper's scale (0.9494, not 94.94).
* **Pass@1** -- the fraction of tasks whose **first** attempt succeeds, i.e. no
  correction round was used at all. This is a first-attempt rate over tasks, NOT
  the Chen et al. unbiased ``pass@k`` estimator over ``k`` independent samples
  (``bench.sequence.pass_at_k``); with one sample per attempt index there is
  nothing to estimate, it is a plain count. Pass@1 <= SUC always, and the
  constructor asserts it.
* **AVG Re** -- the mean number of correction retries consumed, averaged over
  the SUCCESSFUL tasks only. A task that succeeded on its first attempt
  contributes 0. Failed tasks are excluded from this average entirely, which is
  precisely why the number cannot be read alone.

The linked-report rule
----------------------
The paper's own caveat is that AVG Re must be read TOGETHER with SUC: a
configuration can raise retries *while raising final success*, because tasks
that used to fail outright now succeed after two or three corrections and start
contributing to an average they were previously absent from. That is exactly
what its Table 1 shows -- both-memory has BOTH the better SUC (0.9950 vs 0.9494)
and the higher AVG Re (0.3467 vs 0.3018), and reading AVG Re alone would call
the better configuration worse.

This module therefore exposes the triple only as a linked
:class:`StabilityReport`; there is no public function that returns AVG Re on its
own, and :func:`compare` refuses to name a winner from retries alone -- when
success rises together with retries it returns the verdict
``"more_success_more_retries"`` rather than a ranking.

Stdlib only, deterministic, no execution (attempt outcomes are recorded data).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

__all__ = [
    "StabilityReport",
    "attempt_succeeded",
    "first_success_index",
    "stability_report",
    "compare",
    "VERDICTS",
]

#: Every verdict :func:`compare` can return.
VERDICTS = (
    "more_success_more_retries",   # SUC up AND AVG Re up -- not a regression
    "more_success_fewer_retries",  # strictly better on both
    "same_success_fewer_retries",  # SUC tied, cheaper
    "same_success_more_retries",   # SUC tied, costlier
    "same_success_same_retries",
    "less_success",                # SUC down -- retries are not the story
)


def attempt_succeeded(attempt: Mapping) -> bool:
    """One attempt succeeds iff it executed AND passed topological validity."""
    if "executed" not in attempt or "valid" not in attempt:
        raise ValueError(
            "attempt record needs both 'executed' and 'valid'; refusing to read "
            "an executed-but-unchecked attempt as a success (keys: %s)"
            % (sorted(attempt),))
    return bool(attempt["executed"]) and bool(attempt["valid"])


def first_success_index(attempts: Sequence[Mapping],
                        budget: Optional[int] = None) -> Optional[int]:
    """Index of the first succeeding attempt, or ``None`` if the task failed.

    ``budget`` is the number of CORRECTION retries allowed, so at most
    ``budget + 1`` attempts are consulted (attempt 0 is the initial generation).
    ``None`` means "consult every recorded attempt". Attempts beyond the budget
    are ignored even if one of them succeeded -- a success outside the budget is
    not a success under the protocol.
    """
    if budget is not None:
        if budget < 0:
            raise ValueError("correction budget must be non-negative")
        attempts = attempts[:budget + 1]
    for i, attempt in enumerate(attempts):
        if attempt_succeeded(attempt):
            return i
    return None


@dataclass(frozen=True)
class StabilityReport:
    """SUC, Pass@1 and AVG Re over one task set -- always carried together."""

    n_tasks: int
    n_success: int
    n_first_attempt: int
    suc: float
    pass_at_1: float
    avg_retries: float
    budget: Optional[int] = None

    def __post_init__(self) -> None:
        if self.pass_at_1 > self.suc + 1e-12:
            raise ValueError("pass@1 (%r) cannot exceed SUC (%r)"
                             % (self.pass_at_1, self.suc))

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_tasks": self.n_tasks,
            "n_success": self.n_success,
            "n_first_attempt": self.n_first_attempt,
            "suc": self.suc,
            "pass_at_1": self.pass_at_1,
            "avg_retries": self.avg_retries,
            "budget": self.budget,
        }

    def summary(self) -> str:
        """The three numbers in one line -- AVG Re is never printed alone."""
        return ("SUC %.4f / Pass@1 %.4f / AVG Re %.4f over %d tasks "
                "(AVG Re averages the %d successful tasks only)"
                % (self.suc, self.pass_at_1, self.avg_retries,
                   self.n_tasks, self.n_success))


def stability_report(traces: Sequence[Mapping],
                     budget: Optional[int] = None) -> StabilityReport:
    """Build the linked SUC / Pass@1 / AVG Re report over a set of task traces.

    ``traces`` is one entry per task::

        {"task_id": "t3", "attempts": [{"executed": True, "valid": False},
                                       {"executed": True, "valid": True}]}

    A task with no recorded attempts is a failed task (it contributes to the
    SUC denominator), not a skipped one.
    """
    rows: List[Mapping] = list(traces)
    if not rows:
        raise ValueError("stability report is undefined over an empty task set")
    retries: List[int] = []
    n_first = 0
    for trace in rows:
        idx = first_success_index(list(trace.get("attempts", ())), budget)
        if idx is None:
            continue
        retries.append(idx)
        if idx == 0:
            n_first += 1
    n_success = len(retries)
    return StabilityReport(
        n_tasks=len(rows),
        n_success=n_success,
        n_first_attempt=n_first,
        suc=n_success / len(rows),
        pass_at_1=n_first / len(rows),
        avg_retries=(sum(retries) / n_success) if n_success else 0.0,
        budget=budget,
    )


def compare(before: StabilityReport, after: StabilityReport,
            tol: float = 1e-12) -> Dict[str, object]:
    """Compare two configurations, refusing to rank them on AVG Re alone.

    Returns the three deltas plus a ``verdict`` from :data:`VERDICTS` and a
    ``note`` spelling out why. When SUC rises and AVG Re rises with it, the
    verdict is ``"more_success_more_retries"``: the extra retries bought the
    extra successes, and the retry number is not a regression signal.
    """
    d_suc = after.suc - before.suc
    d_p1 = after.pass_at_1 - before.pass_at_1
    d_re = after.avg_retries - before.avg_retries
    if d_suc < -tol:
        verdict = "less_success"
        note = ("SUC fell; AVG Re is not comparable across configurations that "
                "do not solve the same tasks.")
    elif d_suc > tol:
        if d_re > tol:
            verdict = "more_success_more_retries"
            note = ("SUC and AVG Re both rose: tasks that previously failed now "
                    "succeed after corrections and enter an average they were "
                    "absent from. Higher AVG Re here is not a regression.")
        else:
            verdict = "more_success_fewer_retries"
            note = "SUC rose and AVG Re did not; strictly better on both axes."
    elif d_re > tol:
        verdict = "same_success_more_retries"
        note = "SUC tied, so the retry cost is comparable and it went up."
    elif d_re < -tol:
        verdict = "same_success_fewer_retries"
        note = "SUC tied, so the retry cost is comparable and it went down."
    else:
        verdict = "same_success_same_retries"
        note = "SUC and AVG Re both tied."
    return {
        "suc_delta": d_suc,
        "pass_at_1_delta": d_p1,
        "avg_retries_delta": d_re,
        "verdict": verdict,
        "note": note,
    }
