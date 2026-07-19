"""CADReasoner closed-loop program editing (render -> compare -> refine).

Paper: "CADReasoner - Iterative Program Editing for CAD Reverse Engineering",
vs. reporting protocol).

This approach casts reverse engineering as *closed-loop program editing*. At each
iteration ``t`` an editor proposes an updated program from the target, the
previous render, and their geometric discrepancy:

    C_t = f( E(T, S_{t-1}), C_{t-1} ),     S_{t-1} = R(C_{t-1})

The render is fed back to the next step, the loop keeps the *best-so-far* program
by a geometric discrepancy ``D``, and stops when improvement falls below a
threshold or a step budget is reached.

This module is the deterministic *harness* around that loop. Everything learned
in this approach -- the neural editor ``f`` -- is injected as a plain callable, so the
same loop drives a heuristic editor, a stub, or (behind an adapter) a model. The
harness owns exactly the deterministic parts:

  * calling render, catching failures and counting them as *invalid* (the IR
    metric), so a broken program never crashes the loop;
  * computing the discrepancy encoding fed to the editor (via
    ``cadreasoner_discrepancy`` by default);
  * **best-so-far** bookkeeping ``C_{<=t} = argmin_{i<=t} D(T, R(C_i))``;
  * the **selection-vs-reporting** split: ranking/selection use one
    (possibly scan) target while the reported score is measured against a second
    (clean) target, so candidate selection never accesses an oracle;
  * early stopping on a ``min_improvement`` threshold or ``max_steps`` budget.

Determinism: no wall clock, no RNG here. Given the same editor/render/metrics the
trajectory is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Any, Callable, List, Optional, Sequence, Tuple

from harnesscad.domain.editing.discrepancy_encoding import (
    DiscrepancyEncoding,
    encode_discrepancy,
    encode_null_init,
)

Point = Tuple[float, ...]


# --------------------------------------------------------------------------- #
# Per-step record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EditStep:
    """One iteration of the loop.

    - ``t``            : 1-based iteration index.
    - ``program``      : the program C_t the editor emitted this step.
    - ``valid``        : whether R(C_t) rendered a non-degenerate solid.
    - ``select_score`` : D against the *selection* target (``inf`` if invalid).
    - ``report_score`` : D against the *reporting* target (``None`` if invalid or
                         no separate report target was supplied).
    - ``encoding``     : the discrepancy encoding that fed the editor at this step.
    - ``error``        : render/editor error string when invalid.
    """

    t: int
    program: Any
    valid: bool
    select_score: float
    report_score: Optional[float] = None
    encoding: Optional[DiscrepancyEncoding] = None
    error: Optional[str] = None


@dataclass
class EditLoopResult:
    """Full trajectory plus the best-so-far selection."""

    steps: List[EditStep] = field(default_factory=list)
    best_index: int = -1                 # index into ``steps`` of the best-so-far
    best_program: Any = None
    best_select_score: float = inf
    best_report_score: Optional[float] = None
    stopped_reason: str = ""
    invalid_rate: float = 0.0            # fraction of steps that failed to render

    @property
    def converged(self) -> bool:
        return self.best_index >= 0

    def to_dict(self) -> dict:
        return {
            "n_steps": len(self.steps),
            "best_index": self.best_index,
            "best_select_score": self.best_select_score,
            "best_report_score": self.best_report_score,
            "stopped_reason": self.stopped_reason,
            "invalid_rate": self.invalid_rate,
        }


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def run_edit_loop(
    target_points: Sequence[Point],
    initial_program: Any,
    editor: Callable[..., Any],
    render: Callable[[Any], Optional[Sequence[Point]]],
    *,
    select_metric: Callable[[Sequence[Point], Sequence[Point]], Optional[float]],
    report_target_points: Optional[Sequence[Point]] = None,
    report_metric: Optional[
        Callable[[Sequence[Point], Sequence[Point]], Optional[float]]
    ] = None,
    max_steps: int = 5,
    min_improvement: float = 1e-6,
    k: int = 128,
    seed: int = 0,
    encode: Optional[Callable[..., DiscrepancyEncoding]] = None,
) -> EditLoopResult:
    """Drive render -> compare -> refine, keeping the best-so-far program.

    Args:
        target_points: point set sampled on the *selection* target (the scan, at
            inference -- never the clean oracle). Drives ranking and the editor
            feedback.
        initial_program: the program C_0 the loop starts editing from. The first
            edit (t=1) uses the null-prediction encoding when this renders empty.
        editor: ``editor(target_points, prev_render, prev_program, encoding) ->
            next_program``. The injected (learned) refiner. May be called at t=1
            with ``prev_render=None``.
        render: ``render(program) -> point-set`` (the mesh sampled to points) or
            ``None`` / raises on an invalid/degenerate program.
        select_metric: ``D(target_points, render_points) -> float`` used for
            best-so-far selection (lower is better). ``None`` counts as invalid.
        report_target_points / report_metric: optional second target + metric
. When given, the reported score of each step is measured
            against ``report_target_points`` but **selection still uses**
            ``select_metric``, so the loop never selects using the report oracle.
        max_steps: iteration budget ``s``.
        min_improvement: stop once the best select score improves by less than
            this between consecutive steps.
        k: farthest-point budget per side for the discrepancy encoding.
        seed: seed for the t=1 null-init permutation.
        encode: override for the discrepancy encoder (defaults to
            ``encode_discrepancy`` / ``encode_null_init``).

    Returns:
        ``EditLoopResult`` with every step and the best-so-far program/scores.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    report_metric = report_metric or select_metric
    steps: List[EditStep] = []

    prev_program = initial_program
    prev_render = _safe_render(render, initial_program)
    best_index = -1
    best_select = inf
    best_report: Optional[float] = None
    invalid = 0
    reason = "budget-exhausted"

    for t in range(1, max_steps + 1):
        # Build the discrepancy encoding the editor sees this step.
        encoding = _build_encoding(
            target_points, prev_render, k=k, seed=seed, encode=encode)

        # ACT: the injected editor proposes the next program.
        try:
            program = editor(target_points, prev_render, prev_program, encoding)
        except Exception as exc:  # noqa: BLE001 - editor boundary, never fatal
            steps.append(EditStep(
                t=t, program=None, valid=False, select_score=inf,
                encoding=encoding, error=f"editor: {type(exc).__name__}: {exc}"))
            invalid += 1
            prev_program, prev_render = prev_program, None
            continue

        # RENDER + SCORE.
        rendered = _safe_render(render, program)
        if not rendered:
            steps.append(EditStep(
                t=t, program=program, valid=False, select_score=inf,
                encoding=encoding, error="render: invalid or degenerate solid"))
            invalid += 1
            prev_program, prev_render = program, None
            continue

        sel = _safe_metric(select_metric, target_points, rendered)
        if sel is None:
            steps.append(EditStep(
                t=t, program=program, valid=False, select_score=inf,
                encoding=encoding, error="select_metric returned None"))
            invalid += 1
            prev_program, prev_render = program, rendered
            continue

        rep: Optional[float] = None
        if report_target_points is not None:
            rep = _safe_metric(report_metric, report_target_points, rendered)
        else:
            rep = sel

        steps.append(EditStep(
            t=t, program=program, valid=True, select_score=sel,
            report_score=rep, encoding=encoding))

        # BEST-SO-FAR by the selection metric only.
        improvement = best_select - sel
        if sel < best_select:
            best_index = len(steps) - 1
            best_select = sel
            best_report = rep

        prev_program, prev_render = program, rendered

        # Early stop when the best-so-far barely moved.
        if improvement < min_improvement and t > 1:
            reason = "min-improvement"
            break

    result = EditLoopResult(
        steps=steps,
        best_index=best_index,
        best_program=steps[best_index].program if best_index >= 0 else None,
        best_select_score=best_select,
        best_report_score=best_report,
        stopped_reason=reason,
        invalid_rate=invalid / len(steps) if steps else 0.0,
    )
    return result


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _safe_render(render, program) -> Optional[List[Point]]:
    if program is None:
        return None
    try:
        out = render(program)
    except Exception:  # noqa: BLE001 - invalid program is expected, not fatal
        return None
    if not out:
        return None
    return list(out)


def _safe_metric(metric, a, b) -> Optional[float]:
    try:
        v = metric(a, b)
    except Exception:  # noqa: BLE001
        return None
    if v is None:
        return None
    return float(v)


def _build_encoding(target_points, prev_render, *, k, seed, encode):
    if encode is not None:
        return encode(target_points, prev_render, k=k, seed=seed)
    if not prev_render:
        return encode_null_init(target_points, k=k, seed=seed)
    return encode_discrepancy(target_points, prev_render, k=k)
