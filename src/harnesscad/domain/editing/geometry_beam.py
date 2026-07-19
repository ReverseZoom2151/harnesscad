"""CADReasoner stochastic, geometry-guided beam over edit iterations.

Paper: "CADReasoner - Iterative Program Editing for CAD Reverse Engineering",

Greedy decoding can miss higher-quality trajectories, so this approach runs a
stochastic beam pruned by geometry:

  * t = 1: sample ``N`` candidate programs, compile + render each, discard
    invalid generations, rank by the primary geometric metric ``D``, keep the
    top-``N`` survivors.
  * t > 1: from each survivor generate ``N`` children (at most ``N^2`` candidates
    per step), render and score them all, retain the top-``N`` for the next step.
  * The best-so-far program over *all* evaluated candidates is maintained and
    reported. Total renders: ``N + (s-1) * N^2``.

This module is the deterministic harness. The stochastic candidate proposals are
injected as two callables (``seed_generator`` for t=1 and ``child_generator`` for
t>1); the stochasticity lives inside those (a sampling model), while the
harness itself is fully deterministic given them -- it enumerates candidate slots
by explicit integer indices (no wall-clock RNG), renders, scores, and applies a
**stable** top-``N`` selection (ties break by discovery order).

Distinct from ``reliability/strategies/best_of_n`` (a single-step parallel draw
with no render-feedback beam) and from the MCTS strategy: here beams evolve over
*edit iterations*, each child conditioned on its parent's program and render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Any, Callable, List, Optional, Sequence, Tuple

Point = Tuple[float, ...]


@dataclass(frozen=True)
class BeamCandidate:
    """One evaluated candidate program at some iteration."""

    step: int              # 1-based iteration it was produced at
    parent: int            # index of the parent survivor (-1 at t=1)
    slot: int              # candidate slot within its parent's expansion
    program: Any
    valid: bool
    score: float           # D against the target (inf if invalid)
    render: Optional[List[Point]] = None
    error: Optional[str] = None

    @property
    def key(self) -> Tuple[float, int, int]:
        """Stable ranking key: lower score first, then discovery order."""
        return (self.score, self.parent, self.slot)


@dataclass
class BeamResult:
    """Beam trajectory plus the global best-so-far."""

    survivors_per_step: List[Tuple[BeamCandidate, ...]] = field(default_factory=list)
    best: Optional[BeamCandidate] = None
    total_renders: int = 0
    total_invalid: int = 0
    n: int = 0
    steps: int = 0

    @property
    def best_program(self) -> Any:
        return self.best.program if self.best else None

    @property
    def best_score(self) -> float:
        return self.best.score if self.best else inf

    @property
    def expected_render_budget(self) -> int:
        """The N + (s-1) N^2 upper bound on renders."""
        if self.n <= 0 or self.steps <= 0:
            return 0
        return self.n + (self.steps - 1) * self.n * self.n

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "steps": self.steps,
            "total_renders": self.total_renders,
            "total_invalid": self.total_invalid,
            "expected_render_budget": self.expected_render_budget,
            "best_score": self.best_score,
            "survivors_per_step": [len(s) for s in self.survivors_per_step],
        }


def _evaluate(program, render, score, target_points, step, parent, slot):
    """Render + score one program into a BeamCandidate (never raises)."""
    try:
        rendered = render(program)
    except Exception as exc:  # noqa: BLE001
        return BeamCandidate(step, parent, slot, program, False, inf,
                             error=f"render: {type(exc).__name__}: {exc}")
    if not rendered:
        return BeamCandidate(step, parent, slot, program, False, inf,
                             error="render: invalid or degenerate solid")
    rendered = list(rendered)
    try:
        d = score(target_points, rendered)
    except Exception as exc:  # noqa: BLE001
        return BeamCandidate(step, parent, slot, program, False, inf,
                             render=rendered,
                             error=f"score: {type(exc).__name__}: {exc}")
    if d is None:
        return BeamCandidate(step, parent, slot, program, False, inf,
                             render=rendered, error="score returned None")
    return BeamCandidate(step, parent, slot, program, True, float(d), render=rendered)


def run_geometry_beam(
    target_points: Sequence[Point],
    seed_generator: Callable[[Sequence[Point], int], Any],
    child_generator: Callable[..., Any],
    render: Callable[[Any], Optional[Sequence[Point]]],
    score: Callable[[Sequence[Point], Sequence[Point]], Optional[float]],
    *,
    n: int = 5,
    steps: int = 5,
) -> BeamResult:
    """Run the geometry-guided stochastic beam.

    Args:
        target_points: the (selection) target point set; ``score`` ranks against it.
        seed_generator: ``seed_generator(target_points, slot) -> program`` for the
            ``N`` t=1 candidates. ``slot`` is the deterministic candidate index.
        child_generator: ``child_generator(target_points, parent_program,
            parent_render, slot) -> program`` producing one child of a survivor.
        render: ``render(program) -> point-set`` or ``None`` / raises when invalid.
        score: ``D(target_points, render_points) -> float`` (lower is better);
            ``None`` marks the candidate invalid.
        n: beam width ``N`` (candidates kept per step; children per survivor).
        steps: number of iterations ``s``.

    Returns:
        ``BeamResult`` with per-step survivors and the global best-so-far.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    result = BeamResult(n=n, steps=steps)
    best: Optional[BeamCandidate] = None

    def _consider(cand: BeamCandidate) -> None:
        nonlocal best
        result.total_renders += 1
        if not cand.valid:
            result.total_invalid += 1
        # Best-so-far over ALL evaluated candidates (valid only).
        if cand.valid and (best is None or cand.key < best.key):
            best = cand

    # t = 1: N seed candidates.
    first: List[BeamCandidate] = []
    for slot in range(n):
        program = seed_generator(target_points, slot)
        cand = _evaluate(program, render, score, target_points, 1, -1, slot)
        _consider(cand)
        first.append(cand)
    survivors = _top_n(first, n)
    result.survivors_per_step.append(tuple(survivors))

    # t > 1: expand each survivor into N children, keep top-N.
    for t in range(2, steps + 1):
        children: List[BeamCandidate] = []
        for parent_idx, parent in enumerate(survivors):
            if not parent.valid:
                continue
            for slot in range(n):
                program = child_generator(
                    target_points, parent.program, parent.render, slot)
                cand = _evaluate(
                    program, render, score, target_points, t, parent_idx, slot)
                _consider(cand)
                children.append(cand)
        if not children:
            break
        survivors = _top_n(children, n)
        result.survivors_per_step.append(tuple(survivors))

    result.best = best
    return result


def _top_n(candidates: Sequence[BeamCandidate], n: int) -> List[BeamCandidate]:
    """Stable top-``N`` by ascending score, keeping only valid candidates.

    Valid candidates always rank above invalid ones; among valids, ties break by
    discovery order via ``BeamCandidate.key``. If fewer than ``N`` valids exist we
    return just the valids (a fully-invalid step yields an empty beam).
    """
    valid = [c for c in candidates if c.valid]
    valid.sort(key=lambda c: c.key)
    return valid[:n]
