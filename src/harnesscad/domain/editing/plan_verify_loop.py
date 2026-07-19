"""Plan-generate-verify loop: geometry-driven parametric CAD editing.

This loop edits a parametric construction sequence so that its rendered shape
matches a *target* geometric shape, while preserving the original sequence's
structure. It does so by iterating three stages:

  1. **Plan** -- locate the segments that most need editing and mask them
     (:mod:`editing.cadmorph_plan`).
  2. **Generate** -- infill the masks ``N`` times to propose ``N`` candidate
     sequences. This is typically a learned masked-parameter-prediction model;
     here it is any callable ``generate(masked, n, rng) -> [candidate, ...]``
     you supply.
  3. **Verify** -- render each candidate, measure its distance to the target,
     and keep the global best via a cross-round priority queue
     (:mod:`editing.cadmorph_verify`).

The loop terminates when the selected sequence converges (unchanged from the
previous round or its distance falls below a tolerance) or a maximum number of
rounds is reached. This is inference-time test-time scaling with a verifier --
no training, no triplet data.

This orchestrator is model-agnostic: it is distinct from :class:`loop.py`'s
``HarnessSession`` (apply-op -> regen -> verify -> checkpoint), because it does
not apply ops one at a time -- it generates whole candidate *sequences* and
selects among them against a target shape. The two learned components (parameter-to-shape
renderer/embedder and masked-parameter-prediction infiller) enter only through the ``render`` /
``contribution`` / ``generate`` callables, so the whole control flow is
deterministic given a seeded generator.

Stdlib-only; the only randomness is a caller-provided ``random.Random(seed)``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from harnesscad.domain.editing.edit_planning import Contribution, plan_mask
from harnesscad.domain.editing.candidate_verify import (
    CandidateQueue, ScoredCandidate, edit_distance, select_best,
)


Sequence_ = Sequence
Renderer = Callable[[Sequence], object]
Distance = Callable[[object, object], float]
Generator = Callable[[Sequence, int, random.Random], List[Sequence]]


@dataclass(frozen=True)
class RoundRecord:
    """What happened in one plan-generate-verify round."""

    round_index: int
    masked_indices: Tuple[int, ...]
    n_candidates: int
    best_distance: float
    best_score: float
    accepted: bool               # True if this round improved the running best


@dataclass
class LoopResult:
    """Outcome of the full loop."""

    sequence: Tuple                     # the final C'
    distance: float                     # its geometric distance to the target
    rounds: List[RoundRecord] = field(default_factory=list)
    converged: bool = False             # stopped early (vs hitting max rounds)
    queue: Optional[CandidateQueue] = None


class CADMorphLoop:
    """Drive the iterative plan-generate-verify editing loop.

    Parameters (all the learned pieces are injected as callables):

      * ``render(sequence) -> shape`` -- parameter-to-shape ``F`` (parameter-to-shape stand-in).
      * ``distance(shape_a, shape_b) -> float`` -- geometric dissimilarity
        (e.g. :func:`geometry.cadmorph_tsdf.l2_distance`).
      * ``contribution(sequence, shape) -> [float]`` -- per-segment contribution
        ``M`` used by the planner (see
        :func:`editing.cadmorph_plan.leave_one_out_contribution`).
      * ``generate(masked, n, rng) -> [sequence]`` -- candidate infiller (masked-parameter-prediction
        stand-in).

    Tuning: ``n_candidates`` (``N``), ``max_rounds`` (paper uses 10),
    ``queue_size`` (``X``), ``lam`` (structure-preservation weight in the
    objective), and ``tol`` (convergence distance).
    """

    def __init__(self, render: Renderer, distance: Distance,
                 contribution: Contribution, generate: Generator,
                 *, n_candidates: int = 4, max_rounds: int = 10,
                 queue_size: int = 4, lam: float = 0.0,
                 tol: float = 0.0, max_k: Optional[int] = None) -> None:
        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive")
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        self.render = render
        self.distance = distance
        self.contribution = contribution
        self.generate = generate
        self.n_candidates = n_candidates
        self.max_rounds = max_rounds
        self.queue_size = queue_size
        self.lam = lam
        self.tol = tol
        self.max_k = max_k

    def run(self, original: Sequence, target: object,
            *, seed: int = 0) -> LoopResult:
        """Edit ``original`` toward ``target`` and return the best sequence."""
        rng = random.Random(seed)
        original = tuple(original)
        queue: CandidateQueue = CandidateQueue(self.queue_size)

        # Seed the queue/current with the original so a round that produces only
        # worse candidates can never regress below the starting point.
        target_shape = target
        start_dist = self.distance(self.render(original), target_shape)
        queue.push(ScoredCandidate(original, start_dist, start_dist,
                                   round_index=-1, order=0))

        current: Tuple = original
        current_dist = start_dist
        rounds: List[RoundRecord] = []
        converged = False

        def shape_distance(candidate: Sequence) -> float:
            return self.distance(self.render(candidate), target_shape)

        for r in range(self.max_rounds):
            if current_dist <= self.tol:
                converged = True
                break

            # -- Plan: contributions of current segments to current vs target.
            current_shape = self.render(current)
            contrib_current = self.contribution(current, current_shape)
            contrib_target = self.contribution(current, target_shape)
            plan = plan_mask(current, contrib_current, contrib_target,
                             max_k=self.max_k)

            # If the planner found nothing to edit, we have converged.
            if not plan.masked_indices:
                converged = True
                break

            # -- Generate: N candidate sequences from the masked sequence.
            candidates = list(self.generate(plan.masked_sequence,
                                            self.n_candidates, rng))

            # -- Verify: score against target, keep global best via the queue.
            prev_best_score = queue.best().score if queue.best() else None
            if candidates:
                select_best(candidates, shape_distance, original,
                            lam=self.lam, queue=queue, round_index=r)
            best = queue.best()
            assert best is not None  # queue seeded with original

            accepted = (prev_best_score is None
                        or best.score < prev_best_score)
            rounds.append(RoundRecord(
                round_index=r,
                masked_indices=plan.masked_indices,
                n_candidates=len(candidates),
                best_distance=best.geom_distance,
                best_score=best.score,
                accepted=accepted))

            # Convergence: the selected sequence stopped changing.
            new_current = tuple(best.candidate)
            if new_current == current:
                converged = True
                current, current_dist = new_current, best.geom_distance
                break
            current, current_dist = new_current, best.geom_distance

        best = queue.best()
        assert best is not None
        return LoopResult(
            sequence=tuple(best.candidate),
            distance=best.geom_distance,
            rounds=rounds,
            converged=converged,
            queue=queue)
