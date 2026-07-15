"""step_reward_cua — EXACT per-step reward for a GUI trajectory. Credit assignment,
solved.

The synthesis names the hole this closes: "per-step reward is exact (after every op
the document is fully determined), so credit assignment -- the hole in every GUI-RL
approach -- is solved for CAD." A terminal reward of 0 on a 20-action failure punishes
all 20 actions equally; every other system ESTIMATES which step actually broke
(Math-Shepherd samples completions per prefix; Fara has an LLM guess). We do not have
to estimate: after each CAD op the geometry is fully determined, so the first divergent
step is a FACT.

This module does NOT re-derive that fact. The exact first-divergence detector already
exists in :mod:`harnesscad.agents.selftrain.divergence` (the set-difference /
recoverability analysis, validated against the ``trap_hole_oversize`` regression), and
the per-op reward assignment (the book's sec. 12.8.3: +1 to the correct prefix, -1 to
the divergent op ALONE, 0 to the doomed tail) already exists there as
:func:`selftrain.divergence.step_rewards` over :class:`selftrain.trajectory.StepReward`.
This module COMPOSES both and adds the two things the CUA side needs:

* a bridge from a compiled :class:`~harnesscad.agents.cua.verified_trajectory.VerifiedTrajectory`
  to the same per-step reward shape, reading the per-step ORACLE VERDICTS that
  :mod:`harnesscad.agents.cua.trajectory_compiler` already labelled -- so a GUI
  trajectory gets exact credit assignment straight from its verdict track, with no
  second geometric pass (:func:`step_rewards_from_trajectory`,
  :func:`first_divergence_from_trajectory`).
* a single entry point that scores a raw op stream via the geometric detector
  (:func:`score_op_stream`), for the case where only ops + a brief are in hand.

Two exact routes to the same signal, then:

1. **Verdict route** (a compiled trajectory): the per-step oracle already decided each
   step. First divergence = the first non-VERIFIED step. Cheap; no re-measurement.
2. **Geometric route** (a bare op stream): defer to ``selftrain.divergence.analyse``,
   which measures material recoverability per prefix. Slower; needed when no verdict
   track exists yet.

Deterministic, import-safe, no model. The heavy geometric imports live inside
``selftrain.divergence`` and fire only when :func:`score_op_stream` is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

# Compose the existing, validated machinery -- do not duplicate it.
from harnesscad.agents.selftrain.divergence import (
    DivergenceReport,
    analyse,
    step_rewards,
)
from harnesscad.agents.selftrain.trajectory import StepReward
from harnesscad.agents.cua.verified_trajectory import (
    VERIFIED,
    TrajectoryStep,
    VerifiedTrajectory,
)

__all__ = [
    "DivergenceReport",
    "StepReward",
    "analyse",
    "step_rewards",
    "first_divergence_from_trajectory",
    "step_rewards_from_trajectory",
    "score_op_stream",
    "TrajectoryReward",
]


def first_divergence_from_trajectory(traj: VerifiedTrajectory) -> Optional[int]:
    """The first step index the oracle did NOT verify, or ``None`` if all verified.

    Because the compiler's per-step labeller rejects a step the moment the prefix
    stops building (and rejects every step after it), the first non-VERIFIED step IS
    the first divergence -- exact, read straight off the verdict track, no geometry
    re-computed. A trajectory whose steps are all VERIFIED has no divergence.
    """
    for step in traj.steps:
        if step.verdict.label != VERIFIED:
            return step.index
    return None


def _op_tag_of(step: TrajectoryStep) -> str:
    """The op tag carried by a compiled step's action (``action_for_op`` put it there)."""
    verb = step.action.get("verb") if isinstance(step.action, dict) else None
    return str(verb or "?")


def step_rewards_from_trajectory(traj: VerifiedTrajectory) -> List[StepReward]:
    """The book's per-op reward assignment (sec. 12.8.3), sourced from the VERDICTS.

    Mirrors :func:`selftrain.divergence.step_rewards` exactly -- +1 to every step in
    the verified prefix, **-1 to the first divergent step and to it alone**, 0 to the
    doomed tail -- but reads the decision from the trajectory's oracle verdicts instead
    of re-measuring geometry. Same :class:`StepReward` record, so the two routes feed
    the same downstream aggregator.
    """
    d = first_divergence_from_trajectory(traj)
    out: List[StepReward] = []
    for step in traj.steps:
        k = step.index
        applied = step.verdict.label == VERIFIED
        if d is None or k < d:
            reward, divergent = 1.0, False
        elif k == d:
            reward, divergent = -1.0, True
        else:
            reward, divergent = 0.0, False
        out.append(StepReward(
            index=k, op=_op_tag_of(step), applied=applied,
            reward=reward, divergent=divergent,
            detail=step.verdict.detail,
        ))
    return out


def score_op_stream(brief: Any, ops: Sequence[dict], **kw: Any
                    ) -> Tuple[DivergenceReport, List[StepReward]]:
    """Exact per-step reward for a RAW op stream via the geometric detector.

    Composes :func:`selftrain.divergence.analyse` (first-divergence by material
    recoverability) with :func:`selftrain.divergence.step_rewards` (the reward
    assignment). Use this when only ``(brief, ops)`` are in hand and no verdict track
    exists yet; ``kw`` (``samples``, ``seed``, ``tol``) are forwarded to ``analyse``.
    Runs the geometric backend when called; nothing runs at import.
    """
    report = analyse(brief, ops, **kw)
    rewards = step_rewards(report, ops)
    return report, rewards


@dataclass(frozen=True)
class TrajectoryReward:
    """The aggregate per-trajectory view: the step rewards plus their summary.

    ``first_divergence`` is the exact divergent op index (``None`` if none). ``prefix``
    / ``divergent`` / ``tail`` count the three reward bands. ``mean`` is the average
    step reward -- the process-reward term an aggregate reward would combine with the
    outcome term.
    """

    step_rewards: List[StepReward]
    first_divergence: Optional[int]

    @property
    def mean(self) -> float:
        if not self.step_rewards:
            return 0.0
        return sum(s.reward for s in self.step_rewards) / len(self.step_rewards)

    @property
    def prefix(self) -> int:
        return sum(1 for s in self.step_rewards if s.reward > 0.0)

    @property
    def divergent(self) -> int:
        return sum(1 for s in self.step_rewards if s.divergent)

    @property
    def tail(self) -> int:
        return sum(1 for s in self.step_rewards
                   if s.reward == 0.0 and not s.divergent)

    def to_dict(self) -> dict:
        return {
            "first_divergence": self.first_divergence,
            "mean": round(self.mean, 4),
            "prefix": self.prefix,
            "divergent": self.divergent,
            "tail": self.tail,
            "step_rewards": [s.to_dict() for s in self.step_rewards],
        }

    @classmethod
    def from_trajectory(cls, traj: VerifiedTrajectory) -> "TrajectoryReward":
        """Build from a compiled trajectory (the verdict route)."""
        rewards = step_rewards_from_trajectory(traj)
        return cls(step_rewards=rewards,
                   first_divergence=first_divergence_from_trajectory(traj))
