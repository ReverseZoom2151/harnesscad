"""ReCAD hard-question identification and RL objective routing (ReCAD, AAAI 2026).

Before RL training, ReCAD partitions the training set into *easy* and *hard*
questions and routes each to a different GRPO objective (Eq. 11-12):

  * For each question ``q`` it samples ``N`` solutions with the current policy,
    scores each, and takes the **maximum** reward over the group.

  * ``1_hard(q) = 1`` iff ``max{ R(q_i) } < tau_h`` (paper: ``tau_h = 0.8``),
    else 0.  Intuitively a question is "hard" when even the best of ``N`` rollouts
    still fails to reach the reward threshold -- the policy cannot solve it on its
    own.

  * The final training objective (Eq. 11) is a per-question mixture::

        L_RL(q) = 1_hard(q) * J_guided(q; C)  +  (1 - 1_hard(q)) * J_grpo(q)

    Hard questions use the *learn-under-guidance* objective ``J_guided`` (which
    injects off-policy parameterized code as in-context guidance); easy questions
    use standard GRPO ``J``.

This module provides the deterministic *control logic* -- the max-reward
statistic, the hard indicator, and the objective selector -- not the GRPO
gradient itself (see ``dataengine.cadrille_drcppo`` / ``dataengine.cmecad_advantage``
for group-normalized advantages).  This routing-by-max-reward is distinct from
``cmecad_hardneg_buffer`` (which buffers hard *negatives* for contrastive
sampling) and from ``creft_rewards`` difficulty weighting (a per-sample scalar,
not a policy-solvability gate).  Pure stdlib, deterministic.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

DEFAULT_TAU_H = 0.8

# Objective labels returned by the selector.
OBJ_GUIDED = "guided"   # J_guided: learn-under-guidance (hard questions)
OBJ_GRPO = "grpo"       # J: standard GRPO (easy questions)


def max_group_reward(rewards: Sequence[float]) -> float:
    """``max{ R(q_i) }`` over a non-empty group of sampled-solution rewards."""
    vals = [float(r) for r in rewards]
    if not vals:
        raise ValueError("reward group must be non-empty")
    return max(vals)


def is_hard(rewards: Sequence[float], tau_h: float = DEFAULT_TAU_H) -> bool:
    """``1_hard(q)`` (Eq. 12): True iff the group's max reward is below ``tau_h``.

    A question is hard when the best of the sampled rollouts still fails to reach
    the threshold, i.e. the policy cannot reliably solve it unaided.
    """
    return max_group_reward(rewards) < float(tau_h)


def select_objective(rewards: Sequence[float],
                     tau_h: float = DEFAULT_TAU_H) -> str:
    """Route a question to :data:`OBJ_GUIDED` (hard) or :data:`OBJ_GRPO` (easy)."""
    return OBJ_GUIDED if is_hard(rewards, tau_h) else OBJ_GRPO


def objective_value(rewards: Sequence[float],
                    j_guided: float,
                    j_grpo: float,
                    tau_h: float = DEFAULT_TAU_H) -> float:
    """Per-question mixed objective (Eq. 11).

    Returns ``j_guided`` when the question is hard and ``j_grpo`` otherwise --
    the deterministic realization of ``1_hard*J_guided + (1-1_hard)*J``.
    """
    return float(j_guided) if is_hard(rewards, tau_h) else float(j_grpo)


def partition_questions(
    question_rewards: Mapping[object, Sequence[float]],
    tau_h: float = DEFAULT_TAU_H,
) -> dict:
    """Split a mapping ``question -> reward group`` into hard/easy id lists.

    Returns ``{"hard": [...], "easy": [...]}`` with question keys in the input's
    iteration order, so the partition is deterministic for an ordered mapping.
    """
    hard, easy = [], []
    for key, rewards in question_rewards.items():
        (hard if is_hard(rewards, tau_h) else easy).append(key)
    return {"hard": hard, "easy": easy}


def identify_hard_questions(
    questions: Iterable[object],
    sampler: Callable[[object], Sequence[float]],
    tau_h: float = DEFAULT_TAU_H,
) -> dict:
    """End-to-end identification: sample rewards per question, then partition.

    ``sampler(q)`` must return the ``N`` solution rewards for question ``q``
    (the ``N``-sample rollout is external / injected).  Deterministic given a
    deterministic ``sampler`` and an ordered ``questions`` iterable.
    """
    return partition_questions({q: sampler(q) for q in questions}, tau_h)
