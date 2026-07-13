"""CReFT-CAD curriculum reward functions and difficulty-aware attribute rewards.

CReFT-CAD (NeurIPS 2025) trains a VLM with GRPO using three curriculum tasks,
each with a hand-designed, programmatically computable reward. The *training*
loop and the VLM are out of scope (research-heavy/external), but the reward
functions themselves are deterministic verifiers over a candidate answer and the
ground truth — exactly the kind of piece we can build and test. This module
implements all three, matching the paper's equations, plus the difficulty
classification used to weight Task 3.

  * Task 1 — Dichotomous choice (Eq. 1)::

        R_P1 = 1 if every parameter is correct else 0

  * Task 2 — Multiple choice (Eq. 2), set-based over selected value lists::

        R_P2 = 1   if S_selected == S_correct
               0.2 if S_selected ⊆ S_correct but S_selected != S_correct
               0   if S_selected mixes correct and incorrect lists
               0   if S_selected == S_incorrect

  * Task 3 — Parameterization (Eq. 3), difficulty-aware per-attribute reward::

        R_P3 += 1   for each correct EASY attribute
                1.5 for each correct MEDIUM attribute
                2   for each correct HARD attribute
                0   for an incorrect attribute

Difficulty is derived from held-out accuracy per attribute: EASY > 0.8,
0.2 <= MEDIUM <= 0.8, HARD < 0.2.

Pure stdlib, deterministic (parallels :mod:`dataengine.cadrille_reward`).
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, Mapping, Sequence, Set, Tuple

EASY = "easy"
MEDIUM = "medium"
HARD = "hard"

# Difficulty thresholds and their reward weights (Eq. 3).
EASY_THRESHOLD = 0.8
HARD_THRESHOLD = 0.2
DIFFICULTY_REWARD = {EASY: 1.0, MEDIUM: 1.5, HARD: 2.0}


# --------------------------------------------------------------------------- #
# Task 1 — Dichotomous choice
# --------------------------------------------------------------------------- #
def all_parameters_correct(predicted: Mapping[str, object],
                           ground_truth: Mapping[str, object]) -> bool:
    """True iff ``predicted`` matches ``ground_truth`` on every ground-truth key."""
    for key, value in ground_truth.items():
        if key not in predicted or predicted[key] != value:
            return False
    return True


def reward_p1(predicted: Mapping[str, object],
              ground_truth: Mapping[str, object]) -> float:
    """Task 1 reward: 1.0 iff all parameters correct, else 0.0 (Eq. 1)."""
    return 1.0 if all_parameters_correct(predicted, ground_truth) else 0.0


# --------------------------------------------------------------------------- #
# Task 2 — Multiple choice (set-based)
# --------------------------------------------------------------------------- #
def reward_p2(selected: Iterable[Hashable],
              correct: Iterable[Hashable],
              incorrect: Iterable[Hashable]) -> float:
    """Task 2 reward over selected parameter-value lists (Eq. 2).

    ``selected``/``correct``/``incorrect`` are iterables of hashable identifiers
    for candidate value lists. Returns 1.0 / 0.2 / 0.0 per the four cases.
    """
    s_sel: Set[Hashable] = set(selected)
    s_cor: Set[Hashable] = set(correct)
    s_inc: Set[Hashable] = set(incorrect)
    if s_sel == s_cor:
        return 1.0
    # Partial credit: chose only correct lists but missed some.
    if s_sel and s_sel <= s_cor:
        return 0.2
    # Any selection touching an incorrect list, or an empty/other selection: 0.
    return 0.0


# --------------------------------------------------------------------------- #
# Task 3 — Difficulty-aware parameterization
# --------------------------------------------------------------------------- #
def classify_difficulty(accuracy: float) -> str:
    """Map a held-out per-attribute accuracy to EASY / MEDIUM / HARD.

    EASY > 0.8, HARD < 0.2, MEDIUM in between (inclusive of the 0.2 and 0.8 ends).
    """
    a = float(accuracy)
    if not (0.0 <= a <= 1.0):
        raise ValueError("accuracy must be in [0, 1], got %r" % (accuracy,))
    if a > EASY_THRESHOLD:
        return EASY
    if a < HARD_THRESHOLD:
        return HARD
    return MEDIUM


def difficulty_map(accuracies: Mapping[str, float]) -> Dict[str, str]:
    """Classify every attribute's accuracy into a difficulty label."""
    return {name: classify_difficulty(acc) for name, acc in accuracies.items()}


def attribute_reward(difficulty: str) -> float:
    """Reward weight for a correctly predicted attribute of a given difficulty."""
    if difficulty not in DIFFICULTY_REWARD:
        raise ValueError("unknown difficulty %r" % (difficulty,))
    return DIFFICULTY_REWARD[difficulty]


def reward_p3(predicted: Mapping[str, object],
              ground_truth: Mapping[str, object],
              difficulties: Mapping[str, str]) -> float:
    """Task 3 reward: sum difficulty-weighted credit over correct attributes (Eq. 3).

    An attribute contributes its difficulty weight when predicted correctly and 0
    otherwise. Attributes without a difficulty label default to EASY (weight 1).
    """
    total = 0.0
    for key, value in ground_truth.items():
        if key in predicted and predicted[key] == value:
            diff = difficulties.get(key, EASY)
            total += attribute_reward(diff)
    return total


def reward_p3_from_accuracies(predicted: Mapping[str, object],
                              ground_truth: Mapping[str, object],
                              accuracies: Mapping[str, float]) -> float:
    """Task 3 reward deriving each attribute's difficulty from its accuracy."""
    return reward_p3(predicted, ground_truth, difficulty_map(accuracies))


# --------------------------------------------------------------------------- #
# Curriculum aggregation
# --------------------------------------------------------------------------- #
def curriculum_reward(predicted: Mapping[str, object],
                      ground_truth: Mapping[str, object],
                      difficulties: Mapping[str, str],
                      choice_selected: Sequence[Hashable] = (),
                      choice_correct: Sequence[Hashable] = (),
                      choice_incorrect: Sequence[Hashable] = ()) -> Dict[str, float]:
    """Compute all three task rewards for one sample and return them keyed.

    ``choice_*`` supply Task 2's set arguments; when omitted Task 2 scores 0.
    """
    return {
        "p1": reward_p1(predicted, ground_truth),
        "p2": reward_p2(choice_selected, choice_correct, choice_incorrect),
        "p3": reward_p3(predicted, ground_truth, difficulties),
    }
