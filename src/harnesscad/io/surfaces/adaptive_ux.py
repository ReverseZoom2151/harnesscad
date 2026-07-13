"""Deterministic adaptive-UX proficiency estimation and interface policy.

Paper 179 -- "Toward AI-driven Multimodal Interfaces for Industrial CAD
Modeling" (Choi, Jang, Hyun, CHI '25), section 4.2 ("Multimodal Interaction
for Adaptive CAD UX") -- argues that CAD interfaces should adapt to user
proficiency: "experienced designers prefer command-driven workflows, novice
users benefit from visual and natural input methods.  Adaptive UX strategies
help bridge this gap by dynamically adjusting interfaces based on user
proficiency, project complexity, and design intent."

The paper gives the goal, not the mechanism.  This module supplies a
deterministic, learned-model-free mechanism:

* :class:`ProficiencyEstimator` -- maps observable, wall-clock-free
  interaction statistics (commands issued, distinct commands used, errors,
  undos, help requests) to a bounded proficiency score and a tier.
* :func:`recommend_ux` -- a fixed policy mapping (tier, project complexity)
  to a :class:`UxProfile`: which input modalities to surface, explanation
  verbosity, and whether to auto-suggest the next command.
* :class:`ProficiencyStateMachine` -- promotes/demotes the active tier from a
  running feed of interaction stats using hysteresis so the interface does
  not flip-flop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from harnesscad.io.surfaces.modality_fusion import ModalityKind


class ProficiencyTier(str, Enum):
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class Verbosity(str, Enum):
    HIGH = "high"      # full AI explanations, step-by-step guidance
    MEDIUM = "medium"
    LOW = "low"        # terse, expert-facing


@dataclass(frozen=True)
class InteractionStats:
    """Observable counters for one user, free of any wall clock.

    All fields are cumulative counts over a window the caller controls.
    """

    commands_issued: int = 0
    distinct_commands: int = 0
    errors: int = 0          # rejected / malformed commands
    undos: int = 0
    help_requests: int = 0

    def __post_init__(self) -> None:
        for name in (
            "commands_issued", "distinct_commands", "errors",
            "undos", "help_requests",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.distinct_commands > self.commands_issued:
            raise ValueError("distinct_commands cannot exceed commands_issued")


@dataclass(frozen=True)
class UxProfile:
    tier: ProficiencyTier
    primary_modalities: tuple[ModalityKind, ...]
    verbosity: Verbosity
    autosuggest: bool
    show_command_palette: bool


def proficiency_score(stats: InteractionStats) -> float:
    """A bounded score in ``[0, 1]`` (0 = novice-like, 1 = expert-like).

    Deterministic heuristic combining, per issued command:

    * command *fluency* -- more distinct commands relative to total signals a
      broader mastered vocabulary (positive),
    * friction -- errors, undos, and help requests per command drag the score
      down.

    With zero activity the score is 0.0 (treat unknown users as novices).
    """
    n = stats.commands_issued
    if n == 0:
        return 0.0
    fluency = stats.distinct_commands / n            # in [0, 1]
    friction = (stats.errors + stats.undos + stats.help_requests) / n
    raw = fluency - friction
    # clamp to [0, 1]
    return max(0.0, min(1.0, raw))


# Tier thresholds on the score. Kept as module constants so the state machine
# and the one-shot estimator agree.
_INTERMEDIATE_AT = 0.4
_EXPERT_AT = 0.7


def score_to_tier(score: float) -> ProficiencyTier:
    if score >= _EXPERT_AT:
        return ProficiencyTier.EXPERT
    if score >= _INTERMEDIATE_AT:
        return ProficiencyTier.INTERMEDIATE
    return ProficiencyTier.NOVICE


class ProficiencyEstimator:
    """One-shot mapping from :class:`InteractionStats` to a tier."""

    def score(self, stats: InteractionStats) -> float:
        return proficiency_score(stats)

    def tier(self, stats: InteractionStats) -> ProficiencyTier:
        return score_to_tier(self.score(stats))


# Fixed per-tier interface policy (paper section 4.2). Novices get the
# visual/natural channels foregrounded and verbose AI help; experts get the
# command palette and terse output.
_TIER_POLICY: dict[ProficiencyTier, tuple[tuple[ModalityKind, ...], Verbosity, bool, bool]] = {
    ProficiencyTier.NOVICE: (
        (ModalityKind.SKETCH, ModalityKind.VOICE, ModalityKind.GESTURE),
        Verbosity.HIGH, True, False,
    ),
    ProficiencyTier.INTERMEDIATE: (
        (ModalityKind.VOICE, ModalityKind.KEYBOARD, ModalityKind.SKETCH),
        Verbosity.MEDIUM, True, True,
    ),
    ProficiencyTier.EXPERT: (
        (ModalityKind.KEYBOARD, ModalityKind.VOICE),
        Verbosity.LOW, False, True,
    ),
}


def recommend_ux(
    tier: ProficiencyTier, *, project_complexity: int = 0
) -> UxProfile:
    """Fixed policy from tier + project complexity to a :class:`UxProfile`.

    ``project_complexity`` is a non-negative feature count. High-complexity
    projects re-enable auto-suggestion even for experts (the paper's
    "project complexity" adaptation axis), since a large workflow benefits
    from Bayesian next-command hints regardless of skill.
    """
    if project_complexity < 0:
        raise ValueError("project_complexity must be non-negative")
    modalities, verbosity, autosuggest, palette = _TIER_POLICY[tier]
    if project_complexity >= 25:
        autosuggest = True
    return UxProfile(
        tier=tier,
        primary_modalities=modalities,
        verbosity=verbosity,
        autosuggest=autosuggest,
        show_command_palette=palette,
    )


@dataclass
class ProficiencyStateMachine:
    """Stateful tier tracker with hysteresis to avoid flip-flopping.

    A tier only changes when the freshly estimated tier is confirmed for
    ``patience`` consecutive updates. Promotions and demotions both require
    confirmation, so a single fluky window will not move the interface.
    """

    tier: ProficiencyTier = ProficiencyTier.NOVICE
    patience: int = 2
    _pending: Optional[ProficiencyTier] = field(default=None, init=False)
    _streak: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be at least 1")

    def update(self, stats: InteractionStats) -> ProficiencyTier:
        observed = score_to_tier(proficiency_score(stats))
        if observed == self.tier:
            self._pending = None
            self._streak = 0
            return self.tier
        if observed == self._pending:
            self._streak += 1
        else:
            self._pending = observed
            self._streak = 1
        if self._streak >= self.patience:
            self.tier = observed
            self._pending = None
            self._streak = 0
        return self.tier

    def profile(self, *, project_complexity: int = 0) -> UxProfile:
        return recommend_ux(self.tier, project_complexity=project_complexity)
