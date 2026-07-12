"""Gated, weighted multi-component composite reward aggregator.

Ported from SpatialHero (``core/reward_model.py``: ``RewardModel`` weighting
logic), a multi-modal text-to-CAD reward system. The reusable, model-free core
is the *aggregation rule*, which two properties make distinct from the harness's
existing ``quality/cad_reward.py`` (a fixed two-term geometric-chamfer + format
reward):

  1. **Arbitrary named components** combined by a validated weight map that must
     sum to 1.0 (within a small epsilon); and
  2. **Hard gates** -- a set of components (e.g. ``code_valid``,
     ``execution_valid``) that must each clear a threshold, or the whole reward
     collapses to 0.0 regardless of the other terms. This encodes "code that
     does not run earns nothing" without hard-coding which terms gate.

Everything here is stdlib-only, deterministic, and independent of any LLM or
CAD kernel -- callers supply already-computed component scores in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

# SpatialHero's default component weights (must sum to 1.0).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "code_valid": 0.20,
    "dimension_accuracy": 0.30,
    "visual_quality": 0.30,
    "topology_valid": 0.20,
}

# Components that gate the reward: each must be >= GATE_THRESHOLD or reward is 0.
DEFAULT_GATE_KEYS = ("code_valid", "execution_valid")
GATE_THRESHOLD = 0.5
WEIGHT_SUM_EPS = 0.01


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@dataclass(frozen=True)
class CompositeRewardResult:
    """Outcome of a composite-reward computation."""

    total: float
    gated_out: bool          # True when a gate failed and forced total to 0
    contributions: Dict[str, float] = field(default_factory=dict)


class CompositeReward:
    """Weighted sum of named component scores with hard gating.

    Args:
        weights: component-name -> weight; must sum to 1.0 (+/- ``eps``).
        gate_keys: components that must each clear ``gate_threshold`` for the
            reward to be non-zero. Gate components need not carry weight (e.g.
            ``execution_valid`` gates but may not appear in ``weights``).
        gate_threshold: minimum value a gate component must reach.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        gate_keys: Iterable[str] = DEFAULT_GATE_KEYS,
        gate_threshold: float = GATE_THRESHOLD,
        eps: float = WEIGHT_SUM_EPS,
    ):
        self.weights = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)
        s = sum(self.weights.values())
        if abs(s - 1.0) > eps:
            raise ValueError(f"Weights must sum to 1.0, got {s}")
        self.gate_keys = tuple(gate_keys)
        self.gate_threshold = gate_threshold

    def gates_pass(self, components: Dict[str, float]) -> bool:
        """True when every gate component is present and >= the threshold.

        A gate component that is absent counts as failing (0.0 < threshold).
        """
        for key in self.gate_keys:
            if components.get(key, 0.0) < self.gate_threshold:
                return False
        return True

    def compute(self, components: Dict[str, float]) -> CompositeRewardResult:
        """Compute the gated, weighted reward for a component-score dict.

        Component values are clipped to ``[0, 1]`` before weighting. Only keys
        present in ``weights`` contribute to the weighted sum; a weighted key
        missing from ``components`` contributes 0. If any gate fails, ``total``
        is 0.0 and ``gated_out`` is True (contributions are still reported).
        """
        contributions: Dict[str, float] = {}
        total = 0.0
        for key, w in self.weights.items():
            val = _clip01(float(components.get(key, 0.0)))
            contrib = w * val
            contributions[key] = contrib
            total += contrib

        total = _clip01(total)

        if not self.gates_pass(components):
            return CompositeRewardResult(
                total=0.0, gated_out=True, contributions=contributions
            )
        return CompositeRewardResult(
            total=total, gated_out=False, contributions=contributions
        )


def composite_reward(
    components: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    gate_keys: Iterable[str] = DEFAULT_GATE_KEYS,
    gate_threshold: float = GATE_THRESHOLD,
) -> float:
    """Functional shortcut returning just the scalar total reward."""
    return CompositeReward(
        weights=weights, gate_keys=gate_keys, gate_threshold=gate_threshold
    ).compute(components).total
