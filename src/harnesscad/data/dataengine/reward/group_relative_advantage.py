"""GRPO group-relative advantage and CAD reward composition (Guan et al., 2026,
"CAD-Coder: Text-to-CAD Generation with Chain-of-Thought and Geometric Reward").

CAD-Coder fine-tunes with Group Reward Policy Optimization (GRPO) using a
CAD-specific reward that combines a *geometric reward* (from Chamfer Distance) and
a *format reward*. The policy-gradient training is out of scope and non-
deterministic, but the two deterministic numeric cores are buildable and testable:

* **Reward composition.** :func:`cad_reward` maps a Chamfer Distance to a bounded
  geometric reward ``1 / (1 + CD)`` (1 when CD=0, decaying to 0), and combines it
  with a binary format reward (did the CadQuery script parse/execute) via a
  weighted sum. An unparseable script scores 0 geometric reward regardless of CD.

* **GRPO group-relative advantage.** For a *group* of ``G`` sampled completions
  to one prompt with rewards ``r_i``, GRPO normalises within the group::

      A_i = (r_i - mean(r)) / (std(r) + eps)

  removing the need for a value network. :func:`group_relative_advantage` returns
  the per-sample advantages; a degenerate group (all equal rewards) yields all-
  zero advantages.

Deterministic, stdlib-only.
"""

from __future__ import annotations

from statistics import fmean, pstdev
from typing import List, Sequence

__all__ = ["geometric_reward", "cad_reward", "group_relative_advantage"]


def geometric_reward(chamfer_distance: float) -> float:
    """Bounded geometric reward ``1 / (1 + CD)`` in ``(0, 1]`` (1 at CD=0)."""
    cd = float(chamfer_distance)
    if cd < 0:
        raise ValueError("chamfer_distance must be non-negative")
    return 1.0 / (1.0 + cd)


def cad_reward(
    chamfer_distance: float,
    executes: bool,
    w_geometric: float = 0.8,
    w_format: float = 0.2,
) -> float:
    """Combined CAD reward = ``w_geometric * geo + w_format * fmt``.

    ``fmt`` is 1.0 if the script executes else 0.0. A non-executing script gets
    zero geometric reward too (no valid geometry to compare), so its total reward
    is exactly 0. Weights must be non-negative.
    """
    if w_geometric < 0 or w_format < 0:
        raise ValueError("weights must be non-negative")
    if not executes:
        return 0.0
    geo = geometric_reward(chamfer_distance)
    return w_geometric * geo + w_format * 1.0


def group_relative_advantage(
    rewards: Sequence[float], eps: float = 1e-8
) -> List[float]:
    """GRPO per-sample advantages within a group: ``(r_i - mean) / (std + eps)``.

    ``std`` is the population standard deviation over the group. A group with
    fewer than two samples, or with zero variance, returns all zeros (no relative
    signal). Order of the returned advantages matches ``rewards``.
    """
    rs = [float(r) for r in rewards]
    if len(rs) < 2:
        return [0.0] * len(rs)
    mu = fmean(rs)
    sigma = pstdev(rs)
    if sigma == 0.0:
        return [0.0] * len(rs)
    return [(r - mu) / (sigma + eps) for r in rs]
