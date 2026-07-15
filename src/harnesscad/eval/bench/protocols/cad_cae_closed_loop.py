"""Closed-loop CAD-CAE result extraction, cost, and multi-constraint reward.

Paper: *Tool-Augmented Agent for Closed-loop Optimization, Simulation, and
Modeling Orchestration* (COSMO-Agent, Deng et al., 2026). The trained policy is
out of scope, but the paper's Tools 3-4 and reward function are fully
deterministic closed-form formulas (Sec. 3.2-3.3):

  * **Result extractor (Tool 3)** -- reduce CAE field outputs to the scalar
    metrics used for constraint checking:
      - displacement magnitude ``||u_i|| = sqrt(ux^2+uy^2+uz^2)`` and its max;
      - von Mises equivalent stress from the tensor ``(sxx,syy,szz,txy,tyz,tzx)``
        via ``sqrt(0.5*Ds + 3*Dt)`` (eq. 8-10) and its max.
  * **Cost calculator (Tool 4)** -- ``C = rho * V * price`` (eq. 11-12).
  * **Feasibility + reward (Sec. 3.3)** -- three constraints (displacement,
    stress, cost); the count ``N`` of satisfied constraints maps to a piecewise
    ``Rcons`` (0/0.2/0.5/1.0 for N=0..3); a *feasible-then-stop* penalty
    ``-min(lambda*K, lambda_max)`` for tool events after first feasibility; and a
    structured-output ``Rfmt`` bonus. The total is ``Rcons + Rstop + Rfmt``.

Everything is deterministic and stdlib-only. It complements
:mod:`harnesscad.eval.verifiers.simulation` (which holds *analytical* beam/
pressure stress formulas and consumes already-scalar FEA outputs): here we
provide the tensor/field -> scalar reduction and the closed-loop reward those
verifiers do not, matching the COSMO-Agent toolchain exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "displacement_magnitude",
    "max_displacement",
    "von_mises",
    "max_von_mises",
    "material_cost",
    "Feasibility",
    "evaluate_feasibility",
    "constraint_reward",
    "feasible_then_stop_penalty",
    "format_reward",
    "total_reward",
]


# --- Tool 3: result extractor ----------------------------------------------

def displacement_magnitude(u) -> float:
    """Euclidean magnitude of a nodal displacement vector ``(ux, uy, uz)``."""
    return math.sqrt(sum(c * c for c in u))


def max_displacement(nodal_u) -> float:
    """Max displacement magnitude over a sequence of nodal vectors (0 if empty)."""
    return max((displacement_magnitude(u) for u in nodal_u), default=0.0)


def von_mises(stress) -> float:
    """Von Mises equivalent stress from ``(sxx, syy, szz, txy, tyz, tzx)``.

    ``sqrt(0.5*[(sxx-syy)^2+(syy-szz)^2+(szz-sxx)^2] + 3*(txy^2+tyz^2+tzx^2))``
    (eq. 8-10). Raises ``ValueError`` if not six components.
    """
    if len(stress) != 6:
        raise ValueError("stress must have 6 components (sxx,syy,szz,txy,tyz,tzx)")
    sxx, syy, szz, txy, tyz, tzx = stress
    d_sigma = (sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
    d_tau = txy * txy + tyz * tyz + tzx * tzx
    return math.sqrt(0.5 * d_sigma + 3.0 * d_tau)


def max_von_mises(nodal_stresses) -> float:
    """Max von Mises stress over a sequence of stress tensors (0 if empty)."""
    return max((von_mises(s) for s in nodal_stresses), default=0.0)


# --- Tool 4: cost calculator -----------------------------------------------

def material_cost(volume_m3: float, density: float, unit_price: float) -> float:
    """Cost via the volume->mass->price chain: ``rho * V * price`` (eq. 11-12)."""
    return density * volume_m3 * unit_price


# --- feasibility + reward (Sec. 3.3) ---------------------------------------

@dataclass
class Feasibility:
    """Per-constraint satisfaction plus the satisfied count ``N``."""

    displacement_ok: bool
    stress_ok: bool
    cost_ok: bool

    @property
    def n_satisfied(self) -> int:
        return int(self.displacement_ok) + int(self.stress_ok) + int(self.cost_ok)

    @property
    def feasible(self) -> bool:
        return self.n_satisfied == 3


def evaluate_feasibility(u_max: float, sigma_max: float, cost: float,
                         delta: float, sigma_allow: float,
                         kappa: float) -> Feasibility:
    """Check the three COSMO constraints: u<=delta, sigma<=sigma_allow, C<=kappa."""
    return Feasibility(
        displacement_ok=u_max <= delta,
        stress_ok=sigma_max <= sigma_allow,
        cost_ok=cost <= kappa,
    )


_RCONS = {0: 0.0, 1: 0.2, 2: 0.5, 3: 1.0}


def constraint_reward(n_satisfied: int) -> float:
    """Piecewise constraint reward Rcons: 0/0.2/0.5/1.0 for N=0..3 (eq. 15)."""
    if n_satisfied not in _RCONS:
        raise ValueError("n_satisfied must be 0..3")
    return _RCONS[n_satisfied]


def feasible_then_stop_penalty(tool_events_after_feasible: int,
                               lam: float = 0.02,
                               lam_max: float = 0.10) -> float:
    """Rstop = -min(lambda*K, lambda_max) for K tool events after first feasibility.

    ``tool_events_after_feasible`` is 0 (or negative -> clamped) when the policy
    stops immediately after becoming feasible, giving no penalty (eq. 16). If it
    never becomes feasible the caller should pass 0.
    """
    k = max(0, tool_events_after_feasible)
    return -min(lam * k, lam_max)


def format_reward(json_consistent: bool, bonus: float = 0.10) -> float:
    """Rfmt: a bonus if the final JSON's category/material/params are consistent."""
    return bonus if json_consistent else 0.0


def total_reward(n_satisfied: int, tool_events_after_feasible: int,
                 json_consistent: bool, *, feasible_reached: bool = True,
                 lam: float = 0.02, lam_max: float = 0.10,
                 fmt_bonus: float = 0.10) -> float:
    """Combined reward ``Rcons + Rstop + Rfmt`` (eq. 13).

    ``Rstop`` applies only if a feasible triple was reached (``feasible_reached``);
    otherwise it is 0, matching the paper's rule that Rstop=0 when no complete
    feasible triple appears in the trajectory.
    """
    r = constraint_reward(n_satisfied)
    if feasible_reached:
        r += feasible_then_stop_penalty(tool_events_after_feasible, lam, lam_max)
    r += format_reward(json_consistent, fmt_bonus)
    return r
