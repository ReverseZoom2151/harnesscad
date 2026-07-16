"""Fault-parameter identification: sensitivity analysis + analytic estimators.

Mined from **SimCorrect** (``Problem1/parameter_identifier.py`` for the
sensitivity method; ``Problem4`` and ``Problem5`` identifiers for the analytic
estimators). Once a fault is *detected* and *classified*, the remaining
question is which named parameter is wrong and by how much. SimCorrect answers
with two fully classical techniques this module ports:

**Sensitivity analysis with cosine matching** (geometry faults with several
candidates). For each candidate parameter, perturb it by a small fraction,
re-run the same trajectory through an injected simulator callable, and record
the per-joint mean absolute response divided by the perturbation -- the
sensitivity vector. The candidate whose *direction* (cosine similarity) best
matches the observed divergence direction is the culprit, and the magnitude
follows from a one-dimensional least squares projection:

    delta = <s, d> / <s, s>

so the proposed value is ``current + delta``. The simulator is a callable of
``params -> joint trajectory``; nothing in this module knows what runs it.

**Closed-form analytic estimators** for the fault classes whose physics is a
one-line formula:

  * tool-mass mismatch from gravitational sag at two reaches
    (``delta_m = sag * kp / (g * reach)``, averaged, with the 2:1 sag-scaling
    consistency check);
  * encoder zero offset from the Cartesian miss at two reaches
    (``theta = asin(miss / reach)``, averaged);
  * lateral mount offset from drift divided by axis sensitivity.

stdlib-only, deterministic. The identification result dicts are shaped to feed
``harnesscad.domain.spec.caid_artifact.make_patch_from_identification``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "SensitivityResult",
    "identify_by_sensitivity",
    "estimate_mass_from_sag",
    "estimate_zero_offset_from_miss",
    "estimate_lateral_offset",
    "main",
]

Trajectory = Sequence[Sequence[float]]
Simulator = Callable[[Mapping[str, float]], Trajectory]

GRAVITY = 9.81
DEFAULT_PERTURBATION_FRACTION = 0.05
REACH_RATIO_TARGET = 2.0


# --------------------------------------------------------------------------- #
# Sensitivity analysis
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SensitivityResult:
    """Identification verdict shaped for the CAID patch constructor."""

    identified_parameter: str
    confidence: float                    # cosine similarity of best candidate
    current_value: float
    estimated_delta: float
    proposed_value: float
    all_scores: Dict[str, float]
    method: str = "sensitivity_analysis_cosine_similarity"

    def to_dict(self) -> Dict[str, object]:
        return {
            "identified_parameter": self.identified_parameter,
            "confidence": round(self.confidence, 4),
            "current_value": round(self.current_value, 6),
            "estimated_delta": round(self.estimated_delta, 6),
            "proposed_value": round(self.proposed_value, 6),
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
            "method": self.method,
        }


def _mean_abs_columns(a: Trajectory, b: Trajectory) -> List[float]:
    if len(a) != len(b) or not a:
        raise ValueError("trajectories must be equal-length and non-empty")
    width = len(a[0])
    sums = [0.0] * width
    for ra, rb in zip(a, b):
        if len(ra) != width or len(rb) != width:
            raise ValueError("trajectory rows have inconsistent widths")
        for j in range(width):
            sums[j] += abs(ra[j] - rb[j])
    return [s / len(a) for s in sums]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-10)


def identify_by_sensitivity(
    observed_reference: Trajectory,
    observed_suspect: Trajectory,
    candidates: Mapping[str, float],
    simulate: Simulator,
    *,
    perturbation_fraction: float = DEFAULT_PERTURBATION_FRACTION,
) -> SensitivityResult:
    """Which candidate parameter explains the observed joint divergence.

    ``observed_reference`` / ``observed_suspect`` are the paired joint
    trajectories whose divergence is being explained. ``candidates`` maps
    parameter names to their nominal values; ``simulate`` runs the same
    trajectory for any parameter dict. One base run plus one perturbed run per
    candidate: ``len(candidates) + 1`` simulator calls total.
    """
    if not candidates:
        raise ValueError("at least one candidate parameter is required")
    observed = _mean_abs_columns(observed_reference, observed_suspect)
    obs_norm = math.sqrt(sum(x * x for x in observed)) + 1e-10
    obs_dir = [x / obs_norm for x in observed]

    base = dict(candidates)
    base_traj = simulate(base)

    scores: Dict[str, float] = {}
    sensitivities: Dict[str, List[float]] = {}
    for name, value in candidates.items():
        delta = value * perturbation_fraction
        if delta == 0.0:
            raise ValueError(f"candidate '{name}' has zero nominal value; "
                             "cannot form a relative perturbation")
        perturbed = dict(base)
        perturbed[name] = value + delta
        perturbed_traj = simulate(perturbed)
        response = _mean_abs_columns(base_traj, perturbed_traj)
        sens = [r / delta for r in response]
        sensitivities[name] = sens
        scores[name] = _cosine(obs_dir, sens)

    best = max(scores, key=lambda k: scores[k])
    sens = sensitivities[best]
    denom = sum(s * s for s in sens) + 1e-10
    estimated_delta = sum(s * o for s, o in zip(sens, observed)) / denom
    current = float(candidates[best])
    return SensitivityResult(
        identified_parameter=best,
        confidence=scores[best],
        current_value=current,
        estimated_delta=estimated_delta,
        proposed_value=current + estimated_delta,
        all_scores=dict(scores),
    )


# --------------------------------------------------------------------------- #
# Analytic estimators
# --------------------------------------------------------------------------- #
def estimate_mass_from_sag(
    sag_full: float,
    sag_half: float,
    model_mass: float,
    *,
    kp: float,
    reach_full: float,
    reach_half: float,
    gravity: float = GRAVITY,
    scaling_tolerance: float = 0.35,
) -> Dict[str, object]:
    """Actual tool mass from gravitational sag at two arm extensions.

    Physics: ``gravity_torque = delta_m * g * reach`` and
    ``joint_lag = gravity_torque / kp``, so ``delta_m = sag * kp / (g * reach)``.
    The estimate is averaged over both reaches; the 2:1 sag-scaling ratio is
    the mass-mismatch consistency check (``scaling_confirmed``). Sags and
    reaches in metres, kp in Nm/rad, masses in kg.
    """
    if min(reach_full, reach_half) <= 0 or kp <= 0:
        raise ValueError("reaches and kp must be positive")
    ratio = sag_full / (sag_half + 1e-12)
    confirmed = abs(ratio - REACH_RATIO_TARGET) < scaling_tolerance
    dm_full = sag_full * kp / (gravity * reach_full)
    dm_half = sag_half * kp / (gravity * reach_half)
    delta_mass = (dm_full + dm_half) / 2.0
    actual_mass = model_mass + delta_mass
    return {
        "identified_parameter": "tool_mass",
        "proposed_value": actual_mass,
        "current_value": model_mass,
        "estimated_delta": delta_mass,
        "sag_scaling_ratio": ratio,
        "scaling_confirmed": confirmed,
        "uncompensated_torque": delta_mass * gravity * reach_full,
        "method": "gravitational_sag_two_reach",
    }


def estimate_zero_offset_from_miss(
    miss_full: float,
    miss_half: float,
    *,
    reach_full: float,
    reach_half: float,
    scaling_tolerance: float = 0.4,
) -> Dict[str, object]:
    """Joint zero offset (radians) from the Cartesian miss at two reaches.

    A pure rotational fault produces ``miss = reach * sin(theta)``, so
    ``theta = asin(miss / reach)`` at each reach; the estimates are averaged
    and the 2:1 miss-scaling ratio is the rotational-signature check.
    Misses and reaches in metres.
    """
    if min(reach_full, reach_half) <= 0:
        raise ValueError("reaches must be positive")
    for miss, reach in ((miss_full, reach_full), (miss_half, reach_half)):
        if abs(miss) > reach:
            raise ValueError("miss exceeds reach; not a pure rotational fault")
    ratio = miss_full / (miss_half + 1e-12)
    confirmed = abs(ratio - REACH_RATIO_TARGET) < scaling_tolerance
    theta_full = math.asin(miss_full / reach_full)
    theta_half = math.asin(miss_half / reach_half)
    theta = (theta_full + theta_half) / 2.0
    return {
        "identified_parameter": "joint_zero_offset",
        "proposed_value": 0.0,
        "current_value": theta,
        "estimated_delta": -theta,
        "offset_rad": theta,
        "offset_deg": math.degrees(theta),
        "miss_scaling_ratio": ratio,
        "scaling_confirmed": confirmed,
        "method": "rotational_miss_two_reach",
    }


def estimate_lateral_offset(
    lateral_error: float,
    *,
    axis_sensitivity: float = 0.95,
    nominal_offset: float = 0.0,
) -> Dict[str, object]:
    """Mount offset from lateral drift divided by the axis sensitivity."""
    if axis_sensitivity <= 0:
        raise ValueError("axis_sensitivity must be positive")
    offset = lateral_error / axis_sensitivity
    return {
        "identified_parameter": "mount_lateral_offset",
        "proposed_value": nominal_offset,
        "current_value": nominal_offset + offset,
        "estimated_delta": -offset,
        "offset": offset,
        "method": "lateral_drift_sensitivity",
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _linear_simulator(true_params: Mapping[str, float]) -> Simulator:
    """A synthetic simulator: each joint responds linearly to each parameter."""
    gains = {"link1_length": (1.0, 0.2), "link2_length": (0.15, 1.3)}

    def simulate(params: Mapping[str, float]) -> Trajectory:
        rows = []
        for i in range(50):
            t = i * 0.1
            q1 = sum(gains[k][0] * params[k] for k in gains) * math.sin(t)
            q2 = sum(gains[k][1] * params[k] for k in gains) * math.cos(t)
            rows.append((q1, q2))
        return rows

    return simulate


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.eval.quality.physics.fault_identification",
        description="Sensitivity + analytic fault-parameter identification "
                    "(SimCorrect).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="identify an injected link-length fault on a "
                             "synthetic simulator and run all three analytic "
                             "estimators against known ground truth.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    simulate = _linear_simulator({})
    nominal = {"link1_length": 0.30, "link2_length": 0.22}
    truth = {"link1_length": 0.30, "link2_length": 0.25}
    result = identify_by_sensitivity(
        simulate(truth), simulate(nominal), nominal, simulate)
    assert result.identified_parameter == "link2_length", result
    assert abs(result.proposed_value - 0.25) < 0.01, result
    print(f"[selfcheck] sensitivity: {result.identified_parameter} "
          f"{result.current_value:.3f} -> {result.proposed_value:.3f} "
          f"(cosine {result.confidence:.3f})")

    # Sag produced by a +60 g mass error: sag = dm * g * reach / kp.
    sag_full = 0.060 * GRAVITY * 0.75 / 400.0
    mass = estimate_mass_from_sag(sag_full, sag_full / 2.0, 0.100,
                                  kp=400.0, reach_full=0.75, reach_half=0.375)
    assert mass["scaling_confirmed"]
    assert abs(float(mass["estimated_delta"]) - 0.060) < 1e-9
    print(f"[selfcheck] mass: {mass['current_value']:.3f} -> "
          f"{float(mass['proposed_value']):.3f} kg "
          f"(delta +{float(mass['estimated_delta']) * 1000:.1f} g)")

    zero = estimate_zero_offset_from_miss(0.103, 0.052,
                                          reach_full=0.75, reach_half=0.375)
    assert zero["scaling_confirmed"]
    print(f"[selfcheck] zero offset: {float(zero['offset_deg']):.2f} deg "
          f"(ratio {float(zero['miss_scaling_ratio']):.2f})")

    lateral = estimate_lateral_offset(0.150)
    print(f"[selfcheck] lateral offset: {float(lateral['offset']) * 1000:.1f} mm")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
