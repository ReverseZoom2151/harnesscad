"""Simulation-fault taxonomy and signature-based classification (SimCorrect).

Mined from **SimCorrect** (``Problem*/divergence_detector.py`` and the five
problem READMEs), which demonstrates that five physically distinct CAD-model
faults are separable by a small, fully deterministic decision tree over four
measurable signatures -- no learning anywhere:

  1. **joint RMSE** (commanded vs actual): zero for geometry faults, non-zero
     for dynamics faults. This single split is the taxonomy's first branch,
     because no geometry error can produce joint lag.
  2. **velocity dependence** (RMSE at fast vs slow trajectory speed): friction
     consumes torque in proportion to velocity, so a high fast/slow ratio means
     friction; gravity-dependent faults barely change with speed.
  3. **reach scaling** (error at full vs half extension): a pure rotational
     fault (encoder zero offset) and a pure tool-mass fault both scale
     linearly with reach -- the tell-tale 2:1 ratio at 2x reach -- while a
     link-length error is constant and a lateral mount offset is constant.
  4. **error axis** (vertical sag vs lateral drift vs along-reach overshoot):
     separates mass droop, mount offset, and link-length error.

The taxonomy matters to a text-to-CAD harness because these are exactly the
model-vs-reality mismatches a generated CAD model exhibits when its parameters
are wrong: the classifier turns a raw discrepancy into a *named fault class*
with a candidate parameter kind, which is what a correction loop patches.

stdlib-only, deterministic. All signatures are plain floats measured elsewhere
(paired simulation, hardware log, or a differential oracle).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "FaultClass",
    "FaultSignature",
    "FaultVerdict",
    "classify_fault",
    "FAULT_CATALOG",
    "main",
]


class FaultClass(str, Enum):
    """Named fault classes with a candidate parameter kind each."""

    NOMINAL = "NOMINAL"
    GEOMETRIC_LINK_LENGTH = "GEOMETRIC_LINK_LENGTH"
    GEOMETRIC_LATERAL_OFFSET = "GEOMETRIC_LATERAL_OFFSET"
    GEOMETRIC_ZERO_OFFSET = "GEOMETRIC_ZERO_OFFSET"
    DYNAMICS_FRICTION = "DYNAMICS_FRICTION"
    DYNAMICS_MASS_MISMATCH = "DYNAMICS_MASS_MISMATCH"
    GEOMETRIC_UNRESOLVED = "GEOMETRIC_UNRESOLVED"
    DYNAMICS_UNRESOLVED = "DYNAMICS_UNRESOLVED"


#: What each class means and which parameter kind a corrector should target.
FAULT_CATALOG: Dict[FaultClass, Dict[str, str]] = {
    FaultClass.GEOMETRIC_LINK_LENGTH: {
        "signature": "Cartesian overshoot along the reach axis, zero joint RMSE, "
                     "error does not grow with reach ratio beyond geometry",
        "parameter_kind": "link length",
    },
    FaultClass.GEOMETRIC_LATERAL_OFFSET: {
        "signature": "constant lateral Cartesian drift, zero joint RMSE",
        "parameter_kind": "mount offset along the drift axis",
    },
    FaultClass.GEOMETRIC_ZERO_OFFSET: {
        "signature": "rotational Cartesian miss scaling ~2:1 with 2x reach, "
                     "zero joint RMSE",
        "parameter_kind": "joint encoder zero reference",
    },
    FaultClass.DYNAMICS_FRICTION: {
        "signature": "non-zero joint RMSE, strongly velocity-dependent",
        "parameter_kind": "joint damping/friction coefficient",
    },
    FaultClass.DYNAMICS_MASS_MISMATCH: {
        "signature": "non-zero joint RMSE, gravity-dependent vertical sag, "
                     "sag scaling ~2:1 with 2x reach, weak velocity dependence",
        "parameter_kind": "tool/body mass",
    },
}


@dataclass(frozen=True)
class FaultSignature:
    """Measured signatures feeding the classifier. Unmeasured items stay None.

    Units: distances in metres, angles in radians. ``reach ratio`` fields are
    measured at full extension vs half extension (expected 2.0 for faults that
    scale linearly with reach).
    """

    ee_error: float                                 # peak paired EE error (m)
    joint_rmse: float                               # commanded-vs-actual (rad)
    vertical_sag: Optional[float] = None            # signed sag, faulty below GT (m)
    lateral_drift: Optional[float] = None           # drift beyond placement (m)
    velocity_rmse_ratio: Optional[float] = None     # RMSE(fast) / RMSE(slow)
    reach_scaling_ratio: Optional[float] = None     # error(full) / error(half)


@dataclass(frozen=True)
class FaultVerdict:
    fault_class: FaultClass
    detected: bool
    is_dynamics: bool
    reasons: Tuple[str, ...]
    parameter_kind: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "fault_class": self.fault_class.value,
            "detected": self.detected,
            "is_dynamics": self.is_dynamics,
            "parameter_kind": self.parameter_kind,
            "reasons": list(self.reasons),
        }


# Thresholds (SimCorrect's values, all overridable at the call site).
EE_FAULT_THRESHOLD = 0.040          # m -- below this the pair is nominal
RMSE_DYNAMICS_THRESHOLD = 0.005     # rad -- above this the fault is dynamic
VELOCITY_RATIO_THRESHOLD = 2.0      # fast/slow RMSE ratio -> friction
SAG_GRAVITY_THRESHOLD = 0.005       # m of vertical sag implies gravity term
REACH_RATIO_TARGET = 2.0            # linear-with-reach signature
REACH_RATIO_TOLERANCE = 0.4


def classify_fault(
    sig: FaultSignature,
    *,
    ee_threshold: float = EE_FAULT_THRESHOLD,
    rmse_threshold: float = RMSE_DYNAMICS_THRESHOLD,
    velocity_ratio_threshold: float = VELOCITY_RATIO_THRESHOLD,
) -> FaultVerdict:
    """Walk SimCorrect's decision tree over the measured signatures."""
    reasons: List[str] = []

    if sig.ee_error <= ee_threshold and sig.joint_rmse <= rmse_threshold:
        return FaultVerdict(
            FaultClass.NOMINAL, detected=False, is_dynamics=False,
            reasons=("end-effector error and joint RMSE within tolerance",))

    detected = True
    is_dynamics = sig.joint_rmse > rmse_threshold

    if not is_dynamics:
        reasons.append(
            f"joint RMSE {sig.joint_rmse:.4f} rad <= {rmse_threshold}: "
            "geometric fault (joints execute their commands)")
        return FaultVerdict(
            _classify_geometric(sig, reasons), detected, False, tuple(reasons),
            parameter_kind=_parameter_kind(_classify_geometric(sig, [])))

    reasons.append(
        f"joint RMSE {sig.joint_rmse:.4f} rad > {rmse_threshold}: "
        "dynamics fault (joints cannot hold their commands)")
    cls = _classify_dynamics(sig, reasons, velocity_ratio_threshold)
    return FaultVerdict(cls, detected, True, tuple(reasons),
                        parameter_kind=_parameter_kind(cls))


def _classify_geometric(sig: FaultSignature, reasons: List[str]) -> FaultClass:
    if sig.lateral_drift is not None and abs(sig.lateral_drift) > EE_FAULT_THRESHOLD:
        reasons.append(
            f"lateral drift {abs(sig.lateral_drift) * 1000:.1f}mm beyond placement: "
            "mount offset")
        return FaultClass.GEOMETRIC_LATERAL_OFFSET
    if sig.reach_scaling_ratio is not None:
        if abs(sig.reach_scaling_ratio - REACH_RATIO_TARGET) < REACH_RATIO_TOLERANCE:
            reasons.append(
                f"miss scales {sig.reach_scaling_ratio:.2f}:1 at 2x reach "
                "(expected 2.0 for pure rotation): encoder zero offset")
            return FaultClass.GEOMETRIC_ZERO_OFFSET
        reasons.append(
            f"miss scaling ratio {sig.reach_scaling_ratio:.2f} not linear-with-reach: "
            "link length error")
        return FaultClass.GEOMETRIC_LINK_LENGTH
    reasons.append("no reach-scaling or lateral measurements: geometric fault "
                   "unresolved between link length and zero offset")
    return FaultClass.GEOMETRIC_UNRESOLVED


def _classify_dynamics(
    sig: FaultSignature,
    reasons: List[str],
    velocity_ratio_threshold: float,
) -> FaultClass:
    gravity_dependent: Optional[bool] = None

    if sig.velocity_rmse_ratio is not None:
        if sig.velocity_rmse_ratio > velocity_ratio_threshold:
            reasons.append(
                f"velocity ratio {sig.velocity_rmse_ratio:.2f} > "
                f"{velocity_ratio_threshold}: velocity-dependent -> friction")
            return FaultClass.DYNAMICS_FRICTION
        reasons.append(
            f"velocity ratio {sig.velocity_rmse_ratio:.2f} low: gravity-dependent")
        gravity_dependent = True
    elif sig.vertical_sag is not None:
        gravity_dependent = abs(sig.vertical_sag) > SAG_GRAVITY_THRESHOLD
        reasons.append(
            f"no velocity data; vertical sag {abs(sig.vertical_sag) * 1000:.1f}mm "
            f"{'implies' if gravity_dependent else 'does not imply'} gravity term")

    if gravity_dependent:
        if sig.reach_scaling_ratio is not None:
            if abs(sig.reach_scaling_ratio - REACH_RATIO_TARGET) < REACH_RATIO_TOLERANCE:
                reasons.append(
                    f"sag scales {sig.reach_scaling_ratio:.2f}:1 at 2x reach: "
                    "pure mass-error signature confirmed")
            else:
                reasons.append(
                    f"sag scaling ratio {sig.reach_scaling_ratio:.2f} deviates "
                    "from 2.0: mixed fault possible")
        return FaultClass.DYNAMICS_MASS_MISMATCH
    if gravity_dependent is False:
        reasons.append("dynamics fault without gravity dependence: friction")
        return FaultClass.DYNAMICS_FRICTION
    reasons.append("no velocity or sag measurements: dynamics fault unresolved")
    return FaultClass.DYNAMICS_UNRESOLVED


def _parameter_kind(cls: FaultClass) -> Optional[str]:
    entry = FAULT_CATALOG.get(cls)
    return entry["parameter_kind"] if entry else None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
_SELFCHECK_CASES: Tuple[Tuple[str, FaultSignature, FaultClass], ...] = (
    ("nominal",
     FaultSignature(ee_error=0.012, joint_rmse=0.000),
     FaultClass.NOMINAL),
    ("forearm 80mm too long",
     FaultSignature(ee_error=0.080, joint_rmse=0.000, reach_scaling_ratio=1.05),
     FaultClass.GEOMETRIC_LINK_LENGTH),
    ("wrist mounted 150mm off-centre",
     FaultSignature(ee_error=0.150, joint_rmse=0.000, lateral_drift=0.150),
     FaultClass.GEOMETRIC_LATERAL_OFFSET),
    ("encoder 8 degrees off zero",
     FaultSignature(ee_error=0.103, joint_rmse=0.000, reach_scaling_ratio=1.98),
     FaultClass.GEOMETRIC_ZERO_OFFSET),
    ("joint friction doubled",
     FaultSignature(ee_error=0.060, joint_rmse=0.031, velocity_rmse_ratio=3.4),
     FaultClass.DYNAMICS_FRICTION),
    ("gripper 60g heavier than modelled",
     FaultSignature(ee_error=0.095, joint_rmse=0.012, vertical_sag=0.055,
                    velocity_rmse_ratio=1.1, reach_scaling_ratio=2.0),
     FaultClass.DYNAMICS_MASS_MISMATCH),
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.eval.quality.physics.fault_taxonomy",
        description="Simulation-fault taxonomy and classifier (SimCorrect).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="classify all five SimCorrect fault scenarios plus "
                             "a nominal run and check the verdicts.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    for label, sig, expected in _SELFCHECK_CASES:
        verdict = classify_fault(sig)
        status = "OK" if verdict.fault_class is expected else "FAIL"
        print(f"[{status}] {label}: {verdict.fault_class.value}"
              + (f" (target: {verdict.parameter_kind})" if verdict.parameter_kind else ""))
        assert verdict.fault_class is expected, (label, verdict.fault_class)
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
