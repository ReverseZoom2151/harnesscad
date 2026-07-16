"""Paired-simulation divergence detection for CAD-model faults (SimCorrect).

Mined from **SimCorrect** (``Problem*/divergence_detector.py`` and
``Problem1_ForearmLength/trajectory_io.py``). SimCorrect's central diagnostic
device is a *paired simulation*: a ground-truth model and a suspect model run
side by side under identical commands, and the divergence between them -- not
any absolute error -- is the fault signal. Three detection signals matter, and
they separate fault classes cleanly:

  * **sliding-window joint RMSE** between the two joint trajectories: a
    dynamics fault (friction, mass) makes it rise above threshold as soon as
    motion begins; a pure geometry fault leaves it at zero;
  * **end-effector Cartesian error** between the paired end-effector paths:
    geometry faults live entirely here and are invisible in joint space;
  * **lateral drift beyond the expected offset**: when the two arms are
    intentionally placed apart, only the error *beyond* the expected placement
    delta is a fault, and dividing by a known axis sensitivity turns the drift
    directly into a parameter-offset estimate.

The module also carries the trajectory persistence contract: a
:class:`PairedTrajectories` record (times, ground-truth and suspect joint
states and end-effector positions, injected-error metadata) with a JSON round
trip, so detection can run on recorded logs with no simulator installed.

stdlib-only (``math``, ``json``, ``dataclasses``), deterministic, no numpy.
The simulators that produce the trajectories stay outside this module.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

__all__ = [
    "PairedTrajectories",
    "DivergenceReport",
    "sliding_window_rmse",
    "pointwise_distance",
    "detect_divergence",
    "LateralDriftDetector",
    "JointRmseDetector",
    "main",
]

Vector = Sequence[float]
Series = Sequence[Sequence[float]]

DEFAULT_JOINT_RMSE_THRESHOLD = 0.002   # rad, geometric-vs-nominal alarm level
DEFAULT_WINDOW_SIZE = 20               # samples in the sliding RMSE window
DEFAULT_LATERAL_THRESHOLD = 0.050      # m, drift beyond expected placement


# --------------------------------------------------------------------------- #
# Trajectory record + JSON round trip
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairedTrajectories:
    """Recorded ground-truth / suspect trajectory pair under identical commands."""

    times: Tuple[float, ...]
    ground_truth_joints: Tuple[Tuple[float, ...], ...]
    suspect_joints: Tuple[Tuple[float, ...], ...]
    ground_truth_ee: Tuple[Tuple[float, ...], ...]
    suspect_ee: Tuple[Tuple[float, ...], ...]
    ground_truth_params: Dict[str, Any] = field(default_factory=dict)
    suspect_params: Dict[str, Any] = field(default_factory=dict)
    injected_error: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.times)
        for label, series in (
            ("ground_truth_joints", self.ground_truth_joints),
            ("suspect_joints", self.suspect_joints),
            ("ground_truth_ee", self.ground_truth_ee),
            ("suspect_ee", self.suspect_ee),
        ):
            if len(series) != n:
                raise ValueError(f"{label} has {len(series)} samples, expected {n}.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "times": list(self.times),
            "ground_truth": {
                "joint_states": [list(row) for row in self.ground_truth_joints],
                "ee_positions": [list(row) for row in self.ground_truth_ee],
                "params": dict(self.ground_truth_params),
            },
            "suspect": {
                "joint_states": [list(row) for row in self.suspect_joints],
                "ee_positions": [list(row) for row in self.suspect_ee],
                "params": dict(self.suspect_params),
            },
            "injected_error": dict(self.injected_error),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PairedTrajectories":
        gt = payload["ground_truth"]
        fx = payload["suspect"]
        return cls(
            times=tuple(float(t) for t in payload["times"]),
            ground_truth_joints=tuple(tuple(map(float, r)) for r in gt["joint_states"]),
            suspect_joints=tuple(tuple(map(float, r)) for r in fx["joint_states"]),
            ground_truth_ee=tuple(tuple(map(float, r)) for r in gt["ee_positions"]),
            suspect_ee=tuple(tuple(map(float, r)) for r in fx["ee_positions"]),
            ground_truth_params=dict(gt.get("params", {})),
            suspect_params=dict(fx.get("params", {})),
            injected_error=dict(payload.get("injected_error", {})),
        )

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PairedTrajectories":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Core signals
# --------------------------------------------------------------------------- #
def sliding_window_rmse(a: Series, b: Series, window: int = DEFAULT_WINDOW_SIZE) -> List[float]:
    """Per-sample RMSE between two joint series over a trailing window.

    At sample ``i`` the RMSE is computed over samples ``max(0, i-window+1)..i``
    across all joints, exactly SimCorrect's detection signal.
    """
    if len(a) != len(b):
        raise ValueError(f"series lengths differ: {len(a)} vs {len(b)}")
    if window < 1:
        raise ValueError("window must be >= 1")
    out: List[float] = []
    for i in range(len(a)):
        start = max(0, i - window + 1)
        total = 0.0
        count = 0
        for j in range(start, i + 1):
            row_a, row_b = a[j], b[j]
            if len(row_a) != len(row_b):
                raise ValueError(f"joint widths differ at sample {j}")
            for x, y in zip(row_a, row_b):
                d = x - y
                total += d * d
                count += 1
        out.append(math.sqrt(total / count) if count else 0.0)
    return out


def pointwise_distance(a: Series, b: Series) -> List[float]:
    """Euclidean distance between paired points at each sample."""
    if len(a) != len(b):
        raise ValueError(f"series lengths differ: {len(a)} vs {len(b)}")
    return [
        math.sqrt(sum((x - y) ** 2 for x, y in zip(pa, pb)))
        for pa, pb in zip(a, b)
    ]


# --------------------------------------------------------------------------- #
# Batch divergence detection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DivergenceReport:
    """Result of a batch detection pass over a trajectory pair."""

    detected: bool
    first_detection_time: Optional[float]
    peak_joint_rmse: float
    peak_ee_error: float
    mean_ee_error: float
    threshold: float
    window: int
    joint_rmse: Tuple[float, ...]
    ee_error: Tuple[float, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "first_detection_time": self.first_detection_time,
            "peak_joint_rmse": self.peak_joint_rmse,
            "peak_ee_error": self.peak_ee_error,
            "mean_ee_error": self.mean_ee_error,
            "threshold": self.threshold,
            "window": self.window,
        }


def detect_divergence(
    trajectories: PairedTrajectories,
    *,
    threshold: float = DEFAULT_JOINT_RMSE_THRESHOLD,
    window: int = DEFAULT_WINDOW_SIZE,
) -> DivergenceReport:
    """Sliding-RMSE alarm plus end-effector error profile for a recorded pair."""
    rmse = sliding_window_rmse(
        trajectories.ground_truth_joints, trajectories.suspect_joints, window)
    ee_error = pointwise_distance(
        trajectories.ground_truth_ee, trajectories.suspect_ee)
    first_time: Optional[float] = None
    for t, r in zip(trajectories.times, rmse):
        if r > threshold:
            first_time = float(t)
            break
    return DivergenceReport(
        detected=first_time is not None,
        first_detection_time=first_time,
        peak_joint_rmse=max(rmse) if rmse else 0.0,
        peak_ee_error=max(ee_error) if ee_error else 0.0,
        mean_ee_error=(sum(ee_error) / len(ee_error)) if ee_error else 0.0,
        threshold=threshold,
        window=window,
        joint_rmse=tuple(rmse),
        ee_error=tuple(ee_error),
    )


# --------------------------------------------------------------------------- #
# Stateful streaming detectors
# --------------------------------------------------------------------------- #
class LateralDriftDetector:
    """Streaming detector for a lateral placement-offset fault.

    ``update`` receives the paired end-effector positions plus the two arms'
    intended placement coordinates along the watched axis. Only the drift
    beyond the expected placement delta is a fault; dividing by the axis
    sensitivity converts the drift into a parameter-offset estimate
    (SimCorrect Problem 2).
    """

    def __init__(
        self,
        *,
        axis: int = 1,
        threshold: float = DEFAULT_LATERAL_THRESHOLD,
        sensitivity: float = 0.95,
    ) -> None:
        if not 0.0 < sensitivity:
            raise ValueError("sensitivity must be positive")
        self.axis = axis
        self.threshold = threshold
        self.sensitivity = sensitivity
        self.history: List[float] = []
        self.fault_detected = False
        self.estimated_offset = 0.0

    def update(
        self,
        gt_ee: Vector,
        suspect_ee: Vector,
        gt_placement: float,
        suspect_placement: float,
    ) -> bool:
        expected_delta = suspect_placement - gt_placement
        actual_delta = suspect_ee[self.axis] - gt_ee[self.axis]
        lateral_error = actual_delta - expected_delta
        self.history.append(lateral_error)
        if abs(lateral_error) > self.threshold:
            self.fault_detected = True
            self.estimated_offset = lateral_error / self.sensitivity
            return True
        return False

    def report(self) -> Dict[str, Any]:
        mean_abs = (sum(abs(e) for e in self.history) / len(self.history)) \
            if self.history else 0.0
        return {
            "fault_detected": self.fault_detected,
            "mean_lateral_error": mean_abs,
            "estimated_offset": self.estimated_offset,
            "joint_rmse": 0.0,
            "fault_axis": "XYZ"[self.axis] if 0 <= self.axis < 3 else str(self.axis),
            "fault_type": "lateral_offset",
        }

    def reset(self) -> None:
        self.history = []
        self.fault_detected = False
        self.estimated_offset = 0.0


class JointRmseDetector:
    """Streaming detector for dynamics faults visible as commanded-vs-actual lag.

    Each ``update`` compares commanded joint angles against actual joint
    angles; a dynamics fault (excess friction, wrong mass) makes the RMSE rise
    above threshold because the joints physically cannot hold their commands
    (SimCorrect Problem 3).
    """

    def __init__(self, *, threshold: float = 0.015) -> None:
        self.threshold = threshold
        self.history: List[float] = []
        self.fault_detected = False

    def update(self, commanded: Vector, actual: Vector) -> bool:
        if len(commanded) != len(actual):
            raise ValueError("commanded/actual widths differ")
        rmse = math.sqrt(
            sum((c - a) ** 2 for c, a in zip(commanded, actual)) / len(commanded))
        self.history.append(rmse)
        if rmse > self.threshold and not self.fault_detected:
            self.fault_detected = True
            return True
        return False

    def report(self) -> Dict[str, Any]:
        mean = (sum(self.history) / len(self.history)) if self.history else 0.0
        return {
            "fault_detected": self.fault_detected,
            "joint_rmse": mean,
            "fault_type": "dynamics_lag",
        }

    def reset(self) -> None:
        self.history = []
        self.fault_detected = False


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _synthetic_pair() -> PairedTrajectories:
    """A synthetic paired run: geometry fault, zero joint error, EE drift."""
    times = tuple(i * 0.01 for i in range(200))
    joints = tuple((0.4 * math.sin(2.0 * t), 0.3 * math.sin(1.5 * t + 0.5))
                   for t in times)
    gt_ee = tuple((0.25 * math.cos(q1), 0.25 * math.sin(q1), 0.1 + 0.05 * q2)
                  for q1, q2 in joints)
    fx_ee = tuple((x * 1.32, y * 1.32, z) for x, y, z in gt_ee)
    return PairedTrajectories(
        times=times,
        ground_truth_joints=joints,
        suspect_joints=joints,
        ground_truth_ee=gt_ee,
        suspect_ee=fx_ee,
        ground_truth_params={"link2_length": 0.25},
        suspect_params={"link2_length": 0.33},
        injected_error={"parameter": "link2_length",
                        "true_value": 0.25, "faulty_value": 0.33},
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.eval.quality.physics.sim_divergence",
        description="Paired-simulation divergence detection (SimCorrect).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="run detection on a synthetic geometry-fault pair "
                             "and both streaming detectors.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    pair = _synthetic_pair()
    round_tripped = PairedTrajectories.from_dict(pair.to_dict())
    assert round_tripped == pair
    report = detect_divergence(pair)
    assert not report.detected, "identical joints must not alarm in joint space"
    assert report.peak_ee_error > 0.05, "EE drift must be visible"
    print(f"[selfcheck] geometry fault: joint alarm={report.detected} "
          f"peak_ee={report.peak_ee_error * 1000:.1f}mm (invisible in joint space)")

    lat = LateralDriftDetector()
    fired = lat.update((0.52, -0.55, 0.46), (0.52, 0.70, 0.46), -0.55, 0.55)
    assert fired and abs(lat.estimated_offset - 0.15 / 0.95) < 1e-9
    print(f"[selfcheck] lateral drift: offset={lat.estimated_offset * 1000:.1f}mm")

    jr = JointRmseDetector()
    for _ in range(50):
        jr.update((0.0, -0.5, 1.2, 0.1), (0.0, -0.75, 1.2, 0.1))
    assert jr.fault_detected
    print(f"[selfcheck] dynamics lag: rmse={jr.report()['joint_rmse']:.4f} rad")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
