"""Closed detect -> identify -> patch -> revalidate correction loop (SimCorrect).

Mined from **SimCorrect** (``Problem*/correction_and_validation.py`` and
``Problem1/caid_loop.py``), whose end-to-end pipeline is the part worth
transplanting: a fault is never "corrected" until the corrected model has been
re-run and the residual measured. The loop's five phases:

  1. run the suspect model and measure divergence against the reference
     (:mod:`harnesscad.eval.quality.physics.sim_divergence`);
  2. classify the fault (:mod:`...fault_taxonomy`) -- optional, informative;
  3. identify the responsible parameter and its corrected value
     (:mod:`...fault_identification`, or any identification dict);
  4. patch the design through the CAID artifact contract when an artifact is
     supplied (:mod:`harnesscad.domain.spec.caid_artifact`), which resolves
     simulator names to design names and rejects stale writes; otherwise patch
     the flat simulation params directly;
  5. re-run with the corrected params and *verify convergence*: the after-RMSE
     must fall below the acceptance threshold, and the loop reports the
     reduction percentage. A correction that does not converge is a FAIL, not
     a silent success.

The simulator is injected as a callable ``params -> (joint_traj, ee_traj)``;
this module owns only the deterministic bookkeeping. With no simulator
available the loop still produces the patch (phases 1-4 on recorded logs) and
reports validation as SKIPPED -- it never fakes a pass.

stdlib-only, deterministic, absolute imports.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from harnesscad.domain.spec.caid_artifact import (
    apply_parameter_patch,
    apply_patch_to_simulation_params,
    load_artifact,
    make_patch_from_identification,
)
from harnesscad.eval.quality.physics.sim_divergence import PairedTrajectories, detect_divergence

__all__ = ["CorrectionOutcome", "run_correction_loop", "main"]

JointTrajectory = Sequence[Sequence[float]]
EeTrajectory = Sequence[Sequence[float]]
PairSimulator = Callable[[Mapping[str, Any]], Tuple[JointTrajectory, EeTrajectory]]

DEFAULT_CONVERGENCE_RMSE = 0.002  # rad


def _rmse(a: JointTrajectory, b: JointTrajectory) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("trajectories must be equal-length and non-empty")
    total = 0.0
    count = 0
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            d = x - y
            total += d * d
            count += 1
    return math.sqrt(total / count) if count else 0.0


@dataclass(frozen=True)
class CorrectionOutcome:
    """Full record of one correction cycle."""

    verdict: str                                   # PASS | FAIL | SKIPPED
    identified_parameter: str
    proposed_value: Any
    corrected_params: Dict[str, Any]
    patch: Optional[Dict[str, Any]]                # CAID patch when artifact given
    corrected_artifact: Optional[Dict[str, Any]]
    rmse_before: Optional[float]
    rmse_after: Optional[float]
    reduction_pct: Optional[float]
    converged: Optional[bool]
    reasons: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "identified_parameter": self.identified_parameter,
            "proposed_value": self.proposed_value,
            "corrected_params": dict(self.corrected_params),
            "rmse_before": self.rmse_before,
            "rmse_after": self.rmse_after,
            "reduction_pct": self.reduction_pct,
            "converged": self.converged,
            "reasons": list(self.reasons),
        }


def run_correction_loop(
    trajectories: PairedTrajectories,
    identification: Mapping[str, Any],
    suspect_params: Mapping[str, Any],
    *,
    artifact: Optional[Any] = None,
    simulate: Optional[PairSimulator] = None,
    reference_params: Optional[Mapping[str, Any]] = None,
    convergence_rmse: float = DEFAULT_CONVERGENCE_RMSE,
) -> CorrectionOutcome:
    """One full correction cycle over recorded trajectories.

    ``identification`` must carry ``identified_parameter`` and
    ``proposed_value`` (the shape produced by
    :mod:`harnesscad.eval.quality.physics.fault_identification`). When
    ``artifact`` (a CAID artifact source) is given, the patch is routed through
    the contract; otherwise the simulation parameter is written directly.
    Validation runs only when ``simulate`` is supplied; the reference
    trajectory is re-simulated from ``reference_params`` when given, else the
    recorded ground-truth joints are used.
    """
    reasons = []
    param = str(identification["identified_parameter"])
    proposed = identification["proposed_value"]

    # Phase 1: confirm there is something to correct.
    report = detect_divergence(trajectories)
    if not report.detected and report.peak_ee_error <= 0.0:
        return CorrectionOutcome(
            verdict="SKIPPED", identified_parameter=param, proposed_value=proposed,
            corrected_params=dict(suspect_params), patch=None,
            corrected_artifact=None, rmse_before=None, rmse_after=None,
            reduction_pct=None, converged=None,
            reasons=("no divergence detected; nothing to correct",))
    reasons.append(
        f"divergence: joint alarm={report.detected}, "
        f"peak EE error {report.peak_ee_error * 1000:.1f}mm")

    # Phases 3-4: patch.
    patch: Optional[Dict[str, Any]] = None
    corrected_artifact: Optional[Dict[str, Any]] = None
    if artifact is not None:
        loaded = load_artifact(artifact)
        patch = make_patch_from_identification(loaded, dict(identification))
        corrected_artifact = apply_parameter_patch(loaded, patch)
        corrected_params = apply_patch_to_simulation_params(
            loaded, patch, dict(suspect_params))
        reasons.append(
            f"CAID patch: {patch['parameter_patches'][0]['name']} -> {proposed!r}")
    else:
        if param not in suspect_params:
            raise KeyError(f"identified parameter '{param}' not in suspect params")
        corrected_params = dict(suspect_params)
        corrected_params[param] = proposed
        reasons.append(f"direct parameter write: {param} -> {proposed!r} "
                       "(no CAID artifact supplied)")

    # Phase 5: revalidate, or honestly skip.
    if simulate is None:
        reasons.append("no simulator supplied; validation SKIPPED, patch prepared")
        return CorrectionOutcome(
            verdict="SKIPPED", identified_parameter=param, proposed_value=proposed,
            corrected_params=corrected_params, patch=patch,
            corrected_artifact=corrected_artifact, rmse_before=None,
            rmse_after=None, reduction_pct=None, converged=None,
            reasons=tuple(reasons))

    if reference_params is not None:
        reference_joints, _ = simulate(reference_params)
    else:
        reference_joints = trajectories.ground_truth_joints
    before_joints, _ = simulate(suspect_params)
    after_joints, _ = simulate(corrected_params)

    rmse_before = _rmse(reference_joints, before_joints)
    rmse_after = _rmse(reference_joints, after_joints)
    converged = rmse_after < convergence_rmse
    reduction = (1.0 - rmse_after / rmse_before) * 100.0 if rmse_before > 0 else 0.0
    reasons.append(
        f"validation: RMSE {rmse_before:.6f} -> {rmse_after:.6f} rad "
        f"({reduction:.1f}% reduction), converged={converged}")

    return CorrectionOutcome(
        verdict="PASS" if converged else "FAIL",
        identified_parameter=param, proposed_value=proposed,
        corrected_params=corrected_params, patch=patch,
        corrected_artifact=corrected_artifact,
        rmse_before=rmse_before, rmse_after=rmse_after,
        reduction_pct=reduction, converged=converged,
        reasons=tuple(reasons))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _synthetic_world():
    """A toy two-parameter simulator + recorded fault pair + CAID artifact."""
    gains = {"link1_length": (1.0, 0.2), "link2_length": (0.15, 1.3)}

    def simulate(params: Mapping[str, Any]) -> Tuple[JointTrajectory, EeTrajectory]:
        joints = []
        ee = []
        for i in range(60):
            t = i * 0.1
            q1 = sum(gains[k][0] * float(params[k]) for k in gains) * math.sin(t)
            q2 = sum(gains[k][1] * float(params[k]) for k in gains) * math.cos(t)
            joints.append((q1, q2))
            ee.append((0.3 * math.cos(q1), 0.3 * math.sin(q1), 0.1 + 0.05 * q2))
        return joints, ee

    truth = {"link1_length": 0.30, "link2_length": 0.25}
    suspect = {"link1_length": 0.30, "link2_length": 0.22}
    gt_joints, gt_ee = simulate(truth)
    fx_joints, fx_ee = simulate(suspect)
    pair = PairedTrajectories(
        times=tuple(i * 0.1 for i in range(60)),
        ground_truth_joints=tuple(map(tuple, gt_joints)),
        suspect_joints=tuple(map(tuple, fx_joints)),
        ground_truth_ee=tuple(map(tuple, gt_ee)),
        suspect_ee=tuple(map(tuple, fx_ee)),
        ground_truth_params=truth,
        suspect_params=suspect,
        injected_error={"parameter": "link2_length",
                        "true_value": 0.25, "faulty_value": 0.22})
    artifact = {
        "schema_version": 1,
        "artifact_id": "selfcheck-loop",
        "producer": {"name": "harnesscad", "version": "0"},
        "created_at": "2026-01-01T00:00:00Z",
        "feature_tree": {"root_id": "root", "nodes": {}},
        "parameters": {
            "forearm_length": {"name": "forearm_length", "value": 0.22, "unit": "m"},
        },
        "simulation_tags": [
            {"name": "forearm_length", "kind": "parameter", "target": "link2_length"},
        ],
    }
    return simulate, pair, suspect, truth, artifact


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.eval.quality.physics.sim_correction_loop",
        description="Closed detect/identify/patch/revalidate loop (SimCorrect).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the loop on a synthetic fault with a CAID "
                             "artifact (PASS) and without a simulator (SKIPPED).")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    simulate, pair, suspect, truth, artifact = _synthetic_world()
    identification = {
        "identified_parameter": "link2_length",
        "proposed_value": 0.25,
        "method": "sensitivity_analysis",
    }

    outcome = run_correction_loop(
        pair, identification, suspect,
        artifact=artifact, simulate=simulate, reference_params=truth)
    for reason in outcome.reasons:
        print(f"  - {reason}")
    assert outcome.verdict == "PASS", outcome.to_dict()
    assert outcome.corrected_artifact is not None
    assert outcome.corrected_artifact["parameters"]["forearm_length"]["value"] == 0.25
    print(f"[selfcheck] with simulator: {outcome.verdict} "
          f"({outcome.reduction_pct:.1f}% RMSE reduction)")

    prepared = run_correction_loop(pair, identification, suspect, artifact=artifact)
    assert prepared.verdict == "SKIPPED" and prepared.patch is not None
    print(f"[selfcheck] without simulator: {prepared.verdict} (patch prepared, "
          "validation honestly skipped)")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
