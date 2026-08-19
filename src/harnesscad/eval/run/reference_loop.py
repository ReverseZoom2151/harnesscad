"""Close the agent-environment loop with a SCRIPTED REFERENCE POLICY -- no model.

WHAT THIS IS (AND IS NOT)
=========================
The RL environment (``CADGymEnv``), the MCP tool surface, the ReAct trajectory
type, the verifiers and the leaderboards all already EXIST and pass their own
selfchecks -- but until now nothing had ever been driven through them as a full
episode: zero produced trajectories, zero populated leaderboard rows. This module
closes that gap for FREE, without any paid model or external API call, by driving
the real environment with a DETERMINISTIC reference policy: a scripted "agent"
that emits the known-correct CISP op stream for each analytic task (the brief's
own ``reference`` stream, whose ground truth is arithmetic -- see
``eval/corpus/analytic.py``).

This is the standard way to smoke-test an RL env before spending on a model. It
validates reset -> step -> reward -> verify -> trajectory -> done and the eval
harness, reproducibly. It is emphatically NOT a model result: ``model=
"reference-policy"`` on the leaderboard row says so loudly. Plugging a real model
in is then a ONE-LINE change: swap ``reference_policy`` for a model-driven policy
``def model_policy(task): return llm_rollout(task.text)`` over the SAME env, the
SAME ``run_episode``, the SAME ``evaluate`` and the SAME leaderboard plumbing.

WHAT IT PROVES
==============
* Every episode CLOSES (``done=True``) through the real ``CADGymEnv``.
* The reference policy achieves the measured geometric contract (``grade.solved``
  against the brief's closed-form volume/bbox/genus + point probes) on the tasks
  it should.
* A DELIBERATELY WRONG op stream is CAUGHT: the environment happily builds a
  valid-but-wrong solid and hands back a POSITIVE step reward (a plain valid-solid
  reward cannot tell right from wrong -- that is the whole reason a measured
  oracle exists), yet the grader scores it ``solved=False`` and its episode SCORES
  WORSE than the reference. The environment DISCRIMINATES; it does not rubber-stamp.
* The trajectory is a real :class:`~harnesscad.agents.agent.tool_trajectory.ToolTrajectory`
  of (think, tool_call, tool_response) steps carrying the per-step (obs, action,
  reward) structure.
* A real :class:`~harnesscad.eval.leaderboard.hardcorpus_board.Standing` is
  produced from the measured run and ``ranking()`` places it above the wrong policy.

Stdlib-only + the kernel-free ``frep`` backend (no OCCT / cadquery, runs anywhere).
Deterministic (frep samples a fixed grid; no wall clock, no randomness). ASCII.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.agents.agent.tool_schema import InterfaceResult, ToolCall
from harnesscad.agents.agent.tool_trajectory import (ToolTrajectory,
                                                     TrajectoryStep)
from harnesscad.core.cisp.ops import Extrude, Op
from harnesscad.eval.corpus import dev
from harnesscad.eval.corpus.grade import grade
from harnesscad.eval.corpus.spec import Brief
from harnesscad.eval.leaderboard.hardcorpus_board import (Standing, Board,
                                                          ranking)
from harnesscad.io.backends.frep import FRepBackend
from harnesscad.io.surfaces.mcp.gym import CADGymEnv

__all__ = [
    "TASK_IDS", "tasks", "reference_policy", "wrong_policy", "EpisodeResult",
    "run_episode", "evaluate", "build_standing", "leaderboard", "ARTIFACT_DIR",
    "write_artifacts", "selfcheck", "main",
]

Policy = Callable[[Brief], List[Op]]


def _headless() -> None:
    """Keep this frep-only, headless run free of the optional OCCT/cadquery render.

    ``CADGymEnv._observe`` calls the render module on every step to report per-view
    availability; where cadquery/OCP IS installed that import loads an OCCT C++
    runtime that segfaults at interpreter teardown (exit 139), which would leave a
    passing selfcheck/pytest process crashing on the way out. This backend needs no
    kernel, so we make ``import cadquery`` raise -- render then takes its own
    documented headless skip path (image=None, note="rendering unavailable") and
    the geometry, which is pure frep SDF, is untouched. We never clobber a cadquery
    that another caller has already imported.
    """
    if "cadquery" not in sys.modules:
        sys.modules["cadquery"] = None  # type: ignore[assignment]


_headless()

#: The curated task set: analytic dev briefs whose closed-form ground truth the
#: frep sampler can actually RESOLVE (the two thin-wall briefs in the dev split,
#: dev_plate_thin_100x50x3 and dev_hollow_box_120x80x30_t3, are honestly
#: UNMEASURABLE on a 48-cell grid -- see grade.resolvable -- so they are left out
#: of a loop-closure demo rather than charged to the policy). Ten parts spanning
#: prisms, cylinders, through-holes, an annulus, an inward shell, a fillet, a
#: chamfer, a boolean cut and a boolean union -- every measured family.
TASK_IDS: Tuple[str, ...] = (
    "dev_plate_60x40x10",
    "dev_disc_d40_h12",
    "dev_plate_hole_centre",
    "dev_plate_hole_four",
    "dev_spacer_d40_bore14",
    "dev_hollow_box_60x40x20_t3",
    "dev_fillet_plate_60x40x10_r3",
    "dev_chamfer_plate_60x40x10_d2",
    "dev_notched_block_40x40x20",
    "dev_l_bracket_60x40",
)

#: Where the committed "it actually ran" artifacts are written.
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def tasks(ids: Optional[Sequence[str]] = None) -> List[Brief]:
    """The Brief objects for ``ids`` (default :data:`TASK_IDS`), in order."""
    return [dev.by_id(t) for t in (ids if ids is not None else TASK_IDS)]


# ===========================================================================
# Policies -- scripted "agents" over the env's CISP action space
# ===========================================================================
def reference_policy(task: Brief) -> List[Op]:
    """The KNOWN-CORRECT CISP op stream for a task.

    This is the scripted stand-in for a model: it returns the brief's own
    hand-written ``reference`` stream (the SHAPE TARGET whose ground truth is
    arithmetic). A real model policy would map ``task.text`` -> op stream instead;
    everything downstream is identical.
    """
    return list(task.reference)


def wrong_policy(task: Brief) -> List[Op]:
    """A DELIBERATELY WRONG op stream: the same plan with a corrupted extrude.

    The final ``extrude`` distance is cut to 30% of the correct thickness. The
    result is still a perfectly VALID solid -- the environment builds it and pays a
    positive step reward -- but its volume and bounding box no longer match the
    brief's closed form, so the measured oracle scores it ``solved=False``. This is
    the control that proves the environment MEASURES rather than rubber-stamps: a
    valid-solid reward alone would pass this part.
    """
    ops = list(task.reference)
    for i in range(len(ops) - 1, -1, -1):
        op = ops[i]
        if isinstance(op, Extrude):
            ops[i] = dataclasses.replace(op, distance=op.distance * 0.3)
            return ops
    # No extrude to corrupt: fall back to appending a stray zero-length nothing is
    # not valid, so instead scale the whole plan down is out of scope -- every
    # curated task has an extrude, so this branch is unreachable for TASK_IDS.
    return ops


# ===========================================================================
# Episode: drive the real CADGymEnv and record a real trajectory
# ===========================================================================
@dataclass
class EpisodeResult:
    """The measured outcome of ONE episode driven through the real env."""

    task_id: str
    model: str
    completed: bool                 # env reported done=True and every op applied
    n_actions: int
    total_reward: float
    mean_reward: float
    contract_pass: bool             # grade.solved: the measured geometric contract
    built: bool                     # a valid solid was produced at all
    unmeasurable: bool
    bbox_ok: bool
    volume_ok: bool
    genus_ok: bool
    probes_ok: bool
    measured_volume: Optional[float]
    expected_volume: float
    volume_rel_error: float
    measured_bbox: Optional[List[float]]
    expected_bbox: List[float]
    score: float                    # scalar the wrong-vs-right proof compares on
    reasons: List[str] = field(default_factory=list)
    final_digest: str = ""
    trajectory: Optional[ToolTrajectory] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "model": self.model,
            "completed": self.completed, "n_actions": self.n_actions,
            "total_reward": self.total_reward, "mean_reward": self.mean_reward,
            "contract_pass": self.contract_pass, "built": self.built,
            "unmeasurable": self.unmeasurable,
            "bbox_ok": self.bbox_ok, "volume_ok": self.volume_ok,
            "genus_ok": self.genus_ok, "probes_ok": self.probes_ok,
            "measured_volume": self.measured_volume,
            "expected_volume": self.expected_volume,
            "volume_rel_error": self.volume_rel_error,
            "measured_bbox": self.measured_bbox,
            "expected_bbox": self.expected_bbox,
            "score": self.score, "reasons": list(self.reasons),
            "final_digest": self.final_digest,
        }


def _episode_score(contract_pass: bool, volume_rel_error: float) -> float:
    """A scalar in which a wrong episode is strictly below the reference.

    ``1.0`` for a part that satisfies the measured contract, minus the (clamped)
    relative volume error. A reference build lands at ~1.0; a valid-but-wrong build
    with a 70% volume miss lands near -0.7. Purpose-built so the wrong policy
    SCORES WORSE -- the environment's discrimination made a number.
    """
    return (1.0 if contract_pass else 0.0) - min(abs(volume_rel_error), 1.0)


def _trajectory_step(op: Op, obs: Dict[str, Any], reward: float,
                     done: bool, info: Dict[str, Any]) -> TrajectoryStep:
    """Build one real ReAct trajectory step from an env transition.

    ``think`` is deterministic scripted reasoning; ``call`` is the CISP op as a
    typed ToolCall (op tag + its parameters); the ``tool_response`` is the env's
    own (ok, reward, digest) verdict, labelled success/fail exactly as the ReAct
    format prescribes.
    """
    args = {k: v for k, v in op.to_dict().items() if k != "op"}
    think = ("emit the %s op (step %d/%s); it advances the build toward the "
             "target part" % (op.OP, info.get("step", 0), info.get("n_ops", "?")))
    response = json.dumps({
        "ok": bool(info.get("ok")),
        "reward": reward,
        "applied": info.get("applied"),
        "digest": info.get("digest"),
        "validity_ok": obs.get("validity", {}).get("ok"),
        "done": done,
    }, sort_keys=True)
    return TrajectoryStep(
        think=think,
        call=ToolCall(name=op.OP, arguments=args),
        result=InterfaceResult(success=bool(info.get("ok")),
                               description=response,
                               produced_object=info.get("digest")))


def run_episode(env: CADGymEnv, task: Brief, policy: Policy,
                model: str = "reference-policy") -> EpisodeResult:
    """Drive the real ``CADGymEnv`` for one task and return a real outcome.

    reset -> step the policy's ops one at a time -> collect (obs, action, reward,
    verification) per step -> terminate -> measure the final part against the
    brief's closed-form ground truth (the same ``grade`` referee the corpus uses,
    at ``verify_level="core"`` -- it NEVER consults the verifier fleet it is
    scoring). Returns the trajectory object plus everything the verifier reported.
    """
    ops = policy(task)
    # The env sets done=True when it has taken max_steps steps; sizing it to the
    # plan length makes the last op close the episode through the env's own done
    # logic (no manual termination).
    env.max_steps = len(ops)
    obs = env.reset()

    traj = ToolTrajectory()
    total_reward = 0.0
    done = False
    applied_ok = True
    final_digest = ""
    for op in ops:
        obs, reward, done, info = env.step(op)
        total_reward += reward
        final_digest = str(info.get("digest", ""))
        traj.add(_trajectory_step(op, obs, reward, done, info))
        if not info.get("ok"):
            applied_ok = False
    traj.completed = bool(done and applied_ok)

    # Measure the final part against the brief's independent truth. with_shape is
    # off: the ENVELOPE oracle (built + bbox + volume + genus + point probes) is
    # the measured geometric contract, and it is what discriminates here.
    sc = grade(task, ops, backend="frep", with_shape=False)
    measured_vol = sc.measured.get("volume")
    measured_bbox = sc.measured.get("bbox") or None
    rel = (abs((measured_vol or 0.0) - task.volume) / max(task.volume, 1e-9))
    contract_pass = bool(sc.solved and not sc.unmeasurable)

    return EpisodeResult(
        task_id=task.id, model=model,
        completed=traj.completed,
        n_actions=len(ops),
        total_reward=round(total_reward, 6),
        mean_reward=round(total_reward / len(ops), 6) if ops else 0.0,
        contract_pass=contract_pass,
        built=bool(sc.built),
        unmeasurable=bool(sc.unmeasurable),
        bbox_ok=bool(sc.bbox_ok), volume_ok=bool(sc.volume_ok),
        genus_ok=bool(sc.genus_ok), probes_ok=bool(sc.probes_ok),
        measured_volume=measured_vol,
        expected_volume=task.volume,
        volume_rel_error=round(rel, 6),
        measured_bbox=[float(v) for v in measured_bbox] if measured_bbox else None,
        expected_bbox=[float(v) for v in task.bbox],
        score=round(_episode_score(contract_pass, rel), 6),
        reasons=list(sc.reasons),
        final_digest=final_digest,
        trajectory=traj)


# ===========================================================================
# Evaluate: run every episode, emit the eval table
# ===========================================================================
def _new_env() -> CADGymEnv:
    """A fresh env over the kernel-free frep backend (no OCCT / cadquery)."""
    return CADGymEnv(backend=FRepBackend())


def evaluate(task_list: Sequence[Brief], policy: Policy = reference_policy,
             model: str = "reference-policy",
             env: Optional[CADGymEnv] = None) -> Tuple[List[EpisodeResult], dict]:
    """Run ``policy`` over every task and return (results, eval-table dict).

    The eval table has a per-task row {completed, contract_pass, measured vs
    expected, n_actions, reward} plus the aggregate pass-rate.
    """
    env = env if env is not None else _new_env()
    results = [run_episode(env, t, policy, model=model) for t in task_list]
    n = len(results)
    passed = sum(1 for r in results if r.contract_pass)
    closed = sum(1 for r in results if r.completed)
    built = sum(1 for r in results if r.built)
    table = {
        "model": model,
        "backend": "frep",
        "note": ("SCRIPTED REFERENCE POLICY -- not a model. Proves the "
                 "agent-environment loop closes and the eval harness measures. "
                 "Swap reference_policy for a model policy over the same env."),
        "n_tasks": n,
        "episodes_closed": closed,
        "built": built,
        "contract_pass": passed,
        "pass_rate": round(passed / n, 6) if n else 0.0,
        "rows": [
            {
                "task_id": r.task_id,
                "completed": r.completed,
                "n_actions": r.n_actions,
                "total_reward": r.total_reward,
                "contract_pass": r.contract_pass,
                "measured_volume": r.measured_volume,
                "expected_volume": r.expected_volume,
                "volume_rel_error": r.volume_rel_error,
                "measured_bbox": r.measured_bbox,
                "expected_bbox": r.expected_bbox,
                "reasons": r.reasons,
            }
            for r in results
        ],
    }
    return results, table


# ===========================================================================
# Leaderboard: a real Standing from the measured run
# ===========================================================================
def build_standing(name: str, results: Sequence[EpisodeResult]) -> Standing:
    """Convert measured episode results into ONE real leaderboard Standing.

    The columns map honestly onto the hard-corpus board:
      * ``weak_passed``  = the parts that BUILT a valid solid -- the field's weak
        metric ("a valid solid exists") would call each of these a pass.
      * ``oracle_solved`` = the parts the MEASURED oracle solved (bbox + volume +
        genus + point probes) -- the number that is actually true.
      * ``field_fooled`` = built-but-wrong: parts the weak metric passed and the
        oracle failed. For the reference policy this is 0; for the wrong policy it
        is exactly the count it corrupted.
    """
    n = len(results)
    built = sum(1 for r in results if r.built)
    solved = sum(1 for r in results if r.contract_pass)
    fooled = sum(1 for r in results if r.built and not r.contract_pass)
    failed = {r.task_id: r.reasons for r in results if not r.contract_pass}
    return Standing(name=name, n=n, built=built, oracle_solved=solved,
                    weak_passed=built, field_fooled=fooled, failed=failed)


def leaderboard(named_results: Sequence[Tuple[str, Sequence[EpisodeResult]]]
                ) -> Tuple[Board, List[Standing]]:
    """Build a Board from named result sets and return (board, ranked standings)."""
    board = Board()
    for name, results in named_results:
        board.add(build_standing(name, results))
    return board, board.ranked()


# ===========================================================================
# Artifacts
# ===========================================================================
def _trajectory_to_dict(traj: ToolTrajectory) -> dict:
    return {
        "completed": traj.completed,
        "num_success": traj.num_success,
        "num_fail": traj.num_fail,
        "steps": [
            {
                "think": s.think,
                "tool_call": {"name": s.call.name,
                              "arguments": dict(s.call.arguments)},
                "tool_response": s.result.description,
                "label": s.label,
            }
            for s in traj.steps
        ],
    }


def write_artifacts(results: Sequence[EpisodeResult], table: dict,
                    standings: Sequence[Standing],
                    out_dir: Optional[str] = None) -> Dict[str, str]:
    """Write the committed proof artifacts: one trajectory + the eval table.

    Returns the map of {artifact-name: path}. The trajectory chosen is the first
    task's episode -- a complete, real (think, tool_call, tool_response) rollout.
    ``out_dir`` defaults to :data:`ARTIFACT_DIR`, resolved at CALL time so a test
    can redirect it.
    """
    out_dir = out_dir if out_dir is not None else ARTIFACT_DIR
    os.makedirs(out_dir, exist_ok=True)
    example = results[0]
    traj_payload = {
        "model": example.model,
        "task_id": example.task_id,
        "note": ("A real trajectory produced by driving CADGymEnv with the "
                 "scripted reference policy -- NOT a model rollout."),
        "outcome": example.to_dict(),
        "trajectory": _trajectory_to_dict(example.trajectory),
    }
    # Drop the (large, non-serialisable) trajectory object out of the outcome copy.
    traj_payload["outcome"].pop("trajectory", None)

    table_payload = dict(table)
    table_payload["leaderboard"] = {
        "ranking": [s.to_dict() for s in ranking(list(standings))],
    }

    traj_path = os.path.join(out_dir, "reference_trajectory.json")
    table_path = os.path.join(out_dir, "eval_table.json")
    with open(traj_path, "w", encoding="ascii") as fh:
        json.dump(traj_payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(table_path, "w", encoding="ascii") as fh:
        json.dump(table_payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return {"trajectory": traj_path, "eval_table": table_path}


# ===========================================================================
# Verify-first: run the whole thing and assert the loop closes + discriminates
# ===========================================================================
def selfcheck(task_ids: Optional[Sequence[str]] = None,
              write: bool = True, verbose: bool = True) -> int:
    """Run the whole loop deterministically and assert every invariant.

    Returns 0 on success (and, unless ``write=False``, writes the committed
    artifacts). Asserts, in order:
      1. every reference episode CLOSES (done=True) through the real env;
      2. the reference policy achieves the measured contract on every curated task;
      3. the trajectory carries the expected (obs, action, reward) structure;
      4. a DELIBERATELY WRONG op stream on one task is CAUGHT (contract fails);
      5. the wrong episode SCORES WORSE than the reference (the discrimination proof);
      6. a real leaderboard Standing is produced and ``ranking()`` places the
         reference policy above the wrong policy.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    tlist = tasks(task_ids)
    log("[reference-loop] driving CADGymEnv (frep backend) over %d tasks..."
        % len(tlist))

    ref_results, table = evaluate(tlist, reference_policy, model="reference-policy")

    # 1. every episode closes.
    for r in ref_results:
        assert r.completed, "episode %s did not close (done=False)" % r.task_id
    log("[ok] all %d reference episodes closed (done=True)" % len(ref_results))

    # 2. the reference policy achieves the contract on every curated task.
    n_pass = sum(1 for r in ref_results if r.contract_pass)
    assert n_pass == len(ref_results), (
        "reference policy failed the contract on %d/%d tasks: %s"
        % (len(ref_results) - n_pass, len(ref_results),
           [r.task_id for r in ref_results if not r.contract_pass]))
    log("[ok] reference policy passed the measured contract on %d/%d tasks "
        "(pass-rate %.3f)" % (n_pass, len(ref_results), table["pass_rate"]))

    # 3. trajectory structure: (obs, action, reward) per step.
    traj = ref_results[0].trajectory
    assert traj is not None and len(traj) == ref_results[0].n_actions
    for step in traj.steps:
        assert step.think and step.call.name and step.result is not None
        payload = json.loads(step.result.description)
        assert "reward" in payload and "digest" in payload, \
            "trajectory step is missing the (reward, obs-digest) structure"
    assert traj.completed, "the example trajectory is not marked completed"
    log("[ok] trajectory has the expected (think/tool_call/tool_response + "
        "reward + obs-digest) structure over %d steps" % len(traj))

    # 4. + 5. the wrong policy is caught and scores worse -- on the SAME task.
    victim = tlist[0]
    env = _new_env()
    wrong = run_episode(env, victim, wrong_policy, model="wrong-policy")
    ref0 = ref_results[0]
    # The env still built a valid solid and paid a positive reward -- a plain
    # valid-solid reward cannot tell the two apart.
    assert wrong.built, "expected the wrong plan to still build a valid solid"
    assert wrong.total_reward > 0.0, (
        "expected the env to pay a positive step reward for the valid-but-wrong "
        "build (that is exactly why a measured oracle is needed)")
    assert not wrong.contract_pass, (
        "the wrong plan was NOT caught by the measured oracle -- the loop would be "
        "rubber-stamping")
    assert wrong.score < ref0.score, (
        "wrong episode did not score worse than the reference (%.4f vs %.4f)"
        % (wrong.score, ref0.score))
    log("[ok] wrong policy on %s: env still paid reward=%.3f and BUILT a valid "
        "solid, but the oracle caught it (contract_pass=%s)"
        % (victim.id, wrong.total_reward, wrong.contract_pass))
    log("     score: reference=%.4f  wrong=%.4f  (delta=%.4f) -- the environment "
        "DISCRIMINATES" % (ref0.score, wrong.score, ref0.score - wrong.score))
    log("     volume: reference=%.1f mm3  wrong=%.1f mm3  expected=%.1f mm3"
        % (ref0.measured_volume or 0.0, wrong.measured_volume or 0.0,
           victim.volume))

    # 6. a real leaderboard Standing is produced and ranks.
    #    The wrong-policy row uses the wrong plan on EVERY task, so its measured
    #    rate is strictly lower and the field-fooled count is non-zero.
    wrong_results = evaluate(tlist, wrong_policy, model="wrong-policy")[0]
    board, ranked = leaderboard([
        ("reference-policy", ref_results),
        ("wrong-policy", wrong_results),
    ])
    assert len(ranked) == 2
    assert ranked[0].name == "reference-policy", (
        "ranking() did not place the reference policy first: %s"
        % [s.name for s in ranked])
    assert ranked[0].oracle_rate > ranked[1].oracle_rate, \
        "reference oracle-rate is not above the wrong policy's"
    assert ranked[1].field_fooled > 0, (
        "the wrong policy should have fooled the field's weak metric on the parts "
        "it corrupted")
    ref_row = ranked[0]
    log("[ok] leaderboard row produced and ranked:")
    log("     #1 %-16s n=%d oracle_rate=%.3f weak_rate=%.3f fooled=%d"
        % (ref_row.name, ref_row.n, ref_row.oracle_rate, ref_row.weak_rate,
           ref_row.field_fooled))
    log("     #2 %-16s n=%d oracle_rate=%.3f weak_rate=%.3f fooled=%d"
        % (ranked[1].name, ranked[1].n, ranked[1].oracle_rate,
           ranked[1].weak_rate, ranked[1].field_fooled))

    if write:
        paths = write_artifacts(ref_results, table, [ref_row, ranked[1]])
        log("[ok] wrote artifacts:")
        for k, p in paths.items():
            log("     %-11s %s" % (k, p))

    log("[reference-loop] SELFCHECK PASSED -- the agent-environment loop closes, "
        "measures, and ranks.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Drive CADGymEnv with a scripted reference policy: close the "
                    "loop, emit a trajectory + eval table + a leaderboard row.")
    ap.add_argument("--selfcheck", action="store_true",
                    help="run the whole loop deterministically and assert every "
                         "invariant (exit 0 on success).")
    ap.add_argument("--no-write", action="store_true",
                    help="do not (re)write the committed artifacts.")
    ap.add_argument("--json", action="store_true",
                    help="print the eval table as JSON (implies a run).")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.json and not args.selfcheck:
        _, table = evaluate(tasks(), reference_policy)
        print(json.dumps(table, indent=2, sort_keys=True))
        return 0

    # Default action IS the selfcheck: verify-first.
    return selfcheck(write=not args.no_write, verbose=True)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
