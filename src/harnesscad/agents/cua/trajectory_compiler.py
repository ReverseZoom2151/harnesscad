"""trajectory_compiler — COMPILE expert GUI trajectories at p=1.0, don't FILTER them.

This is the key insight of ``audit/cua_synthesis.md`` made executable. Every other
GUI-RL / CUA training pipeline builds its data the same way: sample an agent policy,
keep the rollouts that happen to succeed (rejection sampling). At cold start a
general VLM's success rate in a CAD viewport is a few percent, so you are throwing
away 95%+ of the compute and the surviving data is biased toward the easy tasks the
weak policy could already do. Worse, you cannot even *label* the survivors cheaply:
Fara-7B had to TRAIN a verifier model (and publish CUAVerifierBench to measure how
wrong that model is) because nobody could auto-check "did the trajectory succeed".

HarnessCAD does not sample and does not guess. We OWN a known-correct CISP op stream,
so:

* the **action** at each step is DERIVED from the op (a deterministic op -> GUI-verb
  map, :func:`action_for_op`) rather than predicted by a policy. The policy that
  emits exactly these actions has ``p(action) = 1.0`` at every step -- there is
  nothing to sample and nothing to reject.
* the **verdict** at each step is LABELLED by the exact geometric oracle
  (:mod:`harnesscad.agents.cua.grade` + :mod:`harnesscad.io.gate`), not predicted.
  This is precisely the ``(observation, action, oracle-verdict)`` supervision Fara
  spends a whole model to approximate; we compute it.

The result is a :class:`~harnesscad.agents.cua.verified_trajectory.VerifiedTrajectory`
that is correct by construction -- the compilation of an expert plan, not the
filtrate of a bad one.

Two compilation paths, both RUNNABLE (later) and both import-safe (nothing here runs
at import; a live app / the scripted kernel is touched only when a ``compile*`` method
is actually called):

* :meth:`ExpertTrajectoryCompiler.compile` -- the SCRIPTED path. Observations are
  symbolic (the op prefix built so far); the per-step verdict comes from replaying
  the prefix through the scripted ``FreeCADBackend`` (the kernel that matches ANALYTIC
  to 4.5e-16); the final verdict is the output gate + optional target. No GUI needed.
* :meth:`ExpertTrajectoryCompiler.compile_from_env` -- the GUI path. Drives an
  INJECTED :class:`~harnesscad.core.environment.Environment` (e.g. the live FreeCAD
  GUI) op by op, captures the environment's own read-back as each step's observation
  and verdict, and grades the finished part with :func:`grade_ops`. This is the data
  that proves the GUI DRIVE was faithful, not merely that the plan was sound.

.. warning:: REWARD-HACKING GUARD -- READ BEFORE WIRING ANYTHING.

   This compiler exists because we can reach the app's Python interpreter / scripted
   kernel and thereby label a trajectory for free. THAT CHANNEL IS THE ORACLE AND
   MUST NEVER APPEAR IN A TRAINING ENVIRONMENT'S ACTION SPACE. If a policy being
   trained can reach a Python console (FreeCAD's ``PythonConsole``, Blender's, or a
   scripted backend), the optimal policy is the degenerate one:

       paste ``Part.makeBox(...)`` (or the whole answer script), announce done,
       collect full reward, and learn NOTHING about driving the GUI.

   The console/kernel is legitimate ONLY as (a) the labeller that compiles this
   expert data offline and (b) a Tier-0 *evaluation* agent. Compiling expert
   trajectories with the oracle is the intended use; letting a trained policy call
   the same oracle is the reward hack. Keep the oracle on the data-generation side of
   the wall. See the identical warning in :mod:`harnesscad.eval.grounding.corpus` and
   :mod:`harnesscad.io.cua.viewport`.

Pure stdlib at import time. Deterministic. No model, ever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from harnesscad.core.cisp.ops import Op
from harnesscad.agents.cua.verified_trajectory import (
    OracleVerdict,
    StepLabeller,
    TrajectoryStep,
    VerifiedTrajectory,
    label_trajectory,
)

__all__ = [
    "action_for_op",
    "symbolic_observation",
    "oracle_step_labeller",
    "trusted_prefix_labeller",
    "gate_final_verdict",
    "ExpertTrajectoryCompiler",
    "compile_expert_trajectory",
]


def _op_tag(op: Any) -> str:
    """The stable op tag (``new_sketch``, ``extrude`` ...), duck-typed."""
    return str(getattr(type(op), "OP", "") or getattr(op, "OP", "") or "?")


def action_for_op(op: Any) -> Dict[str, Any]:
    """Derive the GUI action from a CISP op -- deterministic, no policy involved.

    The action is the op's own serialisation split into a ``verb`` (the op tag) and
    its ``params`` (every other field). ``source="compiled"`` marks that this action
    was produced by compilation from a known op, not predicted by a model -- the
    provenance that makes ``p(action) = 1.0`` an honest claim rather than a wish.
    """
    if hasattr(op, "to_dict"):
        d = dict(op.to_dict())
    elif isinstance(op, dict):
        d = dict(op)
    else:  # pragma: no cover - defensive; ops are dataclasses or dicts
        d = {"op": _op_tag(op)}
    verb = d.pop("op", _op_tag(op))
    return {"verb": verb, "params": d, "source": "compiled"}


def symbolic_observation(index: int, built_ops: Sequence[Any],
                         pending_op: Any) -> Dict[str, Any]:
    """A pixel-free observation: what the document holds just before step ``index``.

    The scripted path has no screenshot, and it needs none: the observation a policy
    would be trained against here is the STRUCTURED state (the ops already committed
    and the op about to be driven), which is exactly what the semantic-GUI tier
    (:mod:`harnesscad.io.cua.uia`) exposes anyway. Deterministic given the op stream.
    """
    return {
        "kind": "symbolic",
        "step": int(index),
        "ops_built": [_op_tag(o) for o in built_ops],
        "op_pending": _op_tag(pending_op),
    }


# --------------------------------------------------------------------------- #
# Per-step and final verdict labellers -- the exact oracle, composed.
# --------------------------------------------------------------------------- #
def oracle_step_labeller(ops: Sequence[Op]) -> StepLabeller:
    """A per-step oracle that replays the op PREFIX through the scripted kernel.

    Returns a :data:`StepLabeller` closure. On the call for step ``k`` it applies
    ``ops[k]`` to a scripted :class:`~harnesscad.io.backends.freecad.FreeCADBackend`
    that has been carrying the prefix ``ops[:k]`` -- so a step is VERIFIED iff the
    prefix up to and including it still builds, and REJECTED the moment the kernel
    refuses an op (after which every later step is rejected too, because a broken
    prefix cannot be un-broken).

    This composes the SAME public backend surface the grounding corpus uses
    (``backend.apply`` / ``backend.reset``); it does not touch ``io/backends``. It is
    imported lazily, so this module stays import-safe and the kernel is spun up only
    when a caller actually compiles. ``label_trajectory`` invokes the closure once per
    step in index order, which is the order this stateful labeller assumes.
    """
    from harnesscad.io.backends.freecad import FreeCADBackend

    op_list = list(ops)
    backend = FreeCADBackend()
    backend.reset()
    broken_at: Dict[str, Optional[int]] = {"index": None}

    def label(step: TrajectoryStep) -> OracleVerdict:
        k = step.index
        if broken_at["index"] is not None:
            return OracleVerdict.rejected(
                "the prefix already broke at op %d; this step is unreachable"
                % broken_at["index"])
        if k < 0 or k >= len(op_list):  # pragma: no cover - defensive
            return OracleVerdict.rejected("step index %d is outside the op stream" % k)
        op = op_list[k]
        tag = _op_tag(op)
        result = backend.apply(op)
        if not getattr(result, "ok", False):
            broken_at["index"] = k
            msgs = "; ".join(getattr(d, "message", "")
                             for d in getattr(result, "diagnostics", []) or [])
            return OracleVerdict.rejected(
                "op '%s' rejected by the scripted kernel: %s" % (tag, msgs or "no detail"))
        return OracleVerdict.verified(
            "op '%s' applied; the prefix builds through the scripted kernel" % tag)

    return label


def trusted_prefix_labeller() -> StepLabeller:
    """A labeller that trusts the caller's known-correct contract: every step VERIFIED.

    Valid ONLY when the caller independently guarantees the op stream is correct AND
    the finished part is gate-checked elsewhere (e.g. the corpus generator, which runs
    :func:`gate_final_verdict` on the whole stream). It exists so a trajectory can be
    compiled in pure-stdlib mode with no kernel available; it asserts nothing about the
    geometry and must not be used to MANUFACTURE a verified label for an unchecked
    stream. When in doubt, use :func:`oracle_step_labeller`, which cannot be fooled.
    """
    def label(step: TrajectoryStep) -> OracleVerdict:
        return OracleVerdict.verified(
            "declared-correct op stream (caller-guaranteed; not independently graded "
            "per step)", source="declared")
    return label


def gate_final_verdict(ops: Sequence[Op], target: Any = None) -> OracleVerdict:
    """The trajectory-level verdict: run the stream through the gate + optional target.

    Composes :func:`harnesscad.agents.cua.grade.scripted_measure` (which drives the
    scripted kernel and the output gate) and, when a
    :class:`~harnesscad.agents.cua.briefs.Target` is given, its ``satisfied`` check on
    the measured geometry. Verified iff the gate passes AND the target is met. Imported
    lazily; runs the kernel only when called.
    """
    from harnesscad.agents.cua.grade import scripted_measure

    try:
        metrics, gate_ok, failures = scripted_measure(ops)
    except Exception as exc:  # noqa: BLE001 - unbuildable-by-script is a real verdict
        return OracleVerdict.rejected(
            "the op stream is not buildable by the scripted kernel: %s: %s"
            % (type(exc).__name__, exc))
    reasons: List[str] = []
    if not gate_ok:
        reasons.append("output gate: " + "; ".join(failures))
    if target is not None:
        target_ok, misses = target.satisfied(metrics)
        if not target_ok:
            reasons.append("target: " + "; ".join(misses))
    if reasons:
        return OracleVerdict.rejected(" | ".join(reasons))
    return OracleVerdict.verified("gate passed and target met" if target is not None
                                  else "gate passed")


# --------------------------------------------------------------------------- #
# The compiler.
# --------------------------------------------------------------------------- #
#: An observer maps ``(step_index, ops_built_before, op_pending)`` to an observation
#: dict. The default is :func:`symbolic_observation`; the GUI path injects one that
#: reads the live environment.
Observer = Callable[[int, Sequence[Any], Any], Dict[str, Any]]


@dataclass
class ExpertTrajectoryCompiler:
    """Compiles a known-correct op stream into a p=1.0 :class:`VerifiedTrajectory`.

    ``step_labeller_factory`` builds the per-step oracle from the op list (default:
    :func:`oracle_step_labeller`, the scripted-kernel prefix builder). ``observer``
    produces each step's observation (default: :func:`symbolic_observation`).
    ``final_verdict_fn`` produces the trajectory-level verdict (default:
    :func:`gate_final_verdict`). All three are injection points so the compiler can be
    driven with no kernel (pure test doubles) or against the live GUI, without editing
    anything it composes.
    """

    step_labeller_factory: Callable[[Sequence[Op]], StepLabeller] = oracle_step_labeller
    observer: Observer = symbolic_observation
    final_verdict_fn: Callable[..., OracleVerdict] = staticmethod(gate_final_verdict)

    def compile(self, brief: str, ops: Sequence[Op], *, target: Any = None,
                trajectory_id: str = "",
                step_labeller: Optional[StepLabeller] = None,
                final_verdict: Optional[OracleVerdict] = None) -> VerifiedTrajectory:
        """Compile ``ops`` (known-correct) into a labelled VerifiedTrajectory.

        Each step's action is DERIVED from its op (:func:`action_for_op`) and its
        observation from :attr:`observer`; the per-step and final verdicts come from
        the oracle. Because the actions are compiled and not sampled, this is p=1.0 by
        construction. Runnable (it will spin up the scripted kernel through the default
        labeller / final verdict); nothing runs until this method is called.
        """
        op_list = list(ops)
        steps: List[TrajectoryStep] = []
        for k, op in enumerate(op_list):
            steps.append(TrajectoryStep(
                index=k,
                observation=self.observer(k, op_list[:k], op),
                action=action_for_op(op),
            ))
        traj = VerifiedTrajectory(brief=brief, steps=steps, trajectory_id=trajectory_id)
        labeller = step_labeller or self.step_labeller_factory(op_list)
        fv = final_verdict if final_verdict is not None \
            else self.final_verdict_fn(op_list, target)
        return label_trajectory(traj, labeller, fv)

    def compile_from_env(self, env: Any, brief: str, ops: Sequence[Op], *,
                         target: Any = None, trajectory_id: str = "") -> VerifiedTrajectory:
        """The GUI path: drive an INJECTED environment op by op and record the truth.

        ``env`` is any :class:`~harnesscad.core.environment.Environment` (in practice
        the live :class:`~harnesscad.io.cua.environment_freecad.FreeCADGuiEnvironment`).
        The compiler ``reset``s it, ``step``s each op, and takes the environment's OWN
        read-back (:attr:`StepResult.verified`) as the step verdict and its
        :meth:`observe` as the step observation -- so a step is verified iff the GUI
        proved the action took effect, not iff we hoped it did. The finished part is
        graded with :func:`grade_ops`, and that becomes the final verdict.

        This is injected, never wired into ``environment_freecad`` -- the environment is
        composed from the outside. Runnable; nothing runs until called.
        """
        from harnesscad.agents.cua.grade import grade_ops

        op_list = list(ops)
        env.reset()
        steps: List[TrajectoryStep] = []
        broke = False
        for k, op in enumerate(op_list):
            observation = _observation_dict(env.observe())
            if broke:
                verdict = OracleVerdict.rejected(
                    "a prior GUI step failed; this action was not driven", source="freecad-gui")
                steps.append(TrajectoryStep(index=k, observation=observation,
                                            action=action_for_op(op), verdict=verdict))
                continue
            result = env.step(op)
            if getattr(result, "verified", False):
                verdict = OracleVerdict.verified(
                    "the GUI read back the effect of op '%s'" % _op_tag(op),
                    source="freecad-gui")
            else:
                broke = True
                diags = "; ".join(getattr(d, "message", "")
                                  for d in getattr(result, "diagnostics", []) or [])
                verdict = OracleVerdict.rejected(
                    "the GUI could not verify op '%s': %s" % (_op_tag(op), diags or "no detail"),
                    source="freecad-gui")
            steps.append(TrajectoryStep(index=k, observation=observation,
                                        action=action_for_op(op), verdict=verdict))

        grade = grade_ops(env, op_list, target)
        if getattr(grade, "solved", False):
            final = OracleVerdict.verified(getattr(grade, "reason", "solved"),
                                           source="cad_oracle")
        else:
            final = OracleVerdict.rejected(getattr(grade, "reason", "not solved"),
                                           source="cad_oracle")
        return VerifiedTrajectory(brief=brief, steps=steps, final_verdict=final,
                                  trajectory_id=trajectory_id)


def _observation_dict(observation: Any) -> Dict[str, Any]:
    """An Observation (or anything) -> a plain dict, defensively."""
    if observation is None:
        return {}
    if hasattr(observation, "to_dict"):
        try:
            return dict(observation.to_dict())
        except Exception:  # noqa: BLE001 - an observation must never break compilation
            return {}
    if isinstance(observation, dict):
        return dict(observation)
    return {"repr": str(observation)}


def compile_expert_trajectory(brief: str, ops: Sequence[Op], *, target: Any = None,
                              trajectory_id: str = "") -> VerifiedTrajectory:
    """Convenience: compile with the default (scripted-kernel oracle) compiler.

    The one call most callers need. Equivalent to
    ``ExpertTrajectoryCompiler().compile(brief, ops, target=target, ...)``.
    """
    return ExpertTrajectoryCompiler().compile(
        brief, ops, target=target, trajectory_id=trajectory_id)
