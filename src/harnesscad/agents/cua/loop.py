"""loop — a MODEL in the verified GUI environment, reusing AgentHarness.

The gap this closes
-------------------
The GUI environment (:mod:`harnesscad.io.cua.environment_freecad`) was built,
verified to volume 6000.0000000000, and never had an agent put in it. This
module puts one in it. The loop is::

    brief -> model -> plan -> DRIVE THE GUI -> export -> MEASURE -> correct

Why this reuses AgentHarness instead of being a fifth loop
---------------------------------------------------------
:class:`harnesscad.core.harness.AgentHarness` is the survivor of the loop
collapse and it already owns the exact ReAct spine we need: pre-flight -> plan ->
loop-detect -> dispatch -> verify -> repair, with ``feedback_tiers=`` /
``gated=`` / ``loop_detector=`` as parameters and a clean external-dispatch seam
(``executor.apply_ops(ops) -> ApplyOpsResult``). We reuse it.

But it cannot drive a :class:`~harnesscad.core.environment.Environment` *as-is*,
and here is exactly why: ``AgentHarness.run`` is written against the
``HarnessSession`` surface — it calls ``session.summary()``, ``session.digest()``
and ``session.checkpoint()`` every iteration, and its oracle/contract paths reach
into ``session.backend`` and ``session.opdag``. Two of those are precisely the
``GeometryBackend`` contracts :mod:`harnesscad.core.environment` documents a live
GUI CANNOT honour: ``FreeCADGuiEnvironment.state_digest()`` RAISES ``CapabilityError``
by design (a running Qt app has no content hash of its document, and it will not
fabricate one). So on the first iteration ``AgentHarness._entry`` would call
``session.digest()`` and the run would die.

The fix is an ADAPTER, not a rewrite and not a faked digest:

* :class:`EnvironmentExecutor` implements the harness's dispatch seam by calling
  ``env.step(ops)`` and translating the verified ``StepResult`` into an
  ``ApplyOpsResult``. This is where the GUI actually gets driven.
* :class:`EnvSession` presents the tiny slice of the ``HarnessSession`` surface
  the harness touches. Its ``digest()`` is an HONEST, opaque progress token
  derived from the ops the environment reports it has BUILT — explicitly not a
  content hash, used by the harness only as a trajectory field. It never claims
  to be geometry, because the geometry lives one layer down in the grade.
* :class:`GeometryGradeVerifier` is a harness-level verifier that ignores the
  ``(backend, opdag)`` it is handed and instead measures the Environment's part
  against the brief target. This is how "MEASURE -> correct" rides the harness's
  own verify+repair machinery: a part that does not yet meet spec comes back as
  an ERROR diagnostic, which the gate feeds to the planner for the next attempt.

With ``memory=None`` and ``contract=None`` (both defaults here) the oracle and
contract paths that would dereference ``session.backend`` / ``session.opdag`` are
never taken, so those stay ``None`` and are never touched.

The FINAL, heavyweight adjudication — the full differential against the scripted
backend and the output gate — is :func:`harnesscad.agents.cua.grade.grade_ops`,
run once after the harness converges (or gives up). The harness-level verifier is
the light in-loop check that drives repair; the grade is the verdict.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from harnesscad.agents.cua.grade import GradeResult, grade_ops
from harnesscad.core.cisp.ops import Op
from harnesscad.core.cisp.protocol import ApplyOpsResult
from harnesscad.core.environment import coerce_ops
from harnesscad.eval.verifiers.verify import Diagnostic, Severity


class ActionTier(Enum):
    """The three-tier action space. Always take the highest AVAILABLE.

    * ``SCRIPT`` (tier 0): the app's API. The scripted backend IS this; the GUI
      agent does not use it, on purpose — a policy that can script geometry does
      not learn to drive a GUI.
    * ``SEMANTIC_GUI`` (tier 1): menus, toolbars, dialog fields, resolved by the
      accessibility tree and invoked coordinate-free. The box recipe is here.
    * ``VIEWPORT_PICK`` (tier 2): a click inside the 3D viewport, but the pixel is
      COMPUTED from our own B-rep and the known camera and the pick is adjudicated
      by the app's own ray-picker — never guessed by a vision model. Edge/face
      picks are here (see :mod:`harnesscad.io.cua.picks`).
    """

    SCRIPT = 0
    SEMANTIC_GUI = 1
    VIEWPORT_PICK = 2


@dataclass
class TierCounts:
    """Per-tier action tally, for the report. An action is counted only if the
    environment VERIFIED it (read-back), because an unverified action is not an
    action."""

    script: int = 0
    semantic_gui: int = 0
    viewport_pick: int = 0
    refused: int = 0

    def add(self, tier: ActionTier, n: int = 1) -> None:
        if tier is ActionTier.SCRIPT:
            self.script += n
        elif tier is ActionTier.SEMANTIC_GUI:
            self.semantic_gui += n
        elif tier is ActionTier.VIEWPORT_PICK:
            self.viewport_pick += n

    def to_dict(self) -> dict:
        return {"script": self.script, "semantic_gui": self.semantic_gui,
                "viewport_pick": self.viewport_pick, "refused": self.refused}


class EnvironmentExecutor:
    """The harness's dispatch seam, wired to an :class:`Environment`.

    ``apply_ops(ops) -> ApplyOpsResult`` is the one method
    ``AgentHarness._dispatch`` calls. Here it drives the GUI: ``env.step(ops)``
    opens the toolbar/dialog, writes-and-reads-back every field, confirms, and
    returns a VERIFIED ``StepResult``, which we translate. The op ``digest`` the
    ``ApplyOpsResult`` carries is the same honest progress token :class:`EnvSession`
    uses — never a content hash.
    """

    def __init__(self, env: Any, counts: Optional[TierCounts] = None) -> None:
        self.env = env
        self.counts = counts if counts is not None else TierCounts()

    def apply_ops(self, ops: List[Op]) -> ApplyOpsResult:
        # Read the env's own outcome log around the step so each executed GUI
        # procedure (a "recipe" — one dialog, or one pick-backed feature) is
        # counted exactly ONCE as an action of its tier, even across repair
        # iterations where earlier outcomes persist.
        before = len(getattr(self.env, "_outcomes", []) or [])
        result = self.env.step(list(ops))
        info = result.info or {}
        executed = int(info.get("executed_ops", 0))
        new_outcomes = (getattr(self.env, "_outcomes", []) or [])[before:]
        for outcome in new_outcomes:
            if not outcome.get("ok"):
                continue
            # A box recipe is a tier-1 semantic-GUI action; a pick-backed recipe
            # tags itself "viewport_pick". Attribute what actually executed.
            tier_name = str(outcome.get("tier", "semantic_gui"))
            tier = {"semantic_gui": ActionTier.SEMANTIC_GUI,
                    "viewport_pick": ActionTier.VIEWPORT_PICK,
                    "script": ActionTier.SCRIPT}.get(tier_name, ActionTier.SEMANTIC_GUI)
            self.counts.add(tier, 1)
        if not result.ok:
            self.counts.refused += 1
        return ApplyOpsResult(
            ok=bool(result.ok),
            applied=executed,
            digest=_ops_token(self.env),
            diagnostics=list(result.diagnostics),
            rejected=None if result.ok else {"pending": info.get("pending_ops", 0)},
        )


def _ops_token(env: Any) -> str:
    """An HONEST opaque progress token from what the env reports it has BUILT.

    NOT a content digest — the environment declares it has none and this does not
    pretend otherwise. It is a deterministic function of the built op stream, used
    only as a trajectory marker so the harness's bookkeeping has a stable string.
    """
    try:
        built = env.observe().state.get("ops_built", [])
    except Exception:  # noqa: BLE001 - observation must never break the loop
        built = []
    blob = json.dumps(built, sort_keys=True)
    return "envops-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


class EnvSession:
    """The slice of ``HarnessSession`` that ``AgentHarness`` actually touches.

    Deliberately minimal and honest. ``backend`` / ``opdag`` are ``None`` and are
    never dereferenced because this loop runs the harness with no memory and no
    contract; ``checkpoint`` is a no-op because a live GUI has no checkpoint we
    would be honest calling one; ``digest`` is the opaque progress token.
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self.backend = None
        self.opdag = None

    def summary(self) -> Dict[str, Any]:
        try:
            obs = self.env.observe()
        except Exception:  # noqa: BLE001
            return {}
        state = dict(obs.state)
        state["environment"] = self.env.capabilities().name
        state["supported_ops"] = list(self.env.capabilities().supported_ops)
        return state

    def digest(self) -> str:
        return _ops_token(self.env)

    def checkpoint(self, label: str) -> None:
        # A live GUI has no content checkpoint. Saying so is the honest behaviour;
        # inventing one (or, worse, calling the app's Save) is exactly what the
        # guardrails forbid.
        return None


@dataclass
class _VerifyResult:
    diagnostics: List[Diagnostic] = field(default_factory=list)


class GeometryGradeVerifier:
    """The in-loop MEASURE step, as a harness-level verifier.

    ``AgentHarness._harness_verify`` calls ``v.check(session.backend, session.opdag)``
    and gates on ``.diagnostics``. This verifier IGNORES those arguments (they are
    ``None`` for a GUI, by design) and instead measures the Environment's part
    against the brief target through the real kernel. A part that does not yet
    satisfy the target comes back as an ERROR, which the harness feeds to the
    planner as the correction — that is "MEASURE -> correct", riding the machinery
    the harness already has.

    It does the LIGHT check (target satisfaction + GUI validity), not the full
    differential; the differential is the final grade, run once after the loop.
    """

    def __init__(self, env: Any, target: Any = None) -> None:
        self.env = env
        self.target = target

    def check(self, backend: Any = None, opdag: Any = None) -> _VerifyResult:
        diags: List[Diagnostic] = []
        try:
            metrics = self.env.measure("full")
        except Exception as exc:  # noqa: BLE001 - a failed read is a real correction
            return _VerifyResult([Diagnostic(
                Severity.ERROR, "no-solid",
                "no measurable solid built yet: %s" % exc)])
        # An empty document reports solid_present=False WITHOUT raising (that is a
        # fact, not a read failure); it is still "nothing to grade -> correct".
        if metrics.get("solid_present") is False or "volume" not in metrics:
            return _VerifyResult([Diagnostic(
                Severity.ERROR, "no-solid",
                "no solid in the document yet: %s" % metrics.get("error", ""))])
        validity = self.env.measure("validity")
        if not (validity.get("is_valid") and validity.get("solids", 0) >= 1):
            diags.append(Diagnostic(Severity.ERROR, "invalid-solid",
                                    "the built solid is not valid/closed"))
        if self.target is not None:
            ok, misses = self.target.satisfied(metrics)
            if not ok:
                diags.append(Diagnostic(
                    Severity.ERROR, "target-miss",
                    "the built part does not meet the brief: " + "; ".join(misses)))
        return _VerifyResult(diags)


@dataclass
class CuaSolve:
    """The result of driving one brief end to end."""

    brief_id: str
    model: str
    solved: bool
    harness_ok: bool
    iterations: int
    stop_reason: str
    tier_counts: TierCounts
    grade: Optional[GradeResult]
    planned_ops: List[dict] = field(default_factory=list)
    trajectory: List[dict] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "brief_id": self.brief_id, "model": self.model, "solved": self.solved,
            "harness_ok": self.harness_ok, "iterations": self.iterations,
            "stop_reason": self.stop_reason,
            "tier_counts": self.tier_counts.to_dict(),
            "grade": None if self.grade is None else self.grade.to_dict(),
            "planned_ops": self.planned_ops, "error": self.error,
        }


def build_cua_harness(env: Any, planner: Any, *, target: Any = None,
                      counts: Optional[TierCounts] = None,
                      loop_detector: Any = None, tracer: Any = None,
                      max_iterations: int = 3,
                      feedback_tiers: Any = None):
    """Wire an :class:`AgentHarness` to drive ``env`` through the GUI.

    Returns ``(harness, executor)``; the executor carries the live tier tally.
    ``memory`` and ``contract`` are intentionally absent so the harness never
    dereferences ``session.backend`` / ``session.opdag`` (see the module
    docstring). ``gated=False`` because our executor IS the write gate — the
    Environment's guardrails resolve-before-click and refuse before acting.
    """
    from harnesscad.agents.agent.feedback import MODEL_FACING_TIERS
    from harnesscad.core.harness import AgentHarness

    counts = counts if counts is not None else TierCounts()
    executor = EnvironmentExecutor(env, counts)
    session = EnvSession(env)
    verifier = GeometryGradeVerifier(env, target)
    harness = AgentHarness(
        session=session,
        planner=planner,
        executor=executor,
        loop_detector=loop_detector,
        tracer=tracer,
        verifiers=[verifier],
        max_iterations=max_iterations,
        feedback_tiers=(feedback_tiers if feedback_tiers is not None
                        else MODEL_FACING_TIERS),
        gated=False,
        memory=None,
        oracle=None,
    )
    return harness, executor


def solve(env: Any, llm: Any, brief: Any, *, target: Any = None,
          max_iterations: int = 3, use_tool: bool = False,
          exemplars: int = 2, tracer: Any = None) -> CuaSolve:
    """Drive ONE brief end to end: plan -> DRIVE THE GUI -> measure -> correct,
    then grade the built part against the scripted oracle.

    ``env`` must already be ``reset()`` (a fresh app, document and body). ``brief``
    is either a :class:`~harnesscad.agents.cua.briefs.Brief` or a raw string;
    ``target`` overrides / supplies the acceptance oracle. Never raises: a driver
    or model failure is returned as ``solved=False`` with the error.
    """
    from harnesscad.agents.agent.planner import Planner
    from harnesscad.eval.reliability.loopdetect import LoopDetector

    brief_id = getattr(brief, "id", "adhoc")
    brief_text = getattr(brief, "text", brief)
    if target is None:
        target = getattr(brief, "target", None)
    model_name = getattr(llm, "model", type(llm).__name__)

    planner = Planner(llm, use_tool=use_tool, exemplars=exemplars)
    counts = TierCounts()
    harness, executor = build_cua_harness(
        env, planner, target=target, counts=counts,
        loop_detector=LoopDetector(), tracer=tracer,
        max_iterations=max_iterations)

    result = CuaSolve(brief_id=brief_id, model=model_name, solved=False,
                      harness_ok=False, iterations=0, stop_reason="",
                      tier_counts=counts, grade=None)
    try:
        run = harness.run(brief_text)
    except Exception as exc:  # noqa: BLE001 - a model/driver blow-up is a non-solve
        result.error = "%s: %s" % (type(exc).__name__, exc)
        result.stop_reason = "exception"
        return result

    result.harness_ok = run.ok
    result.iterations = run.iterations
    result.stop_reason = run.stop_reason
    result.trajectory = run.trajectory
    # The ops the agent actually got BUILT in the GUI (verified), for the grade.
    try:
        built = env.observe().state.get("ops_built", [])
        result.planned_ops = built
    except Exception:  # noqa: BLE001
        built = []

    # THE GRADE: the full differential against the scripted backend + the gate.
    ops = _rebuild_ops(built)
    if ops:
        result.grade = grade_ops(env, ops, target)
        result.solved = result.grade.solved
    else:
        result.error = result.error or "the agent built no ops in the GUI"
    return result


def _rebuild_ops(built: Sequence[dict]) -> List[Op]:
    """Turn the env's recorded built-op dicts back into typed Ops for grading."""
    ops: List[Op] = []
    for d in built:
        try:
            ops.extend(coerce_ops(dict(d)))
        except Exception:  # noqa: BLE001 - a malformed record is dropped, not fatal
            continue
    return ops
