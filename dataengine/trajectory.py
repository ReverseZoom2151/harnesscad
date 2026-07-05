"""Trajectory layer — the training-trace record (HARNESS_BLUEPRINT.md sec.17/21).

The harness IS the data flywheel: every session logs
``prompt -> plan -> ops -> geometry -> tests`` off the same trace stream the
loop already emits (``trace.py``: ``{ts, run_id, kind, data}`` events with kinds
run_start / op_applied / verify_result / rejected / checkpoint / run_end).

This module folds that raw event stream into the *one* canonical training
record the blueprint (sec.17) calls for::

    Step        = (S_t, A_t=[reasoning, tool_call], R_t, S_{t+1})
    Trajectory  = ordered [Step] + final scalar reward + sub-goal labels
                  + metadata (prompt / plan)

That single shape feeds GRPO / DPO / STaR (see dataengine/export.py). Per-step
reward comes from the verifier's outcome (pass=+, reject/rollback=-), giving the
"deterministic verifier is simultaneously reward, eval and ceiling" property
(sec.0). Dense / hierarchical credit is supported via **sub-goal labels** (a
reward per verifiable sub-goal), and ``trajectory_slice(to_first_divergence=
True)`` cuts a trace to its first failing step for 3-5x signal density (sec.17).

Design constraints (mirroring trace.py / observe.py): absolute imports, stdlib
only, no wall-clock in any default path. Where available, the op-outcome
reconstruction is delegated to ``observe.Replayer`` (lazy import); a stdlib-only
fallback reads the events directly so this package never hard-depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- reward scale -----------------------------------------------------------
# Per-step reward from the verifier outcome. The magnitudes are the training
# knob; the signs are load-bearing (pass positive, failure negative).
REWARD_PASS: float = 1.0    # op applied + verified ok (a reached sub-goal)
REWARD_FAIL: float = -1.0   # op rolled back (verify-failed) or backend-rejected
REWARD_NEUTRAL: float = 0.0

# Per-op outcome tags (as produced by observe.Replayer / the fallback below).
OUTCOME_APPLIED = "applied"
OUTCOME_ROLLED_BACK = "rolled-back"       # applied then verify-failed -> reverted
OUTCOME_REJECTED = "rejected-backend"      # blocked at the before-tool gate


# =====================================================================
# Action — A_t = [reasoning, tool_call]
# =====================================================================

@dataclass
class Action:
    """The agent's action at step t: its reasoning plus the emitted tool call.

    ``tool_call`` is the typed CISP op dict (``{"op": ..., ...}``) the harness
    executed. ``reasoning`` is the natural-language rationale that produced it
    (empty when the trace did not capture it). The blueprint writes the action
    as the pair ``[reasoning, tool_call]``; ``as_pair()`` returns exactly that.
    """

    reasoning: str = ""
    tool_call: Dict[str, Any] = field(default_factory=dict)

    def as_pair(self) -> list:
        return [self.reasoning, self.tool_call]

    def to_dict(self) -> dict:
        return {"reasoning": self.reasoning, "tool_call": dict(self.tool_call)}


# =====================================================================
# Step — (S_t, A_t, R_t, S_{t+1})
# =====================================================================

@dataclass
class Step:
    """One transition: state_before --action--> reward, state_after.

    State is represented compactly (the feature-tree summary, not a B-rep dump —
    sec.7): a digest + the verify verdict + the op index. ``sub_goal`` is set when
    this step *reached* a verifiable sub-goal (applied + verified + checkpointed),
    which is the unit of dense/hierarchical credit.
    """

    index: int
    state_before: Dict[str, Any]
    action: Action
    reward: float
    state_after: Dict[str, Any]
    outcome: str                                   # applied | rolled-back | rejected-backend
    run_id: Optional[str] = None
    sub_goal: Optional[str] = None                 # label of the sub-goal reached, if any
    diagnostics: List[dict] = field(default_factory=list)

    @property
    def divergent(self) -> bool:
        """A step diverges when its op did not stick (rolled back / rejected)."""
        return self.outcome != OUTCOME_APPLIED

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "state_before": self.state_before,
            "action": self.action.to_dict(),
            "reward": self.reward,
            "state_after": self.state_after,
            "outcome": self.outcome,
            "run_id": self.run_id,
            "sub_goal": self.sub_goal,
            "diagnostics": self.diagnostics,
        }


# =====================================================================
# SubGoal — dense/hierarchical credit unit
# =====================================================================

@dataclass
class SubGoal:
    """A verifiable sub-goal: a checkpointed, verified op. Its ``reward`` is the
    per-sub-goal credit (positive when reached, negative when the matching step
    diverged), giving hierarchical credit assignment (sec.17).
    """

    index: int
    label: str
    reached: bool
    reward: float
    run_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "reached": self.reached,
            "reward": self.reward,
            "run_id": self.run_id,
        }


# =====================================================================
# Trajectory — the canonical training record
# =====================================================================

@dataclass
class Trajectory:
    """Ordered steps + final scalar reward + sub-goal labels + prompt/plan.

    Built by :func:`from_events` off a JsonlTracer event stream. The same record
    is consumed by every exporter (GRPO / DPO / STaR) in dataengine/export.py.
    """

    steps: List[Step] = field(default_factory=list)
    final_reward: float = 0.0
    sub_goal_labels: List[SubGoal] = field(default_factory=list)
    prompt: Optional[str] = None
    plan: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    run_ids: List[str] = field(default_factory=list)

    # --- verdicts --------------------------------------------------------
    @property
    def success(self) -> bool:
        """Verified success: the trace's terminal verdict was ok (final_reward > 0)."""
        return self.final_reward > 0.0

    @property
    def length(self) -> int:
        return len(self.steps)

    def total_reward(self) -> float:
        """Sum of per-step rewards (the undiscounted return over the trace)."""
        return sum(s.reward for s in self.steps)

    def dense_rewards(self) -> List[float]:
        """The per-step reward vector R_0..R_{T-1} (dense credit, sec.17)."""
        return [s.reward for s in self.steps]

    def sub_goal_rewards(self) -> Dict[str, float]:
        """Reward per verifiable sub-goal (hierarchical credit, sec.17)."""
        return {sg.label: sg.reward for sg in self.sub_goal_labels}

    def first_divergence(self) -> Optional[int]:
        """Index of the first step whose op did not stick, or None if all held."""
        for i, s in enumerate(self.steps):
            if s.divergent:
                return i
        return None

    def corrections(self) -> int:
        """Human-corrections-per-plan signal (sec.21 flywheel metric).

        Uses ``metadata['human_corrections']`` when the session logged it
        explicitly; otherwise falls back to the number of divergent steps
        (verify-fail / backend-reject) as a proxy for corrections the agent (or a
        human) had to make against the plan. Should fall over time.
        """
        explicit = self.metadata.get("human_corrections")
        if explicit is not None:
            return int(explicit)
        return sum(1 for s in self.steps if s.divergent)

    # --- signal densification -------------------------------------------
    def trajectory_slice(self, to_first_divergence: bool = True) -> "Trajectory":
        """Cut the trace to its first failing/divergent step (sec.17).

        Returning the prefix up to *and including* the first divergence gives
        3-5x signal density: the interesting learning signal is the step where
        the trajectory went wrong, not the long correct prefix after a later
        recovery. A trace with no divergence is returned unchanged (a full
        verified success is already maximally informative for STaR/RFT).
        """
        if not to_first_divergence:
            return self._clone(self.steps)
        div = self.first_divergence()
        if div is None:
            return self._clone(self.steps)
        sliced = self.steps[: div + 1]
        kept = {s.run_id for s in sliced}
        clone = self._clone(sliced)
        # The slice ends at a divergence, so its terminal verdict is a failure.
        clone.final_reward = REWARD_FAIL if REWARD_FAIL < 0 else 0.0
        clone.sub_goal_labels = [sg for sg in self.sub_goal_labels if sg.index <= div]
        clone.run_ids = [r for r in self.run_ids if r in kept]
        clone.metadata = dict(self.metadata)
        clone.metadata["sliced_to_first_divergence"] = True
        return clone

    def _clone(self, steps: List[Step]) -> "Trajectory":
        return Trajectory(
            steps=list(steps),
            final_reward=self.final_reward,
            sub_goal_labels=list(self.sub_goal_labels),
            prompt=self.prompt,
            plan=self.plan,
            metadata=dict(self.metadata),
            run_ids=list(self.run_ids),
        )

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "plan": self.plan,
            "final_reward": self.final_reward,
            "success": self.success,
            "total_reward": self.total_reward(),
            "corrections": self.corrections(),
            "run_ids": self.run_ids,
            "metadata": self.metadata,
            "steps": [s.to_dict() for s in self.steps],
            "sub_goal_labels": [sg.to_dict() for sg in self.sub_goal_labels],
        }


# =====================================================================
# Event-stream -> per-op outcomes (reuse observe.Replayer when present)
# =====================================================================

def _replay_via_observe(events: List[dict]):
    """Reconstruct per-run op outcomes using observe.Replayer (lazy import).

    Returns ``(runs, ok_by_run)`` where ``runs`` is a list of
    ``(run_id, [OpOutcome-like])`` and ``ok_by_run`` maps run_id -> terminal ok.
    Raises ImportError if observe is unavailable (caller falls back).
    """
    from observe import Replayer  # lazy: never a hard dependency

    replays = Replayer().replay(events)
    runs: List[Tuple[str, list]] = []
    ok_by_run: Dict[str, Optional[bool]] = {}
    for r in replays:
        outcomes = []
        for o in r.ops:
            outcomes.append({
                "index": o.index,
                "op": o.op,
                "outcome": o.outcome,
                "digest": o.digest,
                "verify_ok": o.verify_ok,
                "checkpointed": o.checkpointed,
                "diagnostics": list(o.diagnostics),
            })
        runs.append((r.run_id, outcomes))
        ok_by_run[r.run_id] = r.ok
    return runs, ok_by_run


def _replay_direct(events: List[dict]):
    """Stdlib-only fallback: reconstruct op outcomes straight from the events.

    Mirrors observe.Replayer's semantics so this package works even if observe.py
    is absent. Groups by run_id (first-seen order), then walks each run:
    op_applied -> an applied op; a following verify-failed rejection rolls the
    last op back; a backend-rejected event is a never-applied op.
    """
    order: List[str] = []
    buckets: Dict[str, List[dict]] = {}
    for ev in events:
        rid = ev.get("run_id")
        if rid not in buckets:
            buckets[rid] = []
            order.append(rid)
        buckets[rid].append(ev)

    runs: List[Tuple[str, list]] = []
    ok_by_run: Dict[str, Optional[bool]] = {}
    for rid in order:
        outcomes: List[dict] = []
        ok: Optional[bool] = None
        for ev in buckets[rid]:
            kind = ev.get("kind")
            data = ev.get("data") or {}
            if kind == "op_applied":
                outcomes.append({
                    "index": data.get("index", len(outcomes)),
                    "op": data.get("op", {}),
                    "outcome": OUTCOME_APPLIED,
                    "digest": data.get("digest"),
                    "verify_ok": None,
                    "checkpointed": False,
                    "diagnostics": [],
                })
            elif kind == "verify_result":
                if outcomes:
                    outcomes[-1]["verify_ok"] = data.get("ok")
                    if data.get("diagnostics"):
                        outcomes[-1]["diagnostics"] += list(data["diagnostics"])
            elif kind == "rejected":
                if data.get("reason") == "verify-failed" and outcomes:
                    outcomes[-1]["outcome"] = OUTCOME_ROLLED_BACK
                    if data.get("diagnostics"):
                        outcomes[-1]["diagnostics"] += list(data["diagnostics"])
                else:
                    outcomes.append({
                        "index": len(outcomes),
                        "op": data.get("op", {}),
                        "outcome": OUTCOME_REJECTED,
                        "digest": None,
                        "verify_ok": None,
                        "checkpointed": False,
                        "diagnostics": list(data.get("diagnostics") or []),
                    })
            elif kind == "checkpoint":
                if outcomes:
                    outcomes[-1]["checkpointed"] = True
            elif kind == "run_end":
                ok = data.get("ok")
        runs.append((rid, outcomes))
        ok_by_run[rid] = ok
    return runs, ok_by_run


def _replay(events: List[dict]):
    try:
        return _replay_via_observe(events)
    except ImportError:
        return _replay_direct(events)


def _reasoning_of(op: Dict[str, Any]) -> str:
    """Pull the step's reasoning out of the op dict if the trace captured it.

    The LLM layer may fold ``reasoning`` (or ``rationale``/``thought``) into the
    op payload; if absent, the reasoning is simply empty (older traces).
    """
    if not isinstance(op, dict):
        return ""
    for key in ("reasoning", "rationale", "thought"):
        if op.get(key):
            return str(op[key])
    return ""


# =====================================================================
# from_events — build a Trajectory from a JsonlTracer event stream
# =====================================================================

def from_events(events: Iterable[dict],
                prompt: Optional[str] = None,
                plan: Optional[Any] = None,
                metadata: Optional[Dict[str, Any]] = None,
                final_reward: Optional[float] = None,
                reward_pass: float = REWARD_PASS,
                reward_fail: float = REWARD_FAIL) -> Trajectory:
    """Fold a JsonlTracer event stream into one :class:`Trajectory`.

    The stream may contain several runs (e.g. a failed run followed by a fixed
    one); all ops become steps in order. Each step's reward is assigned from the
    verifier outcome: an applied+verified op earns ``reward_pass`` and registers a
    reached sub-goal; a rolled-back (verify-failed) or backend-rejected op earns
    ``reward_fail`` and registers a missed sub-goal.

    ``prompt`` / ``plan`` default to whatever the trace carried on its first
    ``run_start`` event (keys ``prompt`` / ``plan``) when not passed explicitly.
    ``final_reward`` defaults to the verifier's terminal verdict: +reward_pass if
    the last run ended ok, else 0.0 (a task is a success only when it verifies).
    """
    events = list(events)
    runs, ok_by_run = _replay(events)

    # Prompt/plan: explicit args win, else harvest from the first run_start.
    if prompt is None or plan is None:
        for ev in events:
            if ev.get("kind") == "run_start":
                data = ev.get("data") or {}
                if prompt is None:
                    prompt = data.get("prompt")
                if plan is None:
                    plan = data.get("plan")
                break

    steps: List[Step] = []
    sub_goals: List[SubGoal] = []
    run_ids: List[str] = []
    prev_digest: Optional[str] = None
    idx = 0

    for run_id, outcomes in runs:
        run_ids.append(run_id)
        for o in outcomes:
            op = o.get("op", {}) or {}
            outcome = o.get("outcome", OUTCOME_APPLIED)
            applied_ok = outcome == OUTCOME_APPLIED
            reward = reward_pass if applied_ok else reward_fail

            after_digest = o.get("digest") if o.get("digest") is not None else prev_digest
            state_before = {"digest": prev_digest, "step": idx}
            state_after = {
                "digest": after_digest,
                "verify_ok": o.get("verify_ok"),
                "outcome": outcome,
            }

            label = op.get("op") or op.get("kind") or op.get("type") or f"op-{idx}"
            sub_goal_label = f"{idx}:{label}" if applied_ok else None

            steps.append(Step(
                index=idx,
                state_before=state_before,
                action=Action(reasoning=_reasoning_of(op), tool_call=op),
                reward=reward,
                state_after=state_after,
                outcome=outcome,
                run_id=run_id,
                sub_goal=sub_goal_label,
                diagnostics=list(o.get("diagnostics") or []),
            ))
            sub_goals.append(SubGoal(
                index=idx,
                label=f"{idx}:{label}",
                reached=applied_ok,
                reward=reward,
                run_id=run_id,
            ))

            # A held op advances the state; a diverged one leaves it unchanged.
            if applied_ok:
                prev_digest = after_digest
            idx += 1

    # Terminal scalar reward = the verifier's verdict on the final run.
    if final_reward is None:
        terminal_ok = ok_by_run.get(run_ids[-1]) if run_ids else None
        final_reward = reward_pass if terminal_ok else 0.0

    meta = dict(metadata or {})
    return Trajectory(
        steps=steps,
        final_reward=final_reward,
        sub_goal_labels=sub_goals,
        prompt=prompt,
        plan=plan,
        metadata=meta,
        run_ids=run_ids,
    )
