"""The AGENT surface -- the deterministic half of the agent, dispatchable.

``agents/agent`` and ``agents/agents`` carried the parts of an agent that do NOT
need a model: the approval-gated edit session, the typed plan envelope, the
verifier-gated termination decision, the tool-use metrics and rewards, the
attachment conditioner, the digest-bound observation, the intent resolver, the
iterative-edit policy, the message-branch tree, the async overseer's halt
detector, the tool-knowledge catalogue and the generation contract. All of it was
correct, tested, and reachable from nothing.

This module dispatches into exactly that half. Everything here is deterministic
and runs with no LLM, no host and no network -- which is the point: the parts of
an agent you can TEST are the parts that decide, gate and measure, and they
should not be locked behind a model call.

    envelope(text)          -> the typed five-stage plan an LLM was supposed to emit
    terminate(state, ok)    -> may the agent stop? (the verifier has a veto)
    halt(events)            -> is this agent looping / stagnating? stop it
    metrics(trajectory)     -> tools per task, redundancy, effective progress
    reward(trajectory)      -> format + step-wise + outcome reward
    edit_session(session)   -> propose / approve / reject, with a digest per turn
    intent(text)            -> a resolved intent, or an honest ask-for-clarification
    generation(text, ...)   -> did the model actually finish, or hit its budget?

WHAT IS NOT HERE. The ORCHESTRATORS (``agents.supervisor``,
``agents.vmodel_workflow``, ``agent.planner``, ``agents.roles``) are not routed:
their whole job is to drive an LLM through the roles, and a dispatcher that
'ran' them with no model would be theatre. They stay orphaned, honestly, until a
model is injected.

Adapters only: the agent modules are never modified. Deterministic, stdlib-only,
no network.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "AgentError",
    "envelope",
    "intent",
    "terminate",
    "halt",
    "edit_policy",
    "edit_session",
    "observation",
    "attachment",
    "host_confirm",
    "metrics",
    "reward",
    "trajectory",
    "tool_catalog",
    "dispatch_tools",
    "message_tree",
    "generation",
    "discover",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

_AGT = "harnesscad.agents.agent."
_AGS = "harnesscad.agents.agents."
_LLM = "harnesscad.agents.llm."


class AgentError(ValueError):
    """Base class for every agent-surface failure."""


# --------------------------------------------------------------------------- #
# What the model was supposed to say
# --------------------------------------------------------------------------- #
def envelope(text: str):
    """Parse the strict reasoning/code envelope into a typed five-stage CAD plan.

    A reply that does not obey the envelope is a PARSE FAILURE, not something to
    be regex-scavenged for a code block.
    """
    from harnesscad.agents.agent.plan_envelope import parse_envelope

    return parse_envelope(text)


def intent(text: str, default_unit: str = "mm"):
    """Resolve a natural-language CAD instruction into a typed intent envelope.

    ``needs_clarification()`` is a legitimate outcome. An agent that guesses the
    units is worse than one that asks.
    """
    from harnesscad.agents.agent.intent_resolution import resolve_intent

    return resolve_intent(text, default_unit=default_unit)


def generation(text: str, finish_reason: str, output_tokens: int,
               maximum_tokens: int, require_solid: bool = False):
    """Did the model actually FINISH, or did it just run out of budget?

    A truncated reply that happens to parse is still truncated.
    """
    from harnesscad.agents.llm.generation_contract import assess_generation

    return assess_generation(text, finish_reason=finish_reason,
                             output_tokens=int(output_tokens),
                             maximum_tokens=int(maximum_tokens),
                             require_solid=bool(require_solid))


# --------------------------------------------------------------------------- #
# When may the agent stop, and when must it be stopped
# --------------------------------------------------------------------------- #
def terminate(state: str, verifier_ok: bool, reason: str = ""):
    """The agent says it is done. THE VERIFIER HAS A VETO.

    ``state`` is ``continue`` / ``complete`` / ``blocked``. A ``complete`` over a
    model that does NOT verify is overruled -- the result comes back
    ``continue`` with a ``premature-completion`` diagnostic. Self-report is not
    evidence.
    """
    from harnesscad.agents.agent.termination import TerminationDecision, gate_termination

    return gate_termination(TerminationDecision(state=state, reason=reason),
                            bool(verifier_ok))


def halt(events: Sequence[Mapping[str, Any]], loop_window: int = 6,
         loop_threshold: int = 3, stagnation_rounds: int = 3):
    """Watch an event stream and HALT a looping / stagnating agent.

    Returns the first :class:`Halt` (or ``None``). The overseer has authority:
    it does not suggest, it stops.
    """
    from harnesscad.agents.agents.overseer import AsyncOverseer

    overseer = AsyncOverseer(loop_window=int(loop_window),
                             loop_threshold=int(loop_threshold),
                             stagnation_rounds=int(stagnation_rounds))
    for event in events:
        decision = overseer.observe(dict(event))
        if decision is not None:
            return decision
    return None


def edit_policy(current: Any, candidate: Any, history: Sequence[Any] = (),
                max_rounds: int = 5):
    """Keep the candidate edit, or keep the current model? The iterative-edit policy."""
    from harnesscad.agents.agent.iterative_edit_policy import IterativeEditPolicy

    return IterativeEditPolicy(max_rounds=int(max_rounds)).choose(
        current, candidate, list(history))


# --------------------------------------------------------------------------- #
# The approval-gated edit loop
# --------------------------------------------------------------------------- #
def edit_session(session: Any):
    """A multi-turn, APPROVAL-GATED editing session over a HarnessSession.

    ``propose`` never mutates: it previews. Only ``approve`` applies, and every
    turn carries the digest it produced -- so an edit that changed the model
    cannot later claim it did not.
    """
    from harnesscad.agents.agent.edit_session import EditSession

    return EditSession(session)


def observation(state_digest: str, geometry: Optional[Mapping[str, Any]] = None,
                renders: Optional[Mapping[str, bytes]] = None,
                entity_ids: Sequence[str] = ()):
    """A DIGEST-BOUND multimodal observation.

    ``require_current(digest)`` refuses an observation of a model that has since
    moved on -- an agent must not reason about a state that no longer exists.
    """
    from harnesscad.agents.agent.observation import CADObservation

    return CADObservation(state_digest=str(state_digest),
                          geometry=dict(geometry or {}),
                          renders=dict(renders or {}),
                          entity_ids=tuple(entity_ids))


def attachment(kind: Any, data: bytes, provenance: Any, **kw):
    """Condition a sketch/image attachment: size-capped, root-caged, hash-checked."""
    from harnesscad.agents.agent.attachments import (
        Attachment, DeterministicEncoder, condition_attachment,
    )

    att = Attachment(kind=kind, provenance=provenance, data=data,
                     **{k: v for k, v in kw.items()
                        if k in ("path", "declared_mime", "expected_sha256")})
    return condition_attachment(att, DeterministicEncoder())


def host_confirm(proposal: Any, script: Optional[str] = None):
    """The confirmation-preserving lifecycle for an OPAQUE host script proposal.

    ``refine`` produces a new revision with the lineage intact; a refined
    proposal loses its confirmation, which is the entire point.
    """
    from harnesscad.agents.agent.host_feedback import confirm, refine

    if script is not None:
        return refine(proposal, script)
    return confirm(proposal)


# --------------------------------------------------------------------------- #
# Measuring the tool use
# --------------------------------------------------------------------------- #
def trajectory(calls: Sequence[Any], library: Optional[Any] = None):
    """Roll a list of tool calls out against a tool library into a trajectory."""
    from harnesscad.agents.agent.tool_schema import default_toolcad_library
    from harnesscad.agents.agent.tool_trajectory import rollout

    return rollout(list(calls), library or default_toolcad_library())


def metrics(traj: Any) -> dict:
    """Tools-per-task, success rate, redundant calls, effective progress."""
    from harnesscad.agents.agent.tool_metrics import summarize

    m = summarize(traj)
    return {f: getattr(m, f) for f in type(m).__dataclass_fields__}


def reward(traj: Any, orm_verdict: bool, format_text: str = "") -> dict:
    """Format + step-wise execution + outcome reward for one tool trajectory."""
    from harnesscad.agents.agent.tool_reward import aggregate_reward

    r = aggregate_reward(traj, orm_verdict=bool(orm_verdict),
                         format_text=format_text)
    return {f: getattr(r, f) for f in type(r).__dataclass_fields__}


# --------------------------------------------------------------------------- #
# What the agent knows about its tools
# --------------------------------------------------------------------------- #
def tool_catalog():
    """The tool-knowledge catalogue, seeded with the CISP cards."""
    from harnesscad.agents.agent.tool_knowledge import (
        ToolKnowledgeCatalog, default_cisp_cards,
    )

    cat = ToolKnowledgeCatalog()
    for card in default_cisp_cards():
        cat.register(card)
    return cat


def dispatch_tools(task: str, context: Mapping[str, Any], limit: int = 4,
                   required_tools: Sequence[str] = ()):
    """Which tools does this task need, and what CONTEXT is still missing for them?

    A tool whose required context is absent is not dispatched -- it becomes a
    question.
    """
    return tool_catalog().dispatch(task, dict(context), limit=int(limit),
                                   required_tools=tuple(required_tools))


def message_tree(elements: Sequence[Any], key: str = "id",
                 parent_key: str = "parent_message_id"):
    """The conversation BRANCH tree: siblings, paths to root, leaves, depth.

    ``elements`` are message OBJECTS (anything with the ``key`` and
    ``parent_key`` attributes) -- a branching conversation is a tree, not a list,
    and rewinding to a sibling must not lose the branch you came from.
    """
    from harnesscad.agents.agents.message_tree import MessageTree

    return MessageTree(list(elements), key=key, parent_key=parent_key)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _index() -> Dict[str, Any]:
    return {e.dotted: e for e in capability_registry.index()
            if e.dotted.startswith("harnesscad.agents.agent.")
            or e.dotted.startswith("harnesscad.agents.agents.")
            or e.dotted == _LLM + "generation_contract"}


def _available(dotted: str) -> bool:
    return dotted in _index()


_ROUTES: Tuple[Tuple[str, str, str, str], ...] = (
    ("parse", "envelope", _AGT + "plan_envelope",
     "the strict reasoning/code envelope -> a typed five-stage CAD plan"),
    ("parse", "intent", _AGT + "intent_resolution",
     "an NL instruction -> a typed intent, or an honest clarification request"),
    ("parse", "generation", _LLM + "generation_contract",
     "did the model FINISH, or hit its token budget?"),
    ("gate", "terminate", _AGT + "termination",
     "the agent says it is done -- THE VERIFIER HAS A VETO"),
    ("gate", "halt", _AGS + "overseer",
     "halt a looping / stagnating agent (authority, not advice)"),
    ("gate", "edit_policy", _AGT + "iterative_edit_policy",
     "keep the candidate edit, or keep the current model?"),
    ("edit", "edit_session", _AGT + "edit_session",
     "multi-turn, APPROVAL-GATED editing; propose never mutates"),
    ("edit", "host_confirm", _AGT + "host_feedback",
     "confirmation-preserving lifecycle for an opaque host script"),
    ("input", "observation", _AGT + "observation",
     "digest-bound multimodal observation (refuses a stale state)"),
    ("input", "attachment", _AGT + "attachments",
     "size-capped, root-caged, hash-checked sketch/image conditioning"),
    ("measure", "metrics", _AGT + "tool_metrics",
     "tools-per-task, redundancy, effective progress"),
    ("measure", "reward", _AGT + "tool_reward",
     "format + step-wise execution + outcome reward"),
    ("measure", "trajectory", _AGT + "tool_trajectory",
     "roll tool calls out against a tool library"),
    ("knowledge", "dispatch_tools", _AGT + "tool_knowledge",
     "which tools this task needs, and what context is still missing"),
    ("conversation", "message_tree", _AGS + "message_tree",
     "the conversation branch tree: siblings, paths, leaves"),
)

#: Agent modules deliberately left with NO route, and why.
UNADAPTED_REASONS: Dict[str, str] = {
    _AGT + "planner":
        "needs an LLM -- its whole body is 'build messages, call the model, parse'",
    _AGT + "runner":
        "the plan/apply/repair driver; it takes a Planner, so it needs an LLM",
    _AGT + "system_prompt":
        "a PROMPT (a string builder). The planner is its only caller and needs an LLM",
    _AGS + "roles":
        "Designer / Modeler / Reviewer / RedTeam are LLM-backed roles",
    _AGS + "supervisor":
        "the multi-agent orchestrator: it drives the LLM-backed roles round by "
        "round. Running it with no model would be theatre",
    _AGS + "vmodel_workflow":
        "the Idea-to-CAD collaborative workflow: it drives the LLM-backed "
        "vmodel roles through nested feedback loops",
}


def routed_modules() -> Tuple[str, ...]:
    routed = {m for _g, _n, m, _d in _ROUTES if _available(m)}
    # tool_schema is reached through `trajectory` (the default tool library).
    routed.add(_AGT + "tool_schema")
    return tuple(sorted(d for d in routed if _available(d)))


def discover() -> List[dict]:
    return [{"group": g, "route": n, "module": m, "doc": d,
             "present": _available(m)}
            for (g, n, m, d) in _ROUTES]


def unadapted() -> List[Tuple[str, str]]:
    routed = set(routed_modules())
    return [(d, UNADAPTED_REASONS.get(d, "no route yet")) for d in sorted(_index())
            if d not in routed and not d.endswith(".registry")]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every agent route")
    parser.add_argument("--intent", default=None,
                        help="resolve one natural-language CAD instruction")
    parser.add_argument("--envelope", default=None, metavar="FILE",
                        help="parse a model reply (a file) into a typed plan envelope")
    parser.add_argument("--tools", default=None, metavar="TASK",
                        help="which tools this task needs, and what context is missing")
    parser.add_argument("--unadapted", action="store_true",
                        help="list agent modules with no route, and why")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "unadapted", False):
        for dotted, reason in unadapted():
            print("%s\n    %s" % (dotted, reason))
        return 0

    if getattr(args, "intent", None):
        print(repr(intent(args.intent)))
        return 0

    if getattr(args, "envelope", None):
        with open(args.envelope, "r", encoding="utf-8") as fh:
            print(repr(envelope(fh.read())))
        return 0

    if getattr(args, "tools", None):
        plan = dispatch_tools(args.tools, {})
        print("ready:     %s" % ", ".join(plan.ready_tools))
        for q in plan.questions:
            print("question:  %s" % q)
        return 0

    rows = discover()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(r["route"]) for r in rows)
    for r in rows:
        mark = " " if r["present"] else "-"
        print("%s %-12s %-*s  %s" % (mark, r["group"], width, r["route"], r["doc"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad agent",
        description="agent surface: the deterministic half -- envelopes, gates, "
                    "approval-gated edits, tool metrics")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
