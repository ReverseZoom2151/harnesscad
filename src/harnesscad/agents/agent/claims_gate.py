"""Claim-vs-evidence gating of an agent's final answer (anti-false-success).

THE MECHANISM
-------------
An agent that reports success it did not earn is worse than an agent that
fails: the failure is invisible. This gate binds the *final answer* to what
actually happened:

  * **Mutation guard.** When the run's intent requires a geometry mutation
    (``/build`` -> ``create_geometry``, ``/modify`` -> ``modify_geometry``), a
    bare ``final`` is REJECTED until a CAD mutation tool has actually
    succeeded. A read-only result (a critique, an inspection) does not satisfy
    it. Asking the user (``ask_user``) or reporting a clear blocker is still
    allowed -- a false success is not.
  * **Solver-executed honesty gate.** A ``final`` that claims solver results is
    REJECTED unless ``solver_executed`` is true, and ``solver_executed`` only
    flips true after an **approved, non-error** solver run -- so denied,
    pending and failed runs can never read as success. ``run_solver`` is itself
    blocked until a ``prepare_solver_run`` succeeded, and prepare/run are
    blocked while required inputs are missing.
  * **Read-only / simulation intents SUPPRESS the mutation guard** (a critique
    is not an edit; a simulation step is not a CAD edit, and free text like
    "add a 500N load" must not trip the create heuristic).

EVIDENCE, NOT KEYWORDS
----------------------
The gate never inspects the answer *text*. Every input it reads is structural:

  1. :class:`ToolEvent` records of what the runtime actually executed -- each
     carrying an ``effect`` the *tool registry* assigns (not the model), plus
     the runtime's ``approved`` flag and ``status``. :func:`collect_evidence`
     folds them into a :class:`RunEvidence`.
  2. The resolved :class:`RouteIntent` of the run, from the user's explicit
     slash command / an injected classifier / a keyword heuristic over the
     **user's request** -- never over the agent's answer.
  3. The claims the agent *declares structurally* on its
     :class:`~harnesscad.agents.agent.termination.TerminationDecision`
     (``claims=("solver_results",)``). A declared claim can only ever ADD a
     requirement; omitting it never relaxes the mutation guard, which is driven
     entirely by (1) and (2).

Sniffing the final answer for the word "solver" would be exactly the failure
mode this gate exists to prevent -- a model that says "the simulation ran"
without the word "solver" would slip through, and a model that says "I did not
run the solver" would be blocked. Claims are declared or derived from routing;
evidence comes from the runtime.

RELATION TO EXISTING WORK
-------------------------
:mod:`harnesscad.governance.credibility_tier` and
:mod:`harnesscad.eval.quality.physics.cae_credibility_ladder` already carry the
V&V-40 tiering and a ``solver_executed`` notion -- they classify *a result's
credibility once it exists*. This module is the DELTA: the **termination-time**
gate that decides whether the agent may stop and answer at all. It imports
:func:`~harnesscad.governance.credibility_tier.classify_credibility` so a gated
final carries the same tier vocabulary rather than inventing a second one.

DEFAULT-SAFE
------------
:func:`~harnesscad.agents.agent.termination.gate_termination` keeps its old
two-argument signature and old behaviour; the claim gate only engages when a
caller passes an intent and/or evidence.

Stdlib-only, deterministic, absolute imports. ``--selfcheck`` proves the
guard, the honesty gate, the ordering rules, the abstain path and the
default-safe fallback.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

from harnesscad.governance.credibility_tier import UNVERIFIED, classify_credibility

__all__ = [
    "ToolEffect",
    "ToolEvent",
    "RunEvidence",
    "collect_evidence",
    "RouteIntent",
    "INTENT_REGISTRY",
    "IntentResolution",
    "INTENT_CLARIFY_INSTRUCTION",
    "keyword_classify",
    "resolve_route_intent",
    "ClaimVerdict",
    "SOLVER_CLAIM",
    "gate_claims",
    "main",
]

# --------------------------------------------------------------------------- #
# 1. evidence: what the runtime actually did
# --------------------------------------------------------------------------- #

#: The effect a tool has, assigned by the TOOL REGISTRY -- never by the model.
#: ``mutate_geometry`` is the only effect that satisfies the mutation guard;
#: ``solver_run`` is the only effect that can set ``solver_executed``.
ToolEffect = str

MUTATE_GEOMETRY: ToolEffect = "mutate_geometry"
READ_ONLY: ToolEffect = "read_only"
SOLVER_PREPARE: ToolEffect = "solver_prepare"
SOLVER_RUN: ToolEffect = "solver_run"

_EFFECTS: Tuple[ToolEffect, ...] = (
    MUTATE_GEOMETRY, READ_ONLY, SOLVER_PREPARE, SOLVER_RUN,
)

#: A tool outcome is a success ONLY for this status. Everything else --
#: ``error``, ``denied``, ``pending``, ``timeout``, an unknown string -- is not.
STATUS_OK = "ok"


@dataclass(frozen=True)
class ToolEvent:
    """One executed tool call, as the runtime observed it.

    ``effect`` comes from the tool registry, ``approved`` from the approval
    gate, ``status`` from the executor. The model contributes nothing here --
    that is the point.
    """

    tool: str
    effect: ToolEffect
    approved: bool = True
    status: str = STATUS_OK

    @property
    def succeeded(self) -> bool:
        """Approved AND non-error. Denied/failed/pending never read as success."""
        return bool(self.approved) and self.status == STATUS_OK


@dataclass(frozen=True)
class RunEvidence:
    """What a run has actually earned the right to claim."""

    mutation_succeeded: bool = False
    solver_deck_prepared: bool = False
    solver_executed: bool = False
    solver_status: Optional[str] = None
    read_only_result: bool = False
    denied_or_failed: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "mutation_succeeded": self.mutation_succeeded,
            "solver_deck_prepared": self.solver_deck_prepared,
            "solver_executed": self.solver_executed,
            "solver_status": self.solver_status,
            "read_only_result": self.read_only_result,
            "denied_or_failed": list(self.denied_or_failed),
        }


def collect_evidence(events: Sequence[ToolEvent]) -> RunEvidence:
    """Fold an ordered tool-event log into a :class:`RunEvidence`.

    Rules, ported from the documented workflow state machine:

      * ``mutation_succeeded`` iff some ``mutate_geometry`` event was approved
        and non-error;
      * ``solver_deck_prepared`` iff some ``solver_prepare`` event succeeded;
      * ``solver_executed`` iff an approved, non-error ``solver_run`` happened
        **after** a successful prepare -- ``run_solver`` is blocked until a
        successful ``prepare_solver_run``, so a run without one is not evidence
        even if the executor reported ok;
      * ``solver_status`` is the status of the LAST ``solver_run`` seen (so a
        failed re-run after a good one is reported honestly).
    """
    mutation = False
    prepared = False
    executed = False
    solver_status: Optional[str] = None
    read_only = False
    bad: list = []
    for ev in events:
        if not ev.succeeded:
            bad.append(f"{ev.tool}:{'denied' if not ev.approved else ev.status}")
        if ev.effect == MUTATE_GEOMETRY and ev.succeeded:
            mutation = True
        elif ev.effect == SOLVER_PREPARE and ev.succeeded:
            prepared = True
        elif ev.effect == SOLVER_RUN:
            solver_status = ("denied" if not ev.approved else ev.status)
            if ev.succeeded and prepared:
                executed = True
        elif ev.effect == READ_ONLY and ev.succeeded:
            read_only = True
    return RunEvidence(
        mutation_succeeded=mutation,
        solver_deck_prepared=prepared,
        solver_executed=executed,
        solver_status=solver_status,
        read_only_result=read_only,
        denied_or_failed=tuple(bad),
    )


# --------------------------------------------------------------------------- #
# 2. three-tier intent resolution
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RouteIntent:
    """One routed command: its guard profile and its trigger vocabulary.

    ``INTENT_REGISTRY`` is the single source of truth -- the guard lists and the
    keyword heuristic are both derived from it, so adding an intent is a
    one-entry edit (the source's stated design property).
    """

    command: str
    intent_type: str
    mutation_required: bool = False
    read_only: bool = False
    simulation: bool = False
    triggers: Tuple[str, ...] = ()

    @property
    def suppresses_mutation_guard(self) -> bool:
        """Read-only and simulation intents never demand a geometry mutation."""
        return self.read_only or self.simulation


INTENT_REGISTRY: Dict[str, RouteIntent] = {
    ri.command: ri for ri in (
        RouteIntent("build", "create_geometry", mutation_required=True,
                    triggers=("build", "create", "make", "generate", "model",
                              "design", "draw")),
        RouteIntent("modify", "modify_geometry", mutation_required=True,
                    triggers=("modify", "change", "edit", "adjust", "resize",
                              "increase", "decrease", "widen", "thicken")),
        RouteIntent("critique", "critique_model", read_only=True,
                    triggers=("critique", "review", "check", "inspect",
                              "assess", "evaluate")),
        RouteIntent("explain", "explain_project", read_only=True,
                    triggers=("explain", "describe", "what is", "how does",
                              "summarize", "tell me about")),
        RouteIntent("simulate", "plan_simulation", simulation=True,
                    triggers=("simulate", "simulation", "solver", "fea",
                              "stress", "load case", "analysis")),
    )
}

#: Injected when an inferred intent is actionable but low-confidence or
#: ambiguous: bias the agent to ASK rather than route on a guess.
INTENT_CLARIFY_INSTRUCTION = (
    "The request could not be resolved to a single action with confidence. "
    "Ask the user which is meant instead of guessing; do not claim a result."
)

#: Confidence at or above which a resolved command is routed. Below it the
#: resolver abstains (asks) rather than guess.
CONFIDENCE_ROUTE = 0.6

#: An explicit slash command is ground truth.
CONFIDENCE_EXPLICIT = 1.0

_SLASH_RE = re.compile(r"^\s*/([a-z_]+)\b", re.IGNORECASE)

#: Mutation intent takes precedence over read-only/simulation when both match
#: (source rule: "mutation intent takes precedence").
_KEYWORD_PRECEDENCE: Tuple[str, ...] = (
    "build", "modify", "simulate", "critique", "explain",
)


@dataclass(frozen=True)
class IntentResolution:
    """The resolved routing of ONE user request."""

    intent: Optional[RouteIntent]
    confidence: float
    source: str                       # explicit | classifier | keyword | none
    abstain: bool = False
    clarify_instruction: str = ""
    candidates: Tuple[str, ...] = ()

    @property
    def command(self) -> Optional[str]:
        return self.intent.command if self.intent else None

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "intent_type": self.intent.intent_type if self.intent else None,
            "confidence": self.confidence,
            "source": self.source,
            "abstain": self.abstain,
            "candidates": list(self.candidates),
        }


def _tokens(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9 ]+", " ", text.lower()) + " "


def keyword_classify(text: str) -> Tuple[Optional[str], float, Tuple[str, ...]]:
    """Deterministic tier-3 heuristic over the USER'S request text.

    Returns ``(command, confidence, matched_candidates)``. Confidence is the
    match count of the winning command over the total matches -- a request that
    triggers exactly one intent is confident; one that triggers several is not,
    and the caller abstains. Mutation intents win ties by registry precedence.
    """
    padded = _tokens(text)
    hits: Dict[str, int] = {}
    for command, ri in INTENT_REGISTRY.items():
        n = sum(1 for t in ri.triggers if f" {t} " in padded)
        if n:
            hits[command] = n
    if not hits:
        return None, 0.0, ()
    candidates = tuple(c for c in _KEYWORD_PRECEDENCE if c in hits)
    total = sum(hits.values())
    best = max(candidates, key=lambda c: (hits[c], -_KEYWORD_PRECEDENCE.index(c)))
    return best, hits[best] / total, candidates


def resolve_route_intent(
    text: str,
    *,
    command: Optional[str] = None,
    classifier: Optional[Callable[[str], Tuple[Optional[str], float]]] = None,
    min_confidence: float = CONFIDENCE_ROUTE,
) -> IntentResolution:
    """Three-tier resolution: explicit command > injected classifier > keyword.

    Tier 1 -- an explicit ``/command`` (or an explicit ``command=`` argument)
    ALWAYS wins, at confidence 1.0.
    Tier 2 -- an optional, injected ``classifier(text) -> (command, confidence)``
    (the app wires a model here). It degrades **silently** to tier 3 on any
    exception, on ``None``, or on an unknown command, so a missing provider can
    never change the outcome and tests stay deterministic.
    Tier 3 -- :func:`keyword_classify`.

    An actionable but low-confidence or ambiguous result ABSTAINS: no command
    is routed, :data:`INTENT_CLARIFY_INSTRUCTION` is surfaced, and the caller is
    expected to ask the user. The resolver only ever PROPOSES a command; it
    never relaxes a guard (an abstain leaves the guard exactly where an
    explicit command would have put it -- see :func:`gate_claims`, which treats
    an unresolved intent as "no forced guard" but still honours a declared
    claim).
    """
    explicit = command
    if explicit is None:
        m = _SLASH_RE.match(text or "")
        if m:
            explicit = m.group(1).lower()
    if explicit is not None:
        ri = INTENT_REGISTRY.get(explicit.lower().lstrip("/"))
        if ri is not None:
            return IntentResolution(ri, CONFIDENCE_EXPLICIT, "explicit")

    if classifier is not None:
        try:
            proposed, confidence = classifier(text)
        except Exception:
            proposed, confidence = None, 0.0
        ri = INTENT_REGISTRY.get((proposed or "").lower())
        if ri is not None:
            confidence = float(confidence)
            if confidence >= min_confidence:
                return IntentResolution(ri, confidence, "classifier")
            return IntentResolution(None, confidence, "classifier", True,
                                    INTENT_CLARIFY_INSTRUCTION, (ri.command,))

    best, confidence, candidates = keyword_classify(text or "")
    if best is None:
        return IntentResolution(None, 0.0, "none")
    if confidence < min_confidence or len(candidates) > 1:
        return IntentResolution(None, confidence, "keyword", True,
                                INTENT_CLARIFY_INSTRUCTION, candidates)
    return IntentResolution(INTENT_REGISTRY[best], confidence, "keyword",
                            candidates=candidates)


# --------------------------------------------------------------------------- #
# 3. the claim-vs-evidence gate
# --------------------------------------------------------------------------- #

#: The one declarable claim the harness can check against hard evidence: the
#: final answer reports solver results. Declared STRUCTURALLY on the decision.
SOLVER_CLAIM = "solver_results"

#: Declaring a geometry mutation is optional -- the mutation guard is driven by
#: the routed intent, not by the model's declaration -- but a declaration is
#: checked too, so an agent cannot claim an edit it did not make on a
#: read-only route.
MUTATION_CLAIM = "geometry_mutated"

CHECKABLE_CLAIMS: Tuple[str, ...] = (SOLVER_CLAIM, MUTATION_CLAIM)


@dataclass(frozen=True)
class ClaimVerdict:
    """Whether a final answer's claims are backed by the run's evidence."""

    accepted: bool
    reason: str = ""
    credibility_tier: str = UNVERIFIED
    checked_claims: Tuple[str, ...] = ()
    evidence: RunEvidence = field(default_factory=RunEvidence)

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "credibility_tier": self.credibility_tier,
            "checked_claims": list(self.checked_claims),
            "evidence": self.evidence.to_dict(),
        }


def gate_claims(
    state: str,
    claims: Sequence[str] = (),
    *,
    intent: Optional[RouteIntent] = None,
    evidence: Optional[RunEvidence] = None,
) -> ClaimVerdict:
    """Accept or reject a FINAL answer by claim-vs-evidence comparison.

    ``state`` is the agent's termination state. Only ``complete`` -- the claim
    of success -- is gated. ``continue``, ``blocked`` (a clear blocker) and any
    ask-the-user path are always allowed: the source's invariant is "no false
    success", not "no honest failure".

    Rules, in order:

      1. non-``complete`` states pass through untouched;
      2. **mutation guard** -- an intent whose ``mutation_required`` is set and
         which does not suppress the guard rejects a final unless
         ``evidence.mutation_succeeded``. A read-only result does not satisfy
         it;
      3. **declared-mutation check** -- a final declaring ``geometry_mutated``
         is rejected without a successful mutation, on ANY route;
      4. **solver honesty gate** -- a final declaring ``solver_results`` is
         rejected unless ``evidence.solver_executed`` (approved, non-error run
         after a successful prepare). The reported ``solver_status`` is echoed
         so a denied/failed run is named rather than hidden;
      5. otherwise accept, stamping the credibility tier the evidence earns.

    With no ``intent`` and no ``evidence`` the gate is a no-op accept: an
    un-instrumented caller keeps its old behaviour (default-safe).
    """
    if state != "complete":
        return ClaimVerdict(True, "", UNVERIFIED, (), evidence or RunEvidence())

    ev = evidence if evidence is not None else RunEvidence()
    declared = tuple(c for c in claims if c in CHECKABLE_CLAIMS)

    if (intent is not None
            and intent.mutation_required
            and not intent.suppresses_mutation_guard
            and not ev.mutation_succeeded):
        return ClaimVerdict(
            False,
            f"mutation-required-intent-{intent.intent_type}-without-successful-"
            f"mutation",
            UNVERIFIED, declared, ev)

    if MUTATION_CLAIM in declared and not ev.mutation_succeeded:
        return ClaimVerdict(False, "claimed-geometry-mutation-without-evidence",
                            UNVERIFIED, declared, ev)

    if SOLVER_CLAIM in declared and not ev.solver_executed:
        status = ev.solver_status or ("deck-prepared" if ev.solver_deck_prepared
                                      else "never-run")
        return ClaimVerdict(
            False, f"claimed-solver-results-without-executed-run:{status}",
            UNVERIFIED, declared, ev)

    tier = classify_credibility(
        "solver" if SOLVER_CLAIM in declared else "critique",
        solver_executed=ev.solver_executed or None,
    ).tier
    return ClaimVerdict(True, "", tier, declared, ev)


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Claim-vs-evidence final-answer gate + three-tier intent "
                    "resolution (cad-cae-copilot AGENTS.md reimplementation).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="prove the mutation guard, the solver honesty "
                             "gate, evidence ordering, abstain-on-low-"
                             "confidence, and the default-safe fallback.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    from harnesscad.agents.agent.termination import (
        TerminationDecision, gate_termination,
    )

    build = INTENT_REGISTRY["build"]
    critique = INTENT_REGISTRY["critique"]
    simulate = INTENT_REGISTRY["simulate"]

    # 1. Mutation guard: a bare final on /build with only a read-only result.
    ev = collect_evidence([ToolEvent("cad.critique", READ_ONLY)])
    v = gate_claims("complete", intent=build, evidence=ev)
    assert not v.accepted and "without-successful-mutation" in v.reason, v.to_dict()
    print("[selfcheck] mutation guard: read-only result does not satisfy /build")

    # A failed mutation is not a mutation.
    ev = collect_evidence([ToolEvent("cad.execute", MUTATE_GEOMETRY,
                                     status="error")])
    assert not gate_claims("complete", intent=build, evidence=ev).accepted
    # A denied (unapproved) mutation is not a mutation.
    ev = collect_evidence([ToolEvent("cad.execute", MUTATE_GEOMETRY,
                                     approved=False)])
    assert not gate_claims("complete", intent=build, evidence=ev).accepted
    # A real one is.
    ev = collect_evidence([ToolEvent("cad.execute", MUTATE_GEOMETRY)])
    assert ev.mutation_succeeded
    assert gate_claims("complete", intent=build, evidence=ev).accepted
    print("[selfcheck] mutation guard: only an approved non-error mutation "
          "opens the final")

    # 2. Honest failure is always allowed -- the guard blocks false success only.
    ev = collect_evidence([])
    assert gate_claims("blocked", intent=build, evidence=ev).accepted
    assert gate_claims("continue", intent=build, evidence=ev).accepted
    print("[selfcheck] blocked/continue pass the guard (no honest-failure tax)")

    # 3. Read-only + simulation intents suppress the mutation guard.
    assert critique.suppresses_mutation_guard and simulate.suppresses_mutation_guard
    ev = collect_evidence([ToolEvent("cad.get_source", READ_ONLY)])
    assert gate_claims("complete", intent=critique, evidence=ev).accepted
    assert gate_claims("complete", intent=simulate, evidence=ev).accepted
    print("[selfcheck] read-only/simulation intents suppress the mutation guard")

    # 4. Solver honesty gate.
    prep = ToolEvent("cae.prepare_solver_run", SOLVER_PREPARE)
    ev = collect_evidence([prep])
    v = gate_claims("complete", [SOLVER_CLAIM], intent=simulate, evidence=ev)
    assert not v.accepted and "without-executed-run" in v.reason, v.to_dict()
    # Denied run -> never success.
    ev = collect_evidence([prep, ToolEvent("cae.run_solver", SOLVER_RUN,
                                           approved=False)])
    assert not ev.solver_executed and ev.solver_status == "denied"
    assert not gate_claims("complete", [SOLVER_CLAIM], evidence=ev).accepted
    # Errored run -> never success.
    ev = collect_evidence([prep, ToolEvent("cae.run_solver", SOLVER_RUN,
                                           status="error")])
    assert not ev.solver_executed and ev.solver_status == "error"
    assert not gate_claims("complete", [SOLVER_CLAIM], evidence=ev).accepted
    # Run without a successful prepare -> not evidence (prepare gates run).
    ev = collect_evidence([ToolEvent("cae.run_solver", SOLVER_RUN)])
    assert not ev.solver_executed
    assert not gate_claims("complete", [SOLVER_CLAIM], evidence=ev).accepted
    # Approved, non-error run after a prepare -> the claim is earned.
    ev = collect_evidence([prep, ToolEvent("cae.run_solver", SOLVER_RUN)])
    assert ev.solver_executed and ev.solver_deck_prepared
    v = gate_claims("complete", [SOLVER_CLAIM], intent=simulate, evidence=ev)
    assert v.accepted and v.credibility_tier == "executed_solver_result", v.to_dict()
    print("[selfcheck] solver gate: denied / errored / unprepared runs never "
          "read as success; only approved+prepared+ok earns the claim")

    # A later failed re-run is reported honestly and revokes nothing silently.
    ev = collect_evidence([prep, ToolEvent("cae.run_solver", SOLVER_RUN),
                           ToolEvent("cae.run_solver", SOLVER_RUN,
                                     status="error")])
    assert ev.solver_status == "error" and "cae.run_solver:error" in ev.denied_or_failed
    print("[selfcheck] last solver status reported honestly")

    # 5. Declared mutation claim is checked on any route (read-only included).
    ev = collect_evidence([ToolEvent("cad.critique", READ_ONLY)])
    v = gate_claims("complete", [MUTATION_CLAIM], intent=critique, evidence=ev)
    assert not v.accepted and "without-evidence" in v.reason
    print("[selfcheck] a declared edit on a read-only route is rejected")

    # 6. Three-tier resolution.
    r = resolve_route_intent("/build a 40mm bracket, then explain it")
    assert r.source == "explicit" and r.command == "build" and r.confidence == 1.0
    assert not r.abstain
    r = resolve_route_intent("please build a bracket",
                             classifier=lambda t: ("simulate", 0.9))
    assert r.source == "classifier" and r.command == "simulate"
    r = resolve_route_intent("/build x", classifier=lambda t: ("critique", 1.0))
    assert r.source == "explicit" and r.command == "build"  # explicit always wins
    r = resolve_route_intent("build a bracket",
                             classifier=lambda t: (_ for _ in ()).throw(
                                 RuntimeError("no provider")))
    assert r.source == "keyword" and r.command == "build"  # silent degrade
    r = resolve_route_intent("build a bracket", classifier=lambda t: (None, 0.0))
    assert r.source == "keyword" and r.command == "build"
    print("[selfcheck] intent tiers: explicit > classifier > keyword, "
          "classifier degrades silently")

    # Abstain on low confidence / ambiguity -- and no guard is invented.
    r = resolve_route_intent("build a bracket", classifier=lambda t: ("build", 0.2))
    assert r.abstain and r.command is None and r.clarify_instruction
    r = resolve_route_intent("review the plate and change the wall thickness")
    assert r.abstain and r.command is None and set(r.candidates) >= {"modify",
                                                                     "critique"}
    r = resolve_route_intent("hello there")
    assert r.intent is None and r.source == "none" and not r.abstain
    print("[selfcheck] abstain-on-low-confidence / ambiguity -> ask, never guess")

    # An abstained run forces no mutation guard but still checks declarations.
    r = resolve_route_intent("review the plate and change the wall thickness")
    assert gate_claims("complete", intent=r.intent,
                       evidence=collect_evidence([])).accepted
    assert not gate_claims("complete", [MUTATION_CLAIM], intent=r.intent,
                           evidence=collect_evidence([])).accepted
    print("[selfcheck] abstain never relaxes a declared claim")

    # 7. Default-safe: the legacy two-arg termination call is unchanged.
    assert gate_termination(TerminationDecision("complete"), True).terminal
    assert not gate_termination(TerminationDecision("complete"), False).accepted
    assert gate_termination(TerminationDecision("continue"), False).state == "continue"
    # ...and the wired form gates on evidence.
    res = gate_termination(TerminationDecision("complete"), True, intent=build,
                           evidence=collect_evidence([]))
    assert not res.accepted and res.state == "continue"
    assert "without-successful-mutation" in res.diagnostic
    res = gate_termination(
        TerminationDecision("complete", claims=(SOLVER_CLAIM,)), True,
        intent=simulate,
        evidence=collect_evidence([prep, ToolEvent("cae.run_solver", SOLVER_RUN)]))
    assert res.accepted and res.terminal and res.credibility_tier == (
        "executed_solver_result")
    print("[selfcheck] termination default-safe; wired form gates on evidence")

    # 8. The gate never reads answer text: it has no answer-text input at all.
    import inspect
    params = set(inspect.signature(gate_claims).parameters)
    assert params == {"state", "claims", "intent", "evidence"}, params
    print("[selfcheck] gate_claims takes no answer text -- evidence only")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
