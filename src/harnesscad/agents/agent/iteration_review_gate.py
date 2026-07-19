"""Review-gated iteration contract: fail-first objections, atomic iterations.

The modeling discipline is a deterministic state machine:

* **Single-focus, atomic iterations.** One iteration = one scoped
  deliverable. Generate, run, and review exactly one iteration at a time; a
  second iteration opened while the first is unreviewed marks the run
  INVALID. An iteration that produces what could pass for the complete model
  before ``export_ready`` fails review and must be split or rolled back.
* **Gate-first.** Before modeling, the iteration declares its phase gates;
  after modeling, a review packet records each gate's result
  (``pass`` / ``partial`` / ``unknown`` / ``fail``). Any required gate that
  is not ``pass`` forces a corrective next action (``repair`` / ``refine`` /
  ``replan`` / ``simplify``), never the next phase and never
  ``export_ready``.
* **Fail-first challenge review.** Every review STARTS by listing the
  strongest objections against the model; a gate cannot pass until each
  blocking objection is explicitly defeated with named evidence. An
  undefeated objection blocks export.
* **Export is a hard gate.** ``export_ready`` is refused while any blocking
  objection is undefeated, any required gate is non-pass, or any hard defect
  (visual, functional, disconnected component, primitive-stack evidence) is
  recorded.

Nothing else in the harness models this review-before-next-step discipline:
:class:`~harnesscad.agents.agent.iterative_edit_policy.IterativeEditPolicy`
is accept/rollback over candidate scores, and ``project_iteration`` revises
documents -- neither enforces review-before-next-step or
objection-gated export.

Determinism: all transitions are explicit method calls over recorded state;
the same call sequence always yields the same verdicts. Objection text and
evidence strings are opaque data (UNVERIFIED in harness terms); the gate
logic never interprets them, only their recorded status.

Stdlib only, absolute imports. ``--selfcheck`` replays the skill's rules:
atomicity, gate forcing, objection blocking, and the export hard gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "GATE_RESULTS",
    "NEXT_ACTIONS",
    "CORRECTIVE_ACTIONS",
    "HARD_DEFECT_KINDS",
    "PhaseGate",
    "Objection",
    "HardDefect",
    "ReviewPacket",
    "Iteration",
    "RunInvalid",
    "IterationRun",
    "main",
]

#: The result vocabulary a phase gate may record.
GATE_RESULTS: Tuple[str, ...] = ("pass", "partial", "unknown", "fail")

#: The skill's next-action vocabulary.
NEXT_ACTIONS: Tuple[str, ...] = (
    "brainstorm", "prototype", "repair", "proportion_refine", "surface_refine",
    "detail_add", "simplify", "replan", "export_ready",
)

#: Actions a non-pass required gate forces the choice into.
CORRECTIVE_ACTIONS: Tuple[str, ...] = ("repair", "refine", "replan", "simplify",
                                       "proportion_refine", "surface_refine")

#: Hard-fail defect kinds from the visual/functional audits; any one of these
#: on an open iteration blocks export.
HARD_DEFECT_KINDS: Tuple[str, ...] = (
    "interpenetration", "floating_geometry", "misalignment", "bad_contact",
    "coplanar_overlap", "impossible_assembly", "occluded_feature",
    "scale_mismatch", "view_inconsistency", "disconnected_component",
    "primitive_stack", "missing_moving_axis", "decorative_required_feature",
    "implausible_ratio", "missing_load_path", "blocked_clearance",
)


class RunInvalid(Exception):
    """The atomic-execution rule was broken; the run is invalid."""


@dataclass
class PhaseGate:
    """One declared test for an iteration."""
    name: str
    required: bool = True
    result: str = "unknown"       # one of GATE_RESULTS

    def record(self, result: str) -> "PhaseGate":
        if result not in GATE_RESULTS:
            raise ValueError(f"unknown gate result '{result}' "
                             f"(valid: {GATE_RESULTS})")
        self.result = result
        return self


@dataclass
class Objection:
    """One fail-first challenge: the strongest reason the model fails."""
    text: str
    blocking: bool = True
    defeated: bool = False
    evidence: str = ""

    def defeat(self, evidence: str) -> "Objection":
        """An objection is defeated only with named evidence."""
        if not evidence.strip():
            raise ValueError("an objection cannot be defeated without evidence")
        self.defeated = True
        self.evidence = evidence.strip()
        return self


@dataclass(frozen=True)
class HardDefect:
    """One hard-fail audit finding."""
    kind: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in HARD_DEFECT_KINDS:
            raise ValueError(f"unknown hard defect kind '{self.kind}'")


@dataclass
class ReviewPacket:
    """The compact review evidence for one iteration."""
    gates: List[PhaseGate] = field(default_factory=list)
    objections: List[Objection] = field(default_factory=list)
    defects: List[HardDefect] = field(default_factory=list)
    notes: str = ""

    @property
    def undefeated_blocking(self) -> List[Objection]:
        return [o for o in self.objections if o.blocking and not o.defeated]

    @property
    def failing_required_gates(self) -> List[PhaseGate]:
        return [g for g in self.gates if g.required and g.result != "pass"]

    @property
    def clean(self) -> bool:
        return (not self.undefeated_blocking
                and not self.failing_required_gates
                and not self.defects)

    def to_dict(self) -> dict:
        return {
            "gates": [{"name": g.name, "required": g.required,
                       "result": g.result} for g in self.gates],
            "objections": [{"text": o.text, "blocking": o.blocking,
                            "defeated": o.defeated, "evidence": o.evidence}
                           for o in self.objections],
            "defects": [{"kind": d.kind, "detail": d.detail}
                        for d in self.defects],
            "notes": self.notes,
        }


@dataclass
class Iteration:
    """One single-focus iteration: scope declared up front, reviewed after."""
    index: int
    scope: str
    packet: ReviewPacket = field(default_factory=ReviewPacket)
    reviewed: bool = False
    next_action: str = ""

    def declare_gate(self, name: str, required: bool = True) -> PhaseGate:
        if self.reviewed:
            raise RunInvalid(f"iteration {self.index} is already reviewed; "
                             "gates are declared before modeling")
        gate = PhaseGate(name=name, required=required)
        self.packet.gates.append(gate)
        return gate

    def raise_objection(self, text: str, blocking: bool = True) -> Objection:
        objection = Objection(text=text, blocking=blocking)
        self.packet.objections.append(objection)
        return objection

    def record_defect(self, kind: str, detail: str = "") -> HardDefect:
        defect = HardDefect(kind=kind, detail=detail)
        self.packet.defects.append(defect)
        return defect


class IterationRun:
    """The whole run: enforces atomicity, gate forcing, and the export gate."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.iterations: List[Iteration] = []
        self.exported = False

    # -- lifecycle ----------------------------------------------------------
    def open_iteration(self, scope: str) -> Iteration:
        """Start the next single-focus iteration.

        The atomic-execution rule: a new iteration cannot open while the
        current one is unreviewed. Violating it invalidates the run.
        """
        if not scope.strip():
            raise ValueError("an iteration needs a declared scope")
        if self.iterations and not self.iterations[-1].reviewed:
            raise RunInvalid(
                f"iteration {self.iterations[-1].index} is unreviewed; "
                "generate, run, and review exactly one iteration at a time")
        if self.exported:
            raise RunInvalid("the run already exported; open a new run")
        iteration = Iteration(index=len(self.iterations) + 1, scope=scope.strip())
        self.iterations.append(iteration)
        return iteration

    @property
    def current(self) -> Optional[Iteration]:
        return self.iterations[-1] if self.iterations else None

    # -- review -------------------------------------------------------------
    def allowed_next_actions(self, iteration: Iteration) -> List[str]:
        """What the review evidence permits.

        A required ``fail`` / ``partial`` / ``unknown`` gate, an undefeated
        blocking objection, or a hard defect forces corrective actions only.
        A clean packet permits everything, ``export_ready`` included.
        """
        if iteration.packet.clean:
            return list(NEXT_ACTIONS)
        return [a for a in NEXT_ACTIONS
                if a in CORRECTIVE_ACTIONS or a == "brainstorm"]

    def complete_review(self, iteration: Iteration, next_action: str) -> str:
        """Close the review gate, choosing the next action.

        Refuses a next action the evidence does not permit -- in particular,
        ``export_ready`` while any objection stands, any required gate is
        non-pass, or any hard defect is recorded.
        """
        if iteration.reviewed:
            raise RunInvalid(f"iteration {iteration.index} was already reviewed")
        if not iteration.packet.objections:
            raise RunInvalid(
                "fail-first review: every review starts by listing the "
                "strongest objections; none were raised")
        if next_action not in NEXT_ACTIONS:
            raise ValueError(f"unknown next action '{next_action}' "
                             f"(valid: {NEXT_ACTIONS})")
        allowed = self.allowed_next_actions(iteration)
        if next_action not in allowed:
            packet = iteration.packet
            reasons: List[str] = []
            reasons.extend(f"undefeated objection: {o.text}"
                           for o in packet.undefeated_blocking)
            reasons.extend(f"required gate '{g.name}' is {g.result}"
                           for g in packet.failing_required_gates)
            reasons.extend(f"hard defect: {d.kind}" for d in packet.defects)
            raise RunInvalid(
                f"'{next_action}' is not permitted by the review evidence "
                f"({'; '.join(reasons)}); choose one of {allowed}")
        iteration.reviewed = True
        iteration.next_action = next_action
        if next_action == "export_ready":
            self.exported = True
        return next_action

    # -- reporting ------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "exported": self.exported,
            "iterations": [{
                "index": it.index, "scope": it.scope,
                "reviewed": it.reviewed, "next_action": it.next_action,
                "packet": it.packet.to_dict(),
            } for it in self.iterations],
        }


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------

def _selfcheck() -> int:
    failures: List[str] = []

    def check(cond: bool, message: str) -> None:
        if not cond:
            failures.append(message)

    def expect_invalid(fn, message: str) -> None:
        try:
            fn()
            failures.append(message)
        except RunInvalid:
            pass

    # Atomic execution: no second iteration while the first is unreviewed.
    run = IterationRun("quadcopter frame")
    first = run.open_iteration("main body plate only")
    expect_invalid(lambda: run.open_iteration("arms"),
                   "second unreviewed iteration must invalidate the run")

    # Fail-first: a review without objections is refused.
    gate = first.declare_gate("body_plate_builds")
    gate.record("pass")
    expect_invalid(lambda: run.complete_review(first, "prototype"),
                   "review without objections must be refused")

    # Objections cannot be defeated without evidence.
    objection = first.raise_objection(
        "the plate could pass for the complete model before export_ready")
    try:
        objection.defeat("")
        failures.append("evidence-free defeat must be rejected")
    except ValueError:
        pass
    objection.defeat("review_packet: plate has no arms, motors, or mounts")

    # Clean packet: everything is allowed; move on.
    check("export_ready" in run.allowed_next_actions(first),
          "clean packet permits export_ready")
    check(run.complete_review(first, "prototype") == "prototype",
          "clean review closes")

    # Gate forcing: a required non-pass gate forces corrective actions.
    second = run.open_iteration("four arms with motor mounts")
    second.declare_gate("arms_attach_to_body").record("partial")
    obj2 = second.raise_objection("front-left arm may be floating")
    allowed = run.allowed_next_actions(second)
    check("export_ready" not in allowed and "detail_add" not in allowed,
          "non-pass required gate blocks progress actions")
    check("repair" in allowed and "replan" in allowed,
          "corrective actions stay available")
    expect_invalid(lambda: run.complete_review(second, "export_ready"),
                   "export with a partial required gate must be refused")
    check(not second.reviewed, "refused review leaves the iteration open")

    # Repair, defeat the objection, pass the gate, and a hard defect STILL blocks.
    second.packet.gates[0].record("pass")
    obj2.defeat("geometry_facts: one connected component, arm chord overlaps body")
    second.record_defect("primitive_stack", "arms are bare boxes, no fillets")
    expect_invalid(lambda: run.complete_review(second, "export_ready"),
                   "hard defect must block export")
    check(run.complete_review(second, "surface_refine") == "surface_refine",
          "corrective action accepted alongside a defect")

    # A clean final iteration exports; the run then refuses more work.
    final = run.open_iteration("fillets and export check")
    final.declare_gate("watertight").record("pass")
    final.raise_objection("fillets may have failed on tight edges").defeat(
        "review_packet: fillet edges verified in geometry_facts")
    check(run.complete_review(final, "export_ready") == "export_ready",
          "clean run exports")
    check(run.exported, "run marked exported")
    expect_invalid(lambda: run.open_iteration("more"),
                   "no iterations after export")

    # Vocabulary is closed.
    try:
        final.packet.gates[0].record("meh")
        failures.append("unknown gate result must be rejected")
    except ValueError:
        pass
    try:
        HardDefect(kind="ugly")
        failures.append("unknown defect kind must be rejected")
    except ValueError:
        pass

    # Determinism: the run serialises stably.
    check(run.to_dict() == run.to_dict(), "serialisation stable")
    check([it.next_action for it in run.iterations]
          == ["prototype", "surface_refine", "export_ready"],
          "history preserved in order")

    if failures:
        for f in failures:
            print(f"selfcheck FAIL: {f}")
        return 1
    print("iteration_review_gate selfcheck: OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Review-gated iteration contract (CAD-Agent archive, "
                    "cad-cae-copilot)")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
