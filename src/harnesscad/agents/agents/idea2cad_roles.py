"""The Idea-to-CAD V-model role set and its handoff DAG.

Ocker et al. (2025) mirror a human engineering team with exactly three
specialised agents, each owning one V-model phase, plus the human user closing an
outer validation loop (sec. 3.1, Fig. 2):

  * ``RequirementsEngineer`` — the interface between the user and the technical
    CadEngineer. Clarifies the specification and resolves ambiguities
    interactively (Alg. 1). Owns the REQUIREMENTS phase.
  * ``CadEngineer`` — plans, then writes CadQuery code, checks it, executes it,
    and folds in verification/validation feedback (Alg. 2). Owns DESIGN.
  * ``QualityAssuranceEngineer`` — renders the model to a set of views and
    compares them to the spec, returning discrepancies (Alg. 3). Owns
    VERIFICATION.
  * ``User`` (human) — confirms the model or asks for changes (Alg. 4). Owns
    VALIDATION.

This role set is deliberately *distinct* from the generic
Designer/Modeler/Verifier/DFMCritic/RedTeam/Reviewer supervisor layer already in
``agents.roles``: those are a flat escalate-to-stop panel; these are the paper's
named V-model roles with a specific, cyclic **handoff DAG** and per-role
inject-able behaviour that defaults to deterministic heuristics so the whole
layer runs with no VLM.

The distinctive contribution captured here is the *handoff protocol*: who hands
what artefact to whom, and the direction of the feedback edges that make the DAG
cyclic (QA -> CAD and User -> CAD are the corrective back-edges). ``HandoffDAG``
exposes that graph so an orchestrator can route artefacts without hard-coding it.

stdlib only, absolute imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from harnesscad.agents.agents.idea2cad_blackboard import VPhase
from harnesscad.agents.agents.idea2cad_artifacts import (
    parse_summary,
    detect_ambiguities,
    QAReport,
    default_view_set,
    top_issues,
)


# --------------------------------------------------------------------------- #
# role identity
# --------------------------------------------------------------------------- #
ROLE_RE = "requirements-engineer"
ROLE_CAD = "cad-engineer"
ROLE_QA = "quality-assurance-engineer"
ROLE_USER = "user"

# Each named role owns exactly one V-model phase.
ROLE_PHASE: Dict[str, VPhase] = {
    ROLE_RE: VPhase.REQUIREMENTS,
    ROLE_CAD: VPhase.DESIGN,
    ROLE_QA: VPhase.VERIFICATION,
    ROLE_USER: VPhase.VALIDATION,
}


# --------------------------------------------------------------------------- #
# the handoff DAG
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Handoff:
    """One directed edge in the collaboration graph: ``src`` hands ``artifact``
    to ``dst``. ``kind`` distinguishes a forward hand-off from a corrective
    ``feedback`` back-edge (which is what makes the graph cyclic)."""

    src: str
    dst: str
    artifact: str
    kind: str = "forward"   # "forward" | "feedback"


# The paper's collaboration graph (Fig. 2 + Algorithms 1-4):
#   RE  --spec-->      CAD          (forward)
#   CAD --model-->     QA           (forward)
#   QA  --Fver-->      CAD          (feedback back-edge)
#   CAD --model-->     USER         (forward, outer loop output)
#   USER--Fval-->      CAD          (feedback back-edge)
HANDOFFS: Tuple[Handoff, ...] = (
    Handoff(ROLE_RE, ROLE_CAD, "specification", "forward"),
    Handoff(ROLE_CAD, ROLE_QA, "model", "forward"),
    Handoff(ROLE_QA, ROLE_CAD, "verification-feedback", "feedback"),
    Handoff(ROLE_CAD, ROLE_USER, "model", "forward"),
    Handoff(ROLE_USER, ROLE_CAD, "validation-feedback", "feedback"),
)


class HandoffDAG:
    """The role collaboration graph with routing queries.

    Not a pure DAG (the feedback edges create the verification/validation cycles),
    but the *forward* subgraph is a linear DAG RE -> CAD -> QA/USER. This object
    lets an orchestrator ask "who does RE hand to?" or "what are the corrective
    back-edges into CAD?" instead of hard-coding the topology.
    """

    def __init__(self, handoffs: Tuple[Handoff, ...] = HANDOFFS) -> None:
        self.handoffs = tuple(handoffs)

    def roles(self) -> List[str]:
        """All roles that appear as a source or destination, in first-seen order."""
        seen: List[str] = []
        for h in self.handoffs:
            for r in (h.src, h.dst):
                if r not in seen:
                    seen.append(r)
        return seen

    def out_edges(self, role: str, kind: Optional[str] = None) -> List[Handoff]:
        return [h for h in self.handoffs
                if h.src == role and (kind is None or h.kind == kind)]

    def in_edges(self, role: str, kind: Optional[str] = None) -> List[Handoff]:
        return [h for h in self.handoffs
                if h.dst == role and (kind is None or h.kind == kind)]

    def forward_edges(self) -> List[Handoff]:
        return [h for h in self.handoffs if h.kind == "forward"]

    def feedback_edges(self) -> List[Handoff]:
        return [h for h in self.handoffs if h.kind == "feedback"]

    def forward_order(self) -> List[str]:
        """Topological order of the forward (non-feedback) subgraph.

        Kahn's algorithm over the forward edges only, so the corrective cycles do
        not block the sort. Returns roles in dependency order (RE first).
        """
        fwd = self.forward_edges()
        nodes = self.roles()
        indeg: Dict[str, int] = {n: 0 for n in nodes}
        for h in fwd:
            indeg[h.dst] += 1
        # deterministic: process nodes in first-seen order among zero-indegree
        order: List[str] = []
        indeg = dict(indeg)
        while True:
            ready = [n for n in nodes if indeg.get(n, -1) == 0 and n not in order]
            if not ready:
                break
            n = ready[0]
            order.append(n)
            for h in fwd:
                if h.src == n:
                    indeg[h.dst] -= 1
        # append any remaining (part of a cycle in forward subgraph — shouldn't
        # happen for the canonical DAG) to keep the result total.
        for n in nodes:
            if n not in order:
                order.append(n)
        return order


# --------------------------------------------------------------------------- #
# the three agents (+ user proxy)
# --------------------------------------------------------------------------- #
# Injectable behaviours. Each defaults to a deterministic heuristic so the whole
# role layer runs with no VLM. Signatures mirror the paper's prompt inputs.
ClarifyFn = Callable[[Optional[str], str], List[str]]   # (S, T) -> ambiguities
SummariseFn = Callable[[Optional[str], str], str]        # (S, T) -> <SUMMARY> text
PlanFn = Callable[[str], str]                            # R -> plan
CodeFn = Callable[[str, Optional[str]], str]             # (R, hints) -> code
CheckFn = Callable[[str], bool]                          # code -> executable?
ExecFn = Callable[[str], object]                         # code -> model|None
RenderFn = Callable[[object, Tuple[str, ...]], Dict[str, object]]  # (M, views)->imgs
QAFn = Callable[[str, Dict[str, object]], List[str]]     # (R, views) -> issues
DocHintFn = Callable[[str, str, List[str]], str]         # (code, docs, F) -> hints


class RequirementsEngineer:
    """Owns REQUIREMENTS. Clarifies ``(S, T)`` until no ambiguities remain, then
    emits a ``<SUMMARY>`` addendum (Alg. 1, Listing 1).

    Inject ``clarify_fn`` / ``summarise_fn`` to drive with a real VLM; the
    defaults use the deterministic ``detect_ambiguities`` heuristic and a simple
    summary formatter.
    """

    role = ROLE_RE
    phase = VPhase.REQUIREMENTS

    def __init__(
        self,
        clarify_fn: Optional[ClarifyFn] = None,
        summarise_fn: Optional[SummariseFn] = None,
    ) -> None:
        self.clarify_fn = clarify_fn or (lambda s, t: detect_ambiguities(t))
        self.summarise_fn = summarise_fn or self._default_summary

    @staticmethod
    def _default_summary(sketch: Optional[str], text: str) -> str:
        return f"<SUMMARY>{text.strip()}</SUMMARY>"

    def clarify(self, sketch: Optional[str], text: str) -> List[str]:
        """Return the open ambiguities for ``(S, T)`` (empty == fully specified)."""
        return list(self.clarify_fn(sketch, text))

    def summarise(self, sketch: Optional[str], text: str) -> Optional[str]:
        """Emit the requirements addendum, extracted from a ``<SUMMARY>`` block.

        Returns the inner addendum text, or ``None`` if the summariser did not
        wrap it in a complete ``<SUMMARY>...</SUMMARY>`` block (the paper's
        contract: the keyword is used *only* for the final addendum).
        """
        raw = self.summarise_fn(sketch, text)
        return parse_summary(raw)


class CadEngineer:
    """Owns DESIGN. Plans, writes CadQuery code, checks + executes it, and folds
    in verification/validation feedback via doc-hint retrieval (Alg. 2, Listings
    2-4).

    All four steps are injectable; the defaults are deterministic stand-ins so
    tests need no VLM and no CAD kernel.
    """

    role = ROLE_CAD
    phase = VPhase.DESIGN

    def __init__(
        self,
        plan_fn: Optional[PlanFn] = None,
        code_fn: Optional[CodeFn] = None,
        check_fn: Optional[CheckFn] = None,
        exec_fn: Optional[ExecFn] = None,
        dochint_fn: Optional[DocHintFn] = None,
    ) -> None:
        self.plan_fn = plan_fn or (lambda r: f"1. build model for: {r.strip()[:60]}")
        self.code_fn = code_fn or self._default_code
        self.check_fn = check_fn or _ast_ok
        self.exec_fn = exec_fn or (lambda c: {"code": c})
        self.dochint_fn = dochint_fn or (lambda c, d, f: "; ".join(f))

    @staticmethod
    def _default_code(spec: str, hints: Optional[str]) -> str:
        head = "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)\n"
        if hints:
            head += f"# hints: {hints}\n"
        return head

    def plan(self, spec: str) -> str:
        return self.plan_fn(spec)

    def hints_from_docs(self, code: str, docs: str, feedback: List[str]) -> str:
        return self.dochint_fn(code, docs, list(feedback))

    def generate(self, spec: str, hints: Optional[str] = None) -> str:
        return self.code_fn(spec, hints)

    def check(self, code: str) -> bool:
        """Static executability gate (Alg. 2 ``check(C)``; paper uses ``ast``)."""
        return self.check_fn(code)

    def execute(self, code: str) -> object:
        """Execute checked code into a model ``M`` (or ``None`` on failure)."""
        return self.exec_fn(code)


class QualityAssuranceEngineer:
    """Owns VERIFICATION. Renders the model to the standard seven views and
    compares them to the spec, returning at most the two most relevant
    discrepancies (Alg. 3, Listing 5).

    Inject ``render_fn`` / ``qa_fn`` for a real renderer + VLM; the defaults
    return an empty (acceptable) report.
    """

    role = ROLE_QA
    phase = VPhase.VERIFICATION

    def __init__(
        self,
        render_fn: Optional[RenderFn] = None,
        qa_fn: Optional[QAFn] = None,
        views: Tuple[str, ...] = default_view_set(),
        max_issues: int = 2,
    ) -> None:
        self.render_fn = render_fn or (lambda m, v: {name: None for name in v})
        self.qa_fn = qa_fn or (lambda r, imgs: [])
        self.views = tuple(views)
        self.max_issues = max_issues

    def render(self, model: object) -> Dict[str, object]:
        """Render ``M`` into the configured view set (default: seven views)."""
        return self.render_fn(model, self.views)

    def review(self, spec: str, model: object) -> QAReport:
        """Return a bounded QA report (<= ``max_issues`` discrepancies)."""
        images = self.render(model)
        raw = list(self.qa_fn(spec, images))
        issues = top_issues(raw, self.max_issues)
        return QAReport(issues=issues, views=self.views, acceptable=not issues)


class User:
    """The human user closing the outer VALIDATION loop (Alg. 4).

    Modelled as a proxy so the workflow is testable: inject a ``feedback_fn`` that
    returns a list of change requests given ``(R, M)``; an empty list == the user
    accepts the model and the loop terminates.
    """

    role = ROLE_USER
    phase = VPhase.VALIDATION

    def __init__(self, feedback_fn: Optional[Callable[[str, object], List[str]]] = None) -> None:
        self.feedback_fn = feedback_fn or (lambda r, m: [])

    def validate(self, spec: str, model: object) -> List[str]:
        return list(self.feedback_fn(spec, model))


# --------------------------------------------------------------------------- #
# default static-check gate
# --------------------------------------------------------------------------- #
def _ast_ok(code: str) -> bool:
    """Deterministic executability check: does ``code`` parse under ``ast``?

    Mirrors the paper's pre-execution guard (they use Python's ``ast`` module to
    check generated code before running it). Pure stdlib, no execution.
    """
    import ast
    try:
        ast.parse(code or "")
        return True
    except SyntaxError:
        return False
