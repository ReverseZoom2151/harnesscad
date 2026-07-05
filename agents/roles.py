"""Role personas for the multi-agent supervisor layer (HARNESS_BLUEPRINT sec.12).

Each role is a small, composable object built *over the existing single-agent
pieces* — the Planner, the HarnessSession spine, the plural verifier set, and the
DFM critic — never a re-implementation. The supervisor (supervisor.py) chains them:

    Designer -> Modeler -> Verifier -> DFMCritic -> Reviewer   (+ RedTeam hook)

Design rules honoured here:
  * Every role has a clear typed input and a typed output dataclass.
  * Personas are *injected*: a role that reasons (Designer, Reviewer, RedTeam)
    takes an LLM / callable persona, and falls back to a deterministic heuristic
    default — so the whole layer runs in tests with **no network**.
  * Mechanical roles (Modeler, Verifier, DFMCritic) need no persona at all: they
    delegate straight to the harness they wrap.

Absolute imports, stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from cisp.ops import Op
from cisp.protocol import ApplyOpsResult
from verify import Diagnostic, Severity, VerifyReport
from checks_dfm import DFMCheck, DFMRules
from agent.planner import Planner
from llm.base import LLM


# --------------------------------------------------------------------------- #
# Shared typed vocabulary
# --------------------------------------------------------------------------- #
_SEV_RANK = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}


@dataclass
class Finding:
    """A single normalised observation, tagged with the role that produced it.

    This is the common currency the Reviewer and RedTeam reason over — a superset
    of a :class:`verify.Diagnostic` plus a ``source`` (which role raised it).
    """

    severity: Severity
    code: str
    message: str
    source: str
    where: Optional[str] = None

    @property
    def is_error(self) -> bool:
        return self.severity is Severity.ERROR

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "where": self.where,
        }


def findings_from(diags: List[Diagnostic], source: str) -> List[Finding]:
    """Lift a list of :class:`verify.Diagnostic` into role-tagged Findings."""
    return [Finding(d.severity, d.code, d.message, source, d.where) for d in diags]


def prioritize(findings: List[Finding]) -> List[Finding]:
    """Deterministically order findings: ERROR, then WARNING, then INFO.

    Stable within a severity band so the original (pipeline) order is preserved —
    this is the Reviewer's "self-prioritizes findings" behaviour, factored out so
    it is testable in isolation.
    """
    return sorted(findings, key=lambda f: _SEV_RANK.get(f.severity, 99))


# --------------------------------------------------------------------------- #
# Designer — brief -> plan (wraps the existing Planner)
# --------------------------------------------------------------------------- #
@dataclass
class DesignPlan:
    ops: List[Op]
    ok: bool
    error: Optional[str] = None
    notes: str = ""


# A heuristic persona maps (brief, state_summary, diagnostics) -> ops.
PlanFn = Callable[[str, Optional[dict], Optional[list]], List[Op]]


class Designer:
    """spec -> plan. Wraps a :class:`agent.planner.Planner` (LLM-backed) or a
    heuristic ``plan_fn`` so tests need no network.

    Inject exactly one persona: ``planner=`` (ready Planner), ``llm=`` (wrapped in
    a Planner for you), or ``plan_fn=`` (a pure callable). With none configured the
    Designer degrades to an explicit not-ok plan rather than guessing geometry.
    """

    role = "designer"

    def __init__(
        self,
        planner: Optional[Planner] = None,
        llm: Optional[LLM] = None,
        plan_fn: Optional[PlanFn] = None,
    ) -> None:
        if planner is None and llm is not None:
            planner = Planner(llm)
        self.planner = planner
        self.plan_fn = plan_fn

    def design(
        self,
        brief: str,
        state_summary: Optional[dict] = None,
        diagnostics: Optional[list] = None,
    ) -> DesignPlan:
        if self.plan_fn is not None:
            ops = list(self.plan_fn(brief, state_summary, diagnostics))
            return DesignPlan(ops, ok=True, notes="heuristic plan_fn")
        if self.planner is not None:
            parsed = self.planner.plan_parsed(brief, state_summary, diagnostics)
            if not parsed.ok:
                return DesignPlan([], ok=False, error=parsed.error or "planner produced no ops")
            return DesignPlan(list(parsed.ops), ok=True)
        return DesignPlan([], ok=False, error="Designer has no planner/llm/plan_fn configured")


# --------------------------------------------------------------------------- #
# Modeler — emit ops / apply through the HarnessSession spine
# --------------------------------------------------------------------------- #
@dataclass
class ModelResult:
    result: ApplyOpsResult

    @property
    def ok(self) -> bool:
        return self.result.ok

    @property
    def applied(self) -> int:
        return self.result.applied

    @property
    def digest(self) -> str:
        return self.result.digest

    @property
    def diagnostics(self) -> List[Diagnostic]:
        return list(self.result.diagnostics)


class Modeler:
    """Applies the Designer's ops through the session — which itself does
    block-and-correct + transactional verify + checkpoint. Purely mechanical:
    no persona, no reasoning; the session is the single source of truth."""

    role = "modeler"

    def model(self, session, plan: DesignPlan) -> ModelResult:
        return ModelResult(session.apply_ops(plan.ops))


# --------------------------------------------------------------------------- #
# Verifier — run the PLURAL verifier set (verify.py)
# --------------------------------------------------------------------------- #
@dataclass
class VerifyOutcome:
    report: VerifyReport

    @property
    def ok(self) -> bool:
        return self.report.ok

    @property
    def diagnostics(self) -> List[Diagnostic]:
        return list(self.report.diagnostics)


class Verifier:
    """Runs the plural verifier set against the *current* model state and reports.

    By default it reuses the session's own verifier list (constraint + solid +
    B-rep), so the role sees exactly what the spine sees; a caller may inject an
    explicit verifier list to widen/narrow the panel."""

    role = "verifier"

    def __init__(self, verifiers: Optional[list] = None) -> None:
        self.verifiers = verifiers

    def verify(self, session) -> VerifyOutcome:
        verifiers = self.verifiers if self.verifiers is not None else session.verifiers
        diags: List[Diagnostic] = []
        for v in verifiers:
            diags += v.check(session.backend, session.opdag).diagnostics
        return VerifyOutcome(VerifyReport(diags))


# --------------------------------------------------------------------------- #
# DFMCritic — wraps checks_dfm.DFMCheck (advisory, never an ERROR)
# --------------------------------------------------------------------------- #
@dataclass
class DFMOutcome:
    report: VerifyReport

    @property
    def diagnostics(self) -> List[Diagnostic]:
        return list(self.report.diagnostics)

    @property
    def warnings(self) -> List[Diagnostic]:
        return [d for d in self.report.diagnostics if d.severity is Severity.WARNING]


class DFMCritic:
    """The DFM critic stage: wraps :class:`checks_dfm.DFMCheck` (wall thickness,
    draft, tool-access, min radii — advisory manufacturability findings). Every
    finding is a WARNING/INFO, so the DFM critic can inform but never *block* on
    its own; a veto is the RedTeam's job."""

    role = "dfm-critic"

    def __init__(self, rules: Optional[DFMRules] = None) -> None:
        self.check_impl = DFMCheck(rules)

    def critique(self, session) -> DFMOutcome:
        return DFMOutcome(self.check_impl.check(session.backend, session.opdag))


# --------------------------------------------------------------------------- #
# RedTeam — hunts non-manufacturable geometry / interference; can VETO
# --------------------------------------------------------------------------- #
@dataclass
class RedTeamResult:
    veto: bool
    reasons: List[str] = field(default_factory=list)


# A probe inspects the live session + collected findings and returns veto reasons
# (an empty list / None == no veto).
RedTeamProbe = Callable[[object, List[Finding]], Optional[List[str]]]

# DFM codes that describe geometry that is not merely costly but effectively
# non-manufacturable / degenerate — the default RedTeam treats these as veto-worthy.
_VETO_CODES = frozenset({
    "thin-envelope", "high-aspect-ratio", "oversized", "empty-solid",
    "self-intersection", "non-manifold", "interference", "collision",
})


def default_redteam_probe(session, findings: List[Finding]) -> List[str]:
    """Default adversarial probe: veto on any finding whose code names
    non-manufacturable geometry or interference (see ``_VETO_CODES``)."""
    return [
        f"{f.source}:{f.code} — {f.message}"
        for f in findings
        if f.code in _VETO_CODES
    ]


class RedTeam:
    """Adversarial role with veto authority. Inject a custom ``probe`` to hunt
    domain-specific failure modes; the default flags non-manufacturable geometry
    and interference codes found anywhere in the round's findings."""

    role = "red-team"

    def __init__(self, probe: Optional[RedTeamProbe] = None) -> None:
        self.probe = probe if probe is not None else default_redteam_probe

    def attack(self, session, findings: List[Finding]) -> RedTeamResult:
        reasons = self.probe(session, findings) or []
        if isinstance(reasons, str):
            reasons = [reasons]
        reasons = list(reasons)
        return RedTeamResult(veto=bool(reasons), reasons=reasons)


# --------------------------------------------------------------------------- #
# Reviewer — two-phase critique -> reflection, self-prioritizes findings
# --------------------------------------------------------------------------- #
@dataclass
class ReviewResult:
    approved: bool
    findings: List[Finding]        # prioritized (ERROR -> WARNING -> INFO)
    critique: str                  # phase 1 summary
    reflection: str                # phase 2 decision rationale

    @property
    def blocking(self) -> List[Finding]:
        return [f for f in self.findings if f.is_error]


class Reviewer:
    """Two-phase reviewer.

    Phase 1 (critique): collect every finding surfaced this round — the model's
    apply diagnostics, the verifier report, the DFM critic, and any RedTeam
    reasons — into one tagged list.

    Phase 2 (reflection): *self-prioritize* those findings (ERROR first) and decide
    approval. Approval requires the round to be non-blocking (model applied, verifier
    clean, no RedTeam veto) AND zero ERROR-severity findings. An optional ``judge``
    LLM persona may be injected to author the reflection narrative; the *decision*
    stays deterministic (hard geometry is judged by the kernel, not the LLM — sec.6)."""

    role = "reviewer"

    def __init__(self, judge: Optional[LLM] = None) -> None:
        self.judge = judge

    def review(
        self,
        brief: str,
        findings: List[Finding],
        *,
        blocking_ok: bool,
        veto: bool = False,
    ) -> ReviewResult:
        # -- phase 1: critique -------------------------------------------------
        ordered = prioritize(findings)
        errs = [f for f in ordered if f.is_error]
        warns = [f for f in ordered if f.severity is Severity.WARNING]
        critique = (
            f"critique: {len(errs)} error(s), {len(warns)} warning(s), "
            f"{len(ordered) - len(errs) - len(warns)} info across "
            f"{len({f.source for f in ordered})} role(s)."
        )

        # -- phase 2: reflection + decision -----------------------------------
        approved = blocking_ok and not errs and not veto
        if approved:
            reflection = "reflection: model verified and manufacturable; no blocking findings — APPROVE."
        elif veto:
            reflection = "reflection: RedTeam veto stands; escalate and re-plan — REJECT."
        elif errs:
            top = errs[0]
            reflection = (
                f"reflection: {len(errs)} blocking finding(s); highest priority "
                f"[{top.source}] {top.code}: {top.message} — escalate and re-plan."
            )
        else:
            reflection = "reflection: model not yet in a verified state — escalate and re-plan."

        if self.judge is not None:
            reflection = self._judge_narrative(brief, critique, approved) or reflection

        return ReviewResult(approved=approved, findings=ordered,
                            critique=critique, reflection=reflection)

    def _judge_narrative(self, brief: str, critique: str, approved: bool) -> Optional[str]:
        """Optional subjective narrative from an injected LLM persona. Never flips
        the deterministic decision — it only phrases the rationale."""
        try:
            from llm.base import system, user
            verdict = "APPROVE" if approved else "REJECT"
            res = self.judge.complete([
                system("You are a senior CAD design reviewer. One-sentence rationale."),
                user(f"Brief: {brief}\n{critique}\nVerdict already decided: {verdict}."),
            ])
            text = (res.text or "").strip()
            return f"reflection: {text}" if text else None
        except Exception:  # noqa: BLE001 - a judge hiccup must not break review
            return None
