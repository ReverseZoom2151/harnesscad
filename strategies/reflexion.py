"""Reflexion — Read-Act-Reflect-Write (HARNESS_BLUEPRINT.md sec.8, sec.10).

    "On failed verification, retrieve prior failures, synthesize an insight
     ('this boolean fails when faces are coplanar -> add offset'), store to
     semantic memory, recall next attempt."  — blueprint sec.8

Where Best-of-N spends compute *in parallel* (breadth), Reflexion spends it
*sequentially* (depth): it learns within the run. Each attempt:

  READ    — recall prior insights from semantic memory and prepend them to the
            brief (blueprint sec.7: put learned context at the head).
  ACT     — plan ops for the augmented brief, apply through a FRESH session.
  REFLECT — on failed verify, turn the diagnostics into an actionable insight
            (an injected critic LLM, or the built-in ``heuristic_reflect``).
  WRITE   — store the insight to semantic memory so the next attempt recalls it.

Retries up to ``max_attempts``; returns the full trajectory + whether it converged.
The MemoryStore is injected and used only through its public API
(``get_semantic``/``set_semantic``/``add_episodic``) — never edited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from cisp.ops import Op
from cisp.protocol import ApplyOpsResult
from verify import Diagnostic, Severity


# Default semantic-memory key under which the accumulated insight list lives.
SEMANTIC_INSIGHT_KEY = "reflexion:insights"


# Map a diagnostic code (or message keyword) -> an actionable, reusable insight.
# These encode the block-and-correct / error-recovery ladder from sec.10 as memory.
_INSIGHT_RULES = {
    "over-constrained": "a sketch was over-constrained; drop a redundant dimensional "
                        "constraint so its DOF stays >= 0",
    "under-constrained": "a sketch was left under-constrained; add constraints to pin "
                         "its remaining degrees of freedom",
    "bad-ref": "an op referenced an id that does not exist yet; create the sketch/entity "
               "before referencing it",
    "empty-sketch": "extrude ran on a sketch with no profile; add a closed profile before "
                    "extruding",
    "empty-solid": "a feature produced no solid; ensure the profile is closed and the "
                   "extrude distance is non-zero",
    "bad-value": "an op used an out-of-range value (non-positive radius/distance); use "
                 "positive, manufacturable values",
    "no-solid": "an op needed an existing solid; build a base solid (extrude) before "
                "fillet/boolean",
    "self-intersection": "the boolean/feature self-intersected; simplify the profile or "
                         "offset overlapping faces",
}

# Keyword fallbacks scanned in the diagnostic *message* when no code rule matches.
_MESSAGE_KEYWORDS = {
    "coplanar": "boolean fails when faces are coplanar; add a small offset so the faces "
                "are not coincident",
}


def heuristic_reflect(diagnostics: List[Any], brief: str) -> str:
    """Built-in critic: turn ERROR diagnostics into one actionable insight string.

    A drop-in for the injected ``reflect`` critic. Deterministic and dependency-free
    so the Reflexion loop works with no LLM. Matches each diagnostic's ``code`` (then
    a message-keyword fallback) against ``_INSIGHT_RULES`` and joins the distinct
    lessons; unrecognised codes yield a generic "adjust and retry" insight naming them.
    """
    insights: List[str] = []
    unknown_codes: List[str] = []
    for d in diagnostics:
        code = _diag_field(d, "code")
        message = (_diag_field(d, "message") or "").lower()
        severity = _diag_field(d, "severity")
        # Only ERROR-severity diagnostics block the build; reflect on those.
        if severity not in (Severity.ERROR, Severity.ERROR.value, None):
            continue
        rule = _INSIGHT_RULES.get(code)
        if rule is None:
            for kw, kw_rule in _MESSAGE_KEYWORDS.items():
                if kw in message:
                    rule = kw_rule
                    break
        if rule is not None:
            if rule not in insights:
                insights.append(rule)
        elif code and code not in unknown_codes:
            unknown_codes.append(code)

    if not insights and unknown_codes:
        insights.append(
            "prior attempt failed with " + ", ".join(unknown_codes)
            + "; do not re-emit the same op unchanged — adjust the offending "
              "parameters or fall back to a simpler construction")
    if not insights:
        insights.append(
            "prior attempt failed verification; revise the plan and avoid repeating "
            "the same op unchanged")
    return "; ".join(insights)


def _diag_field(d: Any, name: str) -> Any:
    """Read a field from a Diagnostic object OR its dict form."""
    if isinstance(d, dict):
        return d.get(name)
    return getattr(d, name, None)


@dataclass
class ReflexionAttempt:
    """One pass of the loop: what was tried, the outcome, and any insight learned."""

    index: int
    brief: str                         # the augmented brief actually planned against
    recalled: List[str] = field(default_factory=list)   # insights recalled this pass
    ops: Optional[List[Op]] = None
    result: ApplyOpsResult = None      # type: ignore[assignment]
    insight: Optional[str] = None      # synthesized+stored on failure (None on success)
    error: Optional[str] = None        # set when planning raised

    @property
    def ok(self) -> bool:
        return bool(self.result and self.result.ok)


@dataclass
class ReflexionResult:
    """The full trajectory + convergence flag."""

    converged: bool
    attempts: List[ReflexionAttempt] = field(default_factory=list)
    final_result: Optional[ApplyOpsResult] = None
    insights: List[str] = field(default_factory=list)  # all insights in memory at end


class ReflexionLoop:
    """Read-Act-Reflect-Write over an injected planner / session_factory / memory."""

    def __init__(
        self,
        planner,
        session_factory: Callable[[], object],
        memory,
        reflect: Optional[Callable[[List[Any], str], str]] = None,
        max_attempts: int = 3,
        semantic_key: str = SEMANTIC_INSIGHT_KEY,
        record_episodic: bool = True,
    ) -> None:
        """
        Args:
            planner: has ``plan(brief, state_summary=None, diagnostics=None) -> [Op]``.
            session_factory: zero-arg factory -> a FRESH ``HarnessSession`` per attempt.
            memory: a ``MemoryStore`` (used via get_semantic/set_semantic/add_episodic).
            reflect: critic ``reflect(diagnostics, brief) -> insight``. Defaults to the
                dependency-free ``heuristic_reflect``. Inject an LLM-backed critic here.
            max_attempts: cap on attempts.
            semantic_key: key under which insights accumulate in semantic memory.
            record_episodic: also log each attempt to episodic memory (audit trail).
        """
        self.planner = planner
        self.session_factory = session_factory
        self.memory = memory
        self.reflect = reflect or heuristic_reflect
        self.max_attempts = max_attempts
        self.semantic_key = semantic_key
        self.record_episodic = record_episodic

    # --- memory helpers ---------------------------------------------------
    def _recall_insights(self) -> List[str]:
        stored = self.memory.get_semantic(self.semantic_key, [])
        return list(stored) if stored else []

    def _write_insight(self, insight: str) -> None:
        """Append a distinct insight to the semantic-memory list (dedup, keep order)."""
        stored = self._recall_insights()
        if insight not in stored:
            stored.append(insight)
        self.memory.set_semantic(self.semantic_key, stored)

    def _augment_brief(self, brief: str, insights: List[str]) -> str:
        """Prepend recalled insights to the brief (head placement, sec.7)."""
        if not insights:
            return brief
        lessons = "\n".join(f"- {s}" for s in insights)
        return (
            "PRIOR INSIGHTS (learned from earlier failed attempts — apply them):\n"
            f"{lessons}\n\n{brief}"
        )

    # --- the loop ---------------------------------------------------------
    def run(self, brief: str) -> ReflexionResult:
        """Drive Read-Act-Reflect-Write until verified or ``max_attempts`` exhausted."""
        attempts: List[ReflexionAttempt] = []
        last_diagnostics: Optional[List[Diagnostic]] = None

        for i in range(self.max_attempts):
            # READ: recall prior insights and fold them into this attempt's context.
            recalled = self._recall_insights()
            aug_brief = self._augment_brief(brief, recalled)
            attempt = ReflexionAttempt(index=i, brief=aug_brief, recalled=recalled)

            # ACT: plan against a fresh session, then apply.
            session = self.session_factory()
            try:
                ops = self.planner.plan(
                    aug_brief,
                    state_summary=session.summary(),
                    diagnostics=last_diagnostics,
                )
                attempt.ops = ops
                result = session.apply_ops(ops)
            except Exception as exc:
                attempt.error = f"{type(exc).__name__}: {exc}"
                result = ApplyOpsResult(
                    ok=False, applied=0, digest=session.digest(),
                    diagnostics=[Diagnostic(
                        Severity.ERROR, "plan-error",
                        f"attempt {i} failed to plan: {attempt.error}")],
                    rejected=None)
            attempt.result = result

            if self.record_episodic:
                self.memory.add_episodic(
                    brief=brief,
                    ops=attempt.ops or [],
                    outcome="ok" if result.ok else "failed",
                    digest=result.digest,
                    summary=f"reflexion attempt {i}",
                    attempt=i,
                )

            if result.ok:
                # Converged. Record and stop.
                attempts.append(attempt)
                return ReflexionResult(
                    converged=True,
                    attempts=attempts,
                    final_result=result,
                    insights=self._recall_insights(),
                )

            # REFLECT + WRITE: synthesize an insight and persist it for next attempt.
            insight = self.reflect(result.diagnostics, brief)
            attempt.insight = insight
            self._write_insight(insight)
            last_diagnostics = list(result.diagnostics)
            attempts.append(attempt)

        # Exhausted attempts without convergence.
        return ReflexionResult(
            converged=False,
            attempts=attempts,
            final_result=attempts[-1].result if attempts else None,
            insights=self._recall_insights(),
        )
