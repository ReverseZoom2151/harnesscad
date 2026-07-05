"""Shared design-state blackboard for the Idea-to-CAD V-model MAS.

This implements the *shared design state* that the three role agents of Ocker et
al. (Honda Research Institute, "From Idea to CAD: A Language Model-Driven
Multi-Agent System for Collaborative Design", 2025) read from and write to as
they hand a design around the V-model phases (requirements -> design ->
verification -> validation).

The paper's Algorithms 1-4 all read/write the same handful of state variables:

  * ``S`` / ``T``   — the sketch and text that make up the specification input.
  * ``R``           — the *specification* ``R = (S, T)`` plus the requirements
                      addendum learned interactively (Algorithm 1).
  * ``plan``        — the CadEngineer's coarse textual modeling plan (Alg. 2).
  * ``docs``/``hints`` — retrieved CadQuery documentation and the extracted
                      fix-hints derived from it (Alg. 2).
  * ``C``           — the generated CAD (CadQuery) code (Alg. 2).
  * ``M``           — the produced model / STL handle (Alg. 2).
  * ``Fver``        — verification feedback from the QualityAssuranceEngineer
                      (Alg. 3).
  * ``Fval``        — validation feedback from the human user (Alg. 4).

Crucially, Algorithm 3 feeds the CadEngineer ``design(R, Fval + Fver)`` — the two
feedback channels are *concatenated*, validation-feedback first. This module owns
that composition (``combined_feedback``) so every consumer applies the same rule.

This is a pure, deterministic value container: no VLM, no wall clock, stdlib
only. A monotonic revision counter records every write so a trajectory is
replayable. Absolute imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class VPhase(str, Enum):
    """The four V-model development phases the system walks through (sec. 3.1).

    Subclasses ``str`` so ``phase.value`` round-trips through JSON and a phase
    compares equal to its wire string.
    """

    REQUIREMENTS = "requirements"   # requirement elicitation / specification
    DESIGN = "design"               # model creation (plan + code + exec)
    VERIFICATION = "verification"   # QA compares views to spec
    VALIDATION = "validation"       # user confirms / requests changes


# The canonical phase order (the descending-then-ascending V, flattened to the
# forward workflow order the paper walks).
PHASE_ORDER: Tuple[VPhase, ...] = (
    VPhase.REQUIREMENTS,
    VPhase.DESIGN,
    VPhase.VERIFICATION,
    VPhase.VALIDATION,
)


@dataclass
class Revision:
    """One immutable entry in the blackboard's write log.

    ``rev`` is a monotonic integer (0, 1, 2, ...); ``field`` names the slot
    written; ``phase`` is the phase active at write time; ``summary`` is a short
    human string. The value itself is not copied here (it lives in the slot) —
    this is an audit index, not a full snapshot store.
    """

    rev: int
    field: str
    phase: VPhase
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rev": self.rev,
            "field": self.field,
            "phase": self.phase.value,
            "summary": self.summary,
        }


@dataclass
class DesignBlackboard:
    """The shared design state the three agents collaborate over.

    Every mutation goes through a ``post_*`` method so the revision log stays
    authoritative and the active ``phase`` advances explicitly. Reads are plain
    attribute access (or the derived ``specification`` / ``combined_feedback``).
    """

    # --- specification input (S, T) -------------------------------------
    sketch: Optional[str] = None          # S — an opaque sketch/image handle
    text: str = ""                        # T — the (growing) textual description
    addendum: Optional[str] = None        # requirements addendum from Alg. 1

    # --- design artefacts ------------------------------------------------
    plan: Optional[str] = None            # coarse textual modeling plan
    docs: Optional[str] = None            # retrieved library documentation
    hints: Optional[str] = None           # fix-hints extracted from docs+feedback
    code: Optional[str] = None            # C — generated CAD code
    model: Optional[Any] = None           # M — produced model / STL handle

    # --- feedback channels ----------------------------------------------
    fver: List[str] = field(default_factory=list)   # verification feedback (QA)
    fval: List[str] = field(default_factory=list)   # validation feedback (user)

    # --- bookkeeping -----------------------------------------------------
    phase: VPhase = VPhase.REQUIREMENTS
    log: List[Revision] = field(default_factory=list)
    _rev: int = 0

    # ------------------------------------------------------------------ #
    # write log
    # ------------------------------------------------------------------ #
    def _record(self, field_name: str, summary: str) -> int:
        entry = Revision(self._rev, field_name, self.phase, summary)
        self.log.append(entry)
        self._rev += 1
        return entry.rev

    @property
    def revision(self) -> int:
        """The next revision number that will be assigned (== len(log))."""
        return self._rev

    def enter_phase(self, phase: VPhase) -> "DesignBlackboard":
        """Advance the active V-model phase (recorded in the log)."""
        self.phase = phase
        self._record("phase", f"enter {phase.value}")
        return self

    # ------------------------------------------------------------------ #
    # specification (Algorithm 1)
    # ------------------------------------------------------------------ #
    def post_input(self, sketch: Optional[str], text: str) -> "DesignBlackboard":
        """Seed the initial user specification input ``(S, T)``."""
        self.sketch = sketch
        self.text = text or ""
        self._record("input", f"S={'yes' if sketch else 'none'} |T|={len(self.text)}")
        return self

    def append_text(self, more: str) -> "DesignBlackboard":
        """Append a user clarification turn to ``T`` (Alg. 1: ``T <- T + input``)."""
        if more:
            self.text = (self.text + "\n" + more) if self.text else more
            self._record("text", f"+{len(more)} chars")
        return self

    def post_addendum(self, addendum: str) -> "DesignBlackboard":
        """Record the requirements addendum extracted from the <SUMMARY> block."""
        self.addendum = addendum
        self._record("addendum", f"|addendum|={len(addendum or '')}")
        return self

    @property
    def specification(self) -> str:
        """The specification ``R`` = text ``T`` plus the learned addendum.

        The sketch ``S`` is carried separately (``self.sketch``) as an opaque
        handle; ``R``'s textual portion is what the CadEngineer consumes.
        """
        parts = [p for p in (self.text, self.addendum) if p]
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # design artefacts (Algorithm 2)
    # ------------------------------------------------------------------ #
    def post_plan(self, plan: str) -> "DesignBlackboard":
        self.plan = plan
        self._record("plan", f"|plan|={len(plan or '')}")
        return self

    def post_docs(self, docs: str) -> "DesignBlackboard":
        self.docs = docs
        self._record("docs", f"|docs|={len(docs or '')}")
        return self

    def post_hints(self, hints: str) -> "DesignBlackboard":
        self.hints = hints
        self._record("hints", f"|hints|={len(hints or '')}")
        return self

    def post_code(self, code: str) -> "DesignBlackboard":
        self.code = code
        self._record("code", f"|code|={len(code or '')}")
        return self

    def post_model(self, model: Any) -> "DesignBlackboard":
        self.model = model
        self._record("model", "model produced" if model is not None else "cleared")
        return self

    # ------------------------------------------------------------------ #
    # feedback (Algorithms 3 & 4)
    # ------------------------------------------------------------------ #
    def post_verification_feedback(self, feedback: List[str]) -> "DesignBlackboard":
        """Set ``Fver`` — the QA engineer's discrepancy list for this round."""
        self.fver = list(feedback or [])
        self._record("fver", f"{len(self.fver)} issue(s)")
        return self

    def post_validation_feedback(self, feedback: List[str]) -> "DesignBlackboard":
        """Set ``Fval`` — the user's validation feedback for this round."""
        self.fval = list(feedback or [])
        self._record("fval", f"{len(self.fval)} item(s)")
        return self

    @property
    def combined_feedback(self) -> List[str]:
        """``Fval + Fver`` — the exact concatenation Algorithm 3 feeds to design.

        Validation feedback (the human's) comes first, then verification feedback
        (the QA agent's), matching ``design(R, Fval + Fver)`` in Alg. 3.
        """
        return list(self.fval) + list(self.fver)

    @property
    def has_feedback(self) -> bool:
        return bool(self.fval) or bool(self.fver)

    # ------------------------------------------------------------------ #
    # serialisation
    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict[str, Any]:
        """A JSON-friendly snapshot of the current state (model as a bool flag)."""
        return {
            "phase": self.phase.value,
            "revision": self._rev,
            "sketch": self.sketch,
            "text": self.text,
            "addendum": self.addendum,
            "plan": self.plan,
            "hints": self.hints,
            "code": self.code,
            "has_model": self.model is not None,
            "fver": list(self.fver),
            "fval": list(self.fval),
            "specification": self.specification,
        }

    def log_dicts(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.log]
