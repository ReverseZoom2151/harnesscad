"""Deterministic artefact schemas + parsers for the Idea-to-CAD MAS.

These are the concrete, VLM-free pieces of Ocker et al. (2025) that the role
agents exchange and the workflow branches on:

  * ``parse_summary`` — the ``<SUMMARY>...</SUMMARY>`` addendum contract from the
    RequirementsEngineer prompt (Listing 1). The paper is explicit: the agent may
    use the ``<SUMMARY>`` keyword *only* to return the final addendum, so a
    complete block is the deterministic signal that requirement elicitation has
    converged. This parser extracts that block.

  * ``default_view_set`` / ``SEVEN_VIEWS`` — the QualityAssuranceEngineer's fixed
    render set: "top, bottom, right, left, front, back, and an isometric one"
    (sec. 3.4). This is a deterministic enumeration, not a learned choice.

  * ``top_issues`` — the QA prompt's bounded-feedback rule: "Identify the *two
    most relevant* issues" (Listing 5). We implement the deterministic bounding
    (stable, optionally severity-ranked) that the orchestrator applies to
    whatever the VLM proposes.

  * ``detect_ambiguities`` — a heuristic stand-in for the RequirementsEngineer's
    ambiguity check (dimensions/positions), so the requirements loop terminates
    deterministically in tests without a VLM. Focuses on exactly what the paper
    says the RE cares about: "dimensions and positions" (Listing 1).

  * ``QAReport`` / ``RequirementsAddendum`` — small typed value objects.

stdlib only, absolute imports, no wall clock.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# the seven-view render set (sec. 3.4)
# --------------------------------------------------------------------------- #
SEVEN_VIEWS: Tuple[str, ...] = (
    "top", "bottom", "right", "left", "front", "back", "isometric",
)


def default_view_set() -> Tuple[str, ...]:
    """The QA engineer's default render set — the canonical seven views."""
    return SEVEN_VIEWS


# --------------------------------------------------------------------------- #
# <SUMMARY> addendum parsing (Listing 1)
# --------------------------------------------------------------------------- #
_SUMMARY_RE = re.compile(r"<SUMMARY>(.*?)</SUMMARY>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class RequirementsAddendum:
    """The structured output of the requirements phase: the addendum text plus a
    flag for whether a complete ``<SUMMARY>`` block was present (== converged)."""

    text: Optional[str]
    converged: bool

    @property
    def ok(self) -> bool:
        return self.converged and bool(self.text)


def parse_summary(raw: Optional[str]) -> Optional[str]:
    """Extract the inner text of the *first* complete ``<SUMMARY>...</SUMMARY>``.

    Returns the stripped addendum, or ``None`` if no complete block is present.
    Per the paper's contract the ``<SUMMARY>`` keyword marks the *only* place a
    final addendum may appear, so a ``None`` result means requirement elicitation
    has not yet converged. Case-insensitive; tolerant of surrounding prose and of
    newlines inside the block.
    """
    if not raw:
        return None
    m = _SUMMARY_RE.search(raw)
    if m is None:
        return None
    inner = m.group(1).strip()
    return inner or None


def parse_addendum(raw: Optional[str]) -> RequirementsAddendum:
    """Like :func:`parse_summary` but returns a typed, branch-friendly result."""
    inner = parse_summary(raw)
    return RequirementsAddendum(text=inner, converged=inner is not None)


# --------------------------------------------------------------------------- #
# bounded QA feedback (Listing 5: "the two most relevant issues")
# --------------------------------------------------------------------------- #
def top_issues(
    issues: List[str],
    k: int = 2,
    priorities: Optional[List[int]] = None,
) -> List[str]:
    """Bound a discrepancy list to the ``k`` most relevant issues.

    Deterministic. With no ``priorities`` the first ``k`` non-empty issues are
    kept (stable — preserves the order the QA agent proposed them, which the
    paper treats as relevance order). With ``priorities`` (lower == more
    relevant) the issues are stably ranked by priority first, then truncated.
    Blank/whitespace-only issues are dropped.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    cleaned = [(i, s) for i, s in enumerate(issues) if s and s.strip()]
    if priorities is not None:
        if len(priorities) != len(issues):
            raise ValueError("priorities must align 1:1 with issues")
        # stable sort by (priority, original index) over the surviving items
        cleaned.sort(key=lambda pair: (priorities[pair[0]], pair[0]))
    return [s.strip() for _, s in cleaned[:k]]


@dataclass(frozen=True)
class QAReport:
    """The QualityAssuranceEngineer's verification output for one round.

    ``issues`` is already bounded (<= two). ``acceptable`` is the convergence
    signal for Algorithm 3: an empty issue list (the QA prompt returns an empty
    string when the model is acceptable, Listing 5)."""

    issues: List[str] = field(default_factory=list)
    views: Tuple[str, ...] = SEVEN_VIEWS
    acceptable: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "issues": list(self.issues),
            "views": list(self.views),
            "acceptable": self.acceptable,
        }


# --------------------------------------------------------------------------- #
# ambiguity heuristic (Listing 1 focus: dimensions & positions)
# --------------------------------------------------------------------------- #
# A crude but deterministic detector: the RE cares about dimensions and
# positions. We flag a spec as ambiguous when it lacks any numeric dimension, or
# when it uses hedging language that the paper's example specs resolve ("make
# reasonable assumptions", "other dimensions", etc.).
_DIM_RE = re.compile(r"\d")
_HEDGE_RE = re.compile(
    r"reasonable assumptions|other (dimensions|features)|"
    r"disregard|ignore (further )?details|keep it simple",
    re.IGNORECASE,
)


def detect_ambiguities(text: str) -> List[str]:
    """Heuristic ambiguity list for a textual spec (empty == fully specified).

    Deterministic stand-in for the RE's VLM clarify step. Flags:
      * no numeric dimension present at all (nothing to build to);
      * hedging phrases that defer detail to the modeller.

    The paper's fully-specified example specs (e.g. the angle bracket with every
    leg length, hole diameter and spacing given) produce an empty list; the
    hedged ones (the cap: "make reasonable assumptions") produce a flag.
    """
    t = (text or "").strip()
    issues: List[str] = []
    if not t:
        issues.append("specification is empty: no dimensions or positions given")
        return issues
    if not _DIM_RE.search(t):
        issues.append("no numeric dimensions specified")
    if _HEDGE_RE.search(t):
        issues.append("under-specified: spec defers dimensions/features to assumptions")
    return issues
