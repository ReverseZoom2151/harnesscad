"""CADSmith KB2 — the error-solution pattern knowledge base (CADSmith sec. III-C).

CADSmith augments its Error Refiner with a second knowledge base (KB2): a small
database of common CadQuery / OpenCASCADE failure modes. Each pattern carries
trigger keywords, a root-cause explanation, and a fix instruction with a
before/after code fragment. When the Executor returns a traceback, keywords are
matched against the pattern triggers to retrieve the relevant fixes. The paper
stresses this is deterministic keyword matching, not embedding retrieval — no
extra dependency, reproducible, replaceable by vector search later.

This module implements exactly that: a :class:`ErrorPattern` record, a keyword
retriever ranked deterministically by match count (ties broken by pattern id),
and a seeded default catalogue covering the failure classes the paper names
(fillet radius violations, boolean errors, wire-closure failures, arc
construction, extrusion crashes, selector misuse, and more).

Stdlib only, no LLM, no randomness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Pattern record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ErrorPattern:
    """One KB2 entry: how to recognise a failure mode and how to fix it."""

    id: str
    triggers: Tuple[str, ...]        # lowercase keyword/phrase triggers
    root_cause: str
    fix: str
    before: str = ""
    after: str = ""

    def match_score(self, tokens: Sequence[str], text_lc: str) -> int:
        """Number of triggers present in the traceback.

        A multi-word trigger matches as a substring of the lowercased text; a
        single-word trigger matches a whole token (so ``"arc"`` does not fire on
        ``"search"``).
        """
        score = 0
        token_set = set(tokens)
        for trig in self.triggers:
            if " " in trig:
                if trig in text_lc:
                    score += 1
            elif trig in token_set:
                score += 1
        return score


# --------------------------------------------------------------------------- #
# Tokeniser
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())


# --------------------------------------------------------------------------- #
# Knowledge base
# --------------------------------------------------------------------------- #
class ErrorSolutionKB:
    """Keyword-matched retrieval over :class:`ErrorPattern` records.

    ``retrieve`` ranks patterns by descending match score, breaking ties by
    ascending pattern id for determinism, and drops zero-score patterns.
    """

    def __init__(self, patterns: Sequence[ErrorPattern]):
        ids = [p.id for p in patterns]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate pattern id in KB")
        self._patterns = tuple(patterns)

    def __len__(self) -> int:
        return len(self._patterns)

    @property
    def patterns(self) -> Tuple[ErrorPattern, ...]:
        return self._patterns

    def retrieve(self, traceback: str, *, top_k: int = 3) -> Tuple[ErrorPattern, ...]:
        if top_k <= 0:
            return ()
        text_lc = traceback.lower()
        tokens = _tokenize(traceback)
        scored = []
        for p in self._patterns:
            s = p.match_score(tokens, text_lc)
            if s > 0:
                scored.append((s, p))
        # -score for descending score, then id ascending — fully deterministic.
        scored.sort(key=lambda sp: (-sp[0], sp[1].id))
        return tuple(p for _, p in scored[:top_k])

    def context_for(self, traceback: str, *, top_k: int = 3) -> str:
        """Render the retrieved patterns as an injectable text block for the
        Error Refiner prompt (matches how KB2 context is threaded in)."""
        hits = self.retrieve(traceback, top_k=top_k)
        blocks = []
        for p in hits:
            block = (
                f"[{p.id}]\n"
                f"root cause: {p.root_cause}\n"
                f"fix: {p.fix}"
            )
            if p.before or p.after:
                block += f"\nbefore: {p.before}\nafter: {p.after}"
            blocks.append(block)
        return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Default catalogue (the failure classes CADSmith names)
# --------------------------------------------------------------------------- #
def default_error_patterns() -> Tuple[ErrorPattern, ...]:
    return (
        ErrorPattern(
            id="fillet-radius-too-large",
            triggers=("fillet", "radius", "brep_api_error", "chfi2d"),
            root_cause="Fillet radius exceeds the available edge/face geometry.",
            fix="Reduce the fillet radius below half the smallest adjacent edge "
                "length, or select fewer edges.",
            before=".edges('|Z').fillet(20)",
            after=".edges('|Z').fillet(2)",
        ),
        ErrorPattern(
            id="boolean-empty-result",
            triggers=("boolean", "cut", "union", "fuse", "common", "empty",
                      "null shape", "standard_constructionerror"),
            root_cause="Boolean operands do not overlap, so the result is empty "
                       "or a null shape.",
            fix="Ensure the tool solid actually intersects the base solid; check "
                "the translation and workplane of the cutting body.",
            before=".cut(hole.translate((100, 0, 0)))",
            after=".cut(hole.translate((10, 0, 0)))",
        ),
        ErrorPattern(
            id="wire-not-closed",
            triggers=("wire", "not closed", "close", "openwire",
                      "wire is not closed"),
            root_cause="A sketch wire was extruded/lofted without being closed.",
            fix="Call .close() before .extrude(), or ensure the last point "
                "returns to the start.",
            before=".lineTo(10, 0).lineTo(10, 10).extrude(5)",
            after=".lineTo(10, 0).lineTo(10, 10).close().extrude(5)",
        ),
        ErrorPattern(
            id="arc-construction",
            triggers=("arc", "threepointarc", "radiusarc", "collinear",
                      "gc_makearcofcircle"),
            root_cause="Arc points are collinear or the radius is inconsistent "
                       "with the endpoints.",
            fix="Use non-collinear control points, or a radius large enough to "
                "span the chord.",
            before=".threePointArc((5, 0), (10, 0))",
            after=".threePointArc((5, 2), (10, 0))",
        ),
        ErrorPattern(
            id="extrude-crash",
            triggers=("extrude", "prism", "brepprimapi", "makeprism",
                      "zero", "degenerate"),
            root_cause="Extrusion of a degenerate or zero-area profile.",
            fix="Verify the sketch encloses a positive area before extruding.",
            before=".rect(0, 10).extrude(5)",
            after=".rect(10, 10).extrude(5)",
        ),
        ErrorPattern(
            id="selector-empty",
            triggers=("selector", "no faces", "no edges", "empty selection",
                      "indexerror", "list index out of range"),
            root_cause="A face/edge selector string matched nothing on the "
                       "current solid.",
            fix="Loosen the selector (e.g. '>Z' instead of '>Z and |X'), or "
                "verify the feature exists before selecting.",
            before=".faces('>Z and <X')",
            after=".faces('>Z')",
        ),
        ErrorPattern(
            id="shell-thickness",
            triggers=("shell", "thickness", "brepoffset", "offset",
                      "makethicksolid"),
            root_cause="Shell wall thickness is too large for the solid or a "
                       "removed face is invalid.",
            fix="Use a wall thickness well under the smallest feature size and "
                "remove a valid outer face.",
            before=".shell(-9)",
            after=".shell(-1)",
        ),
        ErrorPattern(
            id="loft-mismatch",
            triggers=("loft", "sections", "profile", "brepoffsetapi",
                      "thrusections"),
            root_cause="Loft sections have incompatible vertex counts or "
                       "orientations.",
            fix="Give each loft section the same wire orientation and comparable "
                "vertex ordering.",
        ),
        ErrorPattern(
            id="workplane-missing",
            triggers=("workplane", "no pending", "attributeerror",
                      "has no attribute"),
            root_cause="An operation was chained on an object without an active "
                       "workplane or pending wire.",
            fix="Insert a .workplane() (or .faces(...).workplane()) before the "
                "sketch operation.",
        ),
        ErrorPattern(
            id="timeout",
            triggers=("timeout", "timed out", "60 second", "killed"),
            root_cause="Script exceeded the 60-second execution budget, often "
                       "from an unbounded loop or excessive tessellation.",
            fix="Reduce iteration counts and avoid per-element boolean chains; "
                "batch geometry where possible.",
        ),
    )


def default_kb() -> ErrorSolutionKB:
    return ErrorSolutionKB(default_error_patterns())
