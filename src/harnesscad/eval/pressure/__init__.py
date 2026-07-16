"""Pressure test — does a typed diagnostic actually beat a blind retry?

The harness's central claim is that returning a NAMED, TYPED error to the model
("shell thickness 9 mm >= available stock 5 mm; the wall consumes the whole
solid") lets it repair a *specific* mistake, where a bare kernel failure only
lets it resample blindly. This package measures that claim.

The design is a paired A/B over one model, one prompt, one seed, one attempt
budget. The ONLY thing that differs between the two arms is the feedback channel:

  loop A ("blind")   apply(verify_level="core")  -> feed back the raw failure text
                     (backend exception / core verifier message), the way a bare
                     kernel would report it: no code, no location, no advice.
  loop B ("harness") apply(verify_level="full")  -> feed back the typed fleet
                     diagnostics: severity + code + message + where.

Grading is deliberately independent of the feedback channel: every attempt's op
stream is re-applied in a fresh ``verify_level="full"`` session and graded
against the brief's declared geometric ground truth (bounding box, volume,
required ops), which NEITHER loop is ever shown. So a loop cannot "win" simply
by being told what the grader wants -- the geometry has to actually be right.

Layout
------
``briefs``   the checked-in corpus (26 briefs, increasing difficulty, including
             traps whose naive reading is geometrically infeasible)
``prompts``  the single system/user prompt both arms share, plus the two
             feedback formatters that are the independent variable
``cache``    content-addressed disk cache of model outputs, keyed by
             (model, seed, attempt, messages) -- a re-run is free and identical
``model``    the ollama-via-litellm client, the caching wrapper, and the
             scripted client the tests use (so the suite never touches ollama)
``loops``    the shared attempt loop; ``feedback=`` selects the arm
``report``   aggregation + the tables
``runner``   the resumable orchestrator

``clarification_bench``  a SECOND, independent probe over the same corpus: it
             mutates a ``Brief.text`` to inject exactly one ambiguity and grades
             the questions an assistant asks back (Matched / Hallucinated /
             Missed). It is deliberately NOT a ``--loop`` arm. Every arm in
             ``loops`` is a model-driven attempt loop that ``run_brief``
             dispatches and ``metrics.grade`` scores against the brief's
             geometric ground truth; this bench runs no model and scores
             QUESTIONS, not geometry, so putting it in ``ALL_LOOPS`` would only
             mean ``--loop all`` handing it to a ``run_brief`` that cannot run
             it. It is registered here, on the package surface, instead.
"""

from __future__ import annotations

from harnesscad.eval.pressure.briefs import BRIEFS, Brief, brief_by_id, briefs_for
from harnesscad.eval.pressure.clarification_bench import (
    DIRECT_CONFLICT,
    UNDER_SPECIFIED,
    ClarificationGrade,
    MisleadingBrief,
    QuestionMatch,
    grade_clarification,
    mutate_direct_conflict,
    mutate_under_specified,
)
from harnesscad.eval.pressure.loops import BLIND, HARNESS, LOOPS, run_brief
from harnesscad.eval.pressure.metrics import AttemptRecord, BriefResult, grade

__all__ = [
    "BRIEFS",
    "Brief",
    "brief_by_id",
    "briefs_for",
    "BLIND",
    "HARNESS",
    "LOOPS",
    "run_brief",
    "AttemptRecord",
    "BriefResult",
    "grade",
    "UNDER_SPECIFIED",
    "DIRECT_CONFLICT",
    "MisleadingBrief",
    "QuestionMatch",
    "ClarificationGrade",
    "mutate_under_specified",
    "mutate_direct_conflict",
    "grade_clarification",
]
