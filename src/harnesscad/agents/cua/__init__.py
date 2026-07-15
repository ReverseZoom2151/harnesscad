"""agents.cua — a computer-use agent whose success signal is GEOMETRY.

This is the package that finally puts a MODEL inside the verified GUI
environment built in :mod:`harnesscad.io.cua`. The loop is::

    brief -> model -> plan -> DRIVE THE GUI -> export -> MEASURE -> correct

and the thing that makes it different from every other computer-use agent is the
last two arrows: the part is exported through the harness's own channel (never
the application's Save) and measured against the *scripted* FreeCAD backend,
which matches ANALYTIC to 4.5e-16 on all 20 CISP ops. Nobody else in the field
grades a CUA on the volume of the solid it built; they grade it on a model's
opinion of a screenshot, and a screenshot cannot tell 37.5 mm from 375 mm.

Public surface:

* :func:`solve` / :class:`CuaSolve` (loop.py) — one brief, driven end to end,
  reusing :class:`harnesscad.core.harness.AgentHarness` (see loop.py for exactly
  how, and exactly why the GUI cannot ride its session spine unmodified).
* :func:`grade_ops` / :class:`GradeResult` (grade.py) — the geometric oracle.
* :data:`BRIEFS` (briefs.py) — the NL brief corpus with machine-checkable targets.
* :func:`discover_models` / :func:`make_llm` (models.py) — local Ollama models.
* :func:`run_campaign` (run.py) — the whole scorecard, per model.
"""

from __future__ import annotations

from harnesscad.agents.cua.grade import (
    GradeResult, differential, grade_ops, scripted_measure,
)
from harnesscad.agents.cua.loop import (
    ActionTier, CuaSolve, EnvironmentExecutor, EnvSession,
    GeometryGradeVerifier, build_cua_harness, solve,
)

__all__ = [
    "ActionTier",
    "CuaSolve",
    "EnvSession",
    "EnvironmentExecutor",
    "GeometryGradeVerifier",
    "GradeResult",
    "build_cua_harness",
    "differential",
    "grade_ops",
    "scripted_measure",
    "solve",
]
