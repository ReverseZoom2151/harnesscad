"""showcase — the end-to-end demo: plain-English brief -> verified, rendered part.

Every other eval package in the repo measures a slice of the harness. This one
runs the whole loop the harness exists for, on real local models:

    brief (natural language)
      -> Planner (agents.agent.planner)             the model writes CISP ops
      -> HarnessSession/CISPServer, verify_level="full"   apply + verify
      -> typed diagnostics fed back on failure      the block-and-correct loop
      -> frep backend                               real geometry, zero deps
      -> io.render                                  a hero PNG
      -> image.validate_png                         the PNG is proved non-blank

Nothing here reimplements planning, verification or the correction loop: it
drives the modules that already do those things and records, per (brief, model),
exactly what happened -- how many attempts, which typed diagnostics were seen,
and whether the model ever reached a verified solid.
"""

from harnesscad.eval.showcase.briefs import BRIEFS, Brief, brief_by_id
from harnesscad.eval.showcase.models import MODELS, SEED, make_llm
from harnesscad.eval.showcase.loop import Attempt, RunRecord, run_brief
from harnesscad.eval.showcase.image import PngStats, load_png, validate_png

__all__ = [
    "BRIEFS",
    "Brief",
    "brief_by_id",
    "MODELS",
    "SEED",
    "make_llm",
    "Attempt",
    "RunRecord",
    "run_brief",
    "PngStats",
    "load_png",
    "validate_png",
]
