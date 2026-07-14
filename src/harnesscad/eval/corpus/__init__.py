"""A benchmark corpus the harness did not write.

THE HOLE THIS PACKAGE EXISTS TO FILL
------------------------------------
``eval/pressure/briefs.py`` was authored by the system under test. Its
``trap_fillet_*`` briefs encoded the harness's own ``preflight-RADIUS_TOO_LARGE``
ceiling as ground truth -- and that rule was unsound (it fired at r = 3.1 on a
6 mm plate and stayed silent at r = 3.0, the true degenerate limit). The grader
was therefore REWARDING A MODEL FOR OBEYING THE HARNESS'S BUG. Its shell briefs
probed their "inside" point exactly ON the outer face, so they could only ever be
satisfied by a part the (broken, dilating) shell had pushed outward; with the
backend fixed, no answer solves them -- not even each brief's own reference
solution. Fleet and corpus shared the blind spot, because one hand wrote both.

Every brief here derives its ground truth from a source OUTSIDE this repository:

``analytic``      arithmetic. A 60x40x10 plate has volume 24000 mm3 because
                  60*40*10 = 24000, and no code in this repo has a vote on that.
                  Steiner's formula gives the exact volume of an all-edge fillet;
                  a through hole adds exactly one handle to the genus. See
                  :mod:`harnesscad.eval.corpus.analytic`.
``standard``      a published engineering standard -- ISO 7089 / DIN 125 plain
                  washers, ISO 273 bolt clearance holes, ISO 261 metric thread
                  pitches, ISO 4032 hex nut widths. The dimensions come from a
                  document nobody here can edit. See
                  :mod:`harnesscad.eval.corpus.standards`.
``metamorphic``   a relation between two runs of the SAME engine (scale every
                  length by k, the volume must go up by k^3). It needs no ground
                  truth at all and holds even when every absolute number the
                  engine reports is wrong. This is the most contamination-
                  resistant oracle in the repository and it is the PRIMARY GATE
                  of :func:`harnesscad.eval.corpus.run.run` -- a corpus score is
                  not even reported when a law is broken.
``differential``  six independent engines. A part all of them agree on is right
                  for reasons that have nothing to do with our brief-writer's
                  opinion. See :mod:`harnesscad.eval.corpus.consensus`.

THE THREE STRUCTURAL FIXES
--------------------------
1. **Every brief carries an EXPECTED BBOX.** ``report.md:92`` names the exact
   hole: the pressure briefs carry ``bbox=None``, so the corpus could not see a
   shell that dilated the part by 3 mm on every face. A brief that does not state
   the envelope it expects cannot catch an envelope bug. :class:`.spec.Brief`
   REFUSES to be constructed without one.

2. **A SHAPE metric beside the envelope metrics.** Volume, bbox and sparse probes
   are all ENVELOPE families and are many-to-one by construction: a hole bored
   10 mm from where the brief asked for it changes neither the bbox nor the
   volume and scores perfectly. :mod:`harnesscad.eval.corpus.shape` adds a
   world-coordinate volumetric IoU against the brief's reference solid. It is
   reported ALONGSIDE the envelope verdict, never instead of it.

3. **A GENUINE HELD-OUT SET.** :mod:`harnesscad.eval.corpus.dev` may be read,
   debugged against and tuned on. :mod:`harnesscad.eval.corpus.heldout` may only
   be SCORED, through :mod:`harnesscad.eval.corpus.score`, and the discipline is
   mechanically enforced by a test that fails if any other module imports it.

WHAT THE GRADER MUST NOT TOUCH
------------------------------
Nothing in this package runs the verifier fleet. The fleet is the thing under
test; a grader that consulted it would be exactly the contamination this package
was built to remove. Grading is: build the op stream on a geometric engine at
``verify_level="core"``, and compare the SOLID against arithmetic.
"""

from __future__ import annotations

__all__ = ["spec", "analytic", "standards", "dev", "shape", "grade",
           "consensus", "run", "score"]
