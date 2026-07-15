"""agents.pdd -- the Parts-Driven Development (PDD) orchestrator.

PDD is HarnessCAD's spec-driven-development variant for CAD generation (see
``audit/pdd_synthesis.md``). It runs SDD's four phases end to end as ONE named
pipeline with the Measured Geometric Contract (MGC) as the spine:

* **Specify** -> compile the part brief into an MGC
  (:mod:`harnesscad.domain.spec.contract`), surfacing every unbound
  ``[NEEDS CLARIFICATION]`` predicate BEFORE anything is built.
* **Plan** -> accept a CISP op program (the model's job, not the pipeline's).
* **Implement** -> apply the ops through an injected executor/backend double,
  so this package hard-depends on no geometry kernel.
* **Validate** -> re-measure through the output gate
  (:mod:`harnesscad.io.gate`), run the differential oracle, check the MGC, and
  run the standing gates (orphan-provenance, mutation-score, coverage-matrix),
  then fold them into one :class:`~harnesscad.agents.pdd.pipeline.PddVerdict`.

The honest residual from the synthesis doc is carried on every verdict: a PASS
means "passes every measured predicate", NOT "matches designer intent" -- the
oracle is many-to-one, so a part can hit every contracted quantity for the wrong
reason. PDD narrows the space; it does not close it.

This package imports its collaborators LAZILY (inside functions, guarded by
``try/except ImportError``): four sibling modules are authored in parallel and
may not exist at import time, so ``import harnesscad.agents.pdd`` must always
succeed and degrade gracefully when a collaborator is absent.

Nothing is imported eagerly here. Import :mod:`harnesscad.agents.pdd.pipeline`
for the public surface (``PddPipeline``, ``run_pdd``, ``PddVerdict``, ...).
"""

from __future__ import annotations

__all__ = ["pipeline"]
