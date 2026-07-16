"""A CAD benchmark a frontier model cannot saturate and a weak oracle cannot score.

THE TWO THINGS THE PUBLISHED BENCHMARKS GET WRONG, AND WHY WE CAN EXPLOIT BOTH
------------------------------------------------------------------------------
**Text2CAD-Bench** (arXiv 2605.18430) curates 600 examples by hand and scores them
with Chamfer Distance, volumetric IoU and an invalidity rate. **MUSE**
(arXiv 2605.28579) curates 106 and scores them in three stages: a code check, a
geometric check (watertight / manifold / self-intersection / overlap), and a
design-intent stage that is **a VLM judge agreeing with a human at r = 0.713**.

Both are limited in the same two ways, and this package is built on both limits.

1. THEIR ORACLES ARE WEAK, AND THE BLIND SPOT IS MEASURED, NOT ASSERTED.

   * ``eval/bench/harness/pressure_correlation.py`` correlated every intrinsic
     metric this repository owns against the one outcome it cares about, over 208
     graded attempts. ``is_valid``, ``manifold``, ``watertight``, ``solid_present``,
     ``built`` and ``parse_ok`` are **literally constant** across that set: solved
     and unsolved parts are INDISTINGUISHABLE by MUSE's entire geometric-check
     family. A metric with zero variance has zero information.
   * A volumetric IoU scores **0.973** on an 8 mm hole where the brief demanded
     12 mm, and **0.957** on a hole bored 20 mm from where it was asked for
     (``eval/corpus/shape.py`` states the second, measured, and refused to move its
     own threshold to hide it). Both PASS. ``io/gate.py`` carries a test pinning
     that the displaced hole passes the gate too.
   * A shell opened on the WRONG FACE of a box has -- exactly, not approximately --
     the same volume, the same bounding box, the same genus, and is equally
     watertight and manifold. Every published metric except a VLM's opinion of a
     picture says it is correct.

   So: :mod:`~harnesscad.eval.hardcorpus.discriminative` DELIBERATELY CONSTRUCTS
   the wrong answers that pass. For each one it reports what IoU, Chamfer,
   watertight, manifold and the invalidity rate say (PASS) beside what the measured
   oracle says (FAIL, and where). **The gap is the result.**

2. THEY HUMAN-CURATE. WE DO NOT HAVE TO.

   :mod:`~harnesscad.eval.hardcorpus.generate` INVERTS THE GENERATOR. It samples a
   hard op stream from a seeded grammar, and because WE AUTHORED THE OP STREAM the
   part's ground truth is exact and free -- the hole is at x = 27.5 because we put
   it there, not because a backend told us so. The brief is then derived FROM the
   op stream, in two registers (a natural one and an expert-procedural one, as
   Text2CAD-Bench does). Zero human labelling, arbitrary difficulty, unbounded
   scale. This is the same trick ``eval/grounding/corpus.py`` already plays for
   click targets (942 verified pairs a minute, no human), and it is reused, not
   reinvented.

WHAT MAKES IT HARD (L3, WHICH IS WHERE THE CLIFF IS)
----------------------------------------------------
Text2CAD-Bench's own numbers: at L3 -- sweeps, lofts, shells, revolves, complex
patterns -- GPT-5.2 is **68% invalid** and Claude-4.5-Sonnet, the best model they
tested, still **fails 70%**. Their conclusion is that "these operations remain
outside most LLMs' pretraining corpus". Our corpus targets exactly those ops and
DEEP chains of them (10-20 ops, not the 4 our old briefs had -- ``qwen2.5-coder:14b``
solves 66.7% of that corpus BLIND, with no harness at all, which is the whole
reason this package exists).

WHERE IoU IS NOT MERELY WEAK BUT DEFINITIONALLY THE WRONG TOOL
--------------------------------------------------------------
:mod:`~harnesscad.eval.hardcorpus.constraints`. "A bracket that takes an M8 bolt,
carries 200 N and fits in 50x50x20." The ground truth is a CONSTRAINT SET, not a
part; MANY parts satisfy it, so a shape metric against one reference answer cannot
score it AT ALL. This is exactly the "fine-grained engineering criteria" band where
MUSE measures closed-source models at **19-21%**. Every constraint we ship is one
we can actually CHECK on the model's own geometry. The ones we could not check
were DROPPED, and they are named in the module docstring, because a constraint you
cannot verify is decoration.

WHAT NOBODY BENCHMARKS AT ALL
-----------------------------
:mod:`~harnesscad.eval.hardcorpus.ambiguous`. A brief with a dimension missing,
where the correct behaviour is TO ASK. A model that confidently invents the number
is worse than one that stops, and every benchmark in the field scores it higher.

THE RULES THIS PACKAGE KEEPS
----------------------------
* **Every brief's own reference stream must BUILD and pass ``io/gate.py``.** This
  is asserted in the test suite, per brief. A corpus whose reference answer does
  not pass its own grader is measuring the engine's bugs and billing the model --
  which is exactly what contaminated v1 (its shell briefs probed a point on the
  OUTER face and only ever passed because of a backend bug).
* **Both scores, always, side by side.** The weak metrics the field uses
  (:mod:`~harnesscad.eval.hardcorpus.weak`) and the measured oracle
  (:mod:`~harnesscad.eval.hardcorpus.oracle`). Reporting only ours would be as
  unfalsifiable as reporting only theirs.
* **Ground truth never comes from this repository.** ``eval/corpus/spec.Brief``
  refuses to be constructed without a declared, non-us provenance and an expected
  bbox; it is IMPORTED, not re-implemented.
* Deterministic and seeded. No wall clock.
"""

from __future__ import annotations

__all__ = ["occt", "weak", "oracle", "generate", "discriminative", "constraints",
           "ambiguous", "score", "report", "contract_grader"]
