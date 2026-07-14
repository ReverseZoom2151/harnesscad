"""The SHAPE metric: volumetric IoU against a reference solid. Reusable.

WHY A SECOND METRIC AT ALL
--------------------------
Bounding box, volume and a handful of probe points are all ENVELOPE families.
They constrain the outline and the amount of material, and they are MANY-TO-ONE
by construction: a hole bored 10 mm from where the brief asked for it changes the
bbox by nothing and the volume by nothing, and it scores perfectly. Probes narrow
that -- a probe on the hole's axis catches a hole that moved far -- but a probe is
a point, and the space of wrong parts that satisfy a dozen points is still large.

IoU integrates over the whole part instead of sampling a dozen places in it::

    IoU = |A and B| / |A or B|

computed in WORLD coordinates on the F-rep backend's exact signed-distance field.
No rigid alignment, no inertia normalisation, no symmetry enumeration -- which is
the deliberate opposite of ``eval/bench/geometry/solid_iou.py``, whose pose-
invariant IoU is right when you are comparing SHAPES and wrong when you are asking
whether the feature landed WHERE THE BRIEF SAID. A hole in the wrong place is
exactly the failure this metric exists to catch, and a pose-invariant score would
rotate the part until the hole lined up and forgive it.

WHAT IoU CANNOT DO, MEASURED AND STATED UP FRONT
------------------------------------------------
**A volumetric IoU is blind to a SMALL misplaced feature, and the blindness is
arithmetic, not a bug.** Move an 8 mm hole 20 mm across a 60x40x12 plate and the
symmetric difference is two holes' worth of material -- 2 * pi * 4^2 * 12 = 1206
mm3 against a union of about 29,400 -- so the IoU is **0.957**. That is a wrong
part, and it sails over any threshold you would dare to set: pushing the bar above
0.957 to catch it would fail correct parts, because a CORRECT rebuild on this
sampled engine already disagrees with itself by a percent or two of surface band.

This was measured, not assumed (``tests/eval/corpus/test_corpus.py``), and the
threshold was NOT moved afterwards to make the number look better. That is the
whole discipline of this package in one decision: a metric that is tuned until it
agrees with the answer you wanted is the pressure corpus again.

So IoU is a check on GROSS shape -- a part built at the wrong scale, a boolean that
took the wrong operand, a feature omitted entirely, a shell that dilated -- and it
is NOT the check that a small hole is in the right place. The PROBE POINTS are
that check (``grade.py``): a probe on a hole's axis is a point assertion, and a
point assertion is exactly what an integral over the whole part cannot make.

The two families are complementary and BOTH are reported. Neither is sufficient,
which is precisely why the pressure corpus's single envelope verdict was not
either.

REPORTED ALONGSIDE, NEVER INSTEAD OF
------------------------------------
The envelope verdict stays. IoU is a SAMPLE of an integral and it is measured
against ONE correct answer (the reference stream), not against the equivalence
class of correct answers -- a brief saying "four holes near the corners" does not
pin them to the micron, so IoU penalises legitimate variation as well as error.
Two numbers, both published. Collapsing them into one would be the same mistake
as the corpus this package replaces: a single number nobody can audit.

REUSABLE ON PURPOSE
-------------------
:func:`iou_of_ops` takes two OP STREAMS and knows nothing about briefs, so
``eval/pressure`` (whose IoU grader is owned by another agent) can import it
rather than grow a second copy. The primitive underneath, :func:`iou_of_backends`,
takes two built solids.

Deterministic: seeded sampling, no wall clock. The same two streams give the same
IoU, bit for bit, forever.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Op

__all__ = ["ShapeScore", "SAMPLES", "SAMPLE_SEED", "IOU_MATCH",
           "iou_of_backends", "iou_of_ops", "build"]

Vec3 = Tuple[float, float, float]

#: Sample count for the Monte-Carlo volume integral. At 20k the standard error on
#: a ratio near 0.9 is ~0.002 -- an order of magnitude below the threshold's
#: distance from 1.0.
SAMPLES = 20000

#: Fixed, so the metric is a function of the geometry and of nothing else.
SAMPLE_SEED = 20260714

#: Padding on the sampled box (mm), so neither solid can touch its boundary.
PAD = 1.0

#: The IoU at or above which two solids are called the same shape. Derived a
#: priori from the F-rep grid's own discretisation error: the engine samples a
#: field on a grid, so a CORRECT rebuild of the same part disagrees with itself by
#: a surface band worth a percent or two of a small part's volume, and a threshold
#: tighter than that fails correct answers.
#:
#: IT WAS NOT MOVED AFTER A SCORE WAS SEEN, and one score in particular: an 8 mm
#: hole displaced 20 mm scores 0.957 and therefore PASSES this threshold. The
#: honest response to that is to write it down (see the module docstring) and let
#: the probe points catch it -- not to raise the bar to 0.96 and fail correct parts
#: to make one wrong part fail. A threshold tuned until the benchmark says what you
#: wanted is the benchmark this package exists to replace.
IOU_MATCH = 0.90


@dataclass
class ShapeScore:
    ok: bool = False                 # both solids exist and an IoU was computed
    iou: Optional[float] = None
    matched: bool = False            # iou >= IOU_MATCH
    reason: str = ""
    samples: int = 0

    def to_dict(self) -> dict:
        return {"ok": self.ok, "iou": self.iou, "matched": self.matched,
                "reason": self.reason, "samples": self.samples}


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
def build(ops: Sequence[Op], backend: str = "frep") -> Optional[Any]:
    """Build an op stream on a fresh engine at ``verify_level='core'``.

    CORE, not FULL: the shape metric must not consult the verifier fleet. The
    fleet is the thing under test, and a grader that asked it for permission
    would be scoring the model on its ability to please the bug.
    """
    from harnesscad.core.loop import HarnessSession
    from harnesscad.eval.selftest.probe import resolve

    engine, _skip = resolve(backend)
    if engine is None:
        return None
    try:
        session = HarnessSession(engine, verify_level="core")
        session.apply_ops(list(ops))
    except Exception:                                          # noqa: BLE001
        return None
    return engine


def _field(backend: Any):
    fn = getattr(backend, "field", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:                                          # noqa: BLE001
        return None


def _bounds(backend: Any) -> Optional[Tuple[Vec3, Vec3]]:
    mesh = getattr(backend, "mesh", None)
    if not callable(mesh):
        return None
    try:
        verts, faces = mesh()
    except Exception:                                          # noqa: BLE001
        return None
    if not verts or not faces:
        return None
    lo = tuple(min(float(v[i]) for v in verts) for i in range(3))
    hi = tuple(max(float(v[i]) for v in verts) for i in range(3))
    return lo, hi                                              # type: ignore[return-value]


def _points(lo: Vec3, hi: Vec3, n: int, seed: int) -> List[Vec3]:
    rnd = random.Random(seed)
    return [(rnd.uniform(lo[0], hi[0]),
             rnd.uniform(lo[1], hi[1]),
             rnd.uniform(lo[2], hi[2])) for _ in range(n)]


# --------------------------------------------------------------------------- #
# the metric
# --------------------------------------------------------------------------- #
def iou_of_backends(candidate: Any, reference: Any,
                    samples: int = SAMPLES,
                    seed: int = SAMPLE_SEED) -> ShapeScore:
    """Volumetric IoU of two BUILT solids, in world coordinates."""
    fa, fb = _field(candidate), _field(reference)
    if fa is None or fb is None:
        return ShapeScore(reason="an engine exposes no signed-distance field; the "
                                 "shape metric needs one (frep does)")
    ba, bb = _bounds(candidate), _bounds(reference)
    if ba is None:
        return ShapeScore(reason="the candidate produced no solid to measure")
    if bb is None:
        return ShapeScore(reason="the reference produced no solid to measure")

    lo = tuple(min(ba[0][i], bb[0][i]) - PAD for i in range(3))
    hi = tuple(max(ba[1][i], bb[1][i]) + PAD for i in range(3))
    if any(hi[i] <= lo[i] for i in range(3)):
        return ShapeScore(reason="degenerate sampling box")

    inter = union = 0
    for p in _points(lo, hi, samples, seed):                   # type: ignore[arg-type]
        a = fa(p) <= 0.0
        b = fb(p) <= 0.0
        if a and b:
            inter += 1
        if a or b:
            union += 1
    if union == 0:
        return ShapeScore(reason="neither solid occupies the sampled box",
                          samples=samples)
    iou = inter / float(union)
    return ShapeScore(ok=True, iou=iou, matched=iou >= IOU_MATCH, samples=samples,
                      reason="shape IoU %.3f against the reference solid "
                             "(match at %.2f)" % (iou, IOU_MATCH))


def iou_of_ops(candidate: Sequence[Op], reference: Sequence[Op],
               backend: str = "frep",
               samples: int = SAMPLES,
               seed: int = SAMPLE_SEED) -> ShapeScore:
    """Volumetric IoU of two OP STREAMS. The entry point other packages import."""
    ref = build(reference, backend)
    if ref is None:
        return ShapeScore(reason="the reference op stream does not build")
    cand = build(candidate, backend)
    if cand is None:
        return ShapeScore(reason="the candidate op stream does not build")
    return iou_of_backends(cand, ref, samples=samples, seed=seed)


#: Reference solids are expensive (a full grid march) and pure, so they are built
#: once per process. Keyed by the canonical op stream, so two briefs with the same
#: reference share one solid and a changed reference can never hit a stale entry.
_REF_CACHE: Dict[Tuple[str, ...], Any] = {}


def reference_solid(reference: Sequence[Op], backend: str = "frep") -> Optional[Any]:
    """Build (once) and memoise the solid of a reference op stream."""
    from harnesscad.core.cisp.ops import canonical_json

    key = (backend,) + tuple(canonical_json(o) for o in reference)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = build(reference, backend)
    return _REF_CACHE[key]
