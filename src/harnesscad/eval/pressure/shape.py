"""A SHAPE metric, to sit beside the envelope metrics -- v2.

The v1 grader checked bounding box, volume, a handful of SDF probes and some
op-level assertions. Every one of those is an ENVELOPE family: they constrain the
outline and the amount of material, and they are many-to-one by construction. A
hole bored 10 mm from where the brief asked for it changes the bbox by nothing
and the volume by nothing, and it scores perfectly.

Every brief already carries a hand-written ``reference`` op stream that satisfies
it (``briefs.Brief.reference``). v1 used it only to prove the corpus was
solvable. Here it is used as what it actually is: a geometric TARGET.

The metric is volumetric IoU, computed on the F-rep backend's exact signed
distance field::

    IoU = |A and B| / |A or B|

estimated by a deterministic, seeded quasi-uniform sample of the union of the two
bounding boxes. Both solids are evaluated in WORLD coordinates -- no rigid
alignment, no inertia normalisation, no symmetry enumeration. That is deliberate
and it is the opposite of what ``eval/bench/geometry/solid_iou.py`` does: that
module aligns the two solids first, which is right when you are comparing shapes
and wrong when you are asking whether the feature landed where the brief said.
A hole in the wrong place is exactly the failure this metric exists to catch, and
a pose-invariant IoU would forgive it.

WHAT THIS METRIC DOES NOT PROVE
-------------------------------
* It is a SAMPLE, not an integral. At :data:`SAMPLES` points the standard error
  on a ratio near 0.9 is about 0.002. It cannot resolve a defect much smaller
  than the sampling density.
* The reference op stream is ONE correct answer, not THE correct answer. A brief
  that says "a hole, centred" pins the hole; a brief that says "four holes near
  the corners" does not pin them to the millimetre. IoU against the reference
  therefore penalises legitimate variation as well as error, and it is reported
  ALONGSIDE the envelope verdict rather than replacing it. Read the two together.
* It says nothing about manufacturability, and nothing about the op stream.

Deterministic: same ops -> same IoU, bit for bit. No wall clock, no unseeded RNG.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]

#: Sample count for the Monte-Carlo volume integral. 20k puts the standard error
#: on an IoU near 0.9 at ~0.002, which is an order of magnitude below the
#: threshold's distance from 1.0. Fixed before any v2 result was seen.
SAMPLES = 20000

#: The RNG seed. Fixed, so the metric is a function of the geometry alone.
SAMPLE_SEED = 20260713

#: Padding on the sampled box, in mm, so neither solid can touch the boundary.
PAD = 1.0

#: The IoU at or above which the shape is called a match. Chosen a priori from
#: the F-rep grid's own discretisation error (the backend samples its field on a
#: grid, so a *correct* rebuild of the same part lands 1-2% off in volume, and
#: the surface band it disagrees on is a few percent of a small part's volume).
#: 0.90 leaves room for that and still fails a misplaced feature: a 12 mm hole
#: displaced entirely off its seat in a 40x40x10 plate costs ~13 points of IoU.
#: NOT tuned after seeing a result. If it were, this experiment would be worth
#: nothing.
IOU_SOLVED = 0.90


@dataclass
class ShapeScore:
    """The shape verdict on one op stream."""

    ok: bool = False                 # both solids exist and the IoU was computed
    iou: Optional[float] = None
    matched: bool = False            # iou >= IOU_SOLVED
    reason: str = ""
    samples: int = 0

    def to_dict(self) -> dict:
        return {"ok": self.ok, "iou": self.iou, "matched": self.matched,
                "reason": self.reason, "samples": self.samples}


def _bounds(backend: Any) -> Optional[Tuple[Vec3, Vec3]]:
    """(lo, hi) of the backend's current solid, in world coordinates."""
    mesh = getattr(backend, "mesh", None)
    if not callable(mesh):
        return None
    try:
        verts, faces = mesh()
    except Exception:                                     # noqa: BLE001
        return None
    if not verts or not faces:
        return None
    lo = tuple(min(float(v[i]) for v in verts) for i in range(3))
    hi = tuple(max(float(v[i]) for v in verts) for i in range(3))
    return lo, hi


def _field(backend: Any):
    fn = getattr(backend, "field", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:                                     # noqa: BLE001
        return None


def _sample_points(lo: Vec3, hi: Vec3, n: int, seed: int) -> List[Vec3]:
    rnd = random.Random(seed)
    return [(rnd.uniform(lo[0], hi[0]),
             rnd.uniform(lo[1], hi[1]),
             rnd.uniform(lo[2], hi[2])) for _ in range(n)]


def iou_of_backends(candidate: Any, reference: Any,
                    samples: int = SAMPLES,
                    seed: int = SAMPLE_SEED,
                    candidate_bounds: Optional[Tuple[Vec3, Vec3]] = None,
                    reference_bounds: Optional[Tuple[Vec3, Vec3]] = None
                    ) -> ShapeScore:
    """Volumetric IoU of two built F-rep solids, in world coordinates.

    ``*_bounds`` let a caller hand in a bounding box it has ALREADY paid for --
    the grader's output-gate pass meshes the candidate anyway and reports its
    bbox_min/bbox_max, and re-tessellating it here would double the cost of the
    most expensive thing in the experiment for no new information.
    """
    fa, fb = _field(candidate), _field(reference)
    if fa is None or fb is None:
        return ShapeScore(reason="a backend exposes no signed-distance field")

    ba = candidate_bounds or _bounds(candidate)
    bb = reference_bounds or _bounds(reference)
    if ba is None:
        return ShapeScore(reason="the candidate produced no solid to measure")
    if bb is None:
        return ShapeScore(reason="the reference produced no solid to measure")

    lo = tuple(min(ba[0][i], bb[0][i]) - PAD for i in range(3))
    hi = tuple(max(ba[1][i], bb[1][i]) + PAD for i in range(3))
    if any(hi[i] <= lo[i] for i in range(3)):
        return ShapeScore(reason="degenerate sampling box")

    inter = union = 0
    for p in _sample_points(lo, hi, samples, seed):
        a = fa(p) <= 0.0
        b = fb(p) <= 0.0
        if a and b:
            inter += 1
        if a or b:
            union += 1
    if union == 0:
        return ShapeScore(reason="neither solid occupies the sampled box",
                          samples=samples)
    iou = inter / union
    return ShapeScore(ok=True, iou=iou, matched=iou >= IOU_SOLVED,
                      samples=samples,
                      reason=("shape IoU %.3f against the brief's reference "
                              "solution (threshold %.2f)" % (iou, IOU_SOLVED)))


# --------------------------------------------------------------------------- #
# reference solids are rebuilt once per brief and reused
# --------------------------------------------------------------------------- #
#: brief id -> (backend, bounds). The reference solid is built and MEASURED once
#: per process; at resolution 96 a tessellation is the single most expensive
#: operation in the experiment and the reference's never changes.
_REF_CACHE: Dict[str, Any] = {}


def reference_backend(brief) -> Optional[Any]:
    """Build (once) the F-rep solid of a brief's hand-written reference stream."""
    entry = _reference(brief)
    return entry[0] if entry else None


def _reference(brief):
    if brief.id in _REF_CACHE:
        return _REF_CACHE[brief.id]
    from harnesscad.eval.pressure.session import frep_server

    server = frep_server("core")          # PINNED mesher -- see session.py
    try:
        server.applyOps([dict(o) for o in brief.reference])
    except Exception:                                     # noqa: BLE001
        _REF_CACHE[brief.id] = None
        return None
    entry = (server.backend, _bounds(server.backend))
    _REF_CACHE[brief.id] = entry
    return entry


def score(brief, candidate_backend: Any,
          candidate_bounds: Optional[Tuple[Vec3, Vec3]] = None) -> ShapeScore:
    """The shape verdict on a candidate, against the brief's own reference."""
    entry = _reference(brief)
    if entry is None or entry[0] is None:
        return ShapeScore(reason="the brief's reference stream does not build")
    ref, ref_bounds = entry
    return iou_of_backends(candidate_backend, ref,
                           candidate_bounds=candidate_bounds,
                           reference_bounds=ref_bounds)
