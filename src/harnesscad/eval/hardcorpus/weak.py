"""THE FIELD'S OWN ORACLE, implemented faithfully and in its STRONGEST form.

This module is not a straw man. Every metric here is the metric a 2026 paper would
actually run, and where a choice existed we took the version that FAVOURS the field:

* **IoU** is the exact OCCT boolean ``vol(A and B) / vol(A or B)``, not a voxel
  count and not a Monte-Carlo estimate. Text2CAD-Bench voxelises. We do not, so
  that "your IoU was noisy" is never an available answer to the table in
  :mod:`~harnesscad.eval.hardcorpus.discriminative`.
* **Chamfer Distance** is the symmetric mean-nearest-neighbour distance over
  area-weighted surface samples -- Text2CAD-Bench's headline number, the one on
  which they report GPT-5.2 at 93.46.
* **The geometric check** is MUSE's: is there a solid, is it watertight, is it
  manifold, does it self-intersect. It is read straight off ``io/gate.py``, which
  already computes exactly those from the exported mesh. We do not reimplement it,
  and we do not weaken it.
* **The invalidity rate** is Text2CAD-Bench's: the fraction of attempts that never
  produced a valid solid at all. It is the number on which they report GPT-5.2 at
  68% and Claude-4.5-Sonnet at a 70% L3 failure rate.

WHY THIS FILE EXISTS
--------------------
So that every brief in this package is graded TWICE and reported side by side.
A benchmark that published only its own oracle's verdict would be exactly as
unfalsifiable as one that published only IoU. The claim this package makes is not
"our metric is better"; it is "**here is the set of wrong parts that their metric
scores as correct, and here is what ours says about the same parts**". You cannot
make that claim without running their metric properly.

THE PRE-REGISTERED THRESHOLDS
-----------------------------
A continuous metric needs a threshold before it can say PASS. Both of ours were
fixed BEFORE any near-miss was scored, and neither has been moved since:

``IOU_MATCH = 0.90``     taken, unchanged, from ``eval/corpus/shape.IOU_MATCH``,
                        where it was already pre-registered and where the module
                        docstring already records that an 8 mm hole displaced by
                        20 mm scores 0.957 and PASSES -- and that the honest
                        response was to write that down rather than raise the bar
                        and start failing correct parts.
``CHAMFER_MATCH``       0.01 -- one percent of the part's own bounding-box
                        diagonal. Scale-free, because a benchmark whose threshold
                        is an absolute millimetre count silently gets easier as
                        parts get bigger.

Both are generous to the field ON PURPOSE. If a near-miss passes a threshold this
tight, no threshold anybody would dare to ship can catch it -- and the discriminative
table reports, for each near-miss, exactly how far the bar would have to move and
what that would cost in false failures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from harnesscad.core.cisp.ops import Op
from harnesscad.eval.corpus.shape import IOU_MATCH
from harnesscad.eval.hardcorpus import occt

__all__ = ["IOU_MATCH", "CHAMFER_MATCH", "CHAMFER_SAMPLES", "WeakScore",
           "score_weak", "invalidity_rate"]

#: Surface samples per solid for Chamfer. 4096 puts the estimator's own noise two
#: orders of magnitude below every near-miss distance in the table.
CHAMFER_SAMPLES = 4096

#: PASS if the symmetric Chamfer Distance is under this fraction of the reference
#: part's bounding-box diagonal. Pre-registered; scale-free; never moved.
CHAMFER_MATCH = 0.01


@dataclass
class WeakScore:
    """Everything the published benchmarks would say about one answer.

    ``valid``     MUSE's geometric check: a solid exists, watertight, manifold, no
                  self-intersection. Read off ``io/gate.py``.
    ``iou``       exact boolean IoU against the reference solid.
    ``chamfer``   symmetric Chamfer Distance, mm.
    ``chamfer_rel``  the same, as a fraction of the reference's bbox diagonal.
    ``passes``    what a Text2CAD-Bench-style grader would conclude: valid AND
                  IoU >= IOU_MATCH AND chamfer_rel <= CHAMFER_MATCH.
    """

    built: bool = False
    valid: bool = False
    watertight: Optional[bool] = None
    manifold: Optional[bool] = None
    self_intersects: Optional[bool] = None
    gate_ok: bool = False
    gate_codes: List[str] = field(default_factory=list)
    iou: Optional[float] = None
    iou_ok: bool = False
    chamfer: Optional[float] = None
    chamfer_rel: Optional[float] = None
    chamfer_ok: bool = False
    passes: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {"built": self.built, "valid": self.valid,
                "watertight": self.watertight, "manifold": self.manifold,
                "self_intersects": self.self_intersects,
                "gate_ok": self.gate_ok, "gate_codes": list(self.gate_codes),
                "iou": self.iou, "iou_ok": self.iou_ok,
                "chamfer": self.chamfer, "chamfer_rel": self.chamfer_rel,
                "chamfer_ok": self.chamfer_ok, "passes": self.passes,
                "reason": self.reason}


def _gate_view(engine: Any) -> Dict[str, Any]:
    """MUSE's geometric stage, straight off ``io/gate.py``. Not reimplemented."""
    from harnesscad.io import gate

    try:
        report = gate.check(engine, source=engine)
    except Exception as exc:                                    # noqa: BLE001
        return {"gate_ok": False, "codes": ["gate-raised: %s" % exc]}
    m = report.measurement or {}
    return {"gate_ok": bool(report.ok),
            "codes": [f.check for f in report.failures],
            "watertight": m.get("watertight"),
            "manifold": m.get("manifold"),
            "self_intersects": m.get("self_intersects",
                                     m.get("self_intersecting"))}


def score_weak(candidate: Sequence[Op], reference: Sequence[Op]) -> WeakScore:
    """Grade ``candidate`` against ``reference`` exactly as the field would."""
    s = WeakScore()
    cand = occt.build(candidate)
    if not cand:
        s.reason = cand.reason or "the candidate did not build"
        return s
    s.built = True
    ref = occt.build(reference)
    if not ref:
        s.reason = ("the REFERENCE did not build: %s -- this brief is broken, not "
                    "the answer" % ref.reason)
        return s

    g = _gate_view(cand.engine)
    s.gate_ok = bool(g["gate_ok"])
    s.gate_codes = list(g.get("codes") or ())
    s.watertight = g.get("watertight")
    s.manifold = g.get("manifold")
    s.self_intersects = g.get("self_intersects")
    # MUSE's geometric check, exactly: a solid, watertight, manifold, no self-
    # intersection. Nothing about WHERE anything is -- which is the whole finding.
    s.valid = bool(s.gate_ok and s.watertight is not False
                   and s.manifold is not False
                   and s.self_intersects is not True)

    s.iou = occt.boolean_iou(cand.shape, ref.shape)
    s.iou_ok = bool(s.iou is not None and s.iou >= IOU_MATCH)

    mc = occt.mesh_of(cand.engine)
    mr = occt.mesh_of(ref.engine)
    if mc and mr and mc[0] and mr[0]:
        pa = occt.sample_surface(mc[0], mc[1], CHAMFER_SAMPLES)
        pb = occt.sample_surface(mr[0], mr[1], CHAMFER_SAMPLES)
        s.chamfer = occt.chamfer_distance(pa, pb)
        dx, dy, dz = occt.extents_of(ref.shape)
        diag = math.sqrt(dx * dx + dy * dy + dz * dz)
        if s.chamfer is not None and diag > 0.0:
            s.chamfer_rel = s.chamfer / diag
            s.chamfer_ok = s.chamfer_rel <= CHAMFER_MATCH

    s.passes = bool(s.valid and s.iou_ok and s.chamfer_ok)
    s.reason = ("valid=%s iou=%s chamfer_rel=%s -> a Text2CAD-Bench-style grader "
                "says %s"
                % (s.valid,
                   "%.4f" % s.iou if s.iou is not None else "n/a",
                   "%.5f" % s.chamfer_rel if s.chamfer_rel is not None else "n/a",
                   "PASS" if s.passes else "FAIL"))
    return s


def invalidity_rate(scores: Sequence[WeakScore]) -> Optional[float]:
    """Text2CAD-Bench's headline: the fraction of attempts with no valid solid.

    The number on which they report GPT-5.2 at 68% invalid at L3. It says nothing
    whatever about whether a part is the RIGHT part, and
    ``eval/bench/harness/pressure_correlation.py`` measured that directly: across
    208 graded attempts, ``is_valid``, ``watertight``, ``manifold``,
    ``solid_present``, ``built`` and ``parse_ok`` were all LITERALLY CONSTANT.
    Solved and unsolved parts were indistinguishable by every one of them.
    """
    if not scores:
        return None
    return sum(0 if s.valid else 1 for s in scores) / float(len(scores))
