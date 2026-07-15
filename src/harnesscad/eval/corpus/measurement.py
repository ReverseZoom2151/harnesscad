"""THE FULL MEASUREMENT VECTOR. Closing the many-to-one hole where it can be closed.

THE HOLE
--------
``assets/pressure/report.md`` (s2.4, gap #5) names the defect precisely: the grader
"checks bbox + volume + a handful of SDF probes + op-count assertions. Four
families, all ENVELOPE families." Volume, bounding box and genus are many-to-one by
construction -- a hole bored 10 mm from where the brief asked changes none of them
and scores perfectly. A grade built only on ``volume + bbox + genus`` cannot tell
the requested part from a large class of wrong ones.

Two metrics collapse that class, and this module composes them into ONE vector:

* the SHAPE metric (``eval/corpus/shape.py``): a volumetric IoU that INTEGRATES over
  the whole part in world coordinates, catching gross error an envelope shares --
  wrong scale, wrong boolean operand, a feature omitted, a shell that dilated.
* the PROBE points (``eval/corpus/grade.py``): point assertions on the exact SDF
  that pin a feature to a LOCATION, catching the small misplaced feature IoU is
  arithmetically blind to (an 8 mm hole moved 20 mm scores IoU 0.957 -- see
  ``shape.py`` -- but fails a probe on its axis).

Neither is sufficient; ``shape.py`` says so in its own docstring. Together they are
the pair the report asked for. This module does not invent a new oracle -- it calls
``grade.grade`` (which already runs every family, the probes AND the shape IoU) and
assembles the result into a single, numeric measurement vector plus a CONJUNCTIVE
verdict, ``matched_full``, that is true only when the envelope, the shape and the
probes all agree. ``grade.Score.solved`` is the envelope verdict and stays; this is
the stricter reading, reported alongside, never instead of it.

WHERE THE HOLE STAYS OPEN, STATED UP FRONT
------------------------------------------
IoU is measured against ONE reference solution, not the equivalence class of correct
answers, so it penalises legitimate variation as well as error; a probe is a point,
and a dozen points still admit a (smaller) space of wrong parts. "Closing the hole
WHERE POSSIBLE" is the honest claim: the vector is strictly more discriminating than
the envelope alone, and it is not a proof of identity. The numbers are all reported;
collapsing them into one score would be the single-number opacity this package
exists to replace.

REUSABLE
--------
:func:`from_score` builds the vector from an already-computed ``grade.Score`` with no
rebuild -- this is what the corpus grader calls. :func:`measure` is the convenience
that grades then wraps. :func:`compare_ops` is brief-free: it takes two op streams
and reports the shape half of the vector directly, for callers (e.g. the pressure
grader) that have a reference stream but no ``Brief``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Op
from harnesscad.eval.corpus import shape as shape_mod
from harnesscad.eval.corpus.grade import Score, grade
from harnesscad.eval.corpus.spec import Brief

__all__ = ["MeasurementVector", "from_score", "measure", "compare_ops"]


@dataclass
class MeasurementVector:
    """The full comparison of one candidate against one brief's ground truth.

    Every field is a residual or a boolean, so the whole thing is a numeric vector
    a correlation study (``eval/bench/harness/metric_correlation.py``) can consume
    directly, and ``matched_full`` is the conjunction a grader can gate on.
    """

    brief: str = ""
    backend: str = "frep"
    built: bool = False

    # -- envelope residuals (signed where meaningful) ---------------------- #
    bbox_residual: Tuple[Optional[float], Optional[float], Optional[float]] = \
        (None, None, None)          # measured - brief, per axis (mm)
    volume_rel_error: Optional[float] = None    # (measured - brief) / brief
    genus_delta: Optional[int] = None           # measured - brief

    # -- the two hole-closing families ------------------------------------- #
    iou: Optional[float] = None                 # shape.py, world-coordinate IoU
    iou_matched: bool = False                   # iou >= shape.IOU_MATCH
    probes_ok: bool = False                     # every probe assertion held

    # -- verdicts ---------------------------------------------------------- #
    envelope_ok: bool = False                   # == grade.Score.solved
    matched_full: bool = False                  # envelope AND iou AND probes

    reasons: List[str] = field(default_factory=list)
    measured: Dict[str, Any] = field(default_factory=dict)

    def as_vector(self) -> List[float]:
        """The residuals as a flat numeric vector; missing entries become NaN."""
        nan = float("nan")
        bx, by, bz = self.bbox_residual
        return [
            bx if bx is not None else nan,
            by if by is not None else nan,
            bz if bz is not None else nan,
            self.volume_rel_error if self.volume_rel_error is not None else nan,
            float(self.genus_delta) if self.genus_delta is not None else nan,
            self.iou if self.iou is not None else nan,
            1.0 if self.probes_ok else 0.0,
        ]

    def to_dict(self) -> dict:
        return {"brief": self.brief, "backend": self.backend, "built": self.built,
                "bbox_residual": list(self.bbox_residual),
                "volume_rel_error": self.volume_rel_error,
                "genus_delta": self.genus_delta,
                "iou": self.iou, "iou_matched": self.iou_matched,
                "probes_ok": self.probes_ok, "envelope_ok": self.envelope_ok,
                "matched_full": self.matched_full,
                "vector": self.as_vector(), "reasons": self.reasons,
                "measured": self.measured}


def from_score(brief: Brief, score: Score) -> MeasurementVector:
    """Assemble the vector from a ``grade.Score`` WITHOUT rebuilding anything.

    ``grade.grade`` already ran the envelope families, the probes and the shape IoU;
    this only reads its outputs and computes the residuals against the brief.
    """
    mv = MeasurementVector(brief=brief.id, backend=score.backend,
                           built=score.built, reasons=list(score.reasons),
                           measured=dict(score.measured))

    measured_bbox = score.measured.get("bbox") or ()
    residual: List[Optional[float]] = [None, None, None]
    for i in range(3):
        if i < len(measured_bbox) and i < len(brief.bbox):
            residual[i] = float(measured_bbox[i]) - float(brief.bbox[i])
    mv.bbox_residual = (residual[0], residual[1], residual[2])

    measured_vol = score.measured.get("volume")
    if measured_vol is not None and brief.volume:
        mv.volume_rel_error = (float(measured_vol) - float(brief.volume)) \
            / float(brief.volume)

    measured_genus = score.measured.get("genus")
    if measured_genus is not None and brief.genus is not None:
        mv.genus_delta = int(measured_genus) - int(brief.genus)

    mv.iou = score.iou
    mv.iou_matched = bool(score.shape.get("matched"))
    mv.probes_ok = bool(score.probes_ok)

    mv.envelope_ok = bool(score.solved)
    mv.matched_full = bool(score.solved and mv.iou_matched and score.probes_ok)
    return mv


def measure(brief: Brief, ops: Sequence[Op], backend: str = "frep") -> MeasurementVector:
    """Grade ``ops`` against ``brief`` and return the full measurement vector."""
    score = grade(brief, list(ops), backend=backend, with_shape=True)
    return from_score(brief, score)


def compare_ops(candidate: Sequence[Op], reference: Sequence[Op],
                backend: str = "frep") -> shape_mod.ShapeScore:
    """Brief-free: the shape half of the vector between two op streams.

    A thin, honest pass-through to ``shape.iou_of_ops`` for callers that hold a
    reference stream but no ``Brief`` (the pressure grader). It exists so those
    callers reuse the one IoU implementation instead of growing a second copy --
    the probe half needs a brief's declared points and lives in :func:`measure`.
    """
    return shape_mod.iou_of_ops(list(candidate), list(reference), backend=backend)
