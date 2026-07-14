"""THE ORACLE CAPABILITY LEDGER -- what the reward can and cannot see.

Read this before you read anything else in this package, and before you generate
a single training pair.

RLVR's *only* named failure mode, in the book's own words (H2 sec. 13.6.5):

    "The only failure mode is if the model finds solutions that pass
    verification but are not genuinely correct (e.g., exploiting test case
    weaknesses)."

We have already shipped that failure once. The F-rep ``shell`` dilated the part
outward by ``t/2``; a 60x40x20 box shelled at t=3 came out 63x43x23; **no
verifier fired**; and the ``shell_box_3mm`` brief carried ``bbox=None``, so the
corpus was blind to it too. Fleet and corpus shared the blind spot. Every
``shell_box_3mm`` "solve" in the pressure run is a part with wrong outside
dimensions.

A reward is a specification, and a model optimised against it will occupy every
cell of the specification's null space. So the null space gets written down
first, in code, and a test asserts it stays written down.

The four instruments, and the exact boundary of each
====================================================

**1. The differential oracle** (``eval/selftest/differential.py``)
    Runs the same op stream on six independently-implemented engines and reports
    where they disagree.

    CAN SEE: an op stream that is *not well-defined* -- one that means different
    things to different kernels, or crashes one of them. This is a ground-truth
    signal with no human, no reference and no model, and it is the repository's
    main asset.
    CANNOT SEE: whether the well-defined thing is the thing the brief asked for.
    Six engines agreeing on a wrong part agree perfectly. It has never read the
    brief. And a bug shared by all six kernels is invisible to it by
    construction (this is not hypothetical: five of the six route through the
    same op semantics).

**2. The output gate** (``io/gate.py``)
    MEASURED: closed, 2-manifold, consistently wound, positive volume, finite
    bbox, no self-intersection. DECLARED: the op log is replayed and the geometry
    is measured either side of every intent-bearing op -- ``shell`` must not grow
    the bbox, ``cut`` must not add volume, the first ``extrude`` must produce the
    declared height.

    CAN SEE: an artifact that is malformed, or that betrays the intent *its own
    op stream declared*. It is the component that would have refused the dilated
    shell.
    CANNOT SEE: an op stream that declares the wrong intent perfectly. A model
    that asks for a 50 mm plate when the brief said 60 mm passes the gate with
    full marks. The gate audits the harness, not the model.

**3. The envelope grader** (``eval/pressure/metrics.py`` -- ``Expect``)
    bbox, volume band, SDF probe points that must be material / must be air, and
    op-level assertions on the plan.

    CAN SEE: gross geometry error, and -- via the ``outside`` probes -- a hole
    that was never cut, or cut in a *probed* place.
    CANNOT SEE, and this is the load-bearing sentence of this file: **it is
    MANY-TO-ONE.** bbox + volume + genus do not pin down a part. A hole displaced
    to an unprobed location changes the bbox by nothing and the volume by
    nothing and scores perfectly. Probes are a finite set of points; the parts
    that pass through the gaps between them are a continuum. And it requires a
    hand-written ``Expect``, so it exists only for the 28 corpus briefs and for
    nothing a user will ever type.

**4. The shape metric** (``eval/pressure/shape.py`` -- volumetric IoU vs. the
   brief's hand-written ``reference`` stream)
    The answer to (3)'s many-to-one-ness, and it is a partial answer.

    CAN SEE: a feature in the wrong *place*. World-coordinate IoU, deliberately
    not pose-invariant.
    CANNOT SEE: a small feature at the right place with the wrong *size*. This is
    measured, not asserted: the 14b's ``trap_hole_oversize`` regression -- an
    8 mm hole where the brief demanded 12 mm, in a 40x40x10 plate -- scores
    **IoU = 0.963**, comfortably above the 0.90 match threshold. The defect is
    628 mm^3 of a ~15,000 mm^3 part. IoU is a *volume ratio* and it is blind in
    proportion to how small the defect is. It also penalises legitimate variation
    (the reference is ONE correct answer, not THE correct answer) -- so it is a
    filter that is simultaneously too lax and too strict, and it is used here
    only in conjunction, never alone.

What NOTHING in the fleet can see
=================================
* **The brief's semantics.** No instrument reads English. "Four holes near the
  corners" is pinned by the ``Expect`` probes and by nothing else.
* **A bug shared by every engine.** A six-way differential is wrong only if all
  six are wrong the same way -- which is rare, and is not never.
* **Manufacturability, cost, material, tolerance stack-up.** None of it.
* **Whether the op stream is a *good* plan.** Two ops or twenty, the geometry is
  graded, the reasoning is not.

The consequence for training, stated before any data is emitted
===============================================================
An RFT set accepted on ENVELOPE alone teaches "wrong-but-envelope-correct is
fine". So :func:`certify` requires the **conjunction** of all four instruments,
and it records, per accepted candidate, which instruments were *silent* -- the
``blind_spots`` field. A candidate accepted only because an instrument could not
see is an accepted candidate we do not trust, and it says so on its face.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Instrument",
    "INSTRUMENTS",
    "Certificate",
    "certify",
    "blind_spots_of_brief",
]


@dataclass(frozen=True)
class Instrument:
    """One reward channel, and the exact boundary of what it perceives."""

    name: str
    module: str
    can_see: Tuple[str, ...]
    cannot_see: Tuple[str, ...]
    reference_free: bool     # True = needs no answer key; works on any user brief

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "module": self.module,
            "can_see": list(self.can_see),
            "cannot_see": list(self.cannot_see),
            "reference_free": self.reference_free,
        }


#: The ledger. If an instrument changes, this changes, and its test fails until
#: it does. This tuple IS the specification of the reward.
INSTRUMENTS: Tuple[Instrument, ...] = (
    Instrument(
        name="differential",
        module="harnesscad.eval.selftest.differential",
        can_see=(
            "an op stream that is not well-defined across six independent kernels",
            "an op stream that crashes a kernel",
            "a volume/bbox disagreement between engines",
        ),
        cannot_see=(
            "whether the well-defined part is the part the brief asked for",
            "a bug shared by all six engines",
        ),
        reference_free=True,
    ),
    Instrument(
        name="gate",
        module="harnesscad.io.gate",
        can_see=(
            "a non-watertight, non-manifold, inverted or self-intersecting solid",
            "a shell that grew the bounding box (the bug that shipped)",
            "a cut that added volume",
            "a first extrude whose height is not the declared distance",
        ),
        cannot_see=(
            "an op stream that declares the WRONG intent perfectly",
            "anything about the brief; it audits the harness, not the model",
        ),
        reference_free=True,
    ),
    Instrument(
        name="envelope",
        module="harnesscad.eval.pressure.metrics",
        can_see=(
            "bounding box outside the brief's tolerance",
            "volume outside the brief's band",
            "a probe point that should be material and is air, or vice versa",
            "a plan missing a required op, or with an op parameter out of range",
        ),
        cannot_see=(
            "MANY-TO-ONE: a feature displaced to an unprobed location",
            "anything at all for a brief with no hand-written Expect "
            "(i.e. everything a user will ever type)",
        ),
        reference_free=False,
    ),
    Instrument(
        name="shape",
        module="harnesscad.eval.pressure.shape",
        can_see=(
            "a feature in the wrong place (world-coordinate volumetric IoU)",
        ),
        cannot_see=(
            "a small feature of the wrong SIZE: the 14b's 8mm-for-12mm hole "
            "scores IoU 0.963, above the 0.90 threshold",
            "the difference between an error and legitimate variation from the "
            "reference, which is ONE correct answer and not THE correct answer",
        ),
        reference_free=False,
    ),
)


#: Instruments that work with no answer key. Only these can ever grade a brief a
#: user typed. Everything the corpus-only instruments contribute is a luxury the
#: production system does not have, and the training set must not depend on it
#: silently.
REFERENCE_FREE = tuple(i.name for i in INSTRUMENTS if i.reference_free)
REFERENCE_BOUND = tuple(i.name for i in INSTRUMENTS if not i.reference_free)


def blind_spots_of_brief(brief: Any) -> List[str]:
    """Which envelope channels this brief has switched OFF.

    The ``shell_box_3mm`` disaster is exactly this: ``expect.bbox is None``, so
    the one check that would have caught a dilated shell was not merely wrong,
    it was absent. A brief that declines to constrain a dimension cannot punish
    a model for getting it wrong, and any candidate it accepts is accepted with
    that hole in the reasoning.
    """
    holes: List[str] = []
    expect = getattr(brief, "expect", None)
    if expect is None:
        return ["brief carries no Expect at all"]
    if getattr(expect, "bbox", None) is None:
        holes.append("expect.bbox is None: outside dimensions are UNCHECKED")
    if getattr(expect, "volume", None) is None:
        holes.append("expect.volume is None: the amount of material is UNCHECKED")
    if not getattr(expect, "outside", ()):
        holes.append("no 'outside' probes: a hole that was never cut is UNCHECKED")
    if not getattr(expect, "inside", ()):
        holes.append("no 'inside' probes: material presence is UNCHECKED")
    if not getattr(expect, "ops", ()):
        holes.append("no op-level assertions: op parameters are UNCHECKED")
    return holes


@dataclass
class Certificate:
    """The oracle's verdict on one candidate, with its own blindness attached.

    ``accepted`` is the CONJUNCTION. A candidate is certified only when every
    instrument that could speak, spoke, and said yes. ``blind_spots`` lists the
    instruments and brief-level checks that were SILENT -- not passing, silent --
    so that a training record can never launder "nothing objected" into
    "verified correct".
    """

    accepted: bool = False
    apply_ok: bool = False
    gate_ok: bool = False
    envelope_ok: bool = False
    shape_ok: bool = False
    shape_iou: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    measurements: Dict[str, Any] = field(default_factory=dict)
    blind_spots: List[str] = field(default_factory=list)
    instruments_consulted: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "apply_ok": self.apply_ok,
            "gate_ok": self.gate_ok,
            "envelope_ok": self.envelope_ok,
            "shape_ok": self.shape_ok,
            "shape_iou": self.shape_iou,
            "reasons": list(self.reasons),
            "measurements": dict(self.measurements),
            "blind_spots": list(self.blind_spots),
            "instruments_consulted": list(self.instruments_consulted),
        }


def certify(brief: Any, ops: Sequence[dict], *,
            differential: bool = False) -> Certificate:
    """Adjudicate one candidate op stream with EVERY instrument, conjunctively.

    The labeller for every dataset in this package. It is deliberately NOT the
    verifier fleet: the fleet's measured false-positive rate is what cost the
    harness 8 briefs in the pressure run, and DPO memorises individual pairs, so
    a fleet-labelled pair set would teach the model to reject washers. The fleet's
    diagnostics are RECORDED (they are the loop's feedback channel and we want to
    study them) and they are never consulted for the label.
    """
    from harnesscad.eval.pressure import metrics as metrics_mod

    grade = metrics_mod.grade(brief, list(ops), shape=True)
    shape = dict(grade.shape or {})

    cert = Certificate()
    cert.apply_ok = bool(grade.apply_ok)
    cert.gate_ok = bool(grade.gate_ok)
    cert.envelope_ok = bool(grade.solved)
    cert.shape_iou = shape.get("iou")
    cert.shape_ok = bool(shape.get("matched"))
    cert.reasons = list(grade.reasons)
    cert.measurements = {
        "bbox": (grade.measure or {}).get("bbox"),
        "volume": (grade.measure or {}).get("volume"),
        "validity": (grade.measure or {}).get("validity"),
        "shape_iou": cert.shape_iou,
        "gate_failures": list(grade.gate_failures or []),
        "applied": grade.applied,
    }
    cert.instruments_consulted = ["gate", "envelope", "shape"]

    # The conjunction. Envelope alone is many-to-one; shape alone is size-blind;
    # the gate alone never read the brief. Only all three together.
    cert.accepted = bool(cert.apply_ok and cert.gate_ok
                         and cert.envelope_ok and cert.shape_ok)

    # The sixth-engine cross-check. OFF by default because it re-executes the
    # stream on every installed kernel and most of them are not installed on this
    # machine (an absent kernel REFUSES, which is a capability gap and is not held
    # against the candidate) -- so at corpus scale it costs a great deal and, here,
    # decides nothing. When it IS on, a disagreement or a crash REVOKES acceptance.
    if differential:
        from harnesscad.eval.pressure import oracle as oracle_mod

        score = oracle_mod.score_ops(list(ops), name=str(getattr(brief, "id", "?")))
        cert.instruments_consulted.append("differential")
        cert.measurements["differential"] = score.to_dict()
        if score.engines_disagreeing or score.engines_crashed:
            cert.accepted = False
            cert.reasons.append(
                "the differential oracle found %d disagreeing and %d crashed "
                "engine(s)" % (score.engines_disagreeing, score.engines_crashed))

    cert.blind_spots = blind_spots_of_brief(brief)
    if shape.get("ok") is not True:
        cert.blind_spots.append(
            "the shape metric did not run (%s): the SHAPE channel is SILENT"
            % (shape.get("reason") or "no reason given"))
    if not differential:
        cert.blind_spots.append(
            "the differential oracle was NOT consulted: acceptance rests on "
            "gate+envelope+shape only")
    if cert.accepted:
        cert.blind_spots.append(
            "IoU cannot resolve a defect below ~2% of part volume "
            "(measured: an 8mm-for-12mm hole scores 0.963)")
    return cert
