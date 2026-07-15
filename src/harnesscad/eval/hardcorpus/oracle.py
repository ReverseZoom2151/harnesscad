"""THE MEASURED ORACLE. It grades WHERE the material is, not how much.

Every published benchmark this package targets grades an ENVELOPE: how much
material, in what outline, of what topology. This oracle grades the one thing they
cannot -- WHERE each feature landed -- by asking the exact B-rep, at points derived
by arithmetic from the brief, whether there is material there or not.

THE FIVE FAMILIES, SCORED AND REPORTED SEPARATELY
-------------------------------------------------
``bbox``    exact extents from OCCT's bounding box against the brief's stated
            envelope. The check the pressure corpus did not have -- its absence is
            why a shell that dilated a 60x40x20 box into 63x43x23 passed.
``volume``  exact ``BRepGProp`` volume against the brief's closed form. Machine
            precision, not a mesh estimate: a 12 mm hole and an 8 mm hole differ by
            1.4% here, which is 10^11 tolerances, not a rounding error.
``genus``   from the exact Euler characteristic of the exported surface,
            ``genus = (2 - chi) / 2``. N through-holes give genus N. A part with the
            right volume, the right box and no holes fails here and only here.
``probes``  THE HEART OF IT. Points in world coordinates, on a hole's axis wall, in
            a cavity, in the body of the part -- classified EXACTLY by
            ``BRepClass3d_SolidClassifier``. This is what an integral over the whole
            part (IoU) and a nearest-surface distance (Chamfer) structurally cannot
            do: assert that a SPECIFIC POINT is or is not material. A hole in the
            wrong place, a counterbore that is a plain hole, a shell open on the
            wrong face -- each of these is a point that classifies wrong and nothing
            else that does.
``closed``  watertight and manifold, off ``io/gate.py``. NOT part of the verdict --
            it is the field's own check, and we measured (over 208 attempts) that
            it is constant across correct and incorrect parts. It is recorded so the
            report can show it saying PASS while the probes say FAIL.

``solved`` is the conjunction of bbox, volume, genus and probes. ``closed`` is
reported beside it, never inside it.

WHY EXACT, AND WHY OCCT
-----------------------
``eval/corpus/grade`` grades on ``frep``, whose sampled field forces a
"unmeasurable" third verdict whenever a wall is thinner than two grid cells. That
caveat is real and honest there, but it also means ``frep`` cannot build a loft or
a sweep at all. This oracle grades on the OCCT B-rep, which is EXACT -- there is no
grid, so there is no third verdict and no feature too fine to resolve -- and which
can build every L3 op the corpus needs. It never consults the verifier fleet
(``verify_level='core'``), for the reason ``grade.py`` sets out at length: the
fleet is a system under test, and a grader that asked it for permission would score
the model on its ability to please a rule that has been wrong before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from harnesscad.core.cisp.ops import Op
from harnesscad.eval.corpus.spec import Brief
from harnesscad.eval.hardcorpus import occt

__all__ = ["OracleScore", "grade", "grade_reference", "PROBE_CLEARANCE"]

#: How far inside/outside a surface a probe must classify to count, in mm. It is a
#: sanity floor on OCCT's own classifier noise, NOT a feature tolerance: every
#: probe this corpus places is at least an order of magnitude clear of any surface,
#: so no verdict is ever decided by this number. A probe that lands within it of a
#: surface is reported as ambiguous rather than silently called either way.
PROBE_CLEARANCE = 0.05


@dataclass
class OracleScore:
    """The measured verdict on one op stream against one brief."""

    brief: str = ""
    built: bool = False
    solved: bool = False              # bbox AND volume AND genus AND probes
    bbox_ok: bool = False
    volume_ok: bool = False
    genus_ok: bool = True             # vacuous when the brief states no genus
    probes_ok: bool = False
    watertight: Optional[bool] = None
    manifold: Optional[bool] = None
    measured: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"brief": self.brief, "built": self.built, "solved": self.solved,
                "bbox_ok": self.bbox_ok, "volume_ok": self.volume_ok,
                "genus_ok": self.genus_ok, "probes_ok": self.probes_ok,
                "watertight": self.watertight, "manifold": self.manifold,
                "measured": self.measured, "reasons": list(self.reasons)}


def _measured_genus(engine: Any) -> Optional[int]:
    """Genus from the exact Euler characteristic of the exported closed surface."""
    from harnesscad.io import gate

    try:
        chi = gate.check(engine, source=engine).measurement.get("euler_characteristic")
    except Exception:                                          # noqa: BLE001
        return None
    if chi is None:
        return None
    return (2 - int(chi)) // 2


def _closed(engine: Any) -> Dict[str, Optional[bool]]:
    from harnesscad.io import gate

    try:
        m = gate.check(engine, source=engine).measurement
    except Exception:                                          # noqa: BLE001
        return {"watertight": None, "manifold": None}
    return {"watertight": m.get("watertight"), "manifold": m.get("manifold")}


#: Tolerances. Exact kernel, so these are numeric guards, not knobs. A B-rep volume
#: is right to machine precision and a bbox to the kernel's fit tolerance; both are
#: set an order of magnitude tighter than the SMALLEST error any near-miss in this
#: corpus introduces, so the corpus cannot be passed by exploiting slack here.
VOLUME_REL = 1e-3
BBOX_ABS = 1e-2


def grade(brief: Brief, ops: Sequence[Op]) -> OracleScore:
    """Rebuild ``ops`` on the exact kernel and measure them against the brief."""
    s = OracleScore(brief=brief.id)
    built = occt.build(ops)
    if not built:
        s.reasons.append(built.reason or "the op stream did not build")
        return s
    s.built = True
    shape = built.shape

    vol = occt.volume_of(shape)
    dx, dy, dz = occt.extents_of(shape)
    s.measured = {"volume": vol, "bbox": [dx, dy, dz]}

    # -- bbox ---------------------------------------------------------------- #
    bad = []
    for axis, want, got in zip("xyz", brief.bbox, (dx, dy, dz)):
        if abs(got - want) > BBOX_ABS:
            bad.append("%s: %.4f mm, brief says %g mm" % (axis, got, want))
    s.bbox_ok = not bad
    for b in bad:
        s.reasons.append("bounding box wrong on " + b)

    # -- volume -------------------------------------------------------------- #
    rel = abs(vol - brief.volume) / max(brief.volume, 1e-9)
    s.volume_ok = rel <= VOLUME_REL
    if not s.volume_ok:
        s.reasons.append("volume %.3f mm3, the closed form says %.3f mm3 "
                         "(off by %+.2f%%)"
                         % (vol, brief.volume, 100.0 * (vol - brief.volume)
                            / brief.volume))

    # -- genus --------------------------------------------------------------- #
    if brief.genus is not None:
        g = _measured_genus(built.engine)
        s.measured["genus"] = g
        if g is not None:
            s.genus_ok = g == brief.genus
            if not s.genus_ok:
                s.reasons.append("genus %d, the brief's topology says %d (the right "
                                 "size, the wrong holes)" % (g, brief.genus))

    # -- probes: the only family that pins a LOCATION ------------------------ #
    s.probes_ok = _probe(brief, shape, s)

    closed = _closed(built.engine)
    s.watertight = closed["watertight"]
    s.manifold = closed["manifold"]

    s.solved = bool(s.built and s.bbox_ok and s.volume_ok and s.genus_ok
                    and s.probes_ok)
    return s


def _probe(brief: Brief, shape: Any, s: OracleScore) -> bool:
    """Classify every probe point EXACTLY. Inside must be material; outside void."""
    if not brief.inside and not brief.outside:
        return True
    ok = True
    for p in brief.inside:
        st = occt.classify(shape, p)
        if st != "in":
            ok = False
            s.reasons.append(
                "point %s must be SOLID MATERIAL (it is the body of the part, or "
                "the middle of a wall) but the kernel classifies it %r -- the "
                "feature is not where the brief put it" % (list(p), st))
    for p in brief.outside:
        st = occt.classify(shape, p)
        if st != "out":
            ok = False
            s.reasons.append(
                "point %s must be EMPTY (it is on a hole's axis, or inside a "
                "cavity) but the kernel classifies it %r -- there is material "
                "where the brief demanded a void" % (list(p), st))
    return ok


def grade_reference(brief: Brief) -> OracleScore:
    """Grade a brief against ITS OWN reference stream.

    A brief whose own hand-written answer does not pass its own oracle is broken,
    and every score taken against it is meaningless. The pressure corpus failed
    exactly this on two shell briefs and shipped anyway because nobody ran it; the
    hardcorpus test suite runs THIS on every brief.
    """
    return grade(brief, list(brief.reference))
