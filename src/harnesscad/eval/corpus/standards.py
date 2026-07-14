"""Brief factories whose dimensions come from a PUBLISHED STANDARD.

An analytic brief is uncontaminated because the truth is arithmetic. These are
uncontaminated for a second, independent reason: the DIMENSIONS THEMSELVES come
from a document written by a standards body, and nobody working on this
repository can edit it. If the harness's fillet rule is wrong, a plate whose
dimensions we chose is a plate we might have chosen to suit the rule. An ISO 7089
washer for an M8 bolt is 8.4 mm inside, 16 mm outside and 1.6 mm thick whatever
the harness thinks, and it was those dimensions before this repository existed.

That matters here more than it looks. The single rule that cost the pressure
experiment 8 briefs -- a hole's diameter compared against the plate's THICKNESS --
rejected a WASHER, and a washer is the most standardised part in mechanical
engineering. A corpus containing one would have caught the rule the day it was
written. The pressure corpus contained no standard parts at all.

The nominal thread dimensions are read from
:mod:`harnesscad.domain.standards.thread_database` (an ISO 261 / ISO 4032 lookup
that already existed here), so the bolt sizes are not retyped either.

SOURCES, cited per brief
------------------------
ISO 7089 / DIN 125-1 form A   plain washers, normal series, 200 HV.
                              M6:  d1 6.4,  d2 12, h 1.6
                              M8:  d1 8.4,  d2 16, h 1.6
                              M10: d1 10.5, d2 20, h 2.0
                              M12: d1 13.0, d2 24, h 2.5
                              M16: d1 17.0, d2 30, h 3.0
ISO 273                       clearance holes for bolts, MEDIUM series.
                              M5 5.5, M6 6.6, M8 9.0, M10 11.0, M12 13.5,
                              M16 17.5
EN 1092-1 / DIN 2576          DN50 PN16 flat-face flange: OD 165 mm, bolt circle
                              125 mm, 4 bolt holes of 18 mm, 18 mm thick.
ISO 261 / ISO 4032            nominal metric thread diameters and hex widths --
                              read from ``domain.standards.thread_database``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from harnesscad.core.cisp.ops import (AddCircle, AddRectangle, Extrude, Hole,
                                      NewSketch, Op)
from harnesscad.domain.standards.thread_database import thread_lookup
from harnesscad.eval.corpus.spec import Brief, Source
from harnesscad.eval.selftest.golden import annulus_volume

__all__ = ["WASHER_ISO7089", "CLEARANCE_ISO273", "FLANGE_DN50_PN16",
           "washer", "clearance_plate", "flange_dn50_pn16", "nominal_diameter"]

PI = math.pi

#: ISO 7089 / DIN 125-1 form A plain washer: bolt -> (inner d1, outer d2, thickness h)
WASHER_ISO7089: Dict[str, Tuple[float, float, float]] = {
    "M6": (6.4, 12.0, 1.6),
    "M8": (8.4, 16.0, 1.6),
    "M10": (10.5, 20.0, 2.0),
    "M12": (13.0, 24.0, 2.5),
    "M16": (17.0, 30.0, 3.0),
}

#: ISO 273 clearance holes, MEDIUM series: bolt -> hole diameter.
CLEARANCE_ISO273: Dict[str, float] = {
    "M5": 5.5, "M6": 6.6, "M8": 9.0, "M10": 11.0, "M12": 13.5, "M16": 17.5,
}

#: EN 1092-1 / DIN 2576, DN50 PN16 flat-face flange.
FLANGE_DN50_PN16 = {
    "outside_diameter": 165.0,
    "bolt_circle": 125.0,
    "bolt_holes": 4,
    "bolt_hole_diameter": 18.0,
    "thickness": 18.0,
    "nominal_bore": 50.0,          # DN50
}

#: The ISO metric coarse thread each washer/clearance size names, so the nominal
#: diameter is READ from the thread database rather than retyped here.
_COARSE: Dict[str, str] = {
    "M5": "M5x0.8", "M6": "M6x1", "M8": "M8x1.25", "M10": "M10x1.5",
    "M12": "M12x1.75", "M16": "M16x2",
}


def nominal_diameter(bolt: str) -> float:
    """The bolt's nominal major diameter, from the ISO 261 lookup already in-repo."""
    return 2.0 * thread_lookup(_COARSE[bolt]).radius


def _sk(plane: str = "XY") -> Op:
    return NewSketch(plane)


def washer(bid: str, split: str, bolt: str) -> Brief:
    """An ISO 7089 plain washer.

    THE PART THE FLEET REJECTED. A washer's bore is wider than the washer is
    thick -- for an M8 washer, 8.4 mm of bore in 1.6 mm of stock, a ratio of more
    than five to one -- and that is not a defect, it is what a washer IS. The rule
    that compared a hole's diameter against the plate's thickness called this
    infeasible, fired 40 times in the pressure run and caused every regression it
    had. The dimensions below are ISO's, so the brief cannot be quietly adjusted
    to suit whatever the rule happens to say next.
    """
    d1, d2, h = WASHER_ISO7089[bolt]
    ro, ri = d2 / 2.0, d1 / 2.0
    mid = (ro + ri) / 2.0
    return Brief(
        id=bid, split=split, source=Source.STANDARD,
        citation=("ISO 7089 / DIN 125-1 form A plain washer for %s: inside "
                  "diameter %g mm, outside diameter %g mm, thickness %g mm "
                  "(nominal bolt diameter %g mm, ISO 261)"
                  % (bolt, d1, d2, h, nominal_diameter(bolt))),
        text=("A plain washer for an %s bolt: a disc %g mm in outside diameter "
              "and %g mm thick, with a %g mm hole straight through the centre."
              % (bolt, d2, h, d1)),
        reference=(_sk(), AddCircle("sk1", 0, 0, ro), Extrude("sk1", h),
                   Hole("sk1", 0.0, 0.0, d1, None, True, "simple")),
        volume=annulus_volume(ro, ri, h), bbox=(d2, d2, h), genus=1,
        inside=((mid, 0.0, h / 2.0),),
        outside=((0.0, 0.0, h / 2.0),),
        min_feature=min(ro - ri, h),
        note=("the bore is %.1fx the stock thickness. The precheck rule that "
              "compared those two orthogonal quantities rejected exactly this "
              "part." % (d1 / h)))


def clearance_plate(bid: str, split: str, bolt: str,
                    a: float, b: float, c: float, margin: float) -> Brief:
    """A plate with four ISO 273 medium-series clearance holes, ``margin`` in.

    The hole DIAMETER is the standard's; the plate and the margin are ours. Only
    the number that the harness has a rule about is taken from outside -- which is
    the point: it is the number we could otherwise have chosen to suit ourselves.
    """
    d = CLEARANCE_ISO273[bolt]
    r = d / 2.0
    if margin <= r or margin >= min(a, b) / 2.0:
        raise ValueError("the hole does not fit that far in from the edge")
    centres = ((margin, margin), (a - margin, margin),
               (margin, b - margin), (a - margin, b - margin))
    ops: List[Op] = [_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c)]
    ops += [Hole("sk1", cx, cy, d, None, True, "simple") for cx, cy in centres]
    return Brief(
        id=bid, split=split, source=Source.STANDARD,
        citation=("ISO 273 clearance hole, medium series, for %s: %g mm "
                  "(nominal bolt diameter %g mm, ISO 261)"
                  % (bolt, d, nominal_diameter(bolt))),
        text=("A mounting plate %g mm by %g mm and %g mm thick, with a clearance "
              "hole for an %s bolt (medium fit) drilled right through at each "
              "corner, %g mm in from both edges." % (a, b, c, bolt, margin)),
        reference=tuple(ops),
        volume=a * b * c - 4 * PI * r * r * c, bbox=(a, b, c), genus=4,
        inside=((a / 2.0, b / 2.0, c / 2.0),),
        outside=tuple((cx, cy, c / 2.0) for cx, cy in centres),
        note="a model that guesses %g mm instead of the standard's %g mm scores a "
             "wrong volume AND misses nothing else -- which is why the diameter is "
             "the standard's, not ours" % (nominal_diameter(bolt), d))


def flange_dn50_pn16(bid: str, split: str) -> Brief:
    """A DN50 PN16 flange blank: EN 1092-1 / DIN 2576 outside dims and bolt circle.

    Every dimension except the bore's roundness comes from the flange table:
    165 mm across, 18 mm thick, four 18 mm bolt holes on a 125 mm circle. The bolt
    holes sit at (+/-K/2, 0) and (0, +/-K/2) -- a four-hole circle straddles the
    axes, so the coordinates are signed, which is a second thing this brief tests
    and the analytic plates cannot.
    """
    f = FLANGE_DN50_PN16
    od = float(f["outside_diameter"])
    k = float(f["bolt_circle"])
    n = int(f["bolt_holes"])
    dh = float(f["bolt_hole_diameter"])
    t = float(f["thickness"])
    bore = float(f["nominal_bore"])
    ro, ri, rh = od / 2.0, bore / 2.0, dh / 2.0
    pcd = k / 2.0
    centres = ((pcd, 0.0), (-pcd, 0.0), (0.0, pcd), (0.0, -pcd))
    ops: List[Op] = [_sk(), AddCircle("sk1", 0, 0, ro), Extrude("sk1", t),
                     Hole("sk1", 0.0, 0.0, bore, None, True, "simple")]
    ops += [Hole("sk1", cx, cy, dh, None, True, "simple") for cx, cy in centres]
    # Material between the bore and the bolt circle, on a diagonal where no bolt is.
    mid = (ri + pcd - rh) / 2.0
    diag = mid / math.sqrt(2.0)
    return Brief(
        id=bid, split=split, source=Source.STANDARD,
        citation=("EN 1092-1 / DIN 2576 DN50 PN16 flat-face flange: outside "
                  "diameter %g mm, %d bolt holes of %g mm on a %g mm bolt circle, "
                  "%g mm thick, DN50 nominal bore" % (od, n, dh, k, t)),
        text=("A DN50 PN16 pipe flange blank: a disc %g mm in outside diameter and "
              "%g mm thick, with a %g mm bore through the centre and four %g mm "
              "bolt holes on a %g mm bolt circle -- one on each axis, so at "
              "(+/-%g, 0) and (0, +/-%g) -- all going right through."
              % (od, t, bore, dh, k, pcd, pcd)),
        reference=tuple(ops),
        volume=(PI * ro * ro - PI * ri * ri - n * PI * rh * rh) * t,
        bbox=(od, od, t), genus=1 + n,
        inside=((diag, diag, t / 2.0),),
        outside=((0.0, 0.0, t / 2.0),) + centres_to_probes(centres, t),
        note="signed bolt-circle coordinates; the bolt holes are probed on axis, "
             "so a flange with its bolt circle at the wrong radius fails even "
             "though its volume and bbox are exactly right")


def centres_to_probes(centres, t: float):
    return tuple((cx, cy, t / 2.0) for cx, cy in centres)
