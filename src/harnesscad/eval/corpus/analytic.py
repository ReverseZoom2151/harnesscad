"""Brief FACTORIES whose ground truth is arithmetic.

Nothing here is measured. Every number a brief carries is computed from the
dimensions in its own prompt by a formula that is written down, so a backend that
misses it is the backend with the bug -- there is no appeal to "what the harness
usually returns".

The closed forms themselves are NOT re-derived here. They already exist, correct
and documented, in :mod:`harnesscad.eval.selftest.golden` (Steiner's formula for
an all-edge fillet, the 12-prisms-and-8-corners chamfer, the sealed inward shell,
the annulus). The brief asked for that module to be EXTENDED rather than
rebuilt, and importing its formulas is the extension: one derivation, two
consumers, and a fix to the formula cannot desynchronise them.

What this module adds on top of ``golden`` is the three things ``golden`` has no
notion of, because it is an oracle for the ENGINE and not a benchmark for a
MODEL:

  * a natural-language PROMPT (``golden`` parts have op streams only);
  * an EXPECTED BBOX carried as a first-class, mandatory field;
  * PROBE POINTS, derived arithmetically, that pin a feature to a LOCATION --
    the mid-plane of a wall, the axis of a bore. Volume and bbox are envelope
    families and are many-to-one; a hole in the wrong place matches both.

Every factory is a pure function of its dimensions, so the dev split and the
held-out split are two different sets of NUMBERS through the same arithmetic --
not two hand-written files that can drift apart.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from harnesscad.core.cisp.ops import (AddCircle, AddRectangle, Boolean, Chamfer,
                                      Extrude, Fillet, Hole, NewSketch, Op, Shell)
from harnesscad.eval.corpus.spec import Brief, Source, Vec3
# The derivations live in golden.py. They are imported, not copied.
from harnesscad.eval.selftest.golden import (annulus_volume, chamfered_box_volume,
                                             rounded_box_volume, shelled_box_volume)

__all__ = ["plate", "plate_with_holes", "disc", "tube", "hollow_box",
           "filleted_plate", "chamfered_plate", "notched_block", "l_bracket"]

PI = math.pi

_CITE_PRISM = "arithmetic: V = a*b*c for a rectangular prism"
_CITE_CYL = "arithmetic: V = pi*r^2*h for a right circular cylinder"
_CITE_ANNULUS = "arithmetic: V = pi*(R^2 - r^2)*h; a through hole adds one handle"
_CITE_SHELL = ("arithmetic: a sealed inward shell of an a x b x c box to wall t "
               "is a*b*c - (a-2t)(b-2t)(c-2t), and the bbox is UNCHANGED "
               "(golden.shelled_box_volume)")
_CITE_FILLET = ("Steiner's formula: an all-edge fillet of radius r is the "
                "Minkowski sum of a ball of radius r with the box shrunk by 2r "
                "(golden.rounded_box_volume); the bbox is unchanged")
_CITE_CHAMFER = ("arithmetic: 12 edge prisms of section d^2/2 plus 8 corner "
                 "solids of (5/6)d^3 (golden.chamfered_box_volume)")


def _sk(plane: str = "XY") -> Op:
    return NewSketch(plane)


# --------------------------------------------------------------------------- #
# the factories
# --------------------------------------------------------------------------- #
def plate(bid: str, split: str, a: float, b: float, c: float) -> Brief:
    """A flat rectangular plate. V = a*b*c, exactly."""
    return Brief(
        id=bid, split=split, source=Source.ANALYTIC, citation=_CITE_PRISM,
        text=("A flat rectangular plate %g mm long, %g mm wide and %g mm thick."
              % (a, b, c)),
        reference=(_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c)),
        volume=a * b * c, bbox=(a, b, c), genus=0,
        inside=((a / 2, b / 2, c / 2),),
        outside=((a / 2, b / 2, c * 1.5), (a * 1.5, b / 2, c / 2)),
        note="the simplest closed form there is; if this fails nothing else means "
             "anything")


def plate_with_holes(bid: str, split: str, a: float, b: float, c: float,
                     dia: float, centres: Tuple[Tuple[float, float], ...]) -> Brief:
    """A plate with N through holes. V = abc - N*pi*(d/2)^2*c; genus = N.

    The ``outside`` probes sit ON THE AXIS OF EVERY HOLE. That is the assertion a
    volume check cannot make: N holes of the right size in the wrong places have
    exactly the right volume and exactly the right bbox.
    """
    n = len(centres)
    if n < 1:
        raise ValueError("plate_with_holes needs at least one centre")
    r = dia / 2.0
    for cx, cy in centres:
        if not (r < cx < a - r and r < cy < b - r):
            raise ValueError(
                "hole (%g, %g) d=%g does not fit inside a %gx%g plate with a wall "
                "around it; the brief would be unbuildable" % (cx, cy, dia, a, b))
    ops: List[Op] = [_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c)]
    ops += [Hole("sk1", cx, cy, dia, None, True, "simple") for cx, cy in centres]
    where = ", ".join("(%g, %g)" % (cx, cy) for cx, cy in centres)
    # A point inside the material: the plate centre when no hole is there, else a
    # corner well clear of every hole.
    inside_pt = (min(cx for cx, _ in centres) / 2.0,
                 min(cy for _, cy in centres) / 2.0, c / 2.0)
    return Brief(
        id=bid, split=split, source=Source.ANALYTIC, citation=_CITE_ANNULUS,
        text=("A plate %g mm by %g mm and %g mm thick, with %d %g mm diameter "
              "through-holes at %s (measured from the corner at the origin)."
              % (a, b, c, n, dia, where)),
        reference=tuple(ops),
        volume=a * b * c - n * PI * r * r * c, bbox=(a, b, c), genus=n,
        inside=(inside_pt,),
        outside=tuple((cx, cy, c / 2.0) for cx, cy in centres),
        note="the outside probes sit on every hole axis: N holes in the WRONG "
             "places have the right volume and the right bbox")


def disc(bid: str, split: str, dia: float, h: float) -> Brief:
    """A solid cylinder. V = pi*r^2*h."""
    r = dia / 2.0
    return Brief(
        id=bid, split=split, source=Source.ANALYTIC, citation=_CITE_CYL,
        text="A solid cylindrical disc, %g mm in diameter and %g mm tall." % (dia, h),
        reference=(_sk(), AddCircle("sk1", 0, 0, r), Extrude("sk1", h)),
        volume=PI * r * r * h, bbox=(dia, dia, h), genus=0,
        inside=((0.0, 0.0, h / 2.0),),
        outside=((0.0, 0.0, h * 1.5), (dia, dia, h / 2.0)))


def tube(bid: str, split: str, od: float, bore: float, h: float) -> Brief:
    """A tube / spacer / washer blank. V = pi*(R^2 - r^2)*h; genus 1.

    The ``inside`` probe sits at the MID-RADIUS of the wall and the ``outside``
    probe on the axis, so the annulus is pinned from both sides: a part that lost
    its bore, and a part that is all bore, both fail.
    """
    if not (0 < bore < od):
        raise ValueError("bore must be inside the outside diameter")
    ro, ri = od / 2.0, bore / 2.0
    mid = (ro + ri) / 2.0
    return Brief(
        id=bid, split=split, source=Source.ANALYTIC, citation=_CITE_ANNULUS,
        text=("A cylindrical spacer %g mm in outside diameter and %g mm tall, "
              "with a %g mm diameter bore straight through the centre."
              % (od, h, bore)),
        reference=(_sk(), AddCircle("sk1", 0, 0, ro), Extrude("sk1", h),
                   Hole("sk1", 0.0, 0.0, bore, None, True, "simple")),
        volume=annulus_volume(ro, ri, h), bbox=(od, od, h), genus=1,
        inside=((mid, 0.0, h / 2.0),),
        outside=((0.0, 0.0, h / 2.0),),
        min_feature=min(ro - ri, h))


def hollow_box(bid: str, split: str, a: float, b: float, c: float, t: float) -> Brief:
    """A sealed hollow box. V = abc - (a-2t)(b-2t)(c-2t); THE BBOX IS UNCHANGED.

    This is the brief the pressure corpus got wrong twice over. Its shell briefs
    carried ``bbox=None`` (so a shell that dilated the part scored a pass) and
    probed their inside point exactly ON the outer face at x = 0 (so only a
    DILATING shell could satisfy them). Here the envelope is stated, and the
    inside probe sits at the MID-PLANE OF THE WALL, x = t/2 -- which pins the wall
    thickness from below as well: a part shelled at t/2 has no material there.
    """
    if 2 * t >= min(a, b, c):
        raise ValueError("2t >= the smallest extent: no cavity survives")
    return Brief(
        id=bid, split=split, source=Source.ANALYTIC, citation=_CITE_SHELL,
        text=("A hollow box. Start from a solid block %g mm by %g mm by %g mm "
              "and shell it out to leave a sealed %g mm wall." % (a, b, c, t)),
        reference=(_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
                   Shell((), t)),
        volume=shelled_box_volume(a, b, c, t), bbox=(a, b, c), genus=None,
        inside=((t / 2.0, b / 2.0, c / 2.0),),          # MID-WALL, not the face
        outside=((a / 2.0, b / 2.0, c / 2.0),),         # the cavity must exist
        # THE WALL is the thinnest thing here, and it is nowhere in the bbox. A
        # sampled engine that cannot resolve it cannot measure this part -- see
        # grade.resolvable, and the finding in the report.
        min_feature=t,
        note="bbox UNCHANGED (a shell hollows inward) and the inside probe is at "
             "the middle of the wall, so a t/2 wall fails and a dilated part fails")


def filleted_plate(bid: str, split: str, a: float, b: float, c: float,
                   r: float) -> Brief:
    """An all-edge fillet. Steiner's formula. The bbox is UNCHANGED.

    NOTE ON THE FEASIBLE CEILING. The true degenerate limit is ``2r == the
    smallest extent``: at that radius the two roundings meet and the face between
    them vanishes. Everything strictly below it is a real part with a real
    volume, and this factory refuses anything else -- which is what makes these
    briefs a check ON the harness's fillet rule rather than an echo OF it. The
    pressure corpus wrote its fillet briefs to agree with a rule that fired at
    r = 3.1 on a 6 mm plate and stayed silent at r = 3.0.
    """
    if 2 * r >= min(a, b, c):
        raise ValueError("2r >= the smallest extent: the fillet is degenerate")
    return Brief(
        id=bid, split=split, source=Source.ANALYTIC, citation=_CITE_FILLET,
        text=("A plate %g mm by %g mm and %g mm thick with every edge rounded off "
              "with a %g mm radius fillet." % (a, b, c, r)),
        reference=(_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
                   Fillet((), r)),
        volume=rounded_box_volume(a, b, c, r), bbox=(a, b, c), genus=0,
        inside=((a / 2.0, b / 2.0, c / 2.0),),
        # The corner is gone: a point at (r/8, r/8, r/8) is outside the rounded
        # corner but inside the sharp box it came from.
        outside=((r / 8.0, r / 8.0, r / 8.0),),
        note="Steiner. The bbox does NOT change: a fillet removes material.")


def chamfered_plate(bid: str, split: str, a: float, b: float, c: float,
                    d: float) -> Brief:
    """An all-edge chamfer: 12 edge prisms + 8 corners. The bbox is UNCHANGED."""
    if 2 * d >= min(a, b, c):
        raise ValueError("2d >= the smallest extent: the chamfer is degenerate")
    return Brief(
        id=bid, split=split, source=Source.ANALYTIC, citation=_CITE_CHAMFER,
        text=("A plate %g mm by %g mm and %g mm thick with a %g mm chamfer on "
              "every edge." % (a, b, c, d)),
        reference=(_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
                   Chamfer((), d)),
        volume=chamfered_box_volume(a, b, c, d), bbox=(a, b, c), genus=0,
        inside=((a / 2.0, b / 2.0, c / 2.0),),
        outside=((d / 8.0, d / 8.0, d / 8.0),))


def notched_block(bid: str, split: str, a: float, b: float, c: float,
                  nw: float, nh: float) -> Brief:
    """A block with a rectangular notch cut out of one corner, through-thickness.

    V = abc - nw*nh*c, exactly. A cut REMOVES material; the bbox is unchanged
    (the notch is interior to the footprint on two sides and flush on two).
    """
    if not (0 < nw < a and 0 < nh < b):
        raise ValueError("the notch must be smaller than the block")
    return Brief(
        id=bid, split=split, source=Source.ANALYTIC, citation=_CITE_PRISM,
        text=("A block %g mm by %g mm by %g mm with a rectangular notch %g mm by "
              "%g mm cut all the way through its thickness at the corner nearest "
              "the origin." % (a, b, c, nw, nh)),
        reference=(_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
                   _sk(), AddRectangle("sk2", 0, 0, nw, nh), Extrude("sk2", c),
                   Boolean("cut", "f1", "f2")),
        volume=a * b * c - nw * nh * c, bbox=(a, b, c), genus=0,
        inside=((a - 1.0, b - 1.0, c / 2.0),),
        outside=((nw / 2.0, nh / 2.0, c / 2.0),),
        note="the outside probe is INSIDE the notch: a cut that removed the wrong "
             "corner has the same volume and the same bbox")


def l_bracket(bid: str, split: str, a: float, b: float, t: float,
              wall_h: float) -> Brief:
    """An L: a flat base plate unioned with an upright wall on one edge.

    Both bodies are extruded from Z = 0 (CISP has no offset start), so the union
    OVERLAPS in the block t x b x t and the volume is
    ``a*b*t + t*b*wall_h - t*b*t`` by inclusion-exclusion -- an arithmetic fact
    about the two prisms, not a measurement. The bbox is (a, b, max(t, wall_h)).
    """
    if not (0 < t < min(a, b) and wall_h > t):
        raise ValueError("degenerate L")
    return Brief(
        id=bid, split=split, source=Source.ANALYTIC,
        citation="arithmetic: inclusion-exclusion on two overlapping prisms",
        text=("An L-shaped bracket, as one body. A base plate %g mm by %g mm and "
              "%g mm thick lies flat. An upright wall, also %g mm thick and the "
              "full %g mm wide, stands on the short edge at the origin and rises "
              "%g mm. Union them." % (a, b, t, t, b, wall_h)),
        reference=(_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", t),
                   _sk(), AddRectangle("sk2", 0, 0, t, b), Extrude("sk2", wall_h),
                   Boolean("union", "f1", "f2")),
        volume=a * b * t + t * b * wall_h - t * b * t,
        bbox=(a, b, max(t, wall_h)), genus=0,
        inside=((a / 2.0, b / 2.0, t / 2.0), (t / 2.0, b / 2.0, wall_h * 0.9)),
        outside=((a / 2.0, b / 2.0, wall_h * 0.9),),
        note="a union ADDS material: the overlap is counted once, by "
             "inclusion-exclusion, not by asking a backend")
