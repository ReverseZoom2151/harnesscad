"""The attack surface: op streams that are CORRECT and that a rule might refuse.

Every attack here is a part a machinist would build without comment. That is the
point -- a red team that submitted broken parts and celebrated when the fleet
caught them would be testing nothing.

HOW THE SEARCH IS AIMED
-----------------------
At the BOUNDARY OF EVERY RULE, and one step either side of it. The rules state
their own boundaries:

  ``check_fillet_radius``    fires when ``2r >= min_extent``   (kernel_preflight)
  ``check_shell_thickness``  fires when ``2t >= min_extent``   (kernel_preflight)
  ``_shell``                 fires when ``t < min_wall`` (0.5) or ``t >= stock``
                                                                (precheck)
  ``_hole_in_plane``         fires when the disc spans the in-plane extent, warns
                             when it crosses the boundary                (precheck)

So the attacks walk ``EPS`` below each threshold, where the part is still real and
the rule must stay silent. ``preflight-RADIUS_TOO_LARGE`` fired at r = 3.1 on a
6 mm plate (2r = 6.2 > 6) and stayed silent at r = 3.0 (2r = 6.0, which IS the
degenerate limit) -- an off-by-one that a corpus of round numbers cannot see and a
boundary walk finds immediately.

``EPS`` is 0.05 mm: comfortably inside any machine shop's tolerance, so a part one
EPS below a limit is unambiguously a real part, and comfortably outside any
floating-point slop, so a rule that fires on it is not merely rounding.

THE UNCOVERED REGIONS
---------------------
The second half of the file is the parts no brief in the repository exercises at
all -- revolves, patterns, multi-body booleans, extreme aspect ratios, chamfers at
their limit, blind holes. A rule cannot have a measured precision on a shape
nobody ever showed it, and "no brief covers it" is exactly where the next
``precheck`` will be wrong.

Seeded and deterministic: the same seed gives the same attacks on every machine,
forever. The randomness only picks WHICH sizes, never whether a part is legal --
every stream is legal by construction, and ``oracle.py`` proves it independently.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import (AddCircle, AddRectangle, Boolean, Chamfer,
                                      Extrude, Fillet, Hole, LinearPattern,
                                      NewSketch, Op, Revolve, Shell,
                                      canonical_json)
from harnesscad.eval.selftest.golden import (annulus_volume, chamfered_box_volume,
                                             rounded_box_volume, shelled_box_volume)

__all__ = ["Attack", "EPS", "SEED", "generate", "FAMILIES"]

PI = math.pi

#: One step below a rule's threshold. Big enough to be a real dimension on a real
#: machine, big enough not to be floating-point slop.
EPS = 0.05

#: The seed. Same attacks, every machine, forever.
SEED = 20260714


@dataclass(frozen=True)
class Attack:
    """One op stream that builds a GOOD part, aimed at one rule's boundary.

    ``why_fine``  the argument, in words, that this part is correct. It is what
                  gets printed next to a false positive, because "verifier X
                  rejected op stream Y" is not a bug report -- it is a bug report
                  only once somebody has said why Y was fine.
    ``volume`` /  the closed form. ``oracle.py`` proves the part builds to it.
    ``bbox``
    ``targets``   the rule(s) this attack is aimed at, for the report. It is NOT
                  a filter: every attack is shown to the WHOLE fleet, because the
                  interesting false positive is the one from a verifier nobody
                  thought was involved.
    """

    name: str
    family: str
    ops: Tuple[Op, ...]
    why_fine: str
    volume: float
    bbox: Tuple[float, float, float]
    min_feature: float
    genus: Optional[int] = None
    targets: Tuple[str, ...] = ()

    def ops_json(self) -> Tuple[str, ...]:
        return tuple(canonical_json(o) for o in self.ops)


def _sk(plane: str = "XY") -> Op:
    return NewSketch(plane)


FAMILIES: Tuple[str, ...] = (
    "fillet_boundary", "chamfer_boundary", "shell_boundary", "shell_min_wall",
    "hole_edge", "hole_thin_plate", "bore_deeper_than_wide", "uncovered",
)


# --------------------------------------------------------------------------- #
# boundary families
# --------------------------------------------------------------------------- #
def _fillet_boundary(rng: random.Random) -> List[Attack]:
    """A fillet just below ``2r == min_extent``. The rule must stay silent.

    THE BUG THIS FAMILY WAS BUILT FOR. ``preflight-RADIUS_TOO_LARGE`` used to fire
    at r = 3.1 on a 50x30x6 plate and not at r = 3.0 -- it was strict-greater on
    the wrong side of its own suggestion. That rule was the ground truth of the
    ``trap_fillet_*`` briefs, so the benchmark rewarded a model for obeying it.
    """
    out: List[Attack] = []
    sizes = [(50.0, 30.0, 6.0), (60.0, 40.0, 10.0), (100.0, 60.0, 4.0),
             (40.0, 40.0, 20.0)]
    for _ in range(4):
        a = float(rng.randrange(30, 120, 5))
        b = float(rng.randrange(20, 80, 5))
        c = float(rng.randrange(4, 30, 2))
        sizes.append((a, b, c))
    for a, b, c in sizes:
        thin = min(a, b, c)
        for label, r in (("just_below_half", thin / 2.0 - EPS),
                         ("well_below_half", thin / 2.0 - min(1.0, thin / 8.0)),
                         ("quarter", thin / 4.0)):
            if r <= 0.2:
                continue
            r = round(r, 3)
            out.append(Attack(
                name="fillet_%gx%gx%g_r%g_%s" % (a, b, c, r, label),
                family="fillet_boundary",
                ops=(_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
                     Fillet((), r)),
                why_fine=("an all-edge fillet of r=%g on a %gx%gx%g plate. The "
                          "degenerate limit is 2r == the smallest extent (%g); "
                          "here 2r = %g < %g, so a face survives on every extent "
                          "and Steiner's formula gives the exact volume %.2f mm3. "
                          "This is a real part."
                          % (r, a, b, c, thin, 2 * r, thin,
                             rounded_box_volume(a, b, c, r))),
                volume=rounded_box_volume(a, b, c, r), bbox=(a, b, c),
                min_feature=min(thin - 2 * r, r), genus=0,
                targets=("kernel-preflight", "precheck")))
    return out


def _chamfer_boundary(rng: random.Random) -> List[Attack]:
    """A chamfer just below its limit. The preflight scores it with the FILLET rule."""
    out: List[Attack] = []
    for _ in range(4):
        a = float(rng.randrange(30, 100, 5))
        b = float(rng.randrange(20, 60, 5))
        c = float(rng.randrange(4, 24, 2))
        thin = min(a, b, c)
        for label, d in (("just_below_half", thin / 2.0 - EPS),
                         ("third", thin / 3.0)):
            d = round(d, 3)
            if d <= 0.2:
                continue
            out.append(Attack(
                name="chamfer_%gx%gx%g_d%g_%s" % (a, b, c, d, label),
                family="chamfer_boundary",
                ops=(_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
                     Chamfer((), d)),
                why_fine=("an all-edge chamfer of %g mm on a %gx%gx%g plate. "
                          "2d = %g < the smallest extent %g, so a face survives; "
                          "the exact volume is %.2f mm3 (12 edge prisms + 8 "
                          "corners)." % (d, a, b, c, 2 * d, thin,
                                         chamfered_box_volume(a, b, c, d))),
                volume=chamfered_box_volume(a, b, c, d), bbox=(a, b, c),
                min_feature=min(thin - 2 * d, d), genus=0,
                targets=("kernel-preflight",)))
    return out


def _shell_boundary(rng: random.Random) -> List[Attack]:
    """A shell just below ``2t == min_extent``: the cavity survives, so it is legal.

    ``preflight-THICKNESS_TOO_LARGE`` is one of only two PROVEN rules in the fleet
    and it is the harness's single structural advantage over a blind loop. A proven
    rule is exactly the one that must not be off by one, so it gets walked too.
    """
    out: List[Attack] = []
    boxes = [(60.0, 40.0, 20.0), (30.0, 30.0, 30.0), (80.0, 60.0, 25.0)]
    for _ in range(4):
        boxes.append((float(rng.randrange(30, 100, 5)),
                      float(rng.randrange(30, 80, 5)),
                      float(rng.randrange(15, 50, 5))))
    for a, b, c in boxes:
        thin = min(a, b, c)
        for label, t in (("just_below_half", thin / 2.0 - EPS),
                         ("third", thin / 3.0),
                         ("quarter", thin / 4.0)):
            t = round(t, 3)
            if t < 0.5:
                continue
            out.append(Attack(
                name="shell_%gx%gx%g_t%g_%s" % (a, b, c, t, label),
                family="shell_boundary",
                ops=(_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
                     Shell((), t)),
                why_fine=("a sealed shell of t=%g in a %gx%gx%g box. The two "
                          "inward offsets of opposite faces meet at 2t == the "
                          "smallest extent (%g); here 2t = %g < %g, so a cavity "
                          "of %.2f mm3 survives and the wall is real. Volume "
                          "%.2f mm3 by closed form."
                          % (t, a, b, c, thin, 2 * t, thin,
                             (a - 2 * t) * (b - 2 * t) * (c - 2 * t),
                             shelled_box_volume(a, b, c, t))),
                volume=shelled_box_volume(a, b, c, t), bbox=(a, b, c),
                min_feature=t, genus=None,
                targets=("kernel-preflight", "precheck", "shell-envelope")))
    return out


def _shell_min_wall(rng: random.Random) -> List[Attack]:
    """A shell wall at exactly ``PrecheckRules.min_wall`` (0.5 mm), and just above.

    ``precheck._shell`` rejects ``t < min_wall``. At ``t == min_wall`` it must NOT
    fire -- 0.5 mm is the stated minimum MANUFACTURABLE wall, and a rule that
    rejects the minimum it itself declares manufacturable is off by one against
    its own constant.
    """
    out: List[Attack] = []
    for a, b, c in ((20.0, 20.0, 10.0), (30.0, 20.0, 8.0), (25.0, 25.0, 15.0)):
        for label, t in (("exactly_min_wall", 0.5), ("just_above", 0.5 + EPS),
                         ("double_min_wall", 1.0)):
            out.append(Attack(
                name="shell_minwall_%gx%gx%g_t%g_%s" % (a, b, c, t, label),
                family="shell_min_wall",
                ops=(_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
                     Shell((), t)),
                why_fine=("a %g mm wall, against a declared minimum manufacturable "
                          "wall of 0.5 mm (PrecheckRules.min_wall). %g >= 0.5, and "
                          "2t = %g is far below the smallest extent %g, so the "
                          "cavity is large. A rule that rejects the minimum it "
                          "itself calls manufacturable is off by one against its "
                          "own constant."
                          % (t, t, 2 * t, min(a, b, c))),
                volume=shelled_box_volume(a, b, c, t), bbox=(a, b, c),
                min_feature=t, genus=None,
                targets=("precheck", "kernel-preflight", "dfm")))
    return out


def _hole_edge(rng: random.Random) -> List[Attack]:
    """A hole walked up to the stock edge: tangent, and one step inside.

    ``precheck._hole_in_plane`` ERRORs when the disc spans the whole in-plane
    extent and WARNs when it crosses the boundary. A hole whose rim is TANGENT to
    the edge leaves zero wall at one point and is a real (if aggressive) feature; a
    hole one EPS inside leaves a real wall and must be silent.
    """
    out: List[Attack] = []
    for a, b, c in ((60.0, 40.0, 10.0), (40.0, 40.0, 12.0), (100.0, 30.0, 8.0)):
        for dia in (8.0, 12.0):
            r = dia / 2.0
            for label, cx in (("one_eps_inside", r + EPS),
                              ("one_mm_inside", r + 1.0),
                              ("two_mm_inside", r + 2.0)):
                out.append(Attack(
                    name="hole_edge_%gx%gx%g_d%g_%s" % (a, b, c, dia, label),
                    family="hole_edge",
                    ops=(_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
                         Hole("sk1", cx, b / 2.0, dia, None, True, "simple")),
                    why_fine=("a %g mm through hole whose rim stops %.2f mm short "
                              "of the x = 0 edge of a %gx%g plate. A wall of %.2f "
                              "mm survives, the disc does not span the in-plane "
                              "extent (%g mm), and the plate is not severed. "
                              "Volume %.2f mm3."
                              % (dia, cx - r, a, b, cx - r, a,
                                 a * b * c - PI * r * r * c)),
                    volume=a * b * c - PI * r * r * c, bbox=(a, b, c),
                    min_feature=min(c, cx - r), genus=1,
                    targets=("precheck", "dfm", "geometry")))
    return out


def _hole_thin_plate(rng: random.Random) -> List[Attack]:
    """A hole WIDER THAN THE PLATE IS THICK. The bug that shipped.

    The deleted precheck rule compared a hole's DIAMETER against the base extrude
    DISTANCE. Those are orthogonal: the diameter lies in the sketch plane, the
    distance along Z. It rejected an 80 mm disc, 8 mm thick, with a 30 mm bore --
    a WASHER -- and it fired 40 times in the pressure run and caused every
    regression the harness had. The rule is gone; this family is the trap that
    stays behind so it cannot come back, in ANY verifier.
    """
    out: List[Attack] = []
    cases = [
        (80.0, 8.0, 30.0, "an ISO-proportioned washer blank: 80 mm across, 8 mm "
                          "thick, 30 mm bore"),
        (16.0, 1.6, 8.4, "an ISO 7089 M8 plain washer, to the millimetre: the bore "
                         "is 5.25x the stock thickness, and that is what a washer IS"),
        (60.0, 5.0, 25.0, "a 5 mm plate with a 25 mm bore: five times the thickness"),
    ]
    for od, h, bore, why in cases:
        ro, ri = od / 2.0, bore / 2.0
        out.append(Attack(
            name="washer_od%g_h%g_bore%g" % (od, h, bore),
            family="hole_thin_plate",
            ops=(_sk(), AddCircle("sk1", 0, 0, ro), Extrude("sk1", h),
                 Hole("sk1", 0.0, 0.0, bore, None, True, "simple")),
            why_fine=("%s. The hole's diameter is an IN-PLANE quantity and the "
                      "plate's thickness is along Z; they are orthogonal and "
                      "comparing them is a category error. %.2f mm of wall "
                      "survives all the way round. Volume %.2f mm3 = "
                      "pi(R^2 - r^2)h." % (why, ro - ri,
                                           annulus_volume(ro, ri, h))),
            volume=annulus_volume(ro, ri, h), bbox=(od, od, h),
            min_feature=min(ro - ri, h), genus=1,
            targets=("precheck", "dfm", "geometry", "plausibility")))
    # And the rectangular version: a hole row in a strip thinner than the holes.
    out.append(Attack(
        name="strip_holes_wider_than_thick",
        family="hole_thin_plate",
        ops=(_sk(), AddRectangle("sk1", 0, 0, 120, 30), Extrude("sk1", 6.0),
             Hole("sk1", 20.0, 15.0, 8.0, None, True, "simple"),
             Hole("sk1", 60.0, 15.0, 8.0, None, True, "simple"),
             Hole("sk1", 100.0, 15.0, 8.0, None, True, "simple")),
        why_fine=("a 6 mm strip with three 8 mm through-holes. Every hole is wider "
                  "than the strip is thick, which is the single most ordinary "
                  "thing in a sheet-metal shop. 11 mm of wall survives above and "
                  "below each hole."),
        volume=120 * 30 * 6 - 3 * PI * 16 * 6, bbox=(120.0, 30.0, 6.0),
        min_feature=6.0, genus=3,
        targets=("precheck", "dfm")))
    return out


def _bore_deeper_than_wide(rng: random.Random) -> List[Attack]:
    """A bore wider than the part is TALL: a bearing housing. And its inverse."""
    out: List[Attack] = []
    out.append(Attack(
        name="bearing_housing_od60_h25_bore40",
        family="bore_deeper_than_wide",
        ops=(_sk(), AddCircle("sk1", 0, 0, 30.0), Extrude("sk1", 25.0),
             Hole("sk1", 0.0, 0.0, 40.0, None, True, "simple")),
        why_fine=("a 60 mm boss, 25 mm tall, bored 40 mm to take a bearing. The "
                  "bore (40) is wider than the part is tall (25) -- and that is "
                  "what a bearing housing IS. 10 mm of wall all round. Volume "
                  "%.2f mm3." % annulus_volume(30.0, 20.0, 25.0)),
        volume=annulus_volume(30.0, 20.0, 25.0), bbox=(60.0, 60.0, 25.0),
        min_feature=10.0, genus=1,
        targets=("precheck", "dfm", "plausibility")))
    out.append(Attack(
        name="deep_tube_d40_bore24_h50",
        family="bore_deeper_than_wide",
        ops=(_sk(), AddCircle("sk1", 0, 0, 20.0), Extrude("sk1", 50.0),
             Hole("sk1", 0.0, 0.0, 24.0, None, True, "simple")),
        why_fine=("a 40 mm tube, 50 mm long, 24 mm bore: an 8 mm wall and a "
                  "depth-to-diameter ratio of about 2. Routine. Volume %.2f mm3."
                  % annulus_volume(20.0, 12.0, 50.0)),
        volume=annulus_volume(20.0, 12.0, 50.0), bbox=(40.0, 40.0, 50.0),
        min_feature=8.0, genus=1,
        targets=("precheck", "dfm", "plausibility")))
    return out


def _uncovered(rng: random.Random) -> List[Attack]:
    """The shapes no brief in this repository exercises.

    A rule cannot have a measured precision on a shape it was never shown, so this
    is where the next false positive is. Revolves, patterns, multi-body booleans,
    extreme aspect ratios, blind holes: all legal, all correct, none of them in the
    pressure corpus.
    """
    out: List[Attack] = []
    out.append(Attack(
        name="revolved_ring",
        family="uncovered",
        ops=(_sk(), AddRectangle("sk1", 10, 0, 5, 20),
             Revolve("sk1", (0.0, 0.0, 0.0, 0.0, 1.0, 0.0), 360.0)),
        why_fine=("a 5x20 profile, offset 10 mm from the axis, revolved 360 "
                  "degrees. It sweeps an annulus of volume pi(15^2 - 10^2)*20 = "
                  "%.2f mm3. Closed, genus 1. No brief in the repository revolves "
                  "anything." % annulus_volume(15.0, 10.0, 20.0)),
        volume=annulus_volume(15.0, 10.0, 20.0), bbox=(30.0, 20.0, 30.0),
        min_feature=5.0, genus=1,
        targets=("precheck", "geometry", "plausibility")))
    out.append(Attack(
        name="linear_pattern_3x",
        family="uncovered",
        ops=(_sk(), AddRectangle("sk1", 0, 0, 10, 10), Extrude("sk1", 5.0),
             LinearPattern("f1", (1.0, 0.0, 0.0), 3, 20.0)),
        why_fine=("a 10x10x5 block patterned 3 times at 20 mm pitch: three "
                  "disjoint bodies, total volume 1500 mm3. A pattern count of 3 is "
                  "above min_pattern_count (2) and the spacing (20) exceeds the "
                  "part (10), so no two instances interfere."),
        volume=3 * 10 * 10 * 5, bbox=(50.0, 10.0, 5.0), min_feature=5.0, genus=-2,
        targets=("precheck", "interference", "plausibility")))
    out.append(Attack(
        name="two_body_union_then_hole",
        family="uncovered",
        ops=(_sk(), AddRectangle("sk1", 0, 0, 60, 40), Extrude("sk1", 8.0),
             _sk(), AddRectangle("sk2", 60, 0, 40, 40), Extrude("sk2", 8.0),
             Boolean("union", "f1", "f2"),
             Hole("sk1", 50.0, 20.0, 10.0, None, True, "simple")),
        why_fine=("two abutting plates unioned into a 100x40x8 body, then a 10 mm "
                  "hole through the middle of the joint. The hole is 45 mm from "
                  "either end and 15 mm from either side: material everywhere. "
                  "Volume 100*40*8 - pi*25*8 = %.2f mm3."
                  % (100 * 40 * 8 - PI * 25 * 8)),
        volume=100 * 40 * 8 - PI * 25 * 8, bbox=(100.0, 40.0, 8.0),
        min_feature=8.0, genus=1,
        targets=("precheck", "geometry")))
    out.append(Attack(
        name="long_thin_bar",
        family="uncovered",
        ops=(_sk(), AddRectangle("sk1", 0, 0, 200, 8), Extrude("sk1", 8.0)),
        why_fine=("a 200x8x8 bar. An aspect ratio of 25:1 is a piece of stock, not "
                  "a defect; every extruded aluminium profile on earth is worse. "
                  "Volume 12800 mm3."),
        volume=200 * 8 * 8, bbox=(200.0, 8.0, 8.0), min_feature=8.0, genus=0,
        targets=("plausibility", "precheck", "dfm")))
    out.append(Attack(
        name="blind_hole",
        family="uncovered",
        ops=(_sk(), AddRectangle("sk1", 0, 0, 50, 50), Extrude("sk1", 20.0),
             Hole("sk1", 25.0, 25.0, 10.0, 12.0, False, "simple")),
        why_fine=("a 10 mm blind hole, 12 mm deep, in a 20 mm block: 8 mm of "
                  "material left in the bottom. A blind hole with a stated depth "
                  "is the ordinary case and no brief in the repository has one."),
        volume=50 * 50 * 20 - PI * 25 * 12, bbox=(50.0, 50.0, 20.0),
        min_feature=8.0, genus=0,
        targets=("precheck", "dfm", "geometry")))
    out.append(Attack(
        name="cut_leaving_thin_web",
        family="uncovered",
        ops=(_sk(), AddRectangle("sk1", 0, 0, 60, 40), Extrude("sk1", 20.0),
             _sk(), AddRectangle("sk2", 0, 10, 60, 20), Extrude("sk2", 20.0),
             Boolean("cut", "f1", "f2")),
        why_fine=("a 60x40x20 block with a 20 mm slot cut clean through it, "
                  "leaving two 10 mm webs. A cut REMOVES material: 48000 - 24000 = "
                  "24000 mm3, and the bbox is unchanged."),
        volume=60 * 40 * 20 - 60 * 20 * 20, bbox=(60.0, 40.0, 20.0),
        min_feature=10.0, genus=0,
        targets=("precheck", "geometry", "plausibility")))
    return out


_GENERATORS: Tuple[Callable[[random.Random], List[Attack]], ...] = (
    _fillet_boundary, _chamfer_boundary, _shell_boundary, _shell_min_wall,
    _hole_edge, _hole_thin_plate, _bore_deeper_than_wide, _uncovered,
)


def generate(seed: int = SEED,
             families: Optional[Sequence[str]] = None,
             limit: Optional[int] = None) -> List[Attack]:
    """Every attack, deterministically. ``limit`` truncates for a fast test run."""
    rng = random.Random(seed)
    out: List[Attack] = []
    wanted = set(families) if families else None
    for gen in _GENERATORS:
        produced = gen(rng)
        if wanted is not None:
            produced = [a for a in produced if a.family in wanted]
        out.extend(produced)
    # Deduplicate on the canonical op stream: two families can land on the same
    # part and one part must not be counted as two findings.
    seen: Dict[Tuple[str, ...], Attack] = {}
    for a in out:
        key = a.ops_json()
        seen.setdefault(key, a)
    unique = sorted(seen.values(), key=lambda a: (a.family, a.name))
    return unique[:limit] if limit else unique
