"""INVERT THE GENERATOR: the op stream IS the ground truth, so the truth is free.

The published benchmarks curate by hand -- Text2CAD-Bench 600 examples, MUSE 106 --
and a hand-curated corpus is bounded by the hand. We do the opposite, exactly as
``eval/grounding/corpus.py`` already does for click targets (942 verified pairs a
minute, no human in the loop): we SAMPLE A HARD OP STREAM from a seeded grammar,
BUILD it on the exact kernel, and because WE AUTHORED THE STREAM the part's ground
truth is arithmetic and free. The hole is at x = 27.5 because we put it there, not
because a backend reported it. Then the brief is derived FROM the stream.

Consequences: zero human labelling, arbitrary difficulty, unbounded scale, and --
the property none of the others have -- a ground truth that is a CLOSED FORM rather
than a measurement, so it cannot defend a backend bug the way a recorded fixture
would (that is the contamination ``eval/corpus`` was built to remove, and this
package inherits its ``Brief`` type wholesale for exactly that reason: ``Brief``
REFUSES to be constructed without a declared non-us provenance and an expected
bbox).

WHAT IS SAMPLED, AND WHY IT IS HARD
-----------------------------------
The families below are Text2CAD-Bench's L3 -- the cliff. On these ops it reports
GPT-5.2 at 68% invalidity and Claude-4.5-Sonnet, its best model, still failing
70%, and concludes they "remain outside most LLMs' pretraining corpus":

  * ``revolve``  an annular ring, exact volume ``pi*(R^2 - r^2)*h``.
  * ``loft``     a rectangular prismatoid (``h/6 * (A0 + 4*Am + A1)``) and a conical
                 frustum (``pi*h/3 * (r0^2 + r0*r1 + r1^2)``), both exact.
  * ``sweep``    a bar swept along a straight path, exact ``area * length``.
  * ``shell``    a box hollowed with ONE FACE OPEN -- exact
                 ``abc - (a-2t)(b-2t)(c-t)`` -- which is also the near-miss engine
                 for ``discriminative`` (a shell open on the wrong face has the
                 identical volume).
  * ``pattern``  a rib array (linear) and a tooth ring (circular), each an exact
                 ``count * unit`` because the units are placed NOT to overlap. The
                 cadquery pattern op replicates the whole body and unions it, so a
                 pattern-of-holes does NOT drill more holes (measured -- see the
                 note on ``DROPPED_OPS``); a pattern of a standalone unit does, and
                 that is what is sampled.
  * DEEP CHAINS  a plate chamfered or filleted on every edge and THEN drilled with
                 six interior holes: 10 ops, exact because the chamfer precedes the
                 holes so the two features are volume-independent (measured: chamfer
                 AFTER holes rounds the hole rims and breaks additivity by 2%; BEFORE
                 holes it is additive to 1e-16). This is the 10-20 op depth the
                 field's 4-op briefs never reach, and the depth our own old corpus
                 never reached -- which is why ``qwen2.5-coder:14b`` solved 66.7% of
                 it blind.

DUAL-STYLE PROMPTS
------------------
Every part is emitted as TWO briefs sharing one reference and one ground truth: a
``_plain`` brief in the register a non-expert would use ("a ring 60 mm across...")
and a ``_proc`` brief in expert-procedural register ("revolve a 10 x 5 mm section
at radius 20 mm through 360 degrees"). Text2CAD-Bench uses exactly this dual-style
design; the pair tests whether a model's competence survives a change of register.

DROPPED, AND SAID SO
--------------------
``DROPPED_OPS`` records the L3 ops we CANNOT ship a brief for because their own
reference solution does not build on any available backend -- today, ``draft``
(the cadquery backend returns ``unsupported-op``; frep too). A brief whose
reference does not build is the exact bug that contaminated v1, so draft is dropped
rather than shipped broken. When a backend gains draft, a factory drops in here and
the drop-note comes out.
"""

from __future__ import annotations

import math
import random
import zlib
from typing import Callable, Dict, List, Optional, Tuple

from harnesscad.core.cisp.ops import (AddCircle, AddLine, AddRectangle, Chamfer,
                                      CircularPattern, Extrude, Fillet, Hole,
                                      LinearPattern, Loft, NewSketch, Op, Revolve,
                                      Shell, Sweep)
from harnesscad.eval.corpus.spec import Brief, Source, Vec3
from harnesscad.eval.selftest.golden import chamfered_box_volume

__all__ = ["DROPPED_OPS", "LEVEL", "FACTORIES", "sample", "all_briefs",
           "GeneratedPart"]

PI = math.pi

#: L3 ops whose reference solution does not build on any backend here, so no brief
#: can be shipped for them without shipping a broken reference. Named, not hidden.
DROPPED_OPS: Dict[str, str] = {
    "draft": ("the cadquery backend returns unsupported-op ('rejected'='draft') "
              "and so does frep; a draft brief would carry a reference stream that "
              "does not build, which is the v1 contamination bug. Dropped until a "
              "backend implements draft."),
}

#: Difficulty tag per family. L1/L2 are calibration (a model that fails these is
#: not being measured on L3 at all); the corpus is deliberately L3-heavy. L4 (a
#: VLM-judged design-intent tier) is NOT emitted -- see the report; we will not
#: ship a tier whose oracle is a model's opinion of a picture.
LEVEL: Dict[str, str] = {}


# --------------------------------------------------------------------------- #
# a generated part: the op stream, its closed form, and its dual prompts
# --------------------------------------------------------------------------- #
class GeneratedPart:
    """Internal: one authored part, from which two Briefs (plain, proc) are cut."""

    def __init__(self, family: str, level: str, ops: Tuple[Op, ...],
                 volume: float, bbox: Vec3, genus: Optional[int],
                 inside: Tuple[Vec3, ...], outside: Tuple[Vec3, ...],
                 plain: str, proc: str, citation: str,
                 min_feature: Optional[float] = None, note: str = "") -> None:
        self.family = family
        self.level = level
        self.ops = ops
        self.volume = volume
        self.bbox = bbox
        self.genus = genus
        self.inside = inside
        self.outside = outside
        self.plain = plain
        self.proc = proc
        self.citation = citation
        self.min_feature = min_feature
        self.note = note
        LEVEL[family] = level

    def briefs(self, bid: str, split: str) -> List[Brief]:
        common = dict(source=Source.ANALYTIC, citation=self.citation,
                      reference=self.ops, volume=self.volume, bbox=self.bbox,
                      genus=self.genus, inside=self.inside, outside=self.outside,
                      min_feature=self.min_feature, note=self.note)
        return [
            Brief(id=bid + "_plain", split=split, text=self.plain, **common),
            Brief(id=bid + "_proc", split=split, text=self.proc, **common),
        ]


def _sk(plane: str = "XY") -> Op:
    return NewSketch(plane)


# --------------------------------------------------------------------------- #
# the factories -- each a pure function of sampled dimensions
# --------------------------------------------------------------------------- #
def revolve_ring(r_in: float, wall: float, h: float) -> GeneratedPart:
    """An annular ring, revolved from a rectangular section. V = pi*(R^2-r^2)*h."""
    r_out = r_in + wall
    ops = (_sk("XZ"), AddRectangle("sk1", r_in, 0.0, wall, h),
           Revolve("sk1", (0.0, 0.0, 0.0, 0.0, 1.0, 0.0), 360.0))
    vol = PI * (r_out * r_out - r_in * r_in) * h
    mid = r_in + wall / 2.0
    return GeneratedPart(
        "revolve", "L2", ops, vol, (2 * r_out, 2 * r_out, h), 1,
        inside=((mid, 0.0, h / 2.0),),
        outside=((0.0, 0.0, h / 2.0), (0.0, 0.0, h * 2.0)),
        plain=("A ring %g mm across the outside and %g mm tall, with a %g mm wide "
               "wall of material around a hollow centre."
               % (2 * r_out, h, wall)),
        proc=("On the XZ plane sketch a rectangle from radius %g mm to radius %g mm "
              "and %g mm tall, then revolve it a full 360 degrees about the "
              "vertical axis to make a ring." % (r_in, r_out, h)),
        citation="arithmetic: a solid of revolution of an annular section, "
                 "V = pi*(R^2 - r^2)*h; a full revolution adds one handle (genus 1)",
        min_feature=min(wall, h),
        note="the inside probe is at mid-wall radius; a ring with the wrong bore "
             "radius has a different volume AND a mis-classified mid-wall point")


def loft_rect_frustum(a0: float, a1: float, h: float) -> GeneratedPart:
    """A square prismatoid lofted between two centred squares. Exact prismatoid."""
    off = (a0 - a1) / 2.0
    ops = (_sk(), AddRectangle("sk1", 0.0, 0.0, a0, a0),
           _sk(), AddRectangle("sk2", off, off, a1, a1),
           Loft(("sk1", "sk2"), True, (0.0, h)))
    A0, A1 = a0 * a0, a1 * a1
    Am = ((a0 + a1) / 2.0) ** 2
    vol = h / 6.0 * (A0 + 4.0 * Am + A1)
    return GeneratedPart(
        "loft_rect", "L3", ops, vol, (a0, a0, h), 0,
        inside=((a0 / 2.0, a0 / 2.0, h / 2.0),),
        outside=((a0 / 20.0, a0 / 20.0, h * 0.95), (a0 / 2.0, a0 / 2.0, h * 1.5)),
        plain=("A tapered block %g mm square at the bottom, narrowing to %g mm "
               "square at the top, %g mm tall." % (a0, a1, h)),
        proc=("Sketch a %g mm square, sketch a %g mm square centred %g mm above it, "
              "and loft between the two profiles." % (a0, a1, h)),
        citation="arithmetic: the prismatoid formula V = h/6*(A0 + 4*Am + A1) for a "
                 "ruled loft between two parallel squares")


def loft_round_frustum(r0: float, r1: float, h: float) -> GeneratedPart:
    """A conical frustum lofted between two circles. V = pi*h/3*(r0^2+r0r1+r1^2)."""
    ops = (_sk(), AddCircle("sk1", 0.0, 0.0, r0),
           _sk(), AddCircle("sk2", 0.0, 0.0, r1),
           Loft(("sk1", "sk2"), True, (0.0, h)))
    vol = PI * h / 3.0 * (r0 * r0 + r0 * r1 + r1 * r1)
    return GeneratedPart(
        "loft_round", "L3", ops, vol, (2 * r0, 2 * r0, h), 0,
        inside=((0.0, 0.0, h / 2.0),),
        outside=((r0 * 0.9, 0.0, h * 0.95), (0.0, 0.0, h * 1.5)),
        plain=("A cone-like plug, %g mm across at the wide base and %g mm across at "
               "the narrow top, %g mm tall." % (2 * r0, 2 * r1, h)),
        proc=("Sketch a circle of radius %g mm, sketch a circle of radius %g mm "
              "centred %g mm above it, and loft between them to make a frustum."
              % (r0, r1, h)),
        citation="arithmetic: the conical-frustum volume "
                 "V = pi*h/3*(r0^2 + r0*r1 + r1^2)")


def sweep_bar(r: float, length: float) -> GeneratedPart:
    """A round bar swept along a straight vertical path. V = pi*r^2*length."""
    ops = (_sk(), AddCircle("sk1", 0.0, 0.0, r),
           _sk("XZ"), AddLine("sk2", 0.0, 0.0, 0.0, length),
           Sweep("sk1", "sk2"))
    vol = PI * r * r * length
    return GeneratedPart(
        "sweep", "L3", ops, vol, (2 * r, 2 * r, length), 0,
        inside=((0.0, 0.0, length / 2.0),),
        outside=((r * 2.0, 0.0, length / 2.0), (0.0, 0.0, length * 1.2)),
        plain=("A round rod %g mm in diameter and %g mm long."
               % (2 * r, length)),
        proc=("Sketch a circle of radius %g mm and sweep it along a straight path "
              "%g mm long to make a bar." % (r, length)),
        citation="arithmetic: a profile of area A swept a distance L along a "
                 "straight path has volume A*L = pi*r^2*L")


def shell_open(a: float, b: float, c: float, t: float,
               face: str = ">Z") -> GeneratedPart:
    """A box hollowed with the TOP face open. V = abc - (a-2t)(b-2t)(c-t).

    This is also the correct answer whose near-miss (shell open on the WRONG face)
    has the identical volume, bbox and genus -- see ``discriminative``.
    """
    ops = (_sk(), AddRectangle("sk1", 0.0, 0.0, a, b), Extrude("sk1", c),
           Shell((face,), t))
    vol = a * b * c - (a - 2 * t) * (b - 2 * t) * (c - t)
    return GeneratedPart(
        "shell_open", "L3", ops, vol, (a, b, c), 0,
        # mid of a side wall; the floor; and the OPEN top must be void.
        inside=((t / 2.0, b / 2.0, c / 2.0), (a / 2.0, b / 2.0, t / 2.0)),
        outside=((a / 2.0, b / 2.0, c - t / 2.0),),
        plain=("An open-topped box %g mm by %g mm by %g mm tall, hollowed out to a "
               "%g mm wall, open at the top." % (a, b, c, t)),
        proc=("Extrude a %g by %g mm rectangle %g mm tall, then shell it to a %g mm "
              "wall removing the top (+Z) face." % (a, b, c, t)),
        citation="arithmetic: a box shelled to wall t with the top face open has "
                 "V = abc - (a-2t)(b-2t)(c-t); the bbox is unchanged",
        min_feature=t,
        note="the open top is probed as VOID and a side wall as material at its "
             "mid-plane: a shell open on the wrong face fails both")


def rib_array(rib_w: float, rib_d: float, rib_h: float, count: int,
              pitch: float) -> GeneratedPart:
    """A linear array of standalone ribs. Non-overlapping, so V = count*unit."""
    ops = (_sk(), AddRectangle("sk1", 0.0, 0.0, rib_w, rib_d),
           Extrude("sk1", rib_h),
           LinearPattern("f1", (1.0, 0.0, 0.0), count, pitch))
    unit = rib_w * rib_d * rib_h
    span = pitch * (count - 1) + rib_w
    # A void between rib 0 (x in [0, rib_w]) and rib 1 (x in [pitch, pitch+rib_w]).
    gap_x = (rib_w + pitch) / 2.0
    # genus is None, not 0: the ribs are SEPARATE bodies (disconnected), and the
    # single-component Euler->genus identity does not apply to a multi-body solid.
    return GeneratedPart(
        "linear_pattern", "L3", ops, count * unit, (span, rib_d, rib_h), None,
        inside=((rib_w / 2.0, rib_d / 2.0, rib_h / 2.0),
                (pitch * (count - 1) + rib_w / 2.0, rib_d / 2.0, rib_h / 2.0)),
        outside=((gap_x, rib_d / 2.0, rib_h / 2.0),),
        plain=("A comb of %d upright fins, each %g mm wide, %g mm deep and %g mm "
               "tall, standing in a row %g mm apart." % (count, rib_w, rib_d,
                                                         rib_h, pitch)),
        proc=("Extrude one %g by %g mm fin %g mm tall, then linear-pattern it %d "
              "times at %g mm pitch along X." % (rib_w, rib_d, rib_h, count, pitch)),
        citation="arithmetic: %d non-overlapping identical ribs, V = count*unit; "
                 "the gap between two ribs is probed as void" % count,
        min_feature=min(rib_w, rib_d),
        note="the between-rib probe is void: a pattern with the wrong count or "
             "pitch fills or misplaces that gap")


def tooth_ring(r_in: float, tooth_len: float, tooth_w: float, h: float,
               count: int) -> GeneratedPart:
    """A circular array of standalone teeth. Non-overlapping, so V = count*unit."""
    ops = (_sk(), AddRectangle("sk1", r_in, -tooth_w / 2.0, tooth_len, tooth_w),
           Extrude("sk1", h),
           CircularPattern("f1", (0.0, 0.0, 0.0, 0.0, 0.0, 1.0), count, 360.0))
    unit = tooth_len * tooth_w * h
    r_out = r_in + tooth_len
    # EXACT bbox: transform the four tooth corners by every pattern angle and take
    # the true (max - min) extent. It is NOT 2*r_out -- the corners sit at radius
    # sqrt(r_out^2 + (w/2)^2), and for a count that is not a multiple of four the
    # arrangement is not mirror-symmetric, so max_x != -min_x. (Verified against the
    # kernel to 1e-4 mm.)
    corners = ((r_in, -tooth_w / 2.0), (r_out, -tooth_w / 2.0),
               (r_out, tooth_w / 2.0), (r_in, tooth_w / 2.0))
    xs: List[float] = []
    ys: List[float] = []
    for k in range(count):
        th = 2.0 * math.pi * k / count
        cos_t, sin_t = math.cos(th), math.sin(th)
        for cx, cy in corners:
            xs.append(cx * cos_t - cy * sin_t)
            ys.append(cx * sin_t + cy * cos_t)
    dx, dy = max(xs) - min(xs), max(ys) - min(ys)
    # One tooth lies along +X from r_in to r_out at y~0; a between-teeth void is at
    # half the angular pitch, at mid-radius.
    half = math.pi / count
    mid_r = r_in + tooth_len / 2.0
    gap = (mid_r * math.cos(half), mid_r * math.sin(half), h / 2.0)
    on_tooth = (mid_r, 0.0, h / 2.0)
    return GeneratedPart(
        "circular_pattern", "L3", ops, count * unit,
        (dx, dy, h), None,
        inside=(on_tooth,),
        outside=(gap, (0.0, 0.0, h / 2.0)),
        plain=("A hub with %d teeth sticking out like a cog, each tooth %g mm long "
               "and %g mm wide, %g mm thick, on a %g mm radius."
               % (count, tooth_len, tooth_w, h, r_in)),
        proc=("Extrude one %g by %g mm tooth %g mm thick starting at radius %g mm, "
              "then circular-pattern it %d times through 360 degrees about Z."
              % (tooth_len, tooth_w, h, r_in, count)),
        citation="arithmetic: %d non-overlapping identical teeth, V = count*unit; "
                 "a between-teeth gap and the hub centre are probed as void" % count,
        min_feature=min(tooth_w, h),
        note="the centre and a between-teeth point are void: a cog with the wrong "
             "tooth count leaves material in the gap or a tooth missing")


def _interior_holes(a: float, b: float, dia: float,
                    inset: float) -> List[Tuple[float, float]]:
    """A 2x3 grid of hole centres, all clear of the edges by more than inset."""
    xs = (inset + dia, a / 2.0, a - inset - dia)
    ys = (inset + dia, b - inset - dia)
    return [(x, y) for y in ys for x in xs]


def chamfered_drilled_plate(a: float, b: float, c: float, cham: float,
                            dia: float) -> GeneratedPart:
    """DEEP CHAIN (10 ops): chamfer every edge, THEN drill six interior holes.

    Chamfer precedes the holes so the two features are volume-independent (chamfer
    after holes rounds the hole rims and is NOT additive -- measured). Exact:
    ``chamfered_box_volume(a,b,c,cham) - 6*pi*(dia/2)^2*c``.
    """
    r = dia / 2.0
    holes = _interior_holes(a, b, dia, cham + dia)
    ops = [_sk(), AddRectangle("sk1", 0.0, 0.0, a, b), Extrude("sk1", c),
           Chamfer((), cham)]
    ops += [Hole("sk1", x, y, dia, None, True, "simple") for x, y in holes]
    vol = chamfered_box_volume(a, b, c, cham) - len(holes) * PI * r * r * c
    return GeneratedPart(
        "deep_chamfer_holes", "L3", tuple(ops), vol, (a, b, c), len(holes),
        inside=((a / 2.0, b / 2.0, c / 2.0),),
        outside=tuple((x, y, c / 2.0) for x, y in holes),
        plain=("A plate %g mm by %g mm and %g mm thick, chamfered %g mm all round "
               "its edges, with %d holes of %g mm drilled through it in a grid."
               % (a, b, c, cham, len(holes), dia)),
        proc=("Extrude a %g by %g mm plate %g mm thick, chamfer every edge %g mm, "
              "then drill %d through-holes of diameter %g mm at a 3x2 interior grid."
              % (a, b, c, cham, len(holes), dia)),
        citation="arithmetic: chamfered_box_volume(a,b,c,d) minus N interior "
                 "through-holes (chamfer precedes drilling, so the two are "
                 "volume-independent -- verified additive to 1e-16)",
        min_feature=cham,
        note="%d hole axes are probed as void on an exactly chamfered blank: a "
             "10-op chain where a single misplaced hole fails one probe and "
             "nothing else" % len(holes))


def filleted_drilled_plate(a: float, b: float, c: float, rad: float,
                           dia: float) -> GeneratedPart:
    """DEEP CHAIN (10 ops): round the 4 vertical edges, THEN drill six holes.

    The fillet is on the four UPRIGHT edges only, for a measured reason: filleting
    every edge of a box makes the three-fillet spherical corner blend, and this
    kernel's tessellator emits a degenerate/non-manifold mesh for it -- the exported
    surface then FAILS ``io/gate.py`` (degenerate-faces, not-watertight), and a
    reference that does not pass its own gate is the exact v1 contamination bug this
    package exists to refuse. A vertical-edge fillet meshes cleanly and is still a
    genuine fillet feature. Chamfer-all-edges (flat cuts, no spherical corner) does
    mesh cleanly, which is why :func:`chamfered_drilled_plate` keeps every edge.

    Exact: each vertical edge replaces a square corner prism (r^2 x c) with a
    quarter-cylinder (pi/4 r^2 x c), so V = abc - 4*(1-pi/4)*r^2*c, minus the holes
    (the fillet precedes the drilling, so the two features are volume-independent).
    """
    r = dia / 2.0
    holes = _interior_holes(a, b, dia, rad + dia)
    ops = [_sk(), AddRectangle("sk1", 0.0, 0.0, a, b), Extrude("sk1", c),
           Fillet(("|Z",), rad)]
    ops += [Hole("sk1", x, y, dia, None, True, "simple") for x, y in holes]
    vol = (a * b * c - 4.0 * (1.0 - PI / 4.0) * rad * rad * c
           - len(holes) * PI * r * r * c)
    return GeneratedPart(
        "deep_fillet_holes", "L3", tuple(ops), vol, (a, b, c), len(holes),
        inside=((a / 2.0, b / 2.0, c / 2.0),),
        outside=tuple((x, y, c / 2.0) for x, y in holes),
        plain=("A plate %g mm by %g mm and %g mm thick with its four upright corner "
               "edges rounded to a %g mm radius, and %d holes of %g mm drilled "
               "through it in a grid." % (a, b, c, rad, len(holes), dia)),
        proc=("Extrude a %g by %g mm plate %g mm thick, fillet the four vertical "
              "edges %g mm, then drill %d through-holes of diameter %g mm at a 3x2 "
              "interior grid." % (a, b, c, rad, len(holes), dia)),
        citation="arithmetic: abc - 4*(1-pi/4)*r^2*c (the 4 vertical-edge fillets) "
                 "minus N interior through-holes (fillet precedes drilling, so the "
                 "two are volume-independent -- additive to machine precision)",
        min_feature=rad,
        note="%d hole axes probed as void on an exactly filleted blank" % len(holes))


#: The sampling grammar. Each entry: a factory and the seeded ranges it draws from.
#: Ranges are chosen so every sample is a real, buildable, non-degenerate part.
FACTORIES: Dict[str, Callable[[random.Random], GeneratedPart]] = {
    "revolve": lambda rng: revolve_ring(
        rng.uniform(15, 30), rng.uniform(4, 10), rng.uniform(5, 20)),
    "loft_rect": lambda rng: loft_rect_frustum(
        rng.uniform(35, 55), rng.uniform(12, 24), rng.uniform(20, 40)),
    "loft_round": lambda rng: loft_round_frustum(
        rng.uniform(18, 28), rng.uniform(6, 14), rng.uniform(20, 40)),
    "sweep": lambda rng: sweep_bar(rng.uniform(4, 9), rng.uniform(30, 70)),
    "shell_open": lambda rng: shell_open(
        rng.uniform(50, 80), rng.uniform(35, 55), rng.uniform(18, 30),
        rng.uniform(2.5, 4.0)),
    "linear_pattern": lambda rng: rib_array(
        rng.uniform(6, 12), rng.uniform(6, 12), rng.uniform(15, 30),
        rng.randint(4, 7), rng.uniform(16, 24)),
    "circular_pattern": lambda rng: tooth_ring(
        rng.uniform(20, 32), rng.uniform(8, 16), rng.uniform(5, 9),
        rng.uniform(8, 14), rng.randint(6, 10)),
    "deep_chamfer_holes": lambda rng: chamfered_drilled_plate(
        rng.uniform(60, 90), rng.uniform(40, 55), rng.uniform(10, 16),
        rng.uniform(1.5, 3.0), rng.uniform(5, 8)),
    "deep_fillet_holes": lambda rng: filleted_drilled_plate(
        rng.uniform(60, 90), rng.uniform(40, 55), rng.uniform(10, 16),
        rng.uniform(2.0, 3.5), rng.uniform(5, 8)),
}


def sample(seed: int, split: str, families: Optional[List[str]] = None) -> List[Brief]:
    """Deterministically sample ONE part per family at ``seed`` -> two briefs each.

    A different seed is a different, equally-valid corpus. This is the property the
    held-out split relies on: the dev and held-out splits are two seeds through the
    same grammar, so they cannot drift apart the way two hand-written files do.
    """
    fams = families if families is not None else list(FACTORIES)
    briefs: List[Brief] = []
    for i, fam in enumerate(fams):
        # zlib.crc32 of the family name -- a STABLE checksum, not the built-in
        # hash(), whose string randomisation (PYTHONHASHSEED) would make "seed
        # 7919" a different corpus every process and break the held-out split.
        fam_key = zlib.crc32(fam.encode("utf-8")) & 0xFFFFFFFF
        rng = random.Random((seed * 1000003) ^ fam_key ^ i)
        part = FACTORIES[fam](rng)
        bid = "gen_%s_s%d" % (fam, seed)
        briefs.extend(part.briefs(bid, split))
    return briefs


def all_briefs(seed: int, split: str) -> List[Brief]:
    """Every family at one seed. The dev corpus's generated section."""
    return sample(seed, split)
