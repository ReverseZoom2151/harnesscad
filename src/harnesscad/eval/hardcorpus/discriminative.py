"""THE NOVEL CONTRIBUTION: briefs whose plausible WRONG answer passes the field.

Every case here is a matched pair -- a CORRECT op stream and a NEAR-MISS, the wrong
answer a competent model would actually emit -- built so that the near-miss defeats
the metric the published benchmarks grade with. Not "a random broken part": a
specific, plausible, wrong part that Text2CAD-Bench's IoU + Chamfer and MUSE's
watertight + manifold + volume + genus call CORRECT, and that our measured oracle
catches on a single point.

Each case reports, side by side, what every weak metric says (it should PASS) and
what the oracle says (FAIL, and exactly where). That table is the headline of the
package. A benchmark only a measured oracle can score is a benchmark nobody with a
weak oracle can pass -- which is the entire proposition.

THE CASES, AND WHAT EACH ONE DEFEATS (all figures MEASURED, this build)
----------------------------------------------------------------------
``dia_hole``      an 8 mm hole where 12 mm was demanded. IoU 0.973, Chamfer within
                  tolerance, valid -- passes EVERYTHING the field runs. The oracle
                  probes the wall at mid-radius (r = 5 mm): void in the 12 mm part,
                  MATERIAL in the 8 mm part.
``pos_hole``      a hole bored at x = 40 instead of x = 20. Volume, bbox, genus,
                  watertight and manifold are IDENTICAL to the micron; IoU 0.935;
                  Chamfer within tolerance. The oracle probes the demanded axis
                  (20, 20): void in the correct part, material in the near-miss.
                  ``io/gate.py`` carries its own test pinning that this displaced
                  hole passes the gate -- so this is the field's blindness measured
                  on the field's own instrument.
``cbore_plain``   a plain hole where a counterbore was specified. IoU 0.979,
                  Chamfer within tolerance, valid. The oracle probes the counterbore
                  step (r = 6 mm, near the top face): void in the counterbore,
                  material in the plain hole.
``fillet_edges``  a fillet rounded on the TOP edges when the vertical edges were
                  asked for. IoU 0.985, Chamfer within tolerance, valid. The oracle
                  probes a rounded-away vertical corner: void when the vertical edge
                  is filleted, material when only the top is.
``shell_face``    a box shelled OPEN ON THE WRONG FACE. This one is aimed straight
                  at MUSE: volume, bbox, genus, watertight and manifold are ALL
                  EXACTLY EQUAL between the two -- MUSE's entire geometric stage is
                  blind to it, which is the constancy
                  ``eval/bench/harness/pressure_correlation.py`` measured over 208
                  attempts. IoU (0.506) does catch this one; it is in the table to
                  show that the geometric-check FAMILY, on which MUSE gates, cannot.

WHY THE WEAK METRICS ARE RUN IN THEIR STRONGEST FORM
----------------------------------------------------
The IoU here is the EXACT OCCT boolean, not a voxel estimate; the thresholds
(``weak.IOU_MATCH = 0.90``, ``weak.CHAMFER_MATCH = 0.01``) were pre-registered in
``eval/corpus/shape`` before any near-miss was scored and have not moved. If a
near-miss passes a bar this tight and exact, no bar anybody would dare to ship can
catch it -- and the alternative, raising the bar until it does, starts failing
CORRECT parts, because a correct rebuild on a sampled engine already disagrees with
itself by a percent or two (``eval/corpus/shape`` measured and documented exactly
that, and refused to move its threshold to hide the 0.957 hole). The honest report
is this table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from harnesscad.core.cisp.ops import (AddRectangle, Extrude, Fillet, Hole,
                                      NewSketch, Op, Shell)
from harnesscad.eval.corpus.spec import Brief, Source, Split, Vec3
from harnesscad.eval.hardcorpus import oracle, weak

__all__ = ["NearMiss", "CASES", "cases", "grade_case", "CaseVerdict", "table"]

PI = math.pi


def _sk(plane: str = "XY") -> Op:
    return NewSketch(plane)


@dataclass(frozen=True)
class NearMiss:
    """A correct part, its plausible wrong twin, and the brief that separates them.

    ``brief``       carries the CORRECT part's closed-form truth and, crucially, the
                    probe points that the near-miss violates. The oracle grades an
                    answer against this.
    ``correct``     the reference op stream (== ``brief.reference``). The positive
                    control: every weak metric AND the oracle must pass it.
    ``near``        the plausible wrong answer.
    ``defeats``     which weak metric(s) this near-miss is aimed at.
    ``near_text``   what the wrong answer got wrong, in one line, for the report.
    """

    id: str
    level: str
    family: str
    brief: Brief
    near: Tuple[Op, ...]
    defeats: str
    near_text: str

    @property
    def correct(self) -> Tuple[Op, ...]:
        return self.brief.reference


# --------------------------------------------------------------------------- #
# the cases
# --------------------------------------------------------------------------- #
def _dia_hole(split: str) -> NearMiss:
    a, b, c, d_ok, d_bad = 60.0, 40.0, 12.0, 12.0, 8.0
    correct = (_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
               Hole("sk1", 20.0, 20.0, d_ok, None, True, "simple"))
    near = (_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
            Hole("sk1", 20.0, 20.0, d_bad, None, True, "simple"))
    r_ok = d_ok / 2.0
    brief = Brief(
        id="disc_dia_hole", split=split, source=Source.ANALYTIC,
        citation="arithmetic: V = abc - pi*(d/2)^2*c; the wall at mid-radius r=5 mm "
                 "is void for a 12 mm hole and solid for an 8 mm hole",
        text=("A %g by %g mm plate %g mm thick with a single %g mm diameter hole "
              "drilled through the middle at (20, 20)." % (a, b, c, d_ok)),
        reference=correct, volume=a * b * c - PI * r_ok * r_ok * c,
        bbox=(a, b, c), genus=1,
        inside=((5.0, 5.0, c / 2.0),),
        # r = 5 mm from the hole axis: inside a 12 mm hole (r=6), outside an 8 mm one.
        outside=((20.0, 20.0, c / 2.0), (25.0, 20.0, c / 2.0)),
        note="the 25,20 probe is 5 mm off the hole axis: void iff the hole is at "
             "least 10 mm across")
    return NearMiss("dia_hole", "L2", "hole", brief, near,
                    defeats="IoU (0.973), Chamfer, watertight, manifold, invalidity",
                    near_text="hole drilled 8 mm instead of the demanded 12 mm")


def _pos_hole(split: str) -> NearMiss:
    a, b, c, d = 60.0, 40.0, 12.0, 10.0
    correct = (_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
               Hole("sk1", 20.0, 20.0, d, None, True, "simple"))
    near = (_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
            Hole("sk1", 40.0, 20.0, d, None, True, "simple"))
    r = d / 2.0
    brief = Brief(
        id="disc_pos_hole", split=split, source=Source.ANALYTIC,
        citation="arithmetic: a hole of fixed size at (20,20) vs (40,20) has the "
                 "SAME volume, bbox and genus; only the axis point distinguishes them",
        text=("A %g by %g mm plate %g mm thick with a %g mm hole drilled through at "
              "(20, 20), measured from the corner at the origin." % (a, b, c, d)),
        reference=correct, volume=a * b * c - PI * r * r * c,
        bbox=(a, b, c), genus=1,
        inside=((40.0, 20.0, c / 2.0),),        # where the WRONG hole put its void
        outside=((20.0, 20.0, c / 2.0),),        # the demanded axis: must be void
        note="volume, bbox, genus, watertight and manifold are identical to the "
             "micron; only the demanded-axis probe separates the two")
    return NearMiss("pos_hole", "L2", "hole", brief, near,
                    defeats="IoU (0.935), Chamfer, watertight, manifold, volume, genus",
                    near_text="hole bored at x = 40 instead of x = 20 "
                              "(same size, same volume, same bbox)")


def _cbore_plain(split: str) -> NearMiss:
    a, b, c = 60.0, 40.0, 12.0
    dh, dcb, depth = 8.0, 16.0, 4.0
    cx, cy = 30.0, 20.0
    correct = (_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
               Hole("sk1", cx, cy, dh, None, True, "counterbore", dcb, depth))
    near = (_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
            Hole("sk1", cx, cy, dh, None, True, "simple"))
    r_cb = dcb / 2.0
    r_h = dh / 2.0
    vol = a * b * c - PI * r_cb * r_cb * depth - PI * r_h * r_h * (c - depth)
    brief = Brief(
        id="disc_cbore_plain", split=split, source=Source.ANALYTIC,
        citation="arithmetic: a counterbore removes pi*r_cb^2*depth + pi*r_h^2*"
                 "(c-depth); the wide step is void near the top and solid in a "
                 "plain hole",
        text=("A %g by %g mm plate %g mm thick with an %g mm hole through the "
              "centre, counterbored to %g mm diameter and %g mm deep from the top "
              "face." % (a, b, c, dh, dcb, depth)),
        reference=correct, volume=vol, bbox=(a, b, c), genus=1,
        inside=((cx, cy, depth / 2.0),),         # NOTE: air in a counterbore; see below
        # r = 6 mm from the axis, 1 mm below the top face: inside the 16 mm
        # counterbore (r=8), solid in an 8 mm plain hole (r=4).
        outside=((cx + 6.0, cy, c - 1.0),),
        note="the counterbore step is probed at r=6 mm near the top face: void in "
             "a counterbore, material in a plain hole")
    # The 'inside' probe above is on the hole axis and is actually VOID, not solid;
    # replace it with a genuine material point clear of the hole.
    brief = _replace_probes(brief, inside=((5.0, 5.0, c / 2.0),))
    return NearMiss("cbore_plain", "L3", "hole", brief, near,
                    defeats="IoU (0.979), Chamfer, watertight, manifold, invalidity",
                    near_text="a plain hole where a counterbore was specified")


def _fillet_edges(split: str) -> NearMiss:
    a, b, c, r = 60.0, 40.0, 12.0, 3.0
    correct = (_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
               Fillet(("|Z",), r))
    near = (_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
            Fillet((">Z",), r))
    vol = a * b * c - 4.0 * (1.0 - PI / 4.0) * r * r * c
    brief = Brief(
        id="disc_fillet_edges", split=split, source=Source.ANALYTIC,
        citation="arithmetic: rounding the 4 vertical edges removes "
                 "4*(1-pi/4)*r^2*c; the corner at mid-height is void iff the "
                 "VERTICAL edge is the one rounded",
        text=("A %g by %g mm plate %g mm thick with the four upright (vertical) "
              "corner edges rounded to a %g mm radius." % (a, b, c, r)),
        reference=correct, volume=vol, bbox=(a, b, c), genus=0,
        inside=((a / 2.0, b / 2.0, c / 2.0),),
        # A point deep in a vertical corner at mid-height: rounded away by a |Z
        # fillet, still solid if only the top (>Z) edges were filleted.
        outside=((0.3, 0.3, c / 2.0),),
        note="probe a vertical corner at mid-height: filleting the top edges "
             "instead leaves this material in place")
    return NearMiss("fillet_edges", "L3", "fillet", brief, near,
                    defeats="IoU (0.985), Chamfer, watertight, manifold, invalidity",
                    near_text="fillet applied to the top edges, not the vertical "
                              "edges the brief named")


def _shell_face(split: str) -> NearMiss:
    a, b, c, t = 60.0, 40.0, 20.0, 3.0
    correct = (_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
               Shell((">Z",), t))
    near = (_sk(), AddRectangle("sk1", 0, 0, a, b), Extrude("sk1", c),
            Shell(("<Z",), t))
    vol = a * b * c - (a - 2 * t) * (b - 2 * t) * (c - t)
    brief = Brief(
        id="disc_shell_face", split=split, source=Source.ANALYTIC,
        citation="arithmetic: a box shelled to wall t open on ONE face has "
                 "V = abc - (a-2t)(b-2t)(c-t) whichever face is opened -- so volume, "
                 "bbox and genus cannot tell which face it was",
        text=("An open-topped box %g by %g by %g mm tall, hollowed to a %g mm wall "
              "with the TOP face left open." % (a, b, c, t)),
        reference=correct, volume=vol, bbox=(a, b, c), genus=0,
        inside=((a / 2.0, b / 2.0, t / 2.0),),   # the FLOOR is solid
        outside=((a / 2.0, b / 2.0, c - t / 2.0),),  # the open TOP is void
        min_feature=t,
        note="THE MUSE CASE: volume, bbox, genus, watertight and manifold are all "
             "exactly equal to the wrong-face shell. Only the floor/ceiling probes "
             "separate them")
    return NearMiss("shell_face", "L3", "shell", brief, near,
                    defeats="watertight, manifold, volume, genus, invalidity "
                            "(the entire MUSE geometric stage)",
                    near_text="box opened on the BOTTOM face instead of the top "
                              "(identical volume, bbox and topology)")


def _replace_probes(brief: Brief, inside=None, outside=None) -> Brief:
    import dataclasses
    kw = {}
    if inside is not None:
        kw["inside"] = inside
    if outside is not None:
        kw["outside"] = outside
    return dataclasses.replace(brief, **kw)


#: The ordered case factories. A split is threaded through so the same cases can be
#: minted for dev or held-out.
_FACTORIES = (_dia_hole, _pos_hole, _cbore_plain, _fillet_edges, _shell_face)


def cases(split: str = Split.DEV) -> List[NearMiss]:
    return [f(split) for f in _FACTORIES]


#: The dev cases, built once.
CASES: Tuple[NearMiss, ...] = tuple(cases(Split.DEV))


# --------------------------------------------------------------------------- #
# grading: BOTH oracles, on BOTH answers, always
# --------------------------------------------------------------------------- #
@dataclass
class CaseVerdict:
    """The four numbers that make the point: weak/oracle x correct/near-miss."""

    id: str
    level: str
    defeats: str
    near_text: str
    weak_correct: Dict = field(default_factory=dict)
    oracle_correct: Dict = field(default_factory=dict)
    weak_near: Dict = field(default_factory=dict)
    oracle_near: Dict = field(default_factory=dict)

    @property
    def demonstrates_gap(self) -> bool:
        """The near-miss passes Text2CAD-Bench's FULL grader (valid + IoU + Chamfer)
        but fails the oracle. The strongest form of the claim."""
        return bool(self.weak_near.get("passes") and not self.oracle_near.get("solved"))

    @property
    def defeats_geometric_family(self) -> bool:
        """The near-miss passes MUSE's geometric stage (a valid, watertight,
        manifold solid) but fails the oracle.

        Weaker than :attr:`demonstrates_gap` -- IoU may still catch the part -- but
        it is the property MUSE actually gates on, and
        ``eval/bench/harness/pressure_correlation.py`` measured that this family is
        constant across correct and incorrect parts. Every case has at least this.
        """
        return bool(self.weak_near.get("valid") and not self.oracle_near.get("solved"))

    @property
    def scored(self) -> str:
        """Which claim this case makes: 'full' (beats IoU+Chamfer too) or 'muse'."""
        if self.demonstrates_gap:
            return "full"
        if self.defeats_geometric_family:
            return "muse"
        return "none"

    @property
    def controls_hold(self) -> bool:
        """The correct answer passes BOTH -- so a FAIL on the near-miss is about the
        near-miss, not a broken brief."""
        return bool(self.weak_correct.get("passes")
                    and self.oracle_correct.get("solved"))

    def to_dict(self) -> dict:
        return {"id": self.id, "level": self.level, "defeats": self.defeats,
                "near_text": self.near_text,
                "weak_correct": self.weak_correct,
                "oracle_correct": self.oracle_correct,
                "weak_near": self.weak_near, "oracle_near": self.oracle_near,
                "demonstrates_gap": self.demonstrates_gap,
                "defeats_geometric_family": self.defeats_geometric_family,
                "scored": self.scored,
                "controls_hold": self.controls_hold}


def grade_case(nm: NearMiss) -> CaseVerdict:
    """Grade the correct answer and the near-miss on BOTH oracles."""
    return CaseVerdict(
        id=nm.id, level=nm.level, defeats=nm.defeats, near_text=nm.near_text,
        weak_correct=weak.score_weak(nm.correct, nm.correct).to_dict(),
        oracle_correct=oracle.grade(nm.brief, nm.correct).to_dict(),
        weak_near=weak.score_weak(nm.near, nm.correct).to_dict(),
        oracle_near=oracle.grade(nm.brief, nm.near).to_dict())


def table(verdicts: Optional[List[CaseVerdict]] = None) -> str:
    """The headline table: near-miss vs every weak metric vs the measured oracle."""
    if verdicts is None:
        verdicts = [grade_case(nm) for nm in CASES]
    rows: List[str] = []
    rows.append("DISCRIMINATIVE TABLE -- the plausible wrong answer vs the graders")
    rows.append("=" * 78)
    rows.append("Each row: a near-miss that the field's oracle scores CORRECT and "
                "ours scores WRONG.")
    rows.append("")
    hdr = ("%-13s %-6s | %-7s %-8s %-6s | %-7s %-8s | verdict"
           % ("case", "level", "valid", "IoU", "cham", "oracle", "probe"))
    rows.append(hdr)
    rows.append("-" * 78)
    for v in verdicts:
        wn = v.weak_near
        on = v.oracle_near
        iou = wn.get("iou")
        cham = wn.get("chamfer_rel")
        if not v.controls_hold:
            verdict = "BROKEN-CONTROL"
        elif v.demonstrates_gap:
            verdict = "GAP (beats IoU+Cham)"
        elif v.defeats_geometric_family:
            verdict = "MUSE-GAP"
        else:
            verdict = "no-gap"
        rows.append(
            "%-13s %-6s | %-7s %-8s %-6s | %-7s %-8s | %s"
            % (v.id, v.level,
               "PASS" if wn.get("valid") else "fail",
               "%.3f" % iou if iou is not None else "n/a",
               "%.3f" % cham if cham is not None else "n/a",
               "FAIL" if not on.get("solved") else "pass",
               "fail" if not on.get("probes_ok") else "ok",
               verdict))
    rows.append("-" * 78)
    rows.append("valid/IoU/cham are the field's grader (Text2CAD-Bench IoU+Chamfer, "
                "MUSE valid).")
    rows.append("oracle is the measured verdict; 'probe' is the exact point-membership "
                "check.")
    rows.append("A 'GAP' row is a wrong part the field passes and only measurement "
                "catches.")
    return "\n".join(rows)
