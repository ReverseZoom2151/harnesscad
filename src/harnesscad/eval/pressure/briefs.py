"""The pressure-test brief corpus — 28 briefs, deterministic and checked in.

Each brief carries its own GROUND TRUTH, and the ground truth is *geometric*,
not diagnostic: bounding box, volume, and a set of probe points that must lie
inside / outside the finished solid. The probes are evaluated against the F-rep
backend's exact signed-distance field (``FRepBackend.field()``), so they are not
subject to mesh discretisation error and they catch the failures a volume check
cannot -- a hole placed outside the plate, a shell that dilated instead of
hollowing, a fillet that ate the part.

NEITHER LOOP EVER SEES ANY OF THIS. The grader is the only consumer. That is the
whole point: an arm cannot win by being handed the answer key through its
feedback channel, so a difference in solve rate is a difference in the model's
ability to *repair its own geometry*.

``reference`` is a hand-written op stream that satisfies the brief. It is never
shown to a model; it exists so ``test_pressure`` can prove every brief in the
corpus is actually solvable and that the grader accepts a correct answer (a
corpus with an unsolvable brief would silently depress both arms and make the
comparison meaningless).

Trap briefs
-----------
Five briefs (``trap_*``) state a value whose naive reading the harness's own
feasibility rules reject. They are the briefs the claim lives or dies on --
under ``verify_level="core"`` the F-rep backend accepts them all without a
murmur, so the blind arm is never told anything is wrong. One of them
(``trap_hole_oversize``) is a deliberate probe of a rule the harness gets WRONG:
a 12 mm through hole in a 10 mm plate is routine machining, but the precheck
compares the hole's diameter against the *plate thickness* and calls it
infeasible. The ground truth there is the geometry the brief asked for, so the
grader scores the harness's false positive honestly -- as a loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
# ground truth
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OpSpec:
    """An op-level assertion: "the plan must contain <count> <tag> ops, and each
    of them must have parameters inside these ranges"."""

    tag: str
    count_min: int = 1
    count_max: Optional[int] = None          # None = no upper bound
    params: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"tag": self.tag, "count_min": self.count_min,
                "count_max": self.count_max,
                "params": {k: list(v) for k, v in self.params.items()}}


@dataclass(frozen=True)
class Expect:
    """The channel-blind acceptance test for a brief.

    ``bbox``     expected (dx, dy, dz) of the finished solid, within ``bbox_tol``
                 (relative). ``None`` skips the check.
    ``volume``   inclusive (min, max) on the solid's volume in mm^3. Mesh volume
                 carries ~1% discretisation error, so the bands are generous;
                 they exist to catch gross errors (a part 4x too big), not to
                 measure precision.
    ``inside``   probe points that MUST be solid material (sdf < -tol).
    ``outside``  probe points that MUST be air (sdf > +tol). This is what
                 actually proves a hole got cut where it was asked for.
    ``ops``      op-level assertions (a shell must exist, and its thickness must
                 be feasible).
    ``probe_tol`` the margin, in mm, a probe must clear the surface by.
    """

    bbox: Optional[Vec3] = None
    bbox_tol: float = 0.04
    volume: Optional[Tuple[float, float]] = None
    inside: Tuple[Vec3, ...] = ()
    outside: Tuple[Vec3, ...] = ()
    ops: Tuple[OpSpec, ...] = ()
    probe_tol: float = 0.35

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox) if self.bbox else None,
            "bbox_tol": self.bbox_tol,
            "volume": list(self.volume) if self.volume else None,
            "inside": [list(p) for p in self.inside],
            "outside": [list(p) for p in self.outside],
            "ops": [o.to_dict() for o in self.ops],
            "probe_tol": self.probe_tol,
        }


@dataclass(frozen=True)
class Brief:
    id: str
    category: str
    difficulty: int                 # 1 easy .. 4 trap
    text: str                       # the ONLY thing the model is shown
    expect: Expect
    reference: Tuple[dict, ...]     # a known-good solution; never shown
    trap: bool = False
    note: str = ""                  # why this brief exists (for the report)

    def to_dict(self) -> dict:
        return {"id": self.id, "category": self.category,
                "difficulty": self.difficulty, "text": self.text,
                "trap": self.trap, "note": self.note,
                "expect": self.expect.to_dict(),
                "reference": [dict(o) for o in self.reference]}


# --------------------------------------------------------------------------- #
# helpers for building the corpus
# --------------------------------------------------------------------------- #
def _sk(plane: str = "XY") -> dict:
    return {"op": "new_sketch", "plane": plane}


def _rect(sk: str, x: float, y: float, w: float, h: float) -> dict:
    return {"op": "add_rectangle", "sketch": sk, "x": x, "y": y, "w": w, "h": h}


def _circ(sk: str, cx: float, cy: float, r: float) -> dict:
    return {"op": "add_circle", "sketch": sk, "cx": cx, "cy": cy, "r": r}


def _ext(sk: str, d: float) -> dict:
    return {"op": "extrude", "sketch": sk, "distance": d}


def _hole(x: float, y: float, dia: float) -> dict:
    return {"op": "hole", "face_or_sketch": "solid", "x": x, "y": y,
            "diameter": dia, "through": True}


def _band(nominal: float, rel: float = 0.10) -> Tuple[float, float]:
    return (nominal * (1.0 - rel), nominal * (1.0 + rel))


def _cyl_vol(d: float, h: float) -> float:
    return math.pi * (d / 2.0) ** 2 * h


# --------------------------------------------------------------------------- #
# the corpus
# --------------------------------------------------------------------------- #
BRIEFS: Tuple[Brief, ...] = (

    # -- tier 1: plates and primitives ------------------------------------- #
    Brief(
        id="plate_60x40x5", category="plate", difficulty=1,
        text="A flat rectangular plate 60 mm long, 40 mm wide and 5 mm thick.",
        expect=Expect(
            bbox=(60.0, 40.0, 5.0), volume=_band(12000.0),
            inside=((30.0, 20.0, 2.5), (2.0, 2.0, 2.5)),
            outside=((30.0, 20.0, 8.0), (70.0, 20.0, 2.5)),
            ops=(OpSpec("extrude", params={"distance": (4.9, 5.1)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 60, 40), _ext("sk1", 5)),
    ),
    Brief(
        id="plate_square_25", category="plate", difficulty=1,
        text="A square plate, 25 mm by 25 mm, 3 mm thick.",
        expect=Expect(
            bbox=(25.0, 25.0, 3.0), volume=_band(1875.0),
            inside=((12.5, 12.5, 1.5),), outside=((12.5, 12.5, 5.0),),
            ops=(OpSpec("extrude", params={"distance": (2.9, 3.1)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 25, 25), _ext("sk1", 3)),
    ),
    Brief(
        id="disc_d30_h8", category="plate", difficulty=1,
        text="A solid cylindrical disc, 30 mm in diameter and 8 mm tall.",
        expect=Expect(
            bbox=(30.0, 30.0, 8.0), volume=_band(_cyl_vol(30, 8), 0.12),
            inside=((0.0, 0.0, 4.0),),
            outside=((0.0, 0.0, 10.0), (20.0, 20.0, 4.0)),
            ops=(OpSpec("add_circle", params={"r": (14.5, 15.5)}),
                 OpSpec("extrude", params={"distance": (7.9, 8.1)})),
        ),
        reference=(_sk(), _circ("sk1", 0, 0, 15), _ext("sk1", 8)),
    ),
    Brief(
        id="bar_100x10x10", category="plate", difficulty=1,
        text="A square bar 100 mm long with a 10 mm by 10 mm cross-section.",
        expect=Expect(
            bbox=(100.0, 10.0, 10.0), volume=_band(10000.0),
            inside=((50.0, 5.0, 5.0),), outside=((50.0, 15.0, 5.0),),
            ops=(OpSpec("extrude"),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 100, 10), _ext("sk1", 10)),
    ),
    Brief(
        id="plate_thin_80x50x2", category="plate", difficulty=1,
        text="A thin shim plate, 80 mm by 50 mm, only 2 mm thick.",
        expect=Expect(
            bbox=(80.0, 50.0, 2.0), volume=_band(8000.0),
            inside=((40.0, 25.0, 1.0),), outside=((40.0, 25.0, 4.0),),
            ops=(OpSpec("extrude", params={"distance": (1.9, 2.1)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 80, 50), _ext("sk1", 2)),
    ),

    # -- tier 2: holes ------------------------------------------------------ #
    Brief(
        id="plate_hole_centre", category="hole", difficulty=2,
        text=("A plate 60 mm by 40 mm and 12 mm thick, with a single 8 mm "
              "diameter hole drilled straight through the middle of it."),
        expect=Expect(
            bbox=(60.0, 40.0, 12.0), volume=_band(28800.0 - _cyl_vol(8, 12), 0.06),
            inside=((5.0, 5.0, 6.0), (55.0, 35.0, 6.0)),
            outside=((30.0, 20.0, 6.0),),          # the hole must actually be there
            ops=(OpSpec("hole", count_max=1, params={"diameter": (7.5, 8.5)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 60, 40), _ext("sk1", 12),
                   _hole(30, 20, 8)),
        note="the simplest test that a hole lands where the brief said it would",
    ),
    Brief(
        id="plate_hole_offcentre", category="hole", difficulty=2,
        text=("A 50 mm by 50 mm plate, 10 mm thick. Drill one 9 mm diameter "
              "hole right through it, centred 15 mm from the left edge and "
              "15 mm from the bottom edge."),
        expect=Expect(
            bbox=(50.0, 50.0, 10.0), volume=_band(25000.0 - _cyl_vol(9, 10), 0.06),
            inside=((40.0, 40.0, 5.0), (30.0, 15.0, 5.0)),
            outside=((15.0, 15.0, 5.0),),
            ops=(OpSpec("hole", count_max=1, params={"diameter": (8.5, 9.5),
                                                     "x": (14.0, 16.0),
                                                     "y": (14.0, 16.0)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 50, 50), _ext("sk1", 10),
                   _hole(15, 15, 9)),
        note="placement, not just presence: the hole has a stated coordinate",
    ),
    Brief(
        id="plate_hole_four", category="hole", difficulty=3,
        text=("A mounting plate 80 mm by 60 mm and 10 mm thick, with four 7 mm "
              "diameter through-holes, one 12 mm in from each corner in both "
              "directions."),
        expect=Expect(
            bbox=(80.0, 60.0, 10.0),
            volume=_band(48000.0 - 4 * _cyl_vol(7, 10), 0.06),
            inside=((40.0, 30.0, 5.0),),
            outside=((12.0, 12.0, 5.0), (68.0, 12.0, 5.0),
                     (12.0, 48.0, 5.0), (68.0, 48.0, 5.0)),
            ops=(OpSpec("hole", count_min=4, count_max=4,
                        params={"diameter": (6.5, 7.5)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 80, 60), _ext("sk1", 10),
                   _hole(12, 12, 7), _hole(68, 12, 7),
                   _hole(12, 48, 7), _hole(68, 48, 7)),
        note="four holes, four coordinates: the arithmetic is the hard part",
    ),
    Brief(
        id="disc_bore", category="hole", difficulty=2,
        text=("A cylindrical spacer 40 mm in outside diameter and 10 mm tall, "
              "with a 14 mm diameter bore straight through the centre."),
        expect=Expect(
            bbox=(40.0, 40.0, 10.0),
            volume=_band(_cyl_vol(40, 10) - _cyl_vol(14, 10), 0.12),
            inside=((15.0, 0.0, 5.0),),
            outside=((0.0, 0.0, 5.0),),
            ops=(OpSpec("hole", count_max=1, params={"diameter": (13.5, 14.5)}),),
        ),
        reference=(_sk(), _circ("sk1", 0, 0, 20), _ext("sk1", 10),
                   _hole(0, 0, 14)),
    ),
    Brief(
        id="strip_hole_row", category="hole", difficulty=3,
        text=("A strip 100 mm by 20 mm and 8 mm thick with a row of three 6 mm "
              "through-holes along its centreline, at x = 20 mm, x = 50 mm and "
              "x = 80 mm (y = 10 mm)."),
        expect=Expect(
            bbox=(100.0, 20.0, 8.0),
            volume=_band(16000.0 - 3 * _cyl_vol(6, 8), 0.06),
            inside=((5.0, 10.0, 4.0), (35.0, 10.0, 4.0)),
            outside=((20.0, 10.0, 4.0), (50.0, 10.0, 4.0), (80.0, 10.0, 4.0)),
            ops=(OpSpec("hole", count_min=3, count_max=3,
                        params={"diameter": (5.5, 6.5), "y": (9.0, 11.0)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 100, 20), _ext("sk1", 8),
                   _hole(20, 10, 6), _hole(50, 10, 6), _hole(80, 10, 6)),
    ),

    # -- tier 3: brackets and booleans -------------------------------------- #
    Brief(
        id="l_bracket", category="bracket", difficulty=3,
        text=("An L-shaped bracket. The base is a plate 60 mm by 40 mm and 6 mm "
              "thick lying flat. Rising from one 40 mm edge is an upright wall, "
              "also 6 mm thick, 30 mm tall and the full 40 mm wide. Union the "
              "two into one body."),
        expect=Expect(
            volume=_band(60 * 40 * 6 + 6 * 40 * 24, 0.20),
            inside=((30.0, 20.0, 3.0), (3.0, 20.0, 20.0)),
            outside=((30.0, 20.0, 20.0),),
            ops=(OpSpec("boolean", params={}), OpSpec("extrude", count_min=2)),
        ),
        reference=(_sk("XY"), _rect("sk1", 0, 0, 60, 40), _ext("sk1", 6),
                   _sk("XY"), _rect("sk2", 0, 0, 6, 40), _ext("sk2", 30),
                   {"op": "boolean", "kind": "union", "target": "f1", "tool": "f2"}),
        note="two solids and an explicit boolean: the first plan with topology",
    ),
    Brief(
        id="step_block", category="bracket", difficulty=3,
        text=("A stepped block, 60 mm by 40 mm overall. One half (x from 0 to "
              "30 mm) is 10 mm tall; the other half (x from 30 to 60 mm) is "
              "20 mm tall. Build it as one body."),
        expect=Expect(
            volume=_band(60 * 40 * 10 + 30 * 40 * 10, 0.12),
            inside=((45.0, 20.0, 15.0), (15.0, 20.0, 5.0)),
            outside=((15.0, 20.0, 15.0),),
            ops=(OpSpec("boolean", params={}),),
        ),
        reference=(_sk("XY"), _rect("sk1", 0, 0, 60, 40), _ext("sk1", 10),
                   _sk("XY"), _rect("sk2", 30, 0, 30, 40), _ext("sk2", 20),
                   {"op": "boolean", "kind": "union", "target": "f1", "tool": "f2"}),
        note=("CISP extrudes from the sketch plane only -- there is no offset "
              "start -- so a 'cut the top off' step must be built as a union of "
              "two blocks. The brief states the two levels rather than the cut."),
    ),
    Brief(
        id="slotted_block", category="bracket", difficulty=3,
        text=("A block 40 mm by 40 mm by 20 mm with a 10 mm wide slot cut all "
              "the way through it, running the full 40 mm length, centred "
              "across the width and open at the top."),
        expect=Expect(
            volume=_band(40 * 40 * 20 - 10 * 40 * 20, 0.25),
            inside=((20.0, 3.0, 10.0), (20.0, 37.0, 10.0)),
            outside=((20.0, 20.0, 10.0),),
            ops=(OpSpec("boolean"),),
        ),
        reference=(_sk("XY"), _rect("sk1", 0, 0, 40, 40), _ext("sk1", 20),
                   _sk("XY"), _rect("sk2", 0, 15, 40, 10), _ext("sk2", 20),
                   {"op": "boolean", "kind": "cut", "target": "f1", "tool": "f2"}),
    ),
    Brief(
        id="spacer_bore", category="bracket", difficulty=2,
        text=("A cubic spacer block, 30 mm on every side, with a 16 mm diameter "
              "bore through the centre from top to bottom."),
        expect=Expect(
            bbox=(30.0, 30.0, 30.0),
            volume=_band(27000.0 - _cyl_vol(16, 30), 0.08),
            inside=((3.0, 3.0, 15.0),),
            outside=((15.0, 15.0, 15.0),),
            ops=(OpSpec("hole", count_max=1, params={"diameter": (15.5, 16.5)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 30, 30), _ext("sk1", 30),
                   _hole(15, 15, 16)),
    ),

    # -- tier 4: flanges ---------------------------------------------------- #
    Brief(
        id="flange_round", category="flange", difficulty=4,
        text=("A round flange: a disc 80 mm in outside diameter and 8 mm thick, "
              "with a 30 mm diameter bore through the centre, plus four 7 mm "
              "bolt holes on a 60 mm bolt circle (so at x,y = (+/-30, 0) and "
              "(0, +/-30)), all going right through."),
        expect=Expect(
            bbox=(80.0, 80.0, 8.0),
            volume=_band(_cyl_vol(80, 8) - _cyl_vol(30, 8) - 4 * _cyl_vol(7, 8), 0.12),
            inside=((20.0, 20.0, 4.0),),
            outside=((0.0, 0.0, 4.0), (30.0, 0.0, 4.0), (-30.0, 0.0, 4.0),
                     (0.0, 30.0, 4.0), (0.0, -30.0, 4.0)),
            ops=(OpSpec("hole", count_min=5, count_max=5),),
        ),
        reference=(_sk(), _circ("sk1", 0, 0, 40), _ext("sk1", 8),
                   _hole(0, 0, 30), _hole(30, 0, 7), _hole(-30, 0, 7),
                   _hole(0, 30, 7), _hole(0, -30, 7)),
        note="the hardest non-trap brief: five holes, signed coordinates",
    ),
    Brief(
        id="flange_square", category="flange", difficulty=3,
        text=("A square flange plate, 70 mm by 70 mm and 8 mm thick, with a "
              "25 mm diameter bore through the middle and a 6 mm bolt hole "
              "10 mm in from each of the four corners."),
        expect=Expect(
            bbox=(70.0, 70.0, 8.0),
            volume=_band(70 * 70 * 8 - _cyl_vol(25, 8) - 4 * _cyl_vol(6, 8), 0.08),
            inside=((20.0, 20.0, 4.0),),
            outside=((35.0, 35.0, 4.0), (10.0, 10.0, 4.0), (60.0, 10.0, 4.0),
                     (10.0, 60.0, 4.0), (60.0, 60.0, 4.0)),
            ops=(OpSpec("hole", count_min=5, count_max=5),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 70, 70), _ext("sk1", 8),
                   _hole(35, 35, 25), _hole(10, 10, 6), _hole(60, 10, 6),
                   _hole(10, 60, 6), _hole(60, 60, 6)),
    ),
    Brief(
        id="flange_thick", category="flange", difficulty=2,
        text=("A thick round flange, 60 mm outside diameter, 14 mm thick, with "
              "a 22 mm diameter bore through the centre."),
        expect=Expect(
            bbox=(60.0, 60.0, 14.0),
            volume=_band(_cyl_vol(60, 14) - _cyl_vol(22, 14), 0.12),
            inside=((25.0, 0.0, 7.0),),
            outside=((0.0, 0.0, 7.0),),
            ops=(OpSpec("hole", count_max=1, params={"diameter": (21.5, 22.5)}),),
        ),
        reference=(_sk(), _circ("sk1", 0, 0, 30), _ext("sk1", 14),
                   _hole(0, 0, 22)),
    ),

    # -- tier 5: shells ----------------------------------------------------- #
    # NOTE the F-rep shell is a two-sided wall (|f| - t/2), so a shelled box
    # straddles the original boundary: it grows OUTWARD by t/2 as well as
    # hollowing inward. The `inside` probes therefore sit ON the original surface
    # (which is the centre of the wall the backend builds) and the `outside`
    # probes sit at the centroid (which must become a cavity). This encodes what
    # this backend genuinely produces for a CORRECT plan; the outward dilation is
    # reported as a BACKEND BUG in the write-up, and is not scored against the
    # model, which has no op with which to avoid it.
    Brief(
        id="shell_box_3mm", category="shell", difficulty=2,
        text=("A hollow box. Start from a solid block 60 mm by 40 mm by 20 mm "
              "and shell it out to leave a 3 mm wall."),
        expect=Expect(
            volume=(8000.0, 45000.0),
            inside=((0.0, 20.0, 10.0),),           # the wall must be there
            outside=((30.0, 20.0, 10.0),),         # the cavity must exist
            ops=(OpSpec("shell", count_max=1, params={"thickness": (0.5, 19.9)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 60, 40), _ext("sk1", 20),
                   {"op": "shell", "faces": [], "thickness": 3}),
    ),
    Brief(
        id="shell_tray_2mm", category="shell", difficulty=2,
        text=("A shallow tray: take a block 80 mm by 60 mm by 25 mm and hollow "
              "it out with a 2.5 mm wall thickness."),
        expect=Expect(
            volume=(8000.0, 70000.0),
            inside=((0.0, 30.0, 12.0),),
            outside=((40.0, 30.0, 12.0),),
            ops=(OpSpec("shell", count_max=1, params={"thickness": (0.5, 24.9)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 80, 60), _ext("sk1", 25),
                   {"op": "shell", "faces": [], "thickness": 2.5}),
    ),
    Brief(
        id="shell_deep_4mm", category="shell", difficulty=2,
        text=("A deep hollow enclosure: a 40 mm by 40 mm by 30 mm block, "
              "shelled to a 4 mm wall."),
        expect=Expect(
            volume=(6000.0, 45000.0),
            inside=((0.0, 20.0, 15.0),),
            outside=((20.0, 20.0, 15.0),),
            ops=(OpSpec("shell", count_max=1, params={"thickness": (0.5, 29.9)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 40, 40), _ext("sk1", 30),
                   {"op": "shell", "faces": [], "thickness": 4}),
    ),

    # -- tier 6: fillets and chamfers --------------------------------------- #
    # The harness's kernel preflight rejects a fillet radius >= half the SMALLEST
    # extent of the solid (for a 10 mm plate that is 5 mm), and the F-rep fillet
    # is a Minkowski rounding of every edge, so that rule is the real limit here.
    Brief(
        id="fillet_plate_3mm", category="fillet", difficulty=2,
        text=("A plate 60 mm by 40 mm and 10 mm thick with all its edges "
              "rounded off with a 3 mm radius fillet."),
        expect=Expect(
            volume=_band(60 * 40 * 10, 0.15),
            inside=((30.0, 20.0, 5.0),),
            outside=((0.2, 0.2, 0.2),),            # the sharp corner is gone
            ops=(OpSpec("fillet", count_max=1, params={"radius": (0.1, 4.99)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 60, 40), _ext("sk1", 10),
                   {"op": "fillet", "edges": [], "radius": 3}),
    ),
    Brief(
        id="fillet_block_5mm", category="fillet", difficulty=2,
        text=("A block 50 mm by 50 mm by 20 mm with a 5 mm radius fillet on "
              "every edge."),
        expect=Expect(
            volume=_band(50 * 50 * 20, 0.15),
            inside=((25.0, 25.0, 10.0),),
            outside=((0.3, 0.3, 0.3),),
            ops=(OpSpec("fillet", count_max=1, params={"radius": (0.1, 9.99)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 50, 50), _ext("sk1", 20),
                   {"op": "fillet", "edges": [], "radius": 5}),
    ),
    Brief(
        id="chamfer_plate_2mm", category="fillet", difficulty=2,
        text=("A plate 60 mm by 40 mm and 10 mm thick with a 2 mm chamfer on "
              "all of its edges."),
        expect=Expect(
            volume=_band(60 * 40 * 10, 0.15),
            inside=((30.0, 20.0, 5.0),),
            ops=(OpSpec("chamfer", count_max=1, params={"distance": (0.1, 4.99)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 60, 40), _ext("sk1", 10),
                   {"op": "chamfer", "edges": [], "distance": 2}),
    ),

    # -- tier 7: the traps -------------------------------------------------- #
    Brief(
        id="trap_shell_too_thick", category="trap_shell", difficulty=4, trap=True,
        text=("A shallow tray made from a plate 60 mm by 40 mm and 5 mm thick, "
              "hollowed out with a 9 mm wall."),
        expect=Expect(
            volume=(1000.0, 16000.0),
            inside=((0.0, 20.0, 2.5),),
            ops=(OpSpec("shell", count_max=1, params={"thickness": (0.5, 4.99)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 60, 40), _ext("sk1", 5),
                   {"op": "shell", "faces": [], "thickness": 1.5}),
        note=("9 mm of wall in 5 mm of stock. The precheck names it exactly "
              "('the wall consumes the whole solid'); the core verifiers do not "
              "notice, and the F-rep backend silently INFLATES the part to "
              "49x39x14. This is the brief the claim was written for."),
    ),
    Brief(
        id="trap_shell_too_thin", category="trap_shell", difficulty=4, trap=True,
        text=("A hollow enclosure from an 80 mm by 50 mm by 20 mm block, "
              "shelled down to a 0.2 mm wall to save weight."),
        expect=Expect(
            volume=(2000.0, 60000.0),
            ops=(OpSpec("shell", count_max=1, params={"thickness": (0.5, 19.9)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 80, 50), _ext("sk1", 20),
                   {"op": "shell", "faces": [], "thickness": 1.5}),
        note="0.2 mm is below the 0.5 mm minimum manufacturable wall",
    ),
    Brief(
        id="trap_fillet_too_big", category="trap_fillet", difficulty=4, trap=True,
        text=("A plate 50 mm by 30 mm and 6 mm thick, with an 8 mm radius "
              "fillet on all edges."),
        expect=Expect(
            volume=_band(50 * 30 * 6, 0.30),
            inside=((25.0, 15.0, 3.0),),
            ops=(OpSpec("fillet", count_max=1, params={"radius": (0.1, 2.99)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 50, 30), _ext("sk1", 6),
                   {"op": "fillet", "edges": [], "radius": 2}),
        note=("8 mm of radius on 6 mm of stock. Core passes it and the F-rep "
              "fillet quietly erodes the plate to 36x16 mm."),
    ),
    Brief(
        id="trap_fillet_thin_plate", category="trap_fillet", difficulty=4, trap=True,
        text=("A plate 100 mm by 60 mm and 4 mm thick with 5 mm radius fillets "
              "on all edges."),
        expect=Expect(
            volume=_band(100 * 60 * 4, 0.30),
            inside=((50.0, 30.0, 2.0),),
            ops=(OpSpec("fillet", count_max=1, params={"radius": (0.1, 1.99)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 100, 60), _ext("sk1", 4),
                   {"op": "fillet", "edges": [], "radius": 1.5}),
        note="5 mm radius, 4 mm stock; the feasible ceiling is 2 mm",
    ),
    Brief(
        id="trap_hole_oversize", category="trap_hole", difficulty=4, trap=True,
        text=("A block 40 mm by 40 mm and 10 mm thick with a single 12 mm "
              "diameter hole bored right through it, centred."),
        expect=Expect(
            bbox=(40.0, 40.0, 10.0),
            volume=_band(40 * 40 * 10 - _cyl_vol(12, 10), 0.08),
            inside=((5.0, 5.0, 5.0),),
            outside=((20.0, 20.0, 5.0),),
            ops=(OpSpec("hole", count_max=1, params={"diameter": (11.5, 12.5)}),),
        ),
        reference=(_sk(), _rect("sk1", 0, 0, 40, 40), _ext("sk1", 10),
                   _hole(20, 20, 12)),
        note=("A FALSE POSITIVE, on purpose. A 12 mm through hole in a 10 mm "
              "plate is routine machining and the F-rep backend builds it "
              "correctly, but the precheck compares the hole DIAMETER against "
              "the plate THICKNESS and raises an ERROR 'infeasible-plan'. The "
              "ground truth here is the geometry the brief asked for, so an arm "
              "that 'fixes' the hole to satisfy the diagnostic LOSES this brief. "
              "This measures the cost of trusting a WRONG typed diagnostic."),
    ),
)


CATEGORIES: Tuple[str, ...] = tuple(dict.fromkeys(b.category for b in BRIEFS))


def brief_by_id(bid: str) -> Brief:
    for b in BRIEFS:
        if b.id == bid:
            return b
    raise KeyError(f"unknown brief '{bid}'; known: {', '.join(b.id for b in BRIEFS)}")


def briefs_for(selector: str = "all") -> List[Brief]:
    """Resolve a --briefs selector: 'all', 'traps', a category, or a CSV of ids."""
    sel = (selector or "all").strip()
    if sel == "all":
        return list(BRIEFS)
    if sel == "traps":
        return [b for b in BRIEFS if b.trap]
    if sel == "notraps":
        return [b for b in BRIEFS if not b.trap]
    if sel in CATEGORIES:
        return [b for b in BRIEFS if b.category == sel]
    return [brief_by_id(x.strip()) for x in sel.split(",") if x.strip()]
