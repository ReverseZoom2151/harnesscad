"""Tier-2 corpus: parts with NO closed form, for the differential oracle.

Why this file exists
--------------------
Every verified part in this repository is a box, a plate, a cylinder or a
washer. That is not an accident and it is not laziness: it is the price of the
strong claim. :mod:`harnesscad.eval.selftest.golden` asserts EXACT correctness
(``volume 22296.000000, delta 0.00e+00``) and it can only do that for a shape
whose volume can be written down in closed form. So the corpus filled up with
shapes whose volume can be written down, the op streams stayed four ops long,
and the harness ended up measured almost entirely on the parts it found easiest
to check.

Meanwhile the repository already owned the tool that dissolves the constraint
and never pointed it at anything hard. :mod:`harnesscad.eval.selftest.
differential` runs one op stream on six independent engines and reports the
spread. **It needs no ground truth at all.** Six engines that were written by
different people, in different languages, on different mathematics (an exact
OCCT B-rep, a CGAL mesher, a mesh kernel, a sampled distance field) agreeing on
the volume of a V-belt sheave is strong evidence about that sheave -- and no
one has to know, or be able to derive, what the right answer was.

That is the second tier, and this module is its corpus.

WHAT TIER 2 PROVES, AND WHAT IT DOES NOT
----------------------------------------
Be precise, because the difference is the whole point.

**Tier 1** (:mod:`~harnesscad.eval.selftest.golden`) proves CORRECTNESS. The
part's volume is a number derived from geometry, the engine's answer is compared
to it, and a delta of 0.00e+00 means the engine is right. It is the strongest
claim in the repo and it is available only on shapes simple enough to integrate
by hand.

**Tier 2** (this corpus, run through ``differential``) proves NO SUCH THING. It
proves only:

    the engines do not disagree.

That is strictly weaker, in a specific and important way: **agreement is not
truth**. Five engines can share a bug. The signature being compared (volume,
bbox, genus, watertightness) is MANY-TO-ONE -- a part with its holes bored in
the wrong places matches every number in the table. An op whose meaning is
underdetermined (what *should* ``shell`` with an empty ``faces`` list do?) can
have all six engines confidently computing six answers to six different
questions, and consensus among them would mean nothing.

So Tier 2 cannot promote a part to "correct". What it CAN do is fail, and a
failure is unambiguous: when two engines that were built independently return
different geometry for the same plan, at least one of them is WRONG, and we
learned that without knowing the answer. Tier 2 is a bug detector, not a
correctness certificate. Used that way it is worth a great deal, because it can
be pointed at a gyroid.

Tier 2 is buttressed by two oracles that are also ground-truth-free:

* the metamorphic laws in :mod:`~harnesscad.eval.selftest.properties` --
  especially ``scale_is_cubic``, which relates two runs of the SAME engine
  (multiply every length in the plan by k; the volume must go up by k^3). It
  holds even for an engine whose absolute numbers are all wrong, so it survives
  exactly the case where consensus is useless: everybody sharing a bug.
* :mod:`harnesscad.io.gate` -- which re-measures the solid that was actually
  WRITTEN TO DISK, not the one the backend says it built.

None of the three needs to know the volume of a gyroid. Together they are the
strongest statement available about a shape that has no closed form, and that
statement is still "we could not find a disagreement", never "this is right".

THE CORPUS
----------
Chosen to break the monoculture on three axes at once.

*Op vocabulary.* ``mirror``, ``add_instance`` and ``mate`` are exercised here
for the first time anywhere in the repo -- golden and the gallery between them
never emitted one. ``loft`` and ``draft`` are exercised for the first time and
are REFUSED by all five engines; that refusal is the finding (see
:data:`CAPABILITY_GAPS`), and the streams stay in the corpus so the gap is
measured every run instead of being remembered.

*Op co-occurrence.* Golden's 22 parts and the gallery's op-parts contain 40
distinct op PAIRS between them, and the ones they never put together are the
ones that break: ``fillet``+``hole``, ``shell``+``hole``, ``shell``+``fillet``,
``revolve``+``hole``, ``revolve``+``chamfer``. Rounding a block is fine.
Drilling a block is fine. Rounding a block and THEN drilling it is a
combination nothing in the repo had ever asked an engine to do.
:func:`novel_pairs` computes that set rather than asserting it, so the claim
cannot rot.

*Depth.* Golden's deepest stream is 8 ops and its median is 4.
``housing-boss-shell`` is 22 and stacks fillet -> shell -> four bosses -> union
-> four blind bores, which is a feature tree, not a plan.

Deterministic and closed-form-free by construction: no randomness, no wall
clock, and deliberately not one analytic volume in the file. If a number in here
could be checked by hand, the part does not belong in this corpus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, FrozenSet, List, Sequence, Tuple

from harnesscad.core.cisp.ops import Op, parse_op

__all__ = [
    "ComplexPart",
    "CORPUS",
    "CAPABILITY_GAPS",
    "names",
    "get",
    "streams",
    "ops_of",
    "baseline_pairs",
    "corpus_pairs",
    "novel_pairs",
    "novel_ops",
    "coverage_report",
]


@dataclass(frozen=True)
class ComplexPart:
    """One Tier-2 part: an op stream whose volume nobody can write down.

    ``why`` states what the part is FOR -- which op, combination or regime it
    puts under the six engines that nothing else in the repo does.

    ``closed_form`` is False for every part in this corpus, and the field exists
    to make that a fact the tests can assert rather than a claim in a docstring.
    A part that acquires a closed form belongs in :mod:`golden`, not here.
    """

    name: str
    summary: str
    why: str
    raw: Tuple[dict, ...]
    closed_form: bool = False
    expect_refusal: Tuple[str, ...] = ()   # ops we EXPECT every engine to refuse
    notes: str = ""

    @property
    def ops(self) -> Tuple[Op, ...]:
        return tuple(parse_op(dict(o)) for o in self.raw)

    @property
    def tags(self) -> Tuple[str, ...]:
        """The op tags this stream emits, in order."""
        return tuple(str(o["op"]) for o in self.raw)

    @property
    def op_set(self) -> FrozenSet[str]:
        return frozenset(self.tags)

    @property
    def depth(self) -> int:
        return len(self.raw)

    def to_dict(self) -> dict:
        return {"name": self.name, "summary": self.summary, "why": self.why,
                "depth": self.depth, "ops": sorted(self.op_set),
                "closed_form": self.closed_form,
                "expect_refusal": list(self.expect_refusal),
                "notes": self.notes}


# ---------------------------------------------------------------------------
# op-stream helpers (dicts, so the corpus stays a literal and stays hashable)
# ---------------------------------------------------------------------------
def _sk(plane: str = "XY") -> dict:
    return {"op": "new_sketch", "plane": plane}


def _rect(sk: str, x: float, y: float, w: float, h: float) -> dict:
    return {"op": "add_rectangle", "sketch": sk, "x": x, "y": y, "w": w, "h": h}


def _circle(sk: str, cx: float, cy: float, r: float) -> dict:
    return {"op": "add_circle", "sketch": sk, "cx": cx, "cy": cy, "r": r}


def _line(sk: str, x1: float, y1: float, x2: float, y2: float) -> dict:
    return {"op": "add_line", "sketch": sk, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _polyline(sk: str, pts: Sequence[Sequence[float]]) -> List[dict]:
    """Close ``pts`` into a loop of ``add_line`` ops (the polygon-profile idiom)."""
    loop = list(pts) + [pts[0]]
    return [_line(sk, float(a[0]), float(a[1]), float(b[0]), float(b[1]))
            for a, b in zip(loop[:-1], loop[1:])]


def _extrude(sk: str, d: float) -> dict:
    return {"op": "extrude", "sketch": sk, "distance": float(d)}


def _hole(ref: str, x: float, y: float, dia: float, depth: float = None) -> dict:
    op = {"op": "hole", "face_or_sketch": ref, "x": float(x), "y": float(y),
          "diameter": float(dia), "kind": "simple"}
    if depth is None:
        op["through"] = True
    else:
        op["through"] = False
        op["depth"] = float(depth)
    return op


def _union() -> dict:
    return {"op": "boolean", "kind": "union", "target": "", "tool": ""}


def _fillet(edges: Sequence[str], r: float) -> dict:
    return {"op": "fillet", "edges": list(edges), "radius": float(r)}


def _chamfer(edges: Sequence[str], d: float) -> dict:
    return {"op": "chamfer", "edges": list(edges), "distance": float(d)}


def _shell(faces: Sequence[str], t: float) -> dict:
    return {"op": "shell", "faces": list(faces), "thickness": float(t)}


def _revolve(sk: str, angle: float = 360.0) -> dict:
    return {"op": "revolve", "sketch": sk,
            "axis": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], "angle": float(angle)}


# ---------------------------------------------------------------------------
# 1. housing-boss-shell -- the deep feature stack (22 ops)
# ---------------------------------------------------------------------------
def _housing() -> List[dict]:
    """80x60x30 housing: rounded corners, shelled 3 mm, four bosses, four bores.

    fillet -> shell -> boss extrudes -> union -> blind bores. Golden's deepest
    stream is 8 ops; this is 22, and every adjacent pair in it
    (fillet+shell, shell+boolean, shell+hole, fillet+hole) is a pair the repo
    had never put in the same stream.
    """
    ops: List[dict] = [
        _sk(), _rect("sk1", -40.0, -30.0, 80.0, 60.0), _extrude("sk1", 30.0),
        _fillet(["|Z"], 6.0),
        _shell([">Z"], 3.0),
    ]
    bosses = ((-28.0, -20.0), (28.0, -20.0), (-28.0, 20.0), (28.0, 20.0))
    for i, (x, y) in enumerate(bosses):
        sk = "sk%d" % (i + 2)
        ops += [_sk(), _circle(sk, x, y, 6.0), _extrude(sk, 27.0)]
    ops.append(_union())
    for (x, y) in bosses:
        ops.append(_hole("f1", x, y, 3.2, depth=20.0))
    return ops


# ---------------------------------------------------------------------------
# 2. shaft-shoulders -- a turned part (revolve + bore + chamfer)
# ---------------------------------------------------------------------------
#: A stepped turned shaft, half-section in the XZ plane: three journals of
#: falling diameter. Revolved, it is a solid of revolution with no elementary
#: volume -- and it is the first time ``revolve`` has been asked to co-occur with
#: ``hole`` or ``chamfer`` anywhere in the repo.
SHAFT_SECTION: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.0), (14.0, 0.0), (14.0, 12.0), (10.0, 12.0),
    (10.0, 40.0), (7.0, 40.0), (7.0, 66.0), (0.0, 66.0),
)


def _shaft() -> List[dict]:
    """A 66 mm turned shaft with two shoulders, bored 6 mm and chamfered."""
    ops: List[dict] = [_sk("XZ")]
    ops += _polyline("sk1", SHAFT_SECTION)
    ops += [_revolve("sk1"), _hole("f1", 0.0, 0.0, 6.0), _chamfer([], 1.0)]
    return ops


# ---------------------------------------------------------------------------
# 3. sheave-vbelt -- revolved V-groove + a ring of lightening holes
# ---------------------------------------------------------------------------
SHEAVE_SECTION: Tuple[Tuple[float, float], ...] = (
    (6.0, 0.0), (40.0, 0.0), (40.0, 5.0), (30.0, 11.0),
    (40.0, 17.0), (40.0, 22.0), (6.0, 22.0),
)
SHEAVE_LIGHTENING = 5


def _sheave() -> List[dict]:
    """An 80 mm V-belt sheave: revolved groove, 12 mm bore, five lightening holes.

    The V-groove is a re-entrant profile, so the solid of revolution is not a
    difference of cylinders and its volume is not an elementary expression. Five
    holes on a bolt circle take it to genus 6.
    """
    ops: List[dict] = [_sk("XZ")]
    ops += _polyline("sk1", SHEAVE_SECTION)
    ops += [_revolve("sk1"), _hole("f1", 0.0, 0.0, 12.0)]
    for i in range(SHEAVE_LIGHTENING):
        a = 2.0 * math.pi * i / SHEAVE_LIGHTENING
        ops.append(_hole("f1", 22.0 * math.cos(a), 22.0 * math.sin(a), 9.0))
    return ops


# ---------------------------------------------------------------------------
# 4. chain-link -- genus 2 from fillet + hole
# ---------------------------------------------------------------------------
def _chain_link() -> List[dict]:
    """A 60x24x8 chain link: a stadium (fillet r=11.5 on a 24 mm bar) twice bored.

    The corner radius is 11.5 on a half-width of 12, so the fillet very nearly
    closes into a semicircle -- a near-degenerate fillet, deliberately, because
    that is where a kernel's edge-blending falls over. Two bores make it genus 2.
    """
    return [
        _sk(), _rect("sk1", -30.0, -12.0, 60.0, 24.0), _extrude("sk1", 8.0),
        _fillet(["|Z"], 11.5),
        _hole("f1", -13.0, 0.0, 15.0),
        _hole("f1", 13.0, 0.0, 15.0),
        _chamfer([">Z"], 0.8),
    ]


# ---------------------------------------------------------------------------
# 5. plate-fillet-holes -- fillet THEN hole (the pair nothing ever emitted)
# ---------------------------------------------------------------------------
def _plate_fillet_holes() -> List[dict]:
    """90x60x12 plate, corners rounded r=8, then five bores (genus 5).

    The minimal witness for ``fillet``+``hole``. Both ops are individually in the
    golden corpus and individually exact on every engine. Nothing had ever run
    them in the same stream.
    """
    ops = [
        _sk(), _rect("sk1", -45.0, -30.0, 90.0, 60.0), _extrude("sk1", 12.0),
        _fillet(["|Z"], 8.0),
    ]
    for (x, y) in ((-32.0, -20.0), (32.0, -20.0), (-32.0, 20.0), (32.0, 20.0)):
        ops.append(_hole("f1", x, y, 6.6))
    ops.append(_hole("f1", 0.0, 0.0, 25.0))
    return ops


# ---------------------------------------------------------------------------
# 6. shell-and-holes -- shell THEN hole
# ---------------------------------------------------------------------------
def _shell_and_holes() -> List[dict]:
    """A 60x40x25 shelled tray, 2.5 mm wall, with three bores through the walls.

    ``shell``+``hole``: boring a hollow part cuts each wall TWICE, so the genus
    is driven by the cavity, not by the hole count. An engine that models the
    shell as a solid pocket rather than a hollow gets the topology wrong here and
    nowhere else.
    """
    return [
        _sk(), _rect("sk1", -30.0, -20.0, 60.0, 40.0), _extrude("sk1", 25.0),
        _shell([">Z"], 2.5),
        _hole("f1", -20.0, 0.0, 5.0),
        _hole("f1", 20.0, 0.0, 5.0),
        _hole("f1", 0.0, 0.0, 8.0),
    ]


# ---------------------------------------------------------------------------
# 7. thinwall-tall -- the high-aspect sub-cell shell regime
# ---------------------------------------------------------------------------
def _thinwall_tall() -> List[dict]:
    """A 30x30x90 tube shelled to 1.5 mm: a wall THINNER THAN THE FREP GRID CELL.

    frep samples on 48 cells across the largest extent, so at 90 mm tall the cell
    is 1.875 mm and a 1.5 mm wall cannot be represented in the field at all. The
    correct behaviour is to REFUSE. The wrong behaviour -- silently building a
    smaller, eroded part and calling it watertight -- is the bug that
    ``properties.shell_does_not_shrink`` was written to catch. This stream keeps
    that regime under measurement instead of under discussion.
    """
    return [
        _sk(), _rect("sk1", -15.0, -15.0, 30.0, 30.0), _extrude("sk1", 90.0),
        _shell([">Z"], 1.5),
    ]


# ---------------------------------------------------------------------------
# 8. mirror-rib -- the ``mirror`` op, used here for the first time
# ---------------------------------------------------------------------------
def _mirror_rib() -> List[dict]:
    """A ribbed base plate: one rib, MIRRORED across YZ, then unioned to a plate.

    ``mirror`` appears in the CISP op set, in the registry and in no corpus,
    no golden part and no gallery part anywhere in the repository. This is its
    first exercise.
    """
    return [
        _sk(), _rect("sk1", 10.0, -4.0, 30.0, 8.0), _extrude("sk1", 25.0),
        {"op": "mirror", "feature_or_body": "f1", "plane": "YZ"},
        _sk(), _rect("sk2", -45.0, -20.0, 90.0, 40.0), _extrude("sk2", 5.0),
        _union(),
        _chamfer([">Z"], 1.0),
    ]


# ---------------------------------------------------------------------------
# 9. assembly-mate -- add_instance + mate, used here for the first time
# ---------------------------------------------------------------------------
def _assembly_mate() -> List[dict]:
    """Two placed instances of a bored block, coupled by a revolute mate.

    ``add_instance`` and ``mate`` are pure ASSEMBLY bookkeeping: they place and
    constrain bodies without changing the solid. So the geometric prediction is
    sharp and worth stating -- the measured volume must be UNCHANGED by them. An
    engine that folds an instance into the solid is wrong, and only a stream that
    actually emits the ops can find that out.
    """
    return [
        _sk(), _rect("sk1", -20.0, -20.0, 40.0, 40.0), _extrude("sk1", 10.0),
        _hole("f1", 0.0, 0.0, 8.0),
        {"op": "add_instance", "part": "solid", "x": 0.0, "y": 0.0, "z": 0.0},
        {"op": "add_instance", "part": "solid", "x": 50.0, "y": 0.0, "z": 0.0},
        {"op": "mate", "kind": "revolute", "a": "i1", "b": "i2"},
    ]


# ---------------------------------------------------------------------------
# 10 / 11. loft + draft -- the ops NO engine realises
# ---------------------------------------------------------------------------
def _loft_duct() -> List[dict]:
    """A round-to-square duct transition. EVERY engine refuses ``loft``.

    Kept in the corpus precisely because it fails. A capability gap that is
    measured every run is a known gap; a capability gap that lives in somebody's
    memory is a surprise waiting for a customer.
    """
    return [
        _sk(), _circle("sk1", 0.0, 0.0, 20.0),
        _sk(), _rect("sk2", -15.0, -15.0, 30.0, 30.0),
        {"op": "loft", "sketches": ["sk1", "sk2"], "ruled": False,
         "offsets": [0.0, 40.0]},
    ]


def _draft_taper() -> List[dict]:
    """A 5-degree draughted core box. EVERY engine refuses ``draft``.

    As ``loft``: the op is in the vocabulary, the registry knows it, and no
    backend realises it. ``cadquery.py`` says so in as many words ("real drafting
    is not yet wired on the current CadQuery/OCCT build").
    """
    return [
        _sk(), _rect("sk1", -20.0, -20.0, 40.0, 40.0), _extrude("sk1", 30.0),
        {"op": "draft", "faces": ["|Z"], "angle": 5.0, "neutral_plane": "<Z"},
    ]


# ---------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------
CORPUS: Tuple[ComplexPart, ...] = (
    ComplexPart(
        name="housing-boss-shell",
        summary="80x60x30 shelled housing, rounded corners, four bosses, four blind bores.",
        why="The deep feature stack: 22 ops (golden's deepest is 8). Puts "
            "fillet+shell, shell+boolean, shell+hole and fillet+hole in one "
            "stream -- four pairs the repo had never co-emitted.",
        raw=tuple(_housing()),
        notes="The part the 'four ops, three families' warning was written about.",
    ),
    ComplexPart(
        name="shaft-shoulders",
        summary="66 mm turned shaft, two shoulders, 6 mm bore, chamfered.",
        why="revolve+hole and revolve+chamfer, neither of which had ever "
            "co-occurred. A stepped solid of revolution has no elementary volume.",
        raw=tuple(_shaft()),
    ),
    ComplexPart(
        name="sheave-vbelt",
        summary="80 mm V-belt sheave: revolved groove, 12 mm bore, five lightening holes.",
        why="A RE-ENTRANT revolved profile (the V-groove) -- not a difference of "
            "cylinders, so not integrable by inspection -- taken to genus 6.",
        raw=tuple(_sheave()),
    ),
    ComplexPart(
        name="chain-link",
        summary="60x24x8 chain link, near-degenerate r=11.5 fillet, twice bored.",
        why="A fillet radius within 0.5 mm of closing into a semicircle, then "
            "fillet+hole, then chamfer. Genus 2.",
        raw=tuple(_chain_link()),
    ),
    ComplexPart(
        name="plate-fillet-holes",
        summary="90x60x12 plate, corners rounded r=8, five bores (genus 5).",
        why="The MINIMAL WITNESS for fillet+hole. Both ops are exact on every "
            "engine alone; nothing had ever run them in one stream.",
        raw=tuple(_plate_fillet_holes()),
    ),
    ComplexPart(
        name="shell-and-holes",
        summary="60x40x25 tray, 2.5 mm wall, three bores through the walls.",
        why="shell+hole. Boring a hollow cuts each wall twice, so the topology is "
            "driven by the cavity -- an engine that shells into a pocket instead "
            "of a hollow gets the genus wrong here and nowhere else.",
        raw=tuple(_shell_and_holes()),
    ),
    ComplexPart(
        name="thinwall-tall",
        summary="30x30x90 tube shelled to 1.5 mm -- a wall thinner than frep's grid cell.",
        why="The high-aspect sub-cell shell regime. The only correct answer is a "
            "REFUSAL; silently eroding the part is the bug properties."
            "shell_does_not_shrink exists to catch.",
        raw=tuple(_thinwall_tall()),
        notes="A refusal here is the PASS condition for a sampled engine.",
    ),
    ComplexPart(
        name="mirror-rib",
        summary="Ribbed base plate: one rib, mirrored across YZ, unioned, chamfered.",
        why="The FIRST use of `mirror` anywhere in the repository -- not in "
            "golden, not in the gallery, not in any corpus.",
        raw=tuple(_mirror_rib()),
    ),
    ComplexPart(
        name="assembly-mate",
        summary="Two instances of a bored block coupled by a revolute mate.",
        why="The FIRST use of `add_instance` and `mate` anywhere. They are "
            "bookkeeping, so the prediction is sharp: the measured volume must "
            "not move.",
        raw=tuple(_assembly_mate()),
    ),
    ComplexPart(
        name="loft-duct",
        summary="Round-to-square duct transition. Every engine refuses `loft`.",
        why="The FIRST use of `loft`. It fails on all five engines, and that "
            "measured, repeated failure is the point.",
        raw=tuple(_loft_duct()),
        expect_refusal=("loft",),
        notes="A KNOWN capability gap, kept under measurement.",
    ),
    ComplexPart(
        name="draft-taper",
        summary="5-degree draughted core box. Every engine refuses `draft`.",
        why="The FIRST use of `draft`. As loft: in the vocabulary, in the "
            "registry, realised by no backend.",
        raw=tuple(_draft_taper()),
        expect_refusal=("draft",),
        notes="A KNOWN capability gap, kept under measurement.",
    ),
)

#: Ops that are in the CISP vocabulary and that NO backend realises. Asserted by
#: the test suite against the live engines, so if somebody implements one, the
#: test goes red and this constant has to be updated -- which is the point.
CAPABILITY_GAPS: Tuple[str, ...] = ("loft", "draft", "sweep")

_BY_NAME: Dict[str, ComplexPart] = {p.name: p for p in CORPUS}


def names() -> List[str]:
    return [p.name for p in CORPUS]


def get(name: str) -> ComplexPart:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError("no Tier-2 part named %r (%d in the corpus: %s)"
                       % (name, len(CORPUS), ", ".join(names()))) from None


def streams() -> List[Tuple[str, Tuple[Op, ...]]]:
    """``(name, ops)`` pairs, the shape :func:`differential.run` wants."""
    return [(p.name, p.ops) for p in CORPUS]


def ops_of(part: ComplexPart) -> FrozenSet[str]:
    return part.op_set


# ---------------------------------------------------------------------------
# op-coverage: computed, never asserted
# ---------------------------------------------------------------------------
def _pairs(tags: Sequence[str]) -> FrozenSet[Tuple[str, str]]:
    return frozenset(combinations(sorted(set(tags)), 2))


def baseline_pairs() -> FrozenSet[Tuple[str, str]]:
    """Every op PAIR that co-occurs in the pre-existing corpora.

    Read live out of :mod:`golden` and the gallery catalogue rather than
    hard-coded, so the "nothing had ever done this" claim is re-derived on every
    run and cannot quietly become false.
    """
    from harnesscad.eval.gallery import parts as gallery_parts
    from harnesscad.eval.selftest import golden

    out: set = set()
    for p in golden.PARTS:
        out |= _pairs([o.OP for o in p.ops])
    for p in gallery_parts.CATALOGUE:
        if p.cisp_ops:
            out |= _pairs(p.cisp_ops)
    return frozenset(out)


def corpus_pairs() -> FrozenSet[Tuple[str, str]]:
    """Every op pair this Tier-2 corpus emits."""
    out: set = set()
    for p in CORPUS:
        out |= _pairs(p.tags)
    return frozenset(out)


def novel_pairs() -> FrozenSet[Tuple[str, str]]:
    """Op pairs this corpus co-emits that NOTHING in the repo did before."""
    return corpus_pairs() - baseline_pairs()


def novel_ops() -> FrozenSet[str]:
    """Ops this corpus emits that no prior corpus emitted at all."""
    from harnesscad.eval.gallery import parts as gallery_parts
    from harnesscad.eval.selftest import golden

    seen: set = set()
    for p in golden.PARTS:
        seen |= {o.OP for o in p.ops}
    for p in gallery_parts.CATALOGUE:
        seen |= set(p.cisp_ops)
    mine: set = set()
    for p in CORPUS:
        mine |= set(p.tags)
    return frozenset(mine - seen)


def coverage_report() -> dict:
    """What this corpus adds, as data -- for the report and for the tests."""
    novel_p = novel_pairs()
    return {
        "parts": len(CORPUS),
        "max_depth": max(p.depth for p in CORPUS),
        "median_depth": sorted(p.depth for p in CORPUS)[len(CORPUS) // 2],
        "novel_ops": sorted(novel_ops()),
        "novel_pairs": sorted("%s+%s" % (a, b) for a, b in novel_p),
        "novel_pair_count": len(novel_p),
        "baseline_pair_count": len(baseline_pairs()),
        "capability_gaps": list(CAPABILITY_GAPS),
        "closed_form_parts": sum(1 for p in CORPUS if p.closed_form),
    }
