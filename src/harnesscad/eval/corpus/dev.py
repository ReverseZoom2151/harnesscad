"""The DEV split -- the corpus you are allowed to look at.

Read it, debug against it, tune on it, overfit it if you like. That is what it is
for, and it is why it exists SEPARATELY from :mod:`harnesscad.eval.corpus.heldout`:
a benchmark you are allowed to iterate against stops being a measurement the first
time you iterate against it, and the only defence is to keep a second one you are
not allowed to touch.

The audit's own words (14.8.1 benchmark contamination, 14.8.2 overfitting to
benchmarks, 14.8.3 Goodhart's Law): maintain a private test set, refresh it. The
pressure experiment had one corpus, twenty-eight briefs, one hand, and no held-out
set at all -- so there was no instrument left that could tell "the harness got
better" from "the harness learned this corpus".

Every brief here is a call into :mod:`harnesscad.eval.corpus.analytic` or
:mod:`harnesscad.eval.corpus.standards`, so its ground truth is a formula or a
standard, and the DIFFERENCE between this split and the held-out split is a set of
NUMBERS -- not a second hand-written file that can drift into a different opinion
about what correct means.
"""

from __future__ import annotations

from typing import List, Tuple

from harnesscad.eval.corpus import analytic as A
from harnesscad.eval.corpus import standards as S
from harnesscad.eval.corpus.spec import Brief, Split, check_unique

__all__ = ["BRIEFS", "by_id", "ids"]

_D = Split.DEV

BRIEFS: Tuple[Brief, ...] = (
    # -- arithmetic: prisms and cylinders ---------------------------------- #
    A.plate("dev_plate_60x40x10", _D, 60.0, 40.0, 10.0),
    A.plate("dev_plate_thin_100x50x3", _D, 100.0, 50.0, 3.0),
    A.disc("dev_disc_d40_h12", _D, 40.0, 12.0),

    # -- arithmetic: holes (genus is the assertion) ------------------------- #
    A.plate_with_holes("dev_plate_hole_centre", _D, 60.0, 40.0, 12.0, 8.0,
                       ((30.0, 20.0),)),
    A.plate_with_holes("dev_plate_hole_four", _D, 80.0, 60.0, 10.0, 7.0,
                       ((12.0, 12.0), (68.0, 12.0), (12.0, 48.0), (68.0, 48.0))),
    A.tube("dev_spacer_d40_bore14", _D, 40.0, 14.0, 10.0),

    # -- arithmetic: shell (bbox UNCHANGED, wall probed at its mid-plane) ---- #
    A.hollow_box("dev_hollow_box_60x40x20_t3", _D, 60.0, 40.0, 20.0, 3.0),
    A.hollow_box("dev_hollow_box_40x40x30_t4", _D, 40.0, 40.0, 30.0, 4.0),
    # A thin wall in a big box: 3 mm of wall across a 120 mm part, which is 1.2
    # cells of the F-rep sampler's 48-cell grid. It is a PERFECTLY GOOD PART and
    # frep cannot measure it -- it loses 4.9% of the volume and says nothing. This
    # brief is here on purpose, on the split you are allowed to look at, so that
    # the same effect on the held-out split can be REPRODUCED HERE and debugged
    # here. That is the whole protocol: a held-out failure you cannot reproduce on
    # dev means the dev split is too weak, and the fix is to add a dev brief --
    # never to go and look at the held-out one.
    A.hollow_box("dev_hollow_box_120x80x30_t3", _D, 120.0, 80.0, 30.0, 3.0),

    # -- arithmetic: fillet / chamfer (Steiner; bbox UNCHANGED) -------------- #
    A.filleted_plate("dev_fillet_plate_60x40x10_r3", _D, 60.0, 40.0, 10.0, 3.0),
    A.filleted_plate("dev_fillet_block_50x50x20_r5", _D, 50.0, 50.0, 20.0, 5.0),
    A.chamfered_plate("dev_chamfer_plate_60x40x10_d2", _D, 60.0, 40.0, 10.0, 2.0),

    # -- arithmetic: booleans ------------------------------------------------ #
    A.notched_block("dev_notched_block_40x40x20", _D, 40.0, 40.0, 20.0, 10.0, 10.0),
    A.l_bracket("dev_l_bracket_60x40", _D, 60.0, 40.0, 6.0, 30.0),

    # -- standards: the parts an engineer looks up ---------------------------- #
    S.washer("dev_washer_iso7089_m8", _D, "M8"),
    S.clearance_plate("dev_clearance_iso273_m8", _D, "M8", 80.0, 60.0, 10.0, 12.0),
    S.flange_dn50_pn16("dev_flange_dn50_pn16", _D),
)

check_unique(BRIEFS)


def ids() -> List[str]:
    return [b.id for b in BRIEFS]


def by_id(bid: str) -> Brief:
    for b in BRIEFS:
        if b.id == bid:
            return b
    raise KeyError("unknown dev brief %r; known: %s" % (bid, ", ".join(ids())))
