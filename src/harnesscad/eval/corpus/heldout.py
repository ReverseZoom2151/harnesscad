"""THE HELD-OUT SPLIT. DO NOT IMPORT THIS MODULE.

If you are reading this file because you are about to debug a failure on it: stop.
That is the one thing it is for, and the one thing that destroys it.

RULES
-----
1.  Only :mod:`harnesscad.eval.corpus.score` may import this module. That is not a
    convention -- it is enforced by ``tests/eval/corpus/test_holdout_isolation.py``,
    which scans every ``.py`` file in the source tree and FAILS if any module other
    than the scorer names this one. Add an import and the suite goes red.
2.  These briefs are SCORED, never inspected. You may read the number that comes
    out. You may not read the brief that produced it and go change the code until
    that brief passes. Tune on :mod:`harnesscad.eval.corpus.dev`; that split exists
    precisely so this one can stay clean.
3.  When a change makes a held-out brief fail, the finding is "the change made
    something worse". Take it to the dev split, reproduce it there, fix it there.
    If it cannot be reproduced on dev, the dev split is too weak -- ADD A DEV BRIEF.
    Do not fix the code against a brief you can only see here.

WHY
---
``assets/pressure/report.md`` is the record of what happens without this. One
corpus, 28 briefs, written by the system it was scoring. Its fillet briefs encoded
the harness's own unsound ceiling as ground truth, so the grader rewarded a model
for obeying the harness's bug; its shell briefs probed a point that only a DILATED
part could satisfy, so with the backend fixed no answer solves them -- not even
each brief's own reference solution. Fleet and corpus shared the blind spot,
because there was no second corpus that could have disagreed.

The book audit prescribes exactly this mitigation for benchmark contamination
(14.8.1) and for Goodhart's Law (14.8.3): a private test set, refreshed. It was
never in place. It is now.

The briefs below come from the SAME factories as the dev split -- the same formulas
and the same standards -- with different numbers. There is nothing special about
them and nothing secret in them. Their only property is that no line of this
repository has ever been changed to make one of them pass.
"""

from __future__ import annotations

from typing import List, Tuple

from harnesscad.eval.corpus import analytic as A
from harnesscad.eval.corpus import standards as S
from harnesscad.eval.corpus.spec import Brief, Split, check_unique

__all__ = ["BRIEFS", "ids"]

_H = Split.HELDOUT

BRIEFS: Tuple[Brief, ...] = (
    # -- arithmetic: prisms and cylinders ---------------------------------- #
    A.plate("ho_plate_90x35x8", _H, 90.0, 35.0, 8.0),
    A.plate("ho_bar_120x12x12", _H, 120.0, 12.0, 12.0),
    A.disc("ho_disc_d55_h9", _H, 55.0, 9.0),

    # -- arithmetic: holes --------------------------------------------------- #
    A.plate_with_holes("ho_strip_hole_row", _H, 110.0, 24.0, 7.0, 6.0,
                       ((22.0, 12.0), (55.0, 12.0), (88.0, 12.0))),
    A.plate_with_holes("ho_plate_hole_offcentre", _H, 50.0, 50.0, 10.0, 9.0,
                       ((15.0, 15.0),)),
    A.tube("ho_tube_d36_bore20_h45", _H, 36.0, 20.0, 45.0),

    # -- arithmetic: shell --------------------------------------------------- #
    A.hollow_box("ho_hollow_box_80x60x25_t2.5", _H, 80.0, 60.0, 25.0, 2.5),
    A.hollow_box("ho_hollow_cube_50_t5", _H, 50.0, 50.0, 50.0, 5.0),

    # -- arithmetic: fillet / chamfer ---------------------------------------- #
    A.filleted_plate("ho_fillet_plate_70x45x12_r4", _H, 70.0, 45.0, 12.0, 4.0),
    A.chamfered_plate("ho_chamfer_block_45x45x15_d3", _H, 45.0, 45.0, 15.0, 3.0),

    # -- arithmetic: booleans ------------------------------------------------ #
    A.notched_block("ho_notched_block_70x50x15", _H, 70.0, 50.0, 15.0, 20.0, 15.0),
    A.l_bracket("ho_l_bracket_80x30", _H, 80.0, 30.0, 5.0, 25.0),

    # -- standards ------------------------------------------------------------ #
    S.washer("ho_washer_iso7089_m12", _H, "M12"),
    S.washer("ho_washer_iso7089_m6", _H, "M6"),
    S.clearance_plate("ho_clearance_iso273_m10", _H, "M10", 100.0, 70.0, 8.0, 15.0),
)

check_unique(BRIEFS)


def ids() -> List[str]:
    return [b.id for b in BRIEFS]
