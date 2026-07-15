"""THE HELD-OUT SPLIT OF THE HARD CORPUS. DO NOT IMPORT THIS MODULE.

If you are reading this file because you want to debug a failure on it: stop. That
is the one use it has and the one use that destroys it. This mirrors
``eval/corpus/heldout.py`` exactly, and the discipline is the same, because the
contamination it prevents is the same.

RULES
-----
1.  Only :mod:`harnesscad.eval.hardcorpus.score` may import this module. That is
    enforced by ``tests/eval/hardcorpus/test_holdout_isolation.py``, which scans
    the whole source and test tree and FAILS if any other module names it.
2.  These briefs are SCORED, never inspected. You may read the number that comes
    out; you may not read the brief that produced it and edit code until it passes.
    Tune on the dev corpus (``eval/hardcorpus/dev`` -- seed 1).
3.  A held-out failure means "a change made something worse". Reproduce it on dev
    and fix it there. If it will not reproduce on dev, the dev split is too weak --
    add a dev sample -- and that is the bug, not the code the held-out brief hit.

WHY A SEED IS ENOUGH
--------------------
The dev and held-out splits are two SEEDS through the same generator
(:mod:`harnesscad.eval.hardcorpus.generate`). There is nothing special in these
briefs and nothing secret -- they are the same families with different numbers, and
their only property is that no line of this repository has been changed to make one
of them pass. That is exactly the design of ``eval/corpus``: the factories are
pure functions of their dimensions, so two seeds cannot drift apart the way two
hand-written files do. A different held-out seed is an equally valid held-out set,
which is why the seed can be refreshed without touching a factory.
"""

from __future__ import annotations

from typing import List, Tuple

from harnesscad.eval.corpus.spec import Brief, Split
from harnesscad.eval.hardcorpus import discriminative as _disc
from harnesscad.eval.hardcorpus import generate as _gen

__all__ = ["BRIEFS", "NEAR_MISSES", "ids"]

#: The held-out generator seed. Distinct from the dev seed (1). Refreshable.
SEED = 7919

#: Held-out generated L3 briefs -- one part per family, two prompt styles each.
BRIEFS: Tuple[Brief, ...] = tuple(_gen.all_briefs(SEED, Split.HELDOUT))

#: Held-out discriminative near-misses. The near-miss construction is the same as
#: the dev cases; the discipline is that their pass/fail is only ever read through
#: the scorer, never opened here to tune a probe.
NEAR_MISSES = tuple(_disc.cases(Split.HELDOUT))


def ids() -> List[str]:
    """The brief ids. A count and a list of ids are not a leak of the answers."""
    return [b.id for b in BRIEFS] + [nm.id for nm in NEAR_MISSES]
