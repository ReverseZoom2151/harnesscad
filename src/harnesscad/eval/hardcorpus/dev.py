"""The DEV split: the part of the hard corpus you may read, debug and tune on.

Everything here is public and inspectable, by design -- it is the split you take a
held-out failure to and reproduce it on. It is seed 1 through the generator, plus
the dev discriminative near-misses, plus the constraint and ambiguous briefs (which
have no held-out twin: a constraint set and a missing dimension are not secrets, and
there is nothing to overfit in a requirement you can already read).

See :mod:`harnesscad.eval.hardcorpus.score` for how the HELD-OUT split is reached --
never from here.
"""

from __future__ import annotations

from typing import List, Tuple

from harnesscad.eval.corpus.spec import Brief, Split
from harnesscad.eval.hardcorpus import ambiguous as _amb
from harnesscad.eval.hardcorpus import constraints as _con
from harnesscad.eval.hardcorpus import discriminative as _disc
from harnesscad.eval.hardcorpus import generate as _gen

__all__ = ["SEED", "GENERATED", "NEAR_MISSES", "CONSTRAINTS", "AMBIGUOUS",
           "counts", "ids"]

#: The dev generator seed. The held-out split uses a different one (see heldout.py).
SEED = 1

#: Generated L3 briefs at the dev seed: one part per family, two prompt styles each.
GENERATED: Tuple[Brief, ...] = tuple(_gen.all_briefs(SEED, Split.DEV))

#: The dev discriminative near-misses (the headline table's source).
NEAR_MISSES = _disc.CASES

#: The constraint-satisfaction briefs.
CONSTRAINTS = _con.BRIEFS

#: The underspecification briefs.
AMBIGUOUS = _amb.BRIEFS


def counts() -> dict:
    """A census of the dev split, by kind and by difficulty level."""
    by_level: dict = {}
    for b in GENERATED:
        lvl = _gen.LEVEL.get(b.id.split("_")[1] if "_" in b.id else "", "L3")
        by_level[lvl] = by_level.get(lvl, 0) + 1
    return {
        "generated": len(GENERATED),
        "generated_by_family": {fam: _gen.LEVEL[fam] for fam in _gen.FACTORIES},
        "near_misses": len(NEAR_MISSES),
        "constraint_briefs": len(CONSTRAINTS),
        "ambiguous_briefs": len(AMBIGUOUS),
        "total_briefs": (len(GENERATED) + len(NEAR_MISSES)
                         + len(CONSTRAINTS) + len(AMBIGUOUS)),
    }


def ids() -> List[str]:
    return ([b.id for b in GENERATED] + [nm.id for nm in NEAR_MISSES]
            + [b.id for b in CONSTRAINTS] + [b.id for b in AMBIGUOUS])
