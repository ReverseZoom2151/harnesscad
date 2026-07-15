"""Public-benchmark leaderboards: the two publishing surfaces this repo owns.

Two scoreboards, one rule -- report the number the field quotes and the number
that is actually true, side by side, and never let one hide the other.

* :mod:`cadspot_board` -- ranks GUI-grounding submissions over
  :mod:`harnesscad.eval.grounding.cadspot`, split by the four surfaces a CAD user
  clicks (toolbar / dialog / tree / viewport). The viewport split is the
  contribution: it is the only self-labelling CAD grounding corpus, so it is the
  one region no other grounding leaderboard can rank at all.
* :mod:`hardcorpus_board` -- ranks text-to-CAD submissions over
  :mod:`harnesscad.eval.hardcorpus`, scoring each on BOTH the weak metrics the
  field uses (valid + IoU + Chamfer) AND the measured oracle (point membership),
  side by side. The gap between the two columns -- parts the field passes and only
  measurement catches -- is the novel, rankable result.

Both modules are pure scaffolding. They rank REPORTS that a future run produces;
they run no model themselves. Everything is deterministic and sorted, so a board
rendered twice from the same inputs is byte-identical.
"""

from __future__ import annotations

__all__ = ["cadspot_board", "hardcorpus_board"]
