"""``python -m harnesscad.eval.entry.judge_human_agreement``

Wires ``eval.bench.judges.judge_human_agreement`` -- Item / Cell / System
human-vs-judge agreement with bootstrap CIs -- into a runnable face. The audit
(row 116, s14.7) set the bar a judge must clear before it may instruct anything:
Cohen's/rank agreement with a human. This is the module that measures it.

Input JSON::

    {"item_pairs": [[human, judge], ...],
     "cell_pairs": [[human, judge, model_label], ...],
     "min_ci_n": 30, "n_boot": 2000, "seed": 42}

Output: the full agreement report (Item / Cell / System levels).
"""

from __future__ import annotations

import sys
from typing import Any

from harnesscad.eval.bench.judges import judge_human_agreement as _mod
from harnesscad.eval.entry._common import build_main

__all__ = ["compute", "main"]

_SELFCHECK: Any = {
    "item_pairs": [[5, 5], [4, 4], [3, 2], [2, 3], [1, 1], [5, 4], [2, 2], [4, 5]],
    "cell_pairs": [[5, 5, "A"], [4, 4, "A"], [3, 2, "B"], [2, 3, "B"],
                   [1, 1, "C"], [5, 4, "C"]],
    "min_ci_n": 4,
    "n_boot": 200,
    "seed": 42,
}


def compute(doc: Any) -> dict:
    item_pairs = [tuple(p) for p in doc.get("item_pairs", [])]
    cell_pairs = [tuple(p) for p in doc.get("cell_pairs", [])]
    return _mod.agreement_report(
        item_pairs, cell_pairs,
        min_ci_n=int(doc.get("min_ci_n", 30)),
        n_boot=int(doc.get("n_boot", 2000)),
        seed=int(doc.get("seed", 42)))


main = build_main(
    "judge_human_agreement",
    "Item / Cell / System human-vs-judge agreement with bootstrap CIs.",
    compute, _SELFCHECK)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
