"""``python -m harnesscad.eval.entry.ablation_matrix``

Wires ``governance.research.ablation_matrix`` -- paired, stratified A/B summaries
-- into a runnable face WITHOUT editing the governance module (owned elsewhere).
This is the instrument the pressure experiment needed and improvised by hand: a
paired treatment-minus-control delta per stratum, with wins and losses counted.
The audit's whole finding is a paired A/B (typed loop vs blind resample); this
module is how such an A/B is summarised honestly.

Input JSON::

    {"rows": [{"stratum": "14b", "pair_id": "brief3",
               "variant": "control", "solved": 0.0}, ...],
     "metric": "solved"}

``variant`` must be ``"control"`` or ``"treatment"``; only pairs that carry both
contribute a delta. Output: per-stratum ``{n, mean_delta, wins, losses}``.
"""

from __future__ import annotations

import sys
from typing import Any

from harnesscad.eval.entry._common import build_main
from harnesscad.governance.research import ablation_matrix as _mod

__all__ = ["compute", "main"]

_SELFCHECK: Any = {
    "rows": [
        {"stratum": "7b", "pair_id": "b1", "variant": "control", "solved": 0.0},
        {"stratum": "7b", "pair_id": "b1", "variant": "treatment", "solved": 1.0},
        {"stratum": "7b", "pair_id": "b2", "variant": "control", "solved": 1.0},
        {"stratum": "7b", "pair_id": "b2", "variant": "treatment", "solved": 0.0},
        {"stratum": "14b", "pair_id": "b3", "variant": "control", "solved": 1.0},
        {"stratum": "14b", "pair_id": "b3", "variant": "treatment", "solved": 1.0},
    ],
    "metric": "solved",
}


def compute(doc: Any) -> dict:
    strata = _mod.compare_ablation(list(doc["rows"]), metric=str(doc["metric"]))
    return {"metric": doc["metric"], "by_stratum": strata}


main = build_main(
    "ablation_matrix",
    "Paired, stratified treatment-minus-control ablation summary.",
    compute, _SELFCHECK)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
