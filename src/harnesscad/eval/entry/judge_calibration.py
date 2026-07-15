"""``python -m harnesscad.eval.entry.judge_calibration``

Wires ``eval.bench.judges.judge_calibration`` -- the threshold sweep the audit
(row 112, s14.3) named as the calibration procedure a judge must pass before it is
trusted -- into a runnable face.

Input JSON::

    {"records": [{"distance": 0.1, "accepted": true}, ...],
     "thresholds": [0.05, 0.1, 0.2]}

Output: the full precision/recall/F1 sweep and the F1-optimal threshold.
"""

from __future__ import annotations

import sys
from typing import Any, Optional, Sequence

from harnesscad.eval.bench.judges import judge_calibration as _mod
from harnesscad.eval.entry._common import build_main

__all__ = ["compute", "main"]

_SELFCHECK: Any = {
    "records": [
        {"distance": 0.02, "accepted": True},
        {"distance": 0.05, "accepted": True},
        {"distance": 0.30, "accepted": False},
        {"distance": 0.40, "accepted": False},
        {"distance": 0.12, "accepted": True},
    ],
    "thresholds": [0.05, 0.1, 0.2, 0.35],
}


def compute(doc: Any) -> dict:
    records = list(doc["records"])
    thresholds = list(doc["thresholds"])
    rows = _mod.calibrate_threshold(records, thresholds)
    selected = _mod.select_threshold(rows) if rows else None
    return {"calibration": list(rows), "selected": selected}


main = build_main(
    "judge_calibration",
    "Calibrate a compiler-judge distance threshold (precision/recall/F1 sweep).",
    compute, _SELFCHECK)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
