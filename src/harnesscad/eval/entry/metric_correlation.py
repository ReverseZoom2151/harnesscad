"""``python -m harnesscad.eval.entry.metric_correlation``

Wires ``eval.bench.harness.metric_correlation`` -- the cross-family correlation
table -- into a runnable face. The audit (row 110, s14.1, gap #10) asks the one
question this module answers and nobody ran: of ~200 intrinsic metrics, WHICH
correlate with the outcome that matters? A metric that predicts nothing is
decoration.

Input JSON::

    {"recon": {"metric_a": [per-shape values...], ...},
     "seg":   {"metric_x": [per-shape values...], ...}}

Output: the Pearson correlation grid {recon_metric: {seg_metric: r}}.
"""

from __future__ import annotations

import sys
from typing import Any

from harnesscad.eval.bench.harness import metric_correlation as _mod
from harnesscad.eval.entry._common import build_main

__all__ = ["compute", "main"]

_SELFCHECK: Any = {
    "recon": {"chamfer": [0.9, 0.8, 0.7, 0.6], "iou": [0.95, 0.85, 0.75, 0.65]},
    "seg": {"solved": [1.0, 1.0, 0.0, 0.0], "miou": [0.88, 0.80, 0.55, 0.40]},
}


def compute(doc: Any) -> dict:
    table = _mod.correlation_table(dict(doc["recon"]), dict(doc["seg"]))
    return {"correlation_table": table}


main = build_main(
    "metric_correlation",
    "Pearson correlation grid between two families of per-shape metrics.",
    compute, _SELFCHECK)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
