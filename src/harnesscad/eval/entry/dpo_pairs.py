"""``python -m harnesscad.eval.entry.dpo_pairs``

Wires ``data.dataengine.preference.dpo_pairs`` -- oracle-labelled preference-pair
construction -- into a runnable face WITHOUT editing the data module (owned
elsewhere). The audit (rows 47, 108, s3.2) calls this "the single cheapest training
axis available" and states the one rule that makes it safe: pairs must be labelled
by the ORACLE, never by the verifier fleet. This wrapper only marshals rewards a
caller already computed; it does not itself decide which labeller produced them,
and it prints that warning next to its output so the rule is not forgotten.

Input JSON::

    {"samples": [{"code": "...", "reward": 1.0}, ...],
     "mode": "all" | "sample", "count": 8, "seed": 0, "prompt": "..."}

Output: ordered (chosen, rejected) pairs and their DPO records.
"""

from __future__ import annotations

import sys
from typing import Any

from harnesscad.data.dataengine.preference import dpo_pairs as _mod
from harnesscad.eval.entry._common import build_main

__all__ = ["compute", "main"]

_WARNING = ("Preference pairs are only as sound as the reward that labelled them. "
            "Per audit s2.4 / s3.2 and DPO failure-mode #4, the reward MUST come "
            "from the differential/golden oracle, NEVER from the verifier fleet -- "
            "DPO memorises individual pairs and a fleet false positive trains the "
            "model to reject correct parts.")

_SELFCHECK: Any = {
    "samples": [
        {"code": "box(10)", "reward": 1.0},
        {"code": "box(11)", "reward": 0.0},
        {"code": "box(9)", "reward": 0.5},
    ],
    "mode": "all",
    "prompt": "a 10mm cube",
}


def compute(doc: Any) -> dict:
    samples = list(doc["samples"])
    mode = str(doc.get("mode", "all"))
    prompt = str(doc.get("prompt", ""))
    if mode == "sample":
        pairs = _mod.sample_preference_pairs(
            samples, int(doc.get("count", 0)), seed=int(doc.get("seed", 0)))
    elif mode == "all":
        pairs = _mod.all_preference_pairs(samples)
    else:
        raise ValueError("mode must be 'all' or 'sample', got %r" % mode)
    records = _mod.to_dpo_records(pairs, prompt=prompt)
    return {"_labeller_warning": _WARNING,
            "n_pairs": len(records),
            "records": records}


main = build_main(
    "dpo_pairs",
    "Construct oracle-labelled DPO preference pairs from K sampled programs.",
    compute, _SELFCHECK)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
