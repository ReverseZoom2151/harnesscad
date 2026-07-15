"""``python -m harnesscad.eval.entry.pass_at_k``

Wires ``eval.bench.sequence.pass_at_k`` -- the unbiased HumanEval pass@k estimator
-- into a runnable face. The audit (row 114, s14.5, gap in s2.6) found the pressure
test reporting a raw solve-rate that "is neither pass@1 nor pass@3 nor pass^k -- it
is an unnamed quantity", while the estimator sat in the repo imported by nothing.

Input JSON, either a batch::

    {"counts": [[n, c], ...], "k": 3}

or a single cell::

    {"n": 10, "c": 3, "k": 3}

Output: the per-cell pass@k and, for a batch, the macro-averaged pass@k. When
``pass_conjunctive`` is requested (``"conjunctive": true``) it also reports pass^k
(ALL k attempts succeed) -- the reliability metric a CAD harness actually needs.
"""

from __future__ import annotations

import sys
from typing import Any

from harnesscad.eval.bench.sequence import pass_at_k as _mod
from harnesscad.eval.entry._common import build_main

__all__ = ["compute", "pass_hat_k", "main"]

_SELFCHECK: Any = {"counts": [[10, 3], [10, 7], [5, 0], [8, 8]], "k": 3,
                   "conjunctive": True}


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k: the probability all k sampled attempts succeed (no replacement).

    The conjunctive counterpart of the module's disjunctive pass@k. Written here,
    against the same combinatorial model, because reliability is a conjunction:
    a part handed to a machine must be right EVERY time, not right at least once.
    """
    from math import comb
    if n < 0 or c < 0 or c > n or k < 1 or k > n:
        raise ValueError("require 0<=c<=n and 1<=k<=n")
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


def compute(doc: Any) -> dict:
    k = int(doc["k"])
    conjunctive = bool(doc.get("conjunctive", False))
    if "counts" in doc:
        counts = [(int(n), int(c)) for n, c in doc["counts"]]
        per_cell = [
            {"n": n, "c": c,
             "pass_at_k": _mod.estimate_pass_at_k(n, c, k),
             **({"pass_hat_k": pass_hat_k(n, c, k)} if conjunctive else {})}
            for n, c in counts]
        out: dict = {"k": k, "per_cell": per_cell,
                     "macro_pass_at_k": _mod.macro_pass_at_k(counts, k)}
        if conjunctive:
            vals = [pass_hat_k(n, c, k) for n, c in counts]
            out["macro_pass_hat_k"] = sum(vals) / len(vals) if vals else None
        return out
    n, c = int(doc["n"]), int(doc["c"])
    out = {"k": k, "n": n, "c": c, "pass_at_k": _mod.estimate_pass_at_k(n, c, k)}
    if conjunctive:
        out["pass_hat_k"] = pass_hat_k(n, c, k)
    return out


main = build_main(
    "pass_at_k",
    "Unbiased pass@k (and conjunctive pass^k) over sampled attempt counts.",
    compute, _SELFCHECK)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
