"""Statistics. v1 had none -- not one interval, not one test.

``grep -nE "confidence|interval|significan|bootstrap|wilson"`` over
``eval/pressure/*.py`` returned zero hits before this file existed, and the
consequence was not that the harness over-claimed. It is that it UNDER-claimed:
the design is MATCHED (same briefs, same models, same seed, same budget), v1
counted 8 harness regressions against 0 harness wins, and the correct paired test
on 8-vs-0 discordant pairs gives p = 2^-7 ~= 0.008 two-sided. A reviewer's first
move against a 24/72-vs-18/72 headline is to point at n and walk away, and that
reviewer would have been wrong.

Four instruments, all stdlib, all exact where an exact answer exists:

``wilson``      the interval the book names (better coverage near 0 and 1 than
                the Wald interval every intro course teaches).
``mcnemar``     the correct test for PAIRED binary outcomes. Computed EXACTLY
                (the binomial sign test on the discordant pairs), not by the
                chi-squared approximation, because the discordant counts here are
                single digits and chi-squared is a large-sample story.
``pass@k``      "did ANY of k attempts work" -- the demo metric. Uses
                ``eval/bench/sequence/pass_at_k.py``, which already existed and
                which the pressure module never imported.
``pass^k``      "did ALL k attempts work" -- the conjunctive metric. This is the
                one a harness that hands a part to a CNC machine actually needs,
                and it is the one nobody reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from harnesscad.eval.bench.sequence.pass_at_k import estimate_pass_at_k

__all__ = ["Interval", "wilson", "McNemar", "mcnemar", "pass_at_k", "pass_hat_k",
           "Z95"]

#: The 97.5th percentile of the standard normal. The 95% two-sided z.
Z95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def to_dict(self) -> dict:
        return {"lo": self.lo, "hi": self.hi}

    def __str__(self) -> str:
        return f"[{100 * self.lo:.1f}%, {100 * self.hi:.1f}%]"


def wilson(successes: int, n: int, z: float = Z95) -> Interval:
    """The Wilson score interval on a proportion.

    Preferred over the Wald interval because its coverage does not collapse near
    p = 0 and p = 1 -- and several cells in this experiment sit at 0/12.
    """
    if n <= 0:
        return Interval(0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    lo, hi = centre - half, centre + half
    # At 0/n the two terms cancel to within float noise; clamp so the interval
    # reads [0.0, x] rather than [2.8e-17, x].
    eps = 1e-12
    return Interval(0.0 if lo < eps else min(1.0, lo),
                    1.0 if hi > 1.0 - eps else max(0.0, hi))


@dataclass(frozen=True)
class McNemar:
    """A paired comparison of two arms over the same cells."""

    b: int          # arm A solved, arm B did not
    c: int          # arm B solved, arm A did not
    n_pairs: int
    p_value: float  # exact, two-sided

    @property
    def discordant(self) -> int:
        return self.b + self.c

    def to_dict(self) -> dict:
        return {"b_only_a": self.b, "c_only_b": self.c,
                "discordant": self.discordant, "n_pairs": self.n_pairs,
                "p_value": self.p_value, "exact": True}


def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p at p0 = 0.5: sum of tails at least as extreme."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> McNemar:
    """Exact McNemar on paired binary outcomes. ``a`` and ``b`` are aligned.

    The exact (binomial sign) form is used deliberately. The chi-squared
    approximation -- even with continuity correction -- is unreliable when the
    discordant count is below ~25, and every comparison in this experiment is.
    """
    if len(a) != len(b):
        raise ValueError("paired arms must have the same number of cells")
    b_only = sum(1 for x, y in zip(a, b) if x and not y)
    c_only = sum(1 for x, y in zip(a, b) if y and not x)
    return McNemar(b=b_only, c=c_only, n_pairs=len(a),
                   p_value=_binom_two_sided(min(b_only, c_only), b_only + c_only))


def pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one of k draws from n samples, c correct, is correct).

    The unbiased HumanEval estimator. Delegates to the module that already
    implemented it and that ``eval/pressure`` never imported.
    """
    return estimate_pass_at_k(n, c, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """P(ALL of k draws from n samples, c correct, are correct) -- "pass^k".

    The conjunctive metric. pass@k is what you report at a demo; pass^k is what
    you need before a part goes to a machine, and it is brutal: a 70% pass@1 is a
    34% pass^3. Unbiased, by the same hypergeometric argument as pass@k.
    """
    if n < 0 or c < 0 or c > n or k < 1 or k > n:
        raise ValueError("require 0 <= c <= n and 1 <= k <= n")
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)
