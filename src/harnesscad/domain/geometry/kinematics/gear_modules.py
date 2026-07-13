"""Standard gear-module series selection (deterministic, stdlib-only).

CAD-GPT's design prompt (``paper_code/prompts/design_prompt.py``) instructs the
LLM to snap any computed gear module to a preferred standard value:

    "all obtained module number can be considered to round up to following
     numbers in priority: 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 14, 16,
     20, 25, 32, 40, 50. If there is no optimal choice, you can also choose
     from: 1.75, 2.25, 2.75, 3.5, 4.5, 5.5, 7, 9, 14, 18, 22, 28, 36"

That snap is a deterministic lookup over the ISO-54 preferred-number series — no
LLM needed. This module implements it as a pure numeric routine. The harness had
no standard-module selection (its ``spur_gear_blank`` takes an arbitrary float
module); this fills that gap.

"Round up" is the paper's stated priority: the chosen standard is the smallest
standard value >= the computed module, so the geometry is never weaker than the
requirement. A ``nearest`` mode is also provided for callers that want the
closest standard regardless of direction.
"""

from __future__ import annotations

from typing import List, Optional

__all__ = [
    "PREFERRED_MODULES",
    "SECONDARY_MODULES",
    "ALL_MODULES",
    "standard_module",
    "nearest_module",
    "is_standard_module",
]

# Series 1 (preferred), from the CAD-GPT design prompt / ISO 54.
PREFERRED_MODULES: List[float] = [
    1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0,
    16.0, 20.0, 25.0, 32.0, 40.0, 50.0,
]

# Series 2 (secondary fallback), from the same prompt.
SECONDARY_MODULES: List[float] = [
    1.75, 2.25, 2.75, 3.5, 4.5, 5.5, 7.0, 9.0, 14.0, 18.0, 22.0, 28.0, 36.0,
]

# Combined sorted series (deduplicated).
ALL_MODULES: List[float] = sorted(set(PREFERRED_MODULES) | set(SECONDARY_MODULES))

_TOL = 1e-9


def is_standard_module(value: float, allow_secondary: bool = True) -> bool:
    """True if ``value`` equals a standard module (within tolerance)."""
    series = ALL_MODULES if allow_secondary else PREFERRED_MODULES
    return any(abs(value - m) <= _TOL for m in series)


def standard_module(computed: float, allow_secondary: bool = False) -> float:
    """Round a computed module UP to the smallest standard value >= ``computed``.

    Preferred (series 1) values are used unless ``allow_secondary`` is set, in
    which case a nearer series-2 value may win. Ties (a value present in both, or
    exactly equal) resolve to the standard value itself.

    Raises ``ValueError`` if ``computed`` exceeds the largest standard module.
    """
    if computed <= 0:
        raise ValueError("computed module must be positive")
    series = ALL_MODULES if allow_secondary else PREFERRED_MODULES
    for m in series:                       # series is ascending
        if m >= computed - _TOL:
            return m
    raise ValueError(
        "computed module %r exceeds the largest standard module %r"
        % (computed, series[-1])
    )


def nearest_module(computed: float, allow_secondary: bool = True) -> float:
    """Return the standard module CLOSEST to ``computed`` (either direction).

    Ties (equidistant) resolve to the larger value, matching the "round up"
    preference of the design prompt.
    """
    if computed <= 0:
        raise ValueError("computed module must be positive")
    series = ALL_MODULES if allow_secondary else PREFERRED_MODULES
    best: Optional[float] = None
    best_d = None
    for m in series:
        d = abs(m - computed)
        if best is None or d < best_d - _TOL or (abs(d - best_d) <= _TOL and m > best):
            best, best_d = m, d
    return best
