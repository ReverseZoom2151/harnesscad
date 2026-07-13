"""Tolerance-stack analyzer for 1-D assembly dimension chains.

A *tolerance chain* is an ordered list of nominal dimensions, each with a
plus/minus tolerance and a direction (+1 adds to the accumulated result,
-1 subtracts). The chain result is a critical assembly quantity — a gap,
an alignment offset, a fit — and the question this module answers is
whether the accumulated tolerance keeps that result inside a functional
requirement window ``[target - tolerance, target + tolerance]``.

Three independent accumulation methods are computed, cheapest-to-truest:

  * **Worst case** — arithmetic sum of every tolerance in its worst
    direction. Guaranteed bound, always conservative; if worst-case
    passes the assembly can never be out of spec.
  * **RSS (root-sum-square)** — statistical combination assuming each
    dimension's tolerance band is a 3-sigma spread and the errors are
    independent. The realistic 99.7% band; far tighter than worst case
    for long chains (the tolerances partially cancel).
  * **Monte Carlo** — draws ``mc_samples`` samples per dimension from a
    per-dimension distribution (normal / uniform / triangular), sums the
    chain, and measures the empirical yield (fraction of samples inside
    the requirement) plus the process-capability index ``Cpk``.

The analyzer also decomposes the total variance by dimension, so the
single tolerance driving the stack is obvious — the highest-leverage
place to tighten a tolerance or re-order the design.

This is the deterministic statistical core of a tolerance gate: it is
pure stdlib, has no CAD-kernel dependency, and is fully reproducible for
a fixed ``seed``. It complements — and does not duplicate — the geometry
verifiers: those check placed solids; this checks a declared dimension
chain before anything is placed.

Example::

    chain = ToleranceChain("motor_face_gap")
    chain.add("beam_length", nominal=1000.0, plus=0.5)
    chain.add("shim",        nominal=4.0,    plus=0.1)
    chain.add("plate",       nominal=5.0,    plus=0.2)
    chain.add("motor_offset", nominal=1009.0, plus=0.3, direction=-1.0)
    result = chain.analyze(target=0.0, tolerance=0.5, seed=7)
    print(result.worst_case_passed, result.rss_passed, result.cpk)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_VALID_DISTRIBUTIONS = ("normal", "uniform", "triangular")


@dataclass
class Dimension:
    """One dimension in a tolerance chain.

    ``plus`` and ``minus`` are stored as positive magnitudes. ``minus``
    defaults to ``plus`` (a symmetric bilateral tolerance) when the chain
    builder is not given an explicit value.
    """
    name: str
    nominal: float
    plus: float
    minus: float
    distribution: str = "normal"
    direction: float = 1.0

    def __post_init__(self) -> None:
        if self.plus < 0 or self.minus < 0:
            raise ValueError("tolerances must be non-negative magnitudes")
        if self.distribution not in _VALID_DISTRIBUTIONS:
            raise ValueError(
                f"unknown distribution {self.distribution!r}; "
                f"choose one of {_VALID_DISTRIBUTIONS}")
        if self.direction not in (1.0, -1.0):
            raise ValueError("direction must be +1.0 or -1.0")

    @property
    def bilateral(self) -> float:
        """Half-width of the tolerance band (mean of plus and minus)."""
        return (self.plus + self.minus) / 2.0

    @property
    def mean(self) -> float:
        """Mean value, shifted from nominal when the tolerance is asymmetric."""
        return self.nominal + (self.plus - self.minus) / 2.0

    def sample(self, rng: random.Random) -> float:
        """Draw one sample from this dimension's distribution."""
        if self.distribution == "uniform":
            return rng.uniform(self.nominal - self.minus,
                               self.nominal + self.plus)
        if self.distribution == "triangular":
            return rng.triangular(self.nominal - self.minus,
                                  self.nominal + self.plus,
                                  self.nominal)
        # normal: treat the bilateral band as a 3-sigma spread
        sigma = self.bilateral / 3.0
        if sigma == 0.0:
            return self.mean
        return rng.gauss(self.mean, sigma)


@dataclass
class StackResult:
    """Outcome of a tolerance-stack analysis."""
    chain_name: str
    nominal_result: float
    target: float
    tolerance: float

    worst_case_min: float
    worst_case_max: float
    worst_case_range: float
    worst_case_passed: bool

    rss_min: float
    rss_max: float
    rss_range: float
    rss_passed: bool

    mc_mean: float
    mc_std: float
    mc_min: float
    mc_max: float
    mc_yield_pct: float
    mc_passed: bool

    cpk: float
    contributors: List[Dict[str, float]] = field(default_factory=list)

    @property
    def dominant_contributor(self) -> Optional[str]:
        """Name of the dimension contributing the most variance, if any."""
        if not self.contributors:
            return None
        return max(self.contributors, key=lambda c: c["variance_pct"])["name"]


class ToleranceChain:
    """Build and analyze a 1-D tolerance chain."""

    def __init__(self, name: str):
        self.name = name
        self.dimensions: List[Dimension] = []

    def add(self, name: str, nominal: float, plus: float,
            minus: Optional[float] = None, distribution: str = "normal",
            direction: float = 1.0) -> "ToleranceChain":
        """Append a dimension; returns ``self`` for a fluent builder."""
        if minus is None:
            minus = plus
        self.dimensions.append(Dimension(
            name=name, nominal=nominal, plus=plus, minus=minus,
            distribution=distribution, direction=direction))
        return self

    def nominal(self) -> float:
        """Signed sum of the nominal dimensions."""
        return sum(d.direction * d.nominal for d in self.dimensions)

    def analyze(self, target: float = 0.0, tolerance: float = 0.5,
                mc_samples: int = 10000, seed: int = 42) -> StackResult:
        if not self.dimensions:
            raise ValueError("cannot analyze an empty tolerance chain")
        if tolerance < 0:
            raise ValueError("tolerance (requirement half-width) must be >= 0")

        lo, hi = target - tolerance, target + tolerance
        nominal_result = self.nominal()

        # --- worst case: tolerances add in their worst direction ---
        wc_plus = sum(d.plus if d.direction > 0 else d.minus
                      for d in self.dimensions)
        wc_minus = sum(d.minus if d.direction > 0 else d.plus
                       for d in self.dimensions)
        wc_max = nominal_result + wc_plus
        wc_min = nominal_result - wc_minus
        wc_passed = wc_min >= lo and wc_max <= hi

        # --- RSS: independent 3-sigma bands combine in quadrature ---
        rss_total = math.sqrt(sum(d.bilateral ** 2 for d in self.dimensions))
        rss_max = nominal_result + rss_total
        rss_min = nominal_result - rss_total
        rss_passed = rss_min >= lo and rss_max <= hi

        # --- Monte Carlo ---
        rng = random.Random(seed)
        samples = [sum(d.direction * d.sample(rng) for d in self.dimensions)
                   for _ in range(mc_samples)]
        mc_mean = sum(samples) / len(samples)
        mc_var = sum((s - mc_mean) ** 2 for s in samples) / len(samples)
        mc_std = math.sqrt(mc_var)
        in_spec = sum(1 for s in samples if lo <= s <= hi)
        mc_yield = 100.0 * in_spec / len(samples)
        mc_passed = mc_yield >= 99.73

        # --- Cpk (process capability, two-sided) ---
        if mc_std > 0:
            cpk = min((hi - mc_mean) / (3 * mc_std),
                      (mc_mean - lo) / (3 * mc_std))
        else:
            cpk = math.inf

        # --- variance decomposition ---
        total_var = sum(d.bilateral ** 2 for d in self.dimensions)
        contributors: List[Dict[str, float]] = []
        for d in self.dimensions:
            var = d.bilateral ** 2
            pct = 100.0 * var / total_var if total_var > 0 else 0.0
            contributors.append({
                "name": d.name,
                "tolerance": d.bilateral,
                "variance_pct": pct,
            })

        return StackResult(
            chain_name=self.name,
            nominal_result=nominal_result,
            target=target,
            tolerance=tolerance,
            worst_case_min=wc_min,
            worst_case_max=wc_max,
            worst_case_range=wc_max - wc_min,
            worst_case_passed=wc_passed,
            rss_min=rss_min,
            rss_max=rss_max,
            rss_range=2 * rss_total,
            rss_passed=rss_passed,
            mc_mean=mc_mean,
            mc_std=mc_std,
            mc_min=min(samples),
            mc_max=max(samples),
            mc_yield_pct=mc_yield,
            mc_passed=mc_passed,
            cpk=cpk,
            contributors=contributors,
        )
