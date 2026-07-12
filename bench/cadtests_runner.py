"""CADTEST suite runner (Mallis et al., "Text-to-CAD Evaluation with CADTESTS",
Sec. 4-5).

The overall evaluation of a suite ``T = {T_1, ..., T_N}`` on a model ``m`` is the
*conjunction* ``T(m) = AND_i T_i(m)`` (Sec. 4): the sample is correct only if it
passes every test, exactly as HumanEval/MBPP-style code benchmarks require a
sample to pass all tests. Tests are further organised into *prompt-requirement
groups* (multiple tests may verify the same requirement, supplementary A), and a
requirement counts as satisfied only if all of its tests pass.

A generated model may also fail to execute (a runtime error during generation).
Unlike prior protocols that discard invalid outputs, CADTESTBENCH treats an
invalid generation as a failure of every metric (Sec. 5, "Evaluation"). Here a
``None`` model represents such an invalid generation.

This module provides:

  * :func:`run_suite` -- run a suite over one (possibly ``None``) model, returning
    per-test results, the conjunction, per-requirement satisfaction and the
    per-sample Pass-Rate / Requirement-Score.
  * :func:`run_passing_set` -- run a suite over a passing set of pose/scale
    augmentations and check invariance (a robust test passes on every variant).

Deterministic, stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from bench.cadtests_assertions import CATEGORIES, TestResult


@dataclass(frozen=True)
class SampleResult:
    """Result of running a whole suite on one sample."""
    results: Tuple[TestResult, ...]
    invalid: bool                 # generation failed to execute (model is None)
    num_tests: int
    num_passed: int
    passed_all: bool              # T(m) = conjunction
    pass_rate: float              # 1.0 iff passed_all (per-sample PR)
    requirement_score: float      # fraction of requirement groups satisfied
    requirement_satisfied: dict   # {requirement: bool}
    category_counts: dict         # {category: (passed, total)}

    @property
    def num_invalid_tests(self):
        """Tests that raised at runtime on this model."""
        return sum(1 for r in self.results if not r.valid)


def requirement_groups(tests):
    """Group tests by their ``requirement`` field (order-preserving).

    Tests with ``requirement is None`` each form their own singleton group keyed
    by test name, so an ungrouped suite still yields well-defined requirements.
    """
    groups = {}
    for t in tests:
        key = t.requirement if t.requirement is not None else "__%s__" % t.name
        groups.setdefault(key, []).append(t)
    return groups


def run_suite(model, tests):
    """Run ``tests`` on ``model`` (or ``None`` for an invalid generation).

    Returns a :class:`SampleResult`. When ``model is None`` every test is treated
    as failed and the sample is marked invalid with zero PR and RS.
    """
    tests = tuple(tests)
    n = len(tests)
    groups = requirement_groups(tests)

    if model is None:
        results = tuple(
            TestResult(t.name, t.category, False,
                       "invalid generation: model failed to execute",
                       t.requirement, error="InvalidGeneration")
            for t in tests)
        req_sat = {k: False for k in groups}
        cat_counts = {c: (0, sum(1 for t in tests if t.category == c))
                      for c in CATEGORIES}
        return SampleResult(results, True, n, 0, False, 0.0, 0.0, req_sat,
                            cat_counts)

    results = tuple(t.evaluate(model) for t in tests)
    by_name = {r.name: r for r in results}

    num_passed = sum(1 for r in results if r.passed)
    passed_all = n > 0 and num_passed == n

    # Requirement satisfaction: all tests in the group must pass.
    req_sat = {}
    for key, grp in groups.items():
        req_sat[key] = all(by_name[t.name].passed for t in grp)
    rs = (sum(1 for v in req_sat.values() if v) / len(req_sat)) if req_sat \
        else 0.0

    # Per-category (passed, total) tallies for accuracy aggregation.
    cat_counts = {}
    for c in CATEGORIES:
        grp = [r for r in results if r.category == c]
        cat_counts[c] = (sum(1 for r in grp if r.passed), len(grp))

    return SampleResult(
        results=results,
        invalid=False,
        num_tests=n,
        num_passed=num_passed,
        passed_all=passed_all,
        pass_rate=1.0 if passed_all else 0.0,
        requirement_score=rs,
        requirement_satisfied=req_sat,
        category_counts=cat_counts,
    )


@dataclass(frozen=True)
class InvarianceResult:
    """Per-test outcome across a passing set of augmentations."""
    per_variant: Tuple[SampleResult, ...]
    invariant_pass: bool          # every variant passes the whole suite
    unstable_tests: Tuple[str, ...]  # tests that flip pass<->fail across poses


def run_passing_set(variants, tests):
    """Run ``tests`` over a passing set of similarity-transformed models.

    ``variants`` is an iterable of models (e.g. from
    :func:`bench.cadtests_model.similarity_augmentations`). Well-formed CADTESTS
    are pose/scale invariant, so a sound suite yields ``invariant_pass=True`` and
    an empty ``unstable_tests``. A test that passes on some poses but fails on
    others is flagged as unstable -- the failure mode the augmentation guards
    against (Sec. 5).
    """
    variants = tuple(variants)
    if not variants:
        raise ValueError("passing set must contain at least one model")
    per_variant = tuple(run_suite(m, tests) for m in variants)

    # A test is unstable if its pass outcome is not constant across variants.
    outcomes = {}
    for sr in per_variant:
        for r in sr.results:
            outcomes.setdefault(r.name, set()).add(r.passed)
    unstable = tuple(sorted(name for name, s in outcomes.items() if len(s) > 1))

    invariant_pass = all(sr.passed_all for sr in per_variant)
    return InvarianceResult(per_variant, invariant_pass, unstable)
