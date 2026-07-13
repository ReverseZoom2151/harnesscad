"""CADTESTBENCH metrics and mutation analysis (Mallis et al., "Text-to-CAD
Evaluation with CADTESTS", Sec. 5-6).

Two families of deterministic scores are implemented.

1. Benchmark evaluation of Text-to-CAD methods (Sec. 5 "Evaluation", Tab. 3).
   Over a set of generated samples, report four complementary metrics:

     * Pass-Rate (PR)        -- fraction of samples where *all* CADTESTS pass.
     * Requirement Score (RS)-- mean over samples of the fraction of prompt
                                requirement groups satisfied (a group is
                                satisfied iff all its tests pass).
     * Accuracy (Acc)        -- per-category fraction of tests passed (Fig. 3),
                                computed over all tests of all samples.
     * Invalid-Ratio (IR)    -- fraction of samples that execute with a runtime
                                error. Invalid generations count as failures in
                                every metric (never excluded).

2. Test-suite quality via mutation analysis (Sec. 4, 6; Tab. 1-2). Given a
   generated suite, a reference model and a set of mutants (hard negatives that
   deliberately violate the prompt), report:

     * Valid  -- fraction of generated tests that execute on the reference
                 without raising.
     * Sound  -- fraction of *valid* tests that pass on the reference.
     * MScore -- mutation score: fraction of mutants *killed*, i.e. detected by
                 at least one sound test (the test fails on the mutant).

These mirror the "Valid / Sound / MScore" columns of Tab. 1 and the refinement
effectiveness of Tab. 2. A suite kills every mutant iff MScore == 1.

Deterministic, stdlib-only. Consumes :class:`bench.cadtests_runner.SampleResult`
and :class:`bench.cadtests_assertions.CadTest`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from harnesscad.eval.bench.protocols.test_assertions import CATEGORIES
from harnesscad.eval.bench.protocols.test_suite_runner import run_suite


# ---------------------------------------------------------------------------
# Benchmark evaluation of Text-to-CAD methods (Tab. 3, Fig. 3)
# ---------------------------------------------------------------------------
def benchmark_scores(sample_results, *, as_percent=False):
    """Aggregate per-sample :class:`SampleResult` objects into Tab.-3 metrics.

    Returns a dict with ``n``, ``pass_rate`` (PR), ``requirement_score`` (RS),
    ``invalid_ratio`` (IR), ``accuracy`` (per-category dict + ``overall``), and
    ``category_accuracy`` alias. With ``as_percent`` the PR/RS/IR and accuracies
    are scaled to 0-100 as in the paper's tables.
    """
    rows = list(sample_results)
    n = len(rows)
    if n == 0:
        raise ValueError("no samples")
    scale = 100.0 if as_percent else 1.0

    pr = scale * sum(r.pass_rate for r in rows) / n
    rs = scale * sum(r.requirement_score for r in rows) / n
    ir = scale * sum(1 for r in rows if r.invalid) / n

    # Per-category accuracy over all tests of all samples. Invalid samples
    # contribute their (0-passed, total) counts, so they lower accuracy too.
    cat_pass = {c: 0 for c in CATEGORIES}
    cat_total = {c: 0 for c in CATEGORIES}
    for r in rows:
        for c in CATEGORIES:
            p, t = r.category_counts.get(c, (0, 0))
            cat_pass[c] += p
            cat_total[c] += t
    accuracy = {}
    for c in CATEGORIES:
        accuracy[c] = (scale * cat_pass[c] / cat_total[c]) if cat_total[c] \
            else None
    tot_p = sum(cat_pass.values())
    tot_t = sum(cat_total.values())
    accuracy["overall"] = (scale * tot_p / tot_t) if tot_t else None

    return {
        "n": n,
        "pass_rate": pr,
        "requirement_score": rs,
        "invalid_ratio": ir,
        "accuracy": accuracy,
        "category_accuracy": accuracy,
    }


def evaluate_method(samples, *, as_percent=False):
    """Convenience: run suites over ``(model, tests)`` pairs then aggregate.

    ``samples`` is an iterable of ``(model, tests)`` where ``model`` may be
    ``None`` for an invalid generation. Returns the same dict as
    :func:`benchmark_scores`, with the individual :class:`SampleResult` objects
    under ``rows``.
    """
    results = [run_suite(model, tests) for (model, tests) in samples]
    out = benchmark_scores(results, as_percent=as_percent)
    out["rows"] = tuple(results)
    return out


# ---------------------------------------------------------------------------
# Test-suite quality via mutation analysis (Tab. 1-2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SuiteQuality:
    """Validity / soundness / mutation-score of a generated CADTEST suite."""
    n_tests: int
    n_valid: int
    n_sound: int
    validity: float           # valid / total
    soundness: float          # sound / valid
    n_mutants: int
    n_killed: int
    mutation_score: float     # killed / mutants
    killed: Tuple[bool, ...]  # per-mutant kill flags (mutant order preserved)
    sound_test_names: Tuple[str, ...]


def analyze_test_suite(tests, reference_model, mutants):
    """Mutation analysis of a generated suite (Sec. 4, Tab. 1-2).

    ``tests``            iterable of :class:`CadTest`.
    ``reference_model``  the ground-truth reference B-rep (the passing model).
    ``mutants``          iterable of mutant models violating the prompt.

    A test is *valid* if it does not raise on the reference, *sound* if it is
    valid and passes on the reference. A mutant is *killed* if at least one sound
    test fails (without error) on it. Returns a :class:`SuiteQuality`.
    """
    tests = tuple(tests)
    n = len(tests)

    ref_results = [(t, t.evaluate(reference_model)) for t in tests]
    n_valid = sum(1 for _, r in ref_results if r.valid)
    sound_tests = [t for t, r in ref_results if r.valid and r.passed]
    n_sound = len(sound_tests)

    mutants = tuple(mutants)
    killed = []
    for mut in mutants:
        detected = False
        for t in sound_tests:
            r = t.evaluate(mut)
            # A mutant is killed by a test that fails on it without erroring.
            if r.valid and not r.passed:
                detected = True
                break
        killed.append(detected)
    n_killed = sum(1 for k in killed if k)
    n_mut = len(mutants)

    return SuiteQuality(
        n_tests=n,
        n_valid=n_valid,
        n_sound=n_sound,
        validity=(n_valid / n) if n else 0.0,
        soundness=(n_sound / n_valid) if n_valid else 0.0,
        n_mutants=n_mut,
        n_killed=n_killed,
        mutation_score=(n_killed / n_mut) if n_mut else 0.0,
        killed=tuple(killed),
        sound_test_names=tuple(t.name for t in sound_tests),
    )


def refinement_gain(before, after):
    """Mutation-score improvement of the refinement loop (Tab. 2).

    ``before`` and ``after`` are :class:`SuiteQuality` from the initial and
    refined suites. Returns the absolute and relative gain in mutation score
    plus the change in the sound-test count.
    """
    delta = after.mutation_score - before.mutation_score
    rel = (delta / before.mutation_score) if before.mutation_score else None
    return {
        "mutation_score_before": before.mutation_score,
        "mutation_score_after": after.mutation_score,
        "absolute_gain": delta,
        "relative_gain": rel,
        "sound_tests_delta": after.n_sound - before.n_sound,
    }
