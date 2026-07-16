import unittest

from harnesscad.agents.generation.prompt_hillclimb import (
    DEFAULT_WEIGHTS,
    ScoreReport,
    composite_score,
    hillclimb,
    main,
)


def scored(table):
    return lambda c: ScoreReport(score=table[c])


def sequence(names):
    it = iter(names)
    return lambda _current, _history: next(it)


class ClimbTests(unittest.TestCase):
    def test_keeps_improving_candidate(self):
        r = hillclimb("v0", sequence(["v1", "v2"]),
                      scored({"v0": 0.5, "v1": 0.7, "v2": 0.9}), iterations=2)
        self.assertEqual(r.best, "v2")
        self.assertEqual(r.best_score, 0.9)

    def test_never_regresses_below_baseline(self):
        r = hillclimb("v0", sequence(["b1", "b2"]),
                      scored({"v0": 0.8, "b1": 0.1, "b2": 0.2}), iterations=2)
        self.assertEqual(r.best, "v0")
        self.assertEqual(r.best_score, 0.8)

    def test_zero_iterations_scores_baseline_only(self):
        r = hillclimb("v0", sequence([]), scored({"v0": 0.42}), iterations=0)
        self.assertEqual(r.best, "v0")
        self.assertEqual(r.best_score, 0.42)
        self.assertEqual(len(r.history), 1)

    def test_negative_iterations_rejected(self):
        with self.assertRaises(ValueError):
            hillclimb("v0", sequence([]), scored({"v0": 1.0}), iterations=-1)

    def test_negative_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            hillclimb("v0", sequence([]), scored({"v0": 1.0}), tolerance=-0.1)

    def test_equal_score_is_kept_as_best(self):
        r = hillclimb("v0", sequence(["v1"]), scored({"v0": 0.5, "v1": 0.5}),
                      iterations=1)
        self.assertEqual(r.best, "v1")
        self.assertTrue(r.history[1].is_best)


class NoRatchetTests(unittest.TestCase):
    def test_lateral_move_kept_but_bar_holds(self):
        r = hillclimb("v0", sequence(["lat"]), scored({"v0": 0.8, "lat": 0.77}),
                      iterations=1, tolerance=0.05)
        self.assertTrue(r.history[1].kept)
        self.assertFalse(r.history[1].is_best)
        self.assertEqual(r.best_score, 0.8)
        self.assertEqual(r.best, "v0")

    def test_chain_of_small_regressions_cannot_walk_bar_down(self):
        table = {"v0": 0.90, "a": 0.86, "b": 0.82, "c": 0.78, "d": 0.74}
        r = hillclimb("v0", sequence(["a", "b", "c", "d"]), scored(table),
                      iterations=4, tolerance=0.05)
        self.assertEqual(r.best, "v0")
        self.assertEqual(r.best_score, 0.90)
        self.assertEqual([v.kept for v in r.history],
                         [True, True, False, False, False])

    def test_outside_tolerance_is_discarded(self):
        r = hillclimb("v0", sequence(["x"]), scored({"v0": 0.8, "x": 0.5}),
                      iterations=1, tolerance=0.05)
        self.assertFalse(r.history[1].kept)
        self.assertEqual(r.history[1].reason, "regressed")

    def test_discard_reverts_proposer_to_all_time_best(self):
        seen = []

        def watcher(current, _history):
            seen.append(current)
            return "bad" if current == "v0" else "x"

        hillclimb("v0", watcher, scored({"v0": 0.9, "bad": 0.1, "x": 0.2}),
                  iterations=2)
        self.assertEqual(seen, ["v0", "v0"])


class InfraExclusionTests(unittest.TestCase):
    def test_infra_error_retried_then_abandoned(self):
        calls = []

        def flaky(c):
            calls.append(c)
            if c == "infra":
                return ScoreReport(infra_error=True, detail="timeout")
            return ScoreReport(score={"v0": 0.5, "good": 0.9}[c])

        r = hillclimb("v0", sequence(["infra", "good"]), flaky, iterations=2,
                      max_infra_retries=3)
        self.assertEqual(calls.count("infra"), 4)
        self.assertEqual(r.infra_failures, 1)
        self.assertEqual(r.best, "good")
        self.assertEqual([v.iteration for v in r.history], [0, 2])

    def test_infra_error_is_not_a_zero_score(self):
        r = hillclimb("v0", sequence(["infra"]),
                      lambda c: (ScoreReport(infra_error=True) if c == "infra"
                                 else ScoreReport(score=0.8)),
                      iterations=1)
        self.assertEqual(r.best, "v0")
        self.assertEqual(r.best_score, 0.8)
        self.assertEqual(len(r.history), 1)

    def test_transient_infra_error_recovers_on_retry(self):
        state = {"n": 0}

        def flaky(c):
            if c == "cand":
                state["n"] += 1
                if state["n"] == 1:
                    return ScoreReport(infra_error=True)
                return ScoreReport(score=0.9)
            return ScoreReport(score=0.5)

        r = hillclimb("v0", sequence(["cand"]), flaky, iterations=1)
        self.assertEqual(r.best, "cand")
        self.assertEqual(r.infra_failures, 0)

    def test_unmeasurable_baseline_returns_input(self):
        r = hillclimb("v0", sequence([]),
                      lambda c: ScoreReport(infra_error=True), iterations=0)
        self.assertEqual(r.best, "v0")
        self.assertEqual(r.best_score, 0.0)
        self.assertEqual(r.history, ())


class CompositeScoreTests(unittest.TestCase):
    def test_no_metrics_is_zero(self):
        self.assertEqual(composite_score({}), 0.0)

    def test_single_metric_renormalises_to_itself(self):
        self.assertAlmostEqual(composite_score({"completion": 1.0}), 1.0)
        self.assertAlmostEqual(composite_score({"visual": 0.5}), 0.5)

    def test_absent_metric_redistributes_not_zeroes(self):
        self.assertAlmostEqual(
            composite_score({"completion": 1.0, "error_rate": 1.0}), 1.0)

    def test_weighting_matches_defaults(self):
        self.assertAlmostEqual(
            composite_score({"completion": 1.0, "error_rate": 0.0}),
            DEFAULT_WEIGHTS["completion"]
            / (DEFAULT_WEIGHTS["completion"] + DEFAULT_WEIGHTS["error_rate"]))

    def test_unknown_metrics_ignored(self):
        self.assertEqual(composite_score({"vibes": 1.0}), 0.0)

    def test_custom_weights(self):
        self.assertAlmostEqual(
            composite_score({"completion": 1.0, "visual": 0.0},
                            {"completion": 1.0, "visual": 1.0}), 0.5)


class DeterminismTests(unittest.TestCase):
    def test_repeated_runs_identical(self):
        def run():
            return hillclimb("v0", sequence(["v1", "v2"]),
                             scored({"v0": 0.5, "v1": 0.7, "v2": 0.9}),
                             iterations=2).to_dict()

        self.assertEqual(run(), run())

    def test_history_records_every_scored_iteration(self):
        r = hillclimb("v0", sequence(["v1", "v2"]),
                      scored({"v0": 0.5, "v1": 0.7, "v2": 0.3}), iterations=2)
        self.assertEqual(len(r.history), 3)
        self.assertEqual([v.is_best for v in r.history], [True, True, False])
        self.assertEqual(r.history[-1].reason, "regressed")


class SelfcheckTests(unittest.TestCase):
    def test_selfcheck_exits_zero(self):
        self.assertEqual(main(["--selfcheck"]), 0)


if __name__ == "__main__":
    unittest.main()
