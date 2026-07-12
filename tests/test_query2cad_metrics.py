import unittest

from bench.query2cad_metrics import (
    benchmark_composition, success_rate, per_difficulty_success,
    refinement_curve, improvement_deltas, first_refinement_dominates,
    failure_breakdown, PAPER_BINS, DIFFICULTIES, FAILURE_MODES,
)


class TestComposition(unittest.TestCase):
    def test_paper_bins_total_57(self):
        c = benchmark_composition()
        self.assertEqual(c["total"], 57)
        self.assertEqual(c["counts"], {"easy": 21, "medium": 20, "hard": 16})

    def test_fractions_sum_to_one(self):
        c = benchmark_composition()
        self.assertAlmostEqual(sum(c["fractions"].values()), 1.0)

    def test_custom_bins(self):
        c = benchmark_composition({"easy": 2, "medium": 2, "hard": 0})
        self.assertEqual(c["total"], 4)

    def test_unknown_difficulty(self):
        with self.assertRaises(ValueError):
            benchmark_composition({"trivial": 3})

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            benchmark_composition({"easy": 0, "medium": 0, "hard": 0})

    def test_paper_bins_constant(self):
        self.assertEqual(PAPER_BINS, {"easy": 21, "medium": 20, "hard": 16})


class TestSuccessRate(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(success_rate(20, 21), 20 / 21)

    def test_zero_total(self):
        with self.assertRaises(ValueError):
            success_rate(0, 0)

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            success_rate(5, 3)


class TestPerDifficulty(unittest.TestCase):
    def test_accounting(self):
        r = per_difficulty_success({
            "easy": [True, True, True, False],
            "hard": [True, False],
        })
        self.assertAlmostEqual(r["per_difficulty"]["easy"], 0.75)
        self.assertAlmostEqual(r["per_difficulty"]["hard"], 0.5)
        self.assertAlmostEqual(r["overall"], 4 / 6)

    def test_empty_bin(self):
        with self.assertRaises(ValueError):
            per_difficulty_success({"easy": []})

    def test_unknown(self):
        with self.assertRaises(ValueError):
            per_difficulty_success({"trivial": [True]})


class TestRefinementCurve(unittest.TestCase):
    def test_gpt4_curve(self):
        r = refinement_curve([0.536, 0.732, 0.767, 0.767])
        self.assertEqual(r["y0"], 0.536)
        self.assertEqual(r["final"], 0.767)
        self.assertAlmostEqual(r["total_gain"], 0.231)
        self.assertEqual(r["iterations"], 3)

    def test_non_decreasing_enforced(self):
        with self.assertRaises(ValueError):
            refinement_curve([0.5, 0.4])

    def test_bounds(self):
        with self.assertRaises(ValueError):
            refinement_curve([0.5, 1.2])

    def test_empty(self):
        with self.assertRaises(ValueError):
            refinement_curve([])


class TestDeltas(unittest.TestCase):
    def test_deltas(self):
        d = improvement_deltas([0.536, 0.732, 0.767, 0.767])
        self.assertAlmostEqual(d[0], 0.196)
        self.assertAlmostEqual(d[1], 0.035)
        self.assertAlmostEqual(d[2], 0.0)

    def test_first_dominates_true(self):
        self.assertTrue(first_refinement_dominates([0.536, 0.732, 0.767, 0.767]))

    def test_first_dominates_gpt35(self):
        self.assertTrue(first_refinement_dominates([0.327, 0.448, 0.517, 0.534]))

    def test_first_dominates_false(self):
        self.assertFalse(first_refinement_dominates([0.3, 0.35, 0.6]))

    def test_no_refinement_false(self):
        self.assertFalse(first_refinement_dominates([0.5]))


class TestFailureBreakdown(unittest.TestCase):
    def test_gpt4_split(self):
        # 13 failures, 69% non-executable ~ 9, 31% wrong-structure ~ 4.
        b = failure_breakdown(9, 4)
        self.assertEqual(b["total"], 13)
        self.assertAlmostEqual(b["fractions"]["non_executable"], 9 / 13)

    def test_modes_constant(self):
        self.assertEqual(FAILURE_MODES, ("non_executable", "wrong_structure"))

    def test_no_failures(self):
        with self.assertRaises(ValueError):
            failure_breakdown(0, 0)

    def test_negative(self):
        with self.assertRaises(ValueError):
            failure_breakdown(-1, 2)


if __name__ == "__main__":
    unittest.main()
