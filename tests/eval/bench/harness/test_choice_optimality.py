import unittest

from harnesscad.eval.bench.harness.choice_optimality import (
    choice_accuracy,
    evaluate_strategy,
    is_optimal_choice,
    n_solved,
    optimal_cost,
    rank_select,
    strategy_choices,
    time_markup,
    total_cost,
)


class TestOptimal(unittest.TestCase):
    def test_optimal_cost(self):
        self.assertEqual(optimal_cost([3.0, 1.0, None, 2.0]), 1.0)

    def test_optimal_cost_all_timeout(self):
        self.assertIsNone(optimal_cost([None, None]))

    def test_is_optimal_choice(self):
        costs = [3.0, 1.0, 2.0]
        self.assertTrue(is_optimal_choice(costs, 1))
        self.assertFalse(is_optimal_choice(costs, 0))

    def test_is_optimal_choice_timeout_pick(self):
        self.assertFalse(is_optimal_choice([None, 2.0], 0))


class TestBasicMetrics(unittest.TestCase):
    def setUp(self):
        # paper Table 1 hypothetical timings, orderings 0..5
        self.inst = [[22.16, 17.14, None, 24.87, 16.06, 22.58]]

    def test_n_solved(self):
        self.assertEqual(n_solved(self.inst, [4]), 1)
        self.assertEqual(n_solved(self.inst, [2]), 0)  # timeout option

    def test_accuracy_optimal_is_index4(self):
        self.assertEqual(choice_accuracy(self.inst, [4]), 1.0)
        self.assertEqual(choice_accuracy(self.inst, [0]), 0.0)

    def test_total_cost_known(self):
        self.assertAlmostEqual(total_cost(self.inst, [0], time_limit=30.0), 22.16)

    def test_total_cost_timeout_penalised(self):
        # option 2 timed out -> 2 * 30
        self.assertAlmostEqual(total_cost(self.inst, [2], time_limit=30.0), 60.0)

    def test_total_cost_bad_limit(self):
        with self.assertRaises(ValueError):
            total_cost(self.inst, [0], time_limit=0.0)


class TestMarkup(unittest.TestCase):
    def test_markup_optimal_is_zero(self):
        inst = [[16.06, 22.16]]
        self.assertAlmostEqual(time_markup(inst, [0], time_limit=30.0), 0.0)

    def test_markup_formula(self):
        # optimal 16.06, chosen 22.16 -> (22.16-16.06)/(16.06+1)
        inst = [[22.16, 16.06]]
        expected = (22.16 - 16.06) / (16.06 + 1.0)
        self.assertAlmostEqual(time_markup(inst, [0], time_limit=30.0), expected)

    def test_markup_small_problem_forgiving(self):
        # a 1 vs 4 choice yields modest markup thanks to +1 guard
        inst = [[4.0, 1.0]]
        self.assertAlmostEqual(time_markup(inst, [0], time_limit=30.0), 3.0 / 2.0)

    def test_markup_skips_all_timeout(self):
        inst = [[None, None]]
        self.assertEqual(time_markup(inst, [0], time_limit=30.0), 0.0)

    def test_markup_timeout_choice_penalised(self):
        inst = [[10.0, None]]
        # chosen timeout -> 2*30=60, optimal 10 -> (60-10)/11
        self.assertAlmostEqual(
            time_markup(inst, [1], time_limit=30.0), (60.0 - 10.0) / 11.0
        )


class TestEvaluateStrategy(unittest.TestCase):
    def test_bundle(self):
        inst = [[3.0, 1.0], [None, 2.0]]
        rep = evaluate_strategy(inst, [1, 1], time_limit=10.0)
        self.assertEqual(rep.n_instances, 2)
        self.assertEqual(rep.solved, 2)
        self.assertEqual(rep.accuracy, 1.0)
        self.assertAlmostEqual(rep.total_cost, 3.0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            n_solved([[1.0]], [0, 0])

    def test_choice_out_of_range(self):
        with self.assertRaises(ValueError):
            n_solved([[1.0, 2.0]], [5])


class TestRankSelect(unittest.TestCase):
    def test_argmin(self):
        self.assertEqual(rank_select([3.0, 1.0, 2.0]), 1)

    def test_tie_lowest_index(self):
        self.assertEqual(rank_select([1.0, 1.0]), 0)

    def test_feasibility_filter(self):
        # cheapest is index 0 but only odd indices feasible
        self.assertEqual(rank_select([0.5, 2.0, 1.0], feasible=lambda i: i % 2 == 1), 1)

    def test_no_feasible(self):
        with self.assertRaises(ValueError):
            rank_select([1.0], feasible=lambda i: False)

    def test_empty(self):
        with self.assertRaises(ValueError):
            rank_select([])

    def test_strategy_choices(self):
        estimates = [[3.0, 1.0], [0.5, 2.0]]
        self.assertEqual(strategy_choices(estimates), [1, 0])

    def test_strategy_choices_with_feasible(self):
        estimates = [[0.5, 2.0]]
        picks = strategy_choices(estimates, feasible=lambda inst, opt: opt == 1)
        self.assertEqual(picks, [1])


if __name__ == "__main__":
    unittest.main()
