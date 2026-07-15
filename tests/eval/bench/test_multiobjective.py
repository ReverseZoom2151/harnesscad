import unittest

from harnesscad.eval.bench.multiobjective import (
    feasibility_mask,
    feasibility_rate,
    hypervolume,
    normalize_objectives,
    score_population,
)


class TestFeasibility(unittest.TestCase):
    def test_mask(self):
        cons = [[-1.0, 0.0], [0.5, -1.0], [-2.0, -3.0]]
        self.assertEqual(feasibility_mask(cons), [True, False, True])

    def test_rate(self):
        cons = [[-1.0], [1.0], [-1.0], [-1.0]]
        self.assertEqual(feasibility_rate(cons), 0.75)

    def test_rate_empty(self):
        self.assertEqual(feasibility_rate([]), 0.0)


class TestNormalize(unittest.TestCase):
    def test_clip_and_scale(self):
        norm = normalize_objectives([[5.0, 2.0], [20.0, 1.0]], [10.0, 4.0])
        self.assertEqual(norm[0], (0.5, 0.5))
        self.assertEqual(norm[1], (1.0, 0.25))  # 20 clipped to 10 -> 1.0

    def test_zero_ref_axis(self):
        norm = normalize_objectives([[0.0]], [0.0])
        self.assertEqual(norm[0], (0.0,))


class TestHypervolume(unittest.TestCase):
    def test_1d(self):
        self.assertAlmostEqual(hypervolume([[0.3], [0.6]], [1.0]), 0.7)

    def test_2d_single_point(self):
        # Rectangle from (0.5,0.5) to ref (1,1) = 0.25.
        self.assertAlmostEqual(hypervolume([[0.5, 0.5]], [1.0, 1.0]), 0.25)

    def test_2d_two_points_staircase(self):
        # Points (0.2,0.8) and (0.8,0.2) with ref (1,1).
        # HV = union of two rectangles = 0.8*0.2 + (1-0.8... ) computed by sweep.
        hv = hypervolume([[0.2, 0.8], [0.8, 0.2]], [1.0, 1.0])
        # rect A: (1-0.2)*(1-0.8)=0.16 ; rect B extra: (1-0.8)*(1-0.2)-overlap
        # Exact union = 0.8*0.2 + 0.2*0.8 - 0.2*0.2 = 0.16+0.16-0.04 = 0.28
        self.assertAlmostEqual(hv, 0.28)

    def test_dominated_point_ignored(self):
        hv_one = hypervolume([[0.5, 0.5]], [1.0, 1.0])
        hv_two = hypervolume([[0.5, 0.5], [0.7, 0.7]], [1.0, 1.0])
        self.assertAlmostEqual(hv_one, hv_two)

    def test_point_at_ref_zero(self):
        self.assertEqual(hypervolume([[1.0, 1.0]], [1.0, 1.0]), 0.0)

    def test_3d_grid_positive(self):
        hv = hypervolume([[0.5, 0.5, 0.5]], [1.0, 1.0, 1.0], grid=20)
        # Approximates 0.125; grid should be within a coarse tolerance.
        self.assertGreater(hv, 0.10)
        self.assertLess(hv, 0.15)


class TestScorePopulation(unittest.TestCase):
    def test_full_card(self):
        objs = [[5.0, 5.0], [2.0, 8.0], [50.0, 50.0]]
        cons = [[-1.0], [-1.0], [1.0]]  # third infeasible
        card = score_population(objs, cons, [10.0, 10.0])
        self.assertEqual(card.n_designs, 3)
        self.assertEqual(card.n_feasible, 2)
        self.assertAlmostEqual(card.feasibility_rate, 2 / 3)
        self.assertGreater(card.hypervolume, 0.0)
        self.assertEqual(len(card.mean_objectives_feasible), 2)

    def test_no_feasible(self):
        card = score_population([[1.0]], [[5.0]], [10.0])
        self.assertEqual(card.n_feasible, 0)
        self.assertEqual(card.hypervolume, 0.0)

    def test_determinism(self):
        objs = [[5.0, 5.0], [2.0, 8.0]]
        cons = [[-1.0], [-1.0]]
        a = score_population(objs, cons, [10.0, 10.0])
        b = score_population(objs, cons, [10.0, 10.0])
        self.assertEqual(a, b)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            score_population([[1.0]], [[-1.0], [-1.0]], [10.0])


if __name__ == "__main__":
    unittest.main()
