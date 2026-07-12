import unittest

from bench.mrcad2_design_distance import (
    design_distance,
    design_distance_asymmetric,
    proportional_improvement,
)
from editing.mrcad_schema import Design, line


class TestDesignDistance(unittest.TestCase):
    def test_identical_designs_zero(self):
        d = Design((line((0, 0), (10, 0)),))
        self.assertAlmostEqual(design_distance(d, d), 0.0)

    def test_empty_source_returns_one(self):
        d = Design((line((0, 0), (10, 0)),))
        self.assertEqual(design_distance_asymmetric(Design.empty(), d), 1.0)

    def test_both_empty_symmetric_one(self):
        # Each direction returns 1.0 for an empty source; mean is 1.0.
        self.assertEqual(design_distance(Design.empty(), Design.empty()), 1.0)

    def test_range_is_unit_interval(self):
        a = Design((line((0, 0), (10, 0)),))
        b = Design((line((0, 5), (10, 5)),))
        d = design_distance(a, b)
        self.assertGreater(d, 0.0)
        self.assertLessEqual(d, 1.0)

    def test_symmetric(self):
        a = Design((line((0, 0), (10, 0)),))
        b = Design((line((0, 5), (10, 5)),))
        self.assertAlmostEqual(design_distance(a, b), design_distance(b, a))

    def test_capped_far_apart_is_one(self):
        # Two horizontal lines 100 apart, well beyond the cap of 10 -> distance 1.
        a = Design((line((0, 0), (10, 0)),))
        b = Design((line((0, 100), (10, 100)),))
        self.assertAlmostEqual(design_distance(a, b), 1.0)

    def test_exact_perpendicular_offset(self):
        # Parallel lines 3 apart; every sampled point is exactly 3 from the other
        # curve (point-to-CURVE, not point-to-sampled-point). 3/10 = 0.3.
        a = Design((line((0, 0), (10, 0)),))
        b = Design((line((0, 3), (10, 3)),))
        self.assertAlmostEqual(design_distance(a, b), 0.3)

    def test_tighter_than_sampled_chamfer(self):
        # The exact point-to-curve distance is <= the point-to-sampled-point one.
        from bench.mrcad_metrics import chamfer_asymmetric

        a = Design((line((0, 0), (10, 0)),))
        b = Design((line((1, 2), (9, 2)),))
        exact = design_distance_asymmetric(a, b)
        # chamfer_asymmetric sums capped/normalised terms over 10 samples; divide
        # by the sample count to get a comparable mean.
        sampled_mean = chamfer_asymmetric(a, b) / 10
        self.assertLessEqual(exact, sampled_mean + 1e-9)


class TestProportionalImprovement(unittest.TestCase):
    def test_improvement_positive(self):
        target = Design((line((0, 0), (10, 0)),))
        before = Design((line((0, 5), (10, 5)),))
        after = Design((line((0, 1), (10, 1)),))
        pi = proportional_improvement(before, after, target)
        self.assertGreater(pi, 0.0)

    def test_already_at_target_zero(self):
        target = Design((line((0, 0), (10, 0)),))
        pi = proportional_improvement(target, target, target)
        self.assertEqual(pi, 0.0)


if __name__ == "__main__":
    unittest.main()
