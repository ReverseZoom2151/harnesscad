"""Tests for the deterministic aerodynamic drag proxy."""
import unittest

from verifiers.llmdesopt_drag_proxy import (
    bounding_box,
    box_extents,
    CarDimensions,
    realign_dimensions,
    car_dimensions_from_points,
    LinearDragModel,
    fit_linear_drag,
    r_squared,
    BaselineNormaliser,
)


class BoundingBoxTests(unittest.TestCase):
    def test_bounding_box_corners(self):
        pts = [(0.0, 0.0, 0.0), (2.0, 3.0, 1.0), (-1.0, 1.0, 4.0)]
        lo, hi = bounding_box(pts)
        self.assertEqual(lo, (-1.0, 0.0, 0.0))
        self.assertEqual(hi, (2.0, 3.0, 4.0))

    def test_extents(self):
        pts = [(0.0, 0.0, 0.0), (2.0, 3.0, 1.0), (-1.0, 1.0, 4.0)]
        self.assertEqual(box_extents(pts), (3.0, 3.0, 4.0))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            bounding_box([])


class RealignTests(unittest.TestCase):
    def test_realign_orders_length_width_height(self):
        dims = realign_dimensions([1.0, 4.0, 2.0])
        self.assertEqual(dims.length, 4.0)
        self.assertEqual(dims.width, 2.0)
        self.assertEqual(dims.height, 1.0)

    def test_frontal_area_is_width_times_height(self):
        dims = CarDimensions(length=5.0, width=2.0, height=1.5)
        self.assertAlmostEqual(dims.frontal_area, 3.0)

    def test_from_points_pipeline(self):
        # extents (3,3,4) -> length 4, width 3, height 3
        pts = [(0.0, 0.0, 0.0), (3.0, 3.0, 4.0)]
        dims = car_dimensions_from_points(pts)
        self.assertEqual(dims.length, 4.0)
        self.assertEqual(dims.width, 3.0)
        self.assertEqual(dims.height, 3.0)
        self.assertAlmostEqual(dims.frontal_area, 9.0)

    def test_wrong_count_raises(self):
        with self.assertRaises(ValueError):
            realign_dimensions([1.0, 2.0])


class LinearDragTests(unittest.TestCase):
    def test_fit_recovers_known_line(self):
        areas = [1.0, 2.0, 3.0, 4.0]
        cds = [2.0 * a + 0.5 for a in areas]
        model = fit_linear_drag(areas, cds)
        self.assertAlmostEqual(model.slope, 2.0)
        self.assertAlmostEqual(model.intercept, 0.5)
        self.assertAlmostEqual(r_squared(areas, cds, model), 1.0)

    def test_cd_of_points_uses_frontal_area(self):
        model = LinearDragModel(slope=0.5, intercept=0.1)
        pts = [(0.0, 0.0, 0.0), (10.0, 2.0, 1.0)]  # width 2, height 1 -> Af 2
        self.assertAlmostEqual(model.cd_of_points(pts), 0.5 * 2.0 + 0.1)

    def test_r_squared_below_one_for_noisy(self):
        areas = [1.0, 2.0, 3.0, 4.0]
        cds = [2.1, 3.9, 6.2, 7.8]
        model = fit_linear_drag(areas, cds)
        r2 = r_squared(areas, cds, model)
        self.assertTrue(0.9 < r2 <= 1.0)

    def test_constant_area_raises(self):
        with self.assertRaises(ValueError):
            fit_linear_drag([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])

    def test_too_few_samples_raises(self):
        with self.assertRaises(ValueError):
            fit_linear_drag([1.0], [1.0])


class NormaliserTests(unittest.TestCase):
    def test_min_max_normalisation(self):
        norm = BaselineNormaliser.from_baseline([10.0, 20.0, 30.0])
        self.assertAlmostEqual(norm.normalise(10.0), 0.0)
        self.assertAlmostEqual(norm.normalise(30.0), 1.0)
        self.assertAlmostEqual(norm.normalise(20.0), 0.5)

    def test_constant_baseline_returns_zero(self):
        norm = BaselineNormaliser.from_baseline([5.0, 5.0])
        self.assertEqual(norm.normalise(5.0), 0.0)

    def test_normalise_all(self):
        norm = BaselineNormaliser.from_baseline([0.0, 4.0])
        self.assertEqual(norm.normalise_all([0.0, 2.0, 4.0]), [0.0, 0.5, 1.0])

    def test_empty_baseline_raises(self):
        with self.assertRaises(ValueError):
            BaselineNormaliser.from_baseline([])


if __name__ == "__main__":
    unittest.main()
