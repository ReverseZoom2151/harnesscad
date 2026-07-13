"""Tests for PHT-CAD Dimension Accuracy (DA) metric."""

import unittest

from harnesscad.domain.reconstruction.evaluate import pht_dimension_accuracy as da


class DimensionTest(unittest.TestCase):
    def test_valid(self):
        d = da.Dimension("length", 10.0, ((0, 0), (10, 0)))
        self.assertEqual(d.dim_type, "length")

    def test_bad_type(self):
        with self.assertRaises(ValueError):
            da.Dimension("weight", 5.0)


class ComponentTest(unittest.TestCase):
    def setUp(self):
        self.gt = da.Dimension("radius", 5.0, ((0, 0), (5, 0)))

    def test_type(self):
        self.assertTrue(da.type_match(self.gt, da.Dimension("radius", 99.0)))
        self.assertFalse(da.type_match(self.gt, da.Dimension("length", 5.0)))

    def test_value(self):
        self.assertTrue(da.value_match(self.gt, da.Dimension("radius", 5.4), 0.5))
        self.assertFalse(da.value_match(self.gt, da.Dimension("radius", 5.6), 0.5))

    def test_element_within_tol(self):
        pred = da.Dimension("radius", 5.0, ((0.1, 0.1), (5.1, -0.1)))
        self.assertTrue(da.element_match(self.gt, pred, 0.2))
        self.assertFalse(da.element_match(self.gt, pred, 0.05))

    def test_element_count_mismatch(self):
        pred = da.Dimension("radius", 5.0, ((0, 0),))
        self.assertFalse(da.element_match(self.gt, pred, 1.0))

    def test_element_empty_passes(self):
        gt = da.Dimension("angle", 90.0)
        self.assertTrue(da.element_match(gt, da.Dimension("angle", 90.0), 0.1))


class IsCorrectTest(unittest.TestCase):
    def test_all_pass(self):
        gt = da.Dimension("length", 10.0, ((0, 0), (10, 0)))
        pred = da.Dimension("length", 10.2, ((0.05, 0), (10.0, 0.05)))
        self.assertTrue(da.is_correct(gt, pred, tau_v=0.5, tau_e=0.1))

    def test_one_fails(self):
        gt = da.Dimension("length", 10.0, ((0, 0), (10, 0)))
        pred = da.Dimension("diameter", 10.0, ((0, 0), (10, 0)))  # type wrong
        self.assertFalse(da.is_correct(gt, pred, tau_v=0.5, tau_e=0.1))


class DAAggregateTest(unittest.TestCase):
    def test_all_correct(self):
        gts = [da.Dimension("length", 10.0, ((0, 0), (10, 0))),
               da.Dimension("radius", 5.0, ((0, 0),))]
        preds = [da.Dimension("length", 10.0, ((0, 0), (10, 0))),
                 da.Dimension("radius", 5.0, ((0, 0),))]
        r = da.dimension_accuracy(gts, preds, tau_v=0.5, tau_e=0.1)
        self.assertEqual(r.correct, 2)
        self.assertAlmostEqual(r.accuracy, 1.0)

    def test_half(self):
        gts = [da.Dimension("length", 10.0), da.Dimension("angle", 90.0)]
        preds = [da.Dimension("length", 10.0), da.Dimension("angle", 80.0)]
        r = da.dimension_accuracy(gts, preds, tau_v=1.0, tau_e=0.1)
        self.assertEqual(r.correct, 1)
        self.assertAlmostEqual(r.accuracy, 0.5)

    def test_none_prediction_incorrect(self):
        gts = [da.Dimension("length", 10.0)]
        r = da.dimension_accuracy(gts, [None], tau_v=1.0, tau_e=0.1)
        self.assertEqual(r.correct, 0)

    def test_empty(self):
        r = da.dimension_accuracy([], [], tau_v=1.0, tau_e=0.1)
        self.assertEqual(r.total, 0)
        self.assertEqual(r.accuracy, 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            da.dimension_accuracy([da.Dimension("length", 1.0)], [], 1.0, 0.1)


if __name__ == "__main__":
    unittest.main()
