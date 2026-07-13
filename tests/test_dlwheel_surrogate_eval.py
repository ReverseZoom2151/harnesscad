"""Tests for quality.dlwheel_surrogate_eval (paper 112 surrogate metrics)."""

import math
import unittest

from harnesscad.eval.quality import dlwheel_surrogate_eval as se


class MinMaxScalerTests(unittest.TestCase):
    def test_scale_and_inverse(self):
        scaler = se.fit_minmax([10.0, 20.0, 30.0])
        self.assertAlmostEqual(scaler.scale(10.0), 0.0)
        self.assertAlmostEqual(scaler.scale(30.0), 1.0)
        self.assertAlmostEqual(scaler.scale(20.0), 0.5)
        self.assertAlmostEqual(scaler.inverse(0.5), 20.0)

    def test_scale_all_roundtrip(self):
        scaler = se.fit_minmax([1.0, 5.0])
        scaled = scaler.scale_all([1.0, 3.0, 5.0])
        self.assertEqual(scaled, [0.0, 0.5, 1.0])
        back = [scaler.inverse(v) for v in scaled]
        for a, b in zip(back, [1.0, 3.0, 5.0]):
            self.assertAlmostEqual(a, b)

    def test_empty_and_constant(self):
        with self.assertRaises(ValueError):
            se.fit_minmax([])
        with self.assertRaises(ValueError):
            se.fit_minmax([4.0, 4.0])


class MetricTests(unittest.TestCase):
    def test_rmse_zero(self):
        self.assertEqual(se.rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0)

    def test_rmse_known(self):
        # errors 3,4 -> mean of 9,16 = 12.5 -> sqrt
        self.assertAlmostEqual(se.rmse([3.0, 4.0], [0.0, 0.0]), math.sqrt(12.5))

    def test_mape_known(self):
        # y=100 yhat=110 -> 10%; y=50 yhat=45 -> 10% -> mean 10
        self.assertAlmostEqual(se.mape([110.0, 45.0], [100.0, 50.0]), 10.0)

    def test_mape_zero_truth(self):
        with self.assertRaises(ValueError):
            se.mape([1.0], [0.0])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            se.rmse([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            se.rmse([], [])


class EnsembleTests(unittest.TestCase):
    def test_ensemble_mean(self):
        self.assertAlmostEqual(se.ensemble_mean([2.0, 4.0, 6.0]), 4.0)
        with self.assertRaises(ValueError):
            se.ensemble_mean([])

    def test_ensemble_predict(self):
        members = [[1.0, 2.0], [3.0, 4.0]]
        self.assertEqual(se.ensemble_predict(members), [2.0, 3.0])

    def test_ensemble_predict_validation(self):
        with self.assertRaises(ValueError):
            se.ensemble_predict([])
        with self.assertRaises(ValueError):
            se.ensemble_predict([[1.0], [1.0, 2.0]])

    def test_evaluate_ensemble(self):
        members = [[9.0, 11.0], [11.0, 9.0]]  # means 10, 10
        r, m = se.evaluate_ensemble(members, [10.0, 10.0])
        self.assertAlmostEqual(r, 0.0)
        self.assertAlmostEqual(m, 0.0)


if __name__ == "__main__":
    unittest.main()
