import unittest

from dataengine.intent2exec_precision_token_loss import (
    DEFAULT_HEAVY_WEIGHT,
    is_numeric_token,
    precision_token_loss,
    sample_precision_loss,
    token_weight,
    token_weights,
)


class TestNumeric(unittest.TestCase):
    def test_numbers(self):
        for t in ("3", "3.5", "-2", ".5", "1e-3", "+4.0"):
            self.assertTrue(is_numeric_token(t), t)

    def test_non_numbers(self):
        for t in ("box", "abc", "3a", ""):
            self.assertFalse(is_numeric_token(t), t)


class TestWeight(unittest.TestCase):
    def test_numeric_heavy(self):
        self.assertEqual(token_weight("12.0"), DEFAULT_HEAVY_WEIGHT)

    def test_geometry_op_heavy(self):
        self.assertEqual(token_weight("extrude"), DEFAULT_HEAVY_WEIGHT)

    def test_plain_light(self):
        self.assertEqual(token_weight("result"), 1.0)

    def test_bad_weights(self):
        with self.assertRaises(ValueError):
            token_weight("x", heavy=0.5, light=1.0)

    def test_vector(self):
        ws = token_weights(["extrude", "10", "foo"])
        self.assertEqual(ws, [2.0, 2.0, 1.0])


class TestSampleLoss(unittest.TestCase):
    def test_normalised(self):
        # weights [2,1], nll [1.0, 1.0] -> (2*1 + 1*1)/3 = 1.0
        self.assertAlmostEqual(sample_precision_loss([1.0, 1.0], [2.0, 1.0]), 1.0)

    def test_heavy_token_dominates(self):
        # high loss on the heavy token pulls the weighted mean up
        light = sample_precision_loss([5.0, 0.0], [1.0, 2.0])
        heavy = sample_precision_loss([0.0, 5.0], [1.0, 2.0])
        self.assertGreater(heavy, light)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            sample_precision_loss([1.0], [1.0, 2.0])

    def test_empty(self):
        with self.assertRaises(ValueError):
            sample_precision_loss([], [])


class TestBatch(unittest.TestCase):
    def test_mean(self):
        val = precision_token_loss([[1.0, 1.0], [2.0]], [[2.0, 1.0], [1.0]])
        self.assertAlmostEqual(val, (1.0 + 2.0) / 2)

    def test_mismatch(self):
        with self.assertRaises(ValueError):
            precision_token_loss([[1.0]], [[1.0], [1.0]])


if __name__ == "__main__":
    unittest.main()
