"""Tests for joint reconstruction+segmentation analysis (joint SDF paper)."""

import unittest

from harnesscad.eval.bench.harness.jointsdf_joint_metrics import (
    pearson,
    correlation_table,
    part_count_agreement,
    joint_score,
    mean_std,
)


class PearsonTest(unittest.TestCase):
    def test_perfect_positive(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)

    def test_perfect_negative(self):
        # CD down <-> mIoU up, as the paper reports (negative r)
        self.assertAlmostEqual(pearson([1, 2, 3], [6, 4, 2]), -1.0)

    def test_zero_variance_fallback(self):
        self.assertEqual(pearson([5, 5, 5], [1, 2, 3]), 0.0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            pearson([1, 2], [1])


class CorrelationTableTest(unittest.TestCase):
    def test_table_shape_and_sign(self):
        recon = {"CD": [0.5, 0.4, 0.3], "NC": [0.9, 0.95, 0.97]}
        seg = {"mIoU": [0.90, 0.95, 0.97]}
        t = correlation_table(recon, seg)
        # lower CD correlates with higher mIoU -> negative
        self.assertLess(t["CD"]["mIoU"], 0.0)
        # higher NC correlates with higher mIoU -> positive
        self.assertGreater(t["NC"]["mIoU"], 0.0)


class PartCountTest(unittest.TestCase):
    def test_agreement(self):
        out = part_count_agreement([3, 4, 5], [3, 4, 6])
        self.assertAlmostEqual(out["exact_match"], 2 / 3)
        self.assertAlmostEqual(out["mae"], 1 / 3)

    def test_mismatch_len(self):
        with self.assertRaises(ValueError):
            part_count_agreement([1], [1, 2])


class JointScoreTest(unittest.TestCase):
    def test_perfect(self):
        # mIoU 1.0 and cd 0 -> both terms 1.0
        self.assertEqual(joint_score(1.0, 0.0), 1.0)

    def test_monotonic_in_cd(self):
        a = joint_score(0.9, 0.1)
        b = joint_score(0.9, 0.5)
        self.assertGreater(a, b)

    def test_negative_cd_raises(self):
        with self.assertRaises(ValueError):
            joint_score(0.9, -1.0)


class MeanStdTest(unittest.TestCase):
    def test_values(self):
        out = mean_std([2.0, 4.0])
        self.assertEqual(out["mean"], 3.0)
        self.assertEqual(out["std"], 1.0)


if __name__ == "__main__":
    unittest.main()
