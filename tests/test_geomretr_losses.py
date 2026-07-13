"""Tests for VICReg and MMCL metric-learning losses."""

from __future__ import annotations

import math
import unittest

from harnesscad.eval.bench.geomretr_losses import (
    variance_loss,
    covariance_loss,
    invariance_loss,
    vicreg_loss,
    mmcl_loss,
)


class VarianceLossTest(unittest.TestCase):
    def test_collapsed_batch_max_penalty(self):
        # all identical -> zero std -> penalty ~ gamma = 1 (minus eps stabiliser)
        batch = [[1.0, 2.0]] * 5
        self.assertAlmostEqual(variance_loss(batch, gamma=1.0), 0.99, places=2)

    def test_high_variance_low_loss(self):
        batch = [[10.0, 0.0], [-10.0, 0.0], [5.0, 0.0], [-5.0, 0.0]]
        # dim 0 has std >> 1 -> contributes 0; dim 1 zero std -> contributes ~1
        loss = variance_loss(batch)
        self.assertAlmostEqual(loss, 0.495, places=2)

    def test_deterministic(self):
        batch = [[1.0, 3.0], [2.0, 1.0], [0.5, 2.0]]
        self.assertEqual(variance_loss(batch), variance_loss(batch))


class CovarianceLossTest(unittest.TestCase):
    def test_decorrelated_zero(self):
        # independent-looking symmetric data with zero covariance
        batch = [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
        self.assertAlmostEqual(covariance_loss(batch), 0.0, places=9)

    def test_correlated_positive(self):
        batch = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
        self.assertGreater(covariance_loss(batch), 0.0)


class InvarianceLossTest(unittest.TestCase):
    def test_identical_views_zero(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        self.assertAlmostEqual(invariance_loss(a, a), 0.0, places=9)

    def test_known_distance(self):
        a = [[0.0, 0.0]]
        b = [[3.0, 4.0]]
        self.assertAlmostEqual(invariance_loss(a, b), 25.0, places=9)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            invariance_loss([[1.0]], [[1.0], [2.0]])


class VICRegTotalTest(unittest.TestCase):
    def test_components_present(self):
        a = [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [2.0, 2.0, 1.0]]
        b = [[1.1, 0.0, 2.0], [0.0, 1.1, 3.0], [2.0, 2.1, 1.0]]
        r = vicreg_loss(a, b)
        for key in ("invariance", "variance", "covariance", "total"):
            self.assertIn(key, r)
        expected = 25.0 * r["invariance"] + 25.0 * r["variance"] + 1.0 * r["covariance"]
        self.assertAlmostEqual(r["total"], expected, places=9)

    def test_deterministic(self):
        a = [[1.0, 2.0], [3.0, 1.0], [0.0, 0.5]]
        b = [[1.0, 2.1], [3.1, 1.0], [0.1, 0.5]]
        self.assertEqual(vicreg_loss(a, b), vicreg_loss(a, b))


class MMCLTest(unittest.TestCase):
    def test_perfect_alignment_low_loss(self):
        # point == text == image, well separated across batch -> low loss
        zp = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        r = mmcl_loss(zp, zp, zp)
        self.assertGreater(r["total"], 0.0)
        # with separated positives, aligned modalities give smaller loss than misaligned
        misaligned = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        r2 = mmcl_loss(zp, misaligned, misaligned)
        self.assertLess(r["total"], r2["total"])

    def test_symmetry_terms(self):
        zp = [[1.0, 0.0], [0.0, 1.0]]
        zt = [[0.9, 0.1], [0.1, 0.9]]
        zi = [[0.8, 0.2], [0.2, 0.8]]
        r = mmcl_loss(zp, zt, zi)
        self.assertAlmostEqual(r["total"],
                               0.25 * (r["pt"] + r["tp"] + r["pi"] + r["ip"]),
                               places=9)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            mmcl_loss([[1.0]], [[1.0]], [[1.0], [2.0]])


if __name__ == "__main__":
    unittest.main()
