"""Tests for domain.reconstruction.brep.a2z_scan_labeling."""

import unittest

from harnesscad.domain.reconstruction.brep.a2z_scan_labeling import (
    classify_tiny_loops,
    skill_amplitude,
    soft_label_membership,
    sph_poly6_weight,
)


class TinyLoopTest(unittest.TestCase):
    def test_flags_small_loops(self):
        # largest = 100; loops at 5 and 8 are < 15% -> tiny.
        self.assertEqual(classify_tiny_loops([100, 5, 60, 8], tau_h=0.15), [1, 3])

    def test_inactive_with_two_or_fewer(self):
        self.assertEqual(classify_tiny_loops([100, 1]), [])

    def test_threshold_is_strict(self):
        # ratio exactly at tau_h is not tiny.
        self.assertEqual(classify_tiny_loops([100, 15, 90], tau_h=0.15), [])

    def test_bad_tau(self):
        with self.assertRaises(ValueError):
            classify_tiny_loops([1, 2, 3], tau_h=0.0)


class SphWeightTest(unittest.TestCase):
    def test_zero_beyond_support(self):
        self.assertEqual(sph_poly6_weight(2.0, 1.0), 0.0)

    def test_monotone_decreasing(self):
        near = sph_poly6_weight(0.1, 1.0)
        far = sph_poly6_weight(0.8, 1.0)
        self.assertGreater(near, far)

    def test_peak_at_zero(self):
        self.assertAlmostEqual(sph_poly6_weight(0.0, 1.0), 1.0)


class SoftLabelTest(unittest.TestCase):
    def test_argmax_nearest_wins(self):
        cands = [(7, [0.1, 0.2]), (9, [0.9, 0.95])]
        out = soft_label_membership(cands, scale_weights=[0.5, 0.5], radii=[1.0, 1.0])
        self.assertEqual(out["label"], 7)
        self.assertGreater(out["probabilities"][7], out["probabilities"][9])

    def test_aggregates_same_entity_over_scales(self):
        cands = [(1, [0.1, 0.1]), (1, [0.2, 0.2]), (2, [0.95, 0.95])]
        out = soft_label_membership(cands, [1.0, 1.0], [1.0, 1.0])
        self.assertEqual(out["label"], 1)

    def test_all_out_of_support(self):
        cands = [(1, [5.0]), (2, [6.0])]
        out = soft_label_membership(cands, [1.0], [1.0])
        self.assertIsNone(out["label"])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            soft_label_membership([(1, [0.1])], [1.0, 1.0], [1.0, 1.0])


class SkillAmplitudeTest(unittest.TestCase):
    def test_lower_skill_larger_amplitude(self):
        crude = skill_amplitude(1, 100.0)
        pro = skill_amplitude(5, 100.0)
        self.assertGreater(crude["base_amplitude"], pro["base_amplitude"])

    def test_alpha_values(self):
        self.assertAlmostEqual(skill_amplitude(1, 10.0)["alpha"], 1.0)
        self.assertAlmostEqual(skill_amplitude(5, 10.0)["alpha"], 0.2)

    def test_bad_kappa(self):
        with self.assertRaises(ValueError):
            skill_amplitude(0, 10.0)


if __name__ == "__main__":
    unittest.main()
