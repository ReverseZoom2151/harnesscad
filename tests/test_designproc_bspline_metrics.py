"""Tests for datagen.designproc_bspline_metrics."""

import math
import unittest

from harnesscad.data.datagen import designproc_bspline_metrics as m
from harnesscad.data.datagen import designproc_procedure as dp
from harnesscad.data.datagen import designproc_program_synthesis as ps


class TestBsplineRatio(unittest.TestCase):
    def test_eq1(self):
        # f=4, fb=2 -> 0.5; e=10, eb=5 -> 0.5; mean 0.5
        self.assertAlmostEqual(m.bspline_ratio(4, 2, 10, 5), 0.5)

    def test_all_bspline(self):
        self.assertAlmostEqual(m.bspline_ratio(3, 3, 6, 6), 1.0)

    def test_none_bspline(self):
        self.assertAlmostEqual(m.bspline_ratio(3, 0, 6, 0), 0.0)

    def test_zero_faces_no_error(self):
        self.assertAlmostEqual(m.bspline_ratio(0, 0, 4, 2), 0.25)

    def test_count_exceeds_total(self):
        with self.assertRaises(ValueError):
            m.bspline_ratio(2, 3, 4, 1)

    def test_negative(self):
        with self.assertRaises(ValueError):
            m.bspline_ratio(-1, 0, 4, 1)


class TestGeometricProperties(unittest.TestCase):
    def _full_programs(self, n=4):
        progs = []
        for i in range(n):
            proc = dp.build_procedure("bracket", "saddle", n_primitives=1 + i % 3)
            progs.append(ps.synthesize_program(proc, seed=i))
        return progs

    def test_full_has_high_bspline_fraction(self):
        props = m.geometric_properties(self._full_programs())
        self.assertEqual(props["frac_with_bspline_faces"], 1.0)
        self.assertGreater(props["mean_bspline_ratio"], 0.0)
        self.assertGreater(props["avg_lines"], 0.0)

    def test_empty(self):
        props = m.geometric_properties([])
        self.assertEqual(props["n"], 0)
        self.assertEqual(props["mean_bspline_ratio"], 0.0)

    def test_baseline_zero_bspline(self):
        progs = []
        for i in range(3):
            proc = dp.build_procedure("bracket", "saddle", n_primitives=2,
                                      with_reference_surface=False,
                                      with_fillet=False)
            progs.append(ps.synthesize_program(proc, seed=i))
        props = m.geometric_properties(progs)
        self.assertEqual(props["frac_with_bspline_faces"], 0.0)
        self.assertEqual(props["mean_bspline_ratio"], 0.0)


class TestHistogramDiversity(unittest.TestCase):
    def test_histogram_bins(self):
        hist = m.ratio_histogram([0.0, 0.05, 0.5, 0.99, 1.0], n_bins=10)
        self.assertEqual(sum(hist), 5)
        self.assertEqual(hist[0], 2)   # 0.0, 0.05
        self.assertEqual(hist[9], 2)   # 0.99, 1.0
        self.assertEqual(hist[5], 1)   # 0.5

    def test_ratio_out_of_range(self):
        with self.assertRaises(ValueError):
            m.ratio_histogram([1.5])

    def test_entropy_spike_vs_even(self):
        spike = [10, 0, 0, 0]
        even = [3, 3, 3, 3]
        self.assertAlmostEqual(m.distribution_entropy(spike), 0.0)
        self.assertAlmostEqual(m.distribution_entropy(even), 2.0)  # log2(4)
        self.assertAlmostEqual(m.normalized_diversity(even), 1.0)
        self.assertAlmostEqual(m.normalized_diversity(spike), 0.0)

    def test_normalized_diversity_single_bin(self):
        self.assertEqual(m.normalized_diversity([5]), 0.0)

    def test_diversity_report(self):
        progs = [ps.synthesize_program(
            dp.build_procedure("bracket", "saddle", n_primitives=2), seed=i)
            for i in range(5)]
        rep = m.diversity_report(progs, n_bins=10)
        self.assertEqual(rep["n"], 5)
        self.assertEqual(len(rep["histogram"]), 10)
        self.assertEqual(sum(rep["histogram"]), 5)
        self.assertGreaterEqual(rep["normalized_diversity"], 0.0)


class TestValidity(unittest.TestCase):
    def test_valid_full_program(self):
        prog = ps.synthesize_program(
            dp.build_procedure("bracket", "saddle", n_primitives=2), seed=1)
        self.assertTrue(m.program_is_valid(prog))
        self.assertTrue(m.program_is_valid(prog, require_bspline=True))

    def test_empty_invalid(self):
        self.assertFalse(m.program_is_valid([]))

    def test_no_export_invalid(self):
        self.assertFalse(m.program_is_valid([{"op": "sketch", "faces": 1}]))

    def test_require_bspline_rejects_baseline(self):
        prog = ps.synthesize_program(
            dp.build_procedure("bracket", "saddle", n_primitives=2,
                               with_reference_surface=False, with_fillet=False),
            seed=1)
        self.assertTrue(m.program_is_valid(prog))
        self.assertFalse(m.program_is_valid(prog, require_bspline=True))

    def test_filter_valid(self):
        good = ps.synthesize_program(
            dp.build_procedure("bracket", "saddle"), seed=1)
        bad = [{"op": "sketch", "faces": 1}]
        kept = m.filter_valid([good, bad], require_bspline=True)
        self.assertEqual(len(kept), 1)


class TestAugmentationGain(unittest.TestCase):
    def test_gain_positive(self):
        baseline = [ps.synthesize_program(
            dp.build_procedure("bracket", "saddle", n_primitives=2,
                               with_reference_surface=False, with_fillet=False),
            seed=i) for i in range(4)]
        augmented = [ps.synthesize_program(
            dp.build_procedure("bracket", "saddle", n_primitives=2), seed=i)
            for i in range(4)]
        gain = m.augmentation_gain(baseline, augmented)
        self.assertGreater(gain["delta_mean_bspline_ratio"], 0.0)
        self.assertGreater(gain["delta_frac_bspline_faces"], 0.0)
        self.assertEqual(gain["baseline"]["frac_with_bspline_faces"], 0.0)
        self.assertEqual(gain["augmented"]["frac_with_bspline_faces"], 1.0)


if __name__ == "__main__":
    unittest.main()
