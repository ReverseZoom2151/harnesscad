import unittest

from bench.mambacad_length_metrics import (
    effective_length, average_length, length_distribution,
    long_sequence_ratio, length_report, DEEPCAD_BUCKETS,
)


class TestEffectiveLength(unittest.TestCase):
    def test_stops_at_eos(self):
        seq = ["L", "L", "E", "<EOS>", "<EOS>"]
        self.assertEqual(effective_length(seq, "<EOS>"), 3)

    def test_no_eos_full_length(self):
        seq = ["L", "A", "E"]
        self.assertEqual(effective_length(seq, "<EOS>"), 3)

    def test_eos_first(self):
        self.assertEqual(effective_length(["<EOS>"], "<EOS>"), 0)

    def test_integer_tokens(self):
        self.assertEqual(effective_length([5, 6, 0, 0], 0), 2)


class TestAverageLength(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(average_length([10, 20, 30]), 20.0)

    def test_empty(self):
        self.assertEqual(average_length([]), 0.0)

    def test_single(self):
        self.assertEqual(average_length([38]), 38.0)


class TestLengthDistribution(unittest.TestCase):
    def test_buckets_sum_and_fractions(self):
        lengths = [5, 15, 15, 30, 50, 70]
        dist = length_distribution(lengths)
        self.assertAlmostEqual(dist["1-10"], 1 / 6)
        self.assertAlmostEqual(dist["11-25"], 2 / 6)
        self.assertAlmostEqual(dist["26-40"], 1 / 6)
        self.assertAlmostEqual(dist["41-60"], 1 / 6)
        self.assertAlmostEqual(dist["60-128"], 1 / 6)
        self.assertAlmostEqual(sum(dist.values()), 1.0)

    def test_all_buckets_present_when_empty(self):
        dist = length_distribution([])
        self.assertEqual(set(dist), {b[0] for b in DEEPCAD_BUCKETS})
        self.assertTrue(all(v == 0.0 for v in dist.values()))

    def test_boundary_inclusive(self):
        # 10 is in [1-10], 11 in [11-25], 60 in [41-60], 61 in [60-128].
        dist = length_distribution([10, 11, 60, 61])
        self.assertAlmostEqual(dist["1-10"], 0.25)
        self.assertAlmostEqual(dist["11-25"], 0.25)
        self.assertAlmostEqual(dist["41-60"], 0.25)
        self.assertAlmostEqual(dist["60-128"], 0.25)

    def test_out_of_range_ignored_but_counted(self):
        # length 200 falls in no bucket, but still inflates the denominator.
        dist = length_distribution([50, 200])
        self.assertAlmostEqual(dist["41-60"], 0.5)
        self.assertLess(sum(dist.values()), 1.0)

    def test_custom_buckets(self):
        buckets = (("lo", 1, 5), ("hi", 6, None))
        dist = length_distribution([1, 2, 100], buckets)
        self.assertAlmostEqual(dist["lo"], 2 / 3)
        self.assertAlmostEqual(dist["hi"], 1 / 3)


class TestLongSequenceRatio(unittest.TestCase):
    def test_basic(self):
        # GT lengths: two >=60 (idx 1,3). Recon: idx1 valid&long, idx3 invalid.
        gt = [30, 80, 40, 90]
        valid = [True, True, True, False]
        recon = [30, 85, 40, 90]
        # denom = 2 (idx1, idx3); numer = 1 (idx1). ratio = 0.5
        self.assertAlmostEqual(long_sequence_ratio(gt, valid, recon), 0.5)

    def test_recon_too_short_excluded(self):
        gt = [70]
        valid = [True]
        recon = [55]  # valid but reconstructed length < 60
        self.assertAlmostEqual(long_sequence_ratio(gt, valid, recon), 0.0)

    def test_no_long_gt_returns_zero(self):
        self.assertEqual(long_sequence_ratio([10, 20], [True, True], [10, 20]), 0.0)

    def test_custom_threshold(self):
        gt = [40, 40]
        valid = [True, False]
        recon = [40, 40]
        self.assertAlmostEqual(long_sequence_ratio(gt, valid, recon, threshold=40), 0.5)

    def test_all_valid_and_long(self):
        gt = [60, 100, 128]
        valid = [True, True, True]
        recon = [60, 100, 128]
        self.assertAlmostEqual(long_sequence_ratio(gt, valid, recon), 1.0)

    def test_misaligned_raises(self):
        with self.assertRaises(ValueError):
            long_sequence_ratio([1, 2], [True], [1, 2])


class TestLengthReport(unittest.TestCase):
    def test_report_fields(self):
        rep = length_report([15, 30, 70])
        self.assertEqual(rep["total"], 3)
        self.assertAlmostEqual(rep["average_length"], (15 + 30 + 70) / 3)
        self.assertIn("distribution", rep)
        self.assertAlmostEqual(rep["distribution"]["60-128"], 1 / 3)

    def test_empty_report(self):
        rep = length_report([])
        self.assertEqual(rep["total"], 0)
        self.assertEqual(rep["average_length"], 0.0)


if __name__ == "__main__":
    unittest.main()
