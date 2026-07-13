import unittest

from harnesscad.eval.bench.seekcad_metrics import (
    NO,
    UNSURE,
    YES,
    band_histogram,
    complexity_band,
    feedback_accounting,
    g_score,
    is_novel,
    novelty_rate,
    per_band_means,
)


class TestNovelty(unittest.TestCase):
    def test_novel_when_mostly_dissimilar(self):
        # 4 of 5 below tau=0.8 -> fraction 0.8 >= rho=0.8 -> novel
        self.assertTrue(is_novel([0.1, 0.2, 0.3, 0.7, 0.9]))

    def test_not_novel_when_similar(self):
        # only 1 of 5 below tau -> 0.2 < 0.8
        self.assertFalse(is_novel([0.9, 0.85, 0.82, 0.81, 0.1]))

    def test_boundary_tau_strict(self):
        # exactly 0.8 is NOT below tau (strict <)
        self.assertFalse(is_novel([0.8, 0.8, 0.8, 0.8, 0.8]))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            is_novel([])

    def test_out_of_range_similarity(self):
        with self.assertRaises(ValueError):
            is_novel([1.5])

    def test_bad_threshold(self):
        with self.assertRaises(ValueError):
            is_novel([0.1], tau=2.0)

    def test_novelty_rate(self):
        corpus = [
            [0.1, 0.2, 0.3],  # novel
            [0.9, 0.9, 0.9],  # not novel
        ]
        self.assertEqual(novelty_rate(corpus), 0.5)

    def test_novelty_rate_empty(self):
        with self.assertRaises(ValueError):
            novelty_rate([])


class TestComplexity(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(complexity_band(0), "Low")
        self.assertEqual(complexity_band(30), "Low")
        self.assertEqual(complexity_band(31), "Medium")
        self.assertEqual(complexity_band(70), "Medium")
        self.assertEqual(complexity_band(71), "High")

    def test_negative(self):
        with self.assertRaises(ValueError):
            complexity_band(-1)

    def test_histogram(self):
        h = band_histogram([5, 40, 100, 200, 20])
        self.assertEqual(h, {"Low": 2, "Medium": 1, "High": 2})


class TestGScore(unittest.TestCase):
    def test_mean(self):
        self.assertAlmostEqual(g_score([3.0, 4.0, 5.0]), 4.0)

    def test_range_check(self):
        with self.assertRaises(ValueError):
            g_score([0.5])
        with self.assertRaises(ValueError):
            g_score([5.5])

    def test_empty(self):
        with self.assertRaises(ValueError):
            g_score([])


class TestFeedbackAccounting(unittest.TestCase):
    def test_split(self):
        verdicts = [YES] * 67 + [NO] * 374 + [UNSURE] * 59
        acc = feedback_accounting(verdicts)
        self.assertEqual(acc["helpful"], 441)
        self.assertEqual(acc["useless"], 59)
        self.assertAlmostEqual(acc["helpful_fraction"], 441 / 500)
        self.assertAlmostEqual(acc["useless_fraction"], 59 / 500)
        self.assertEqual(acc["counts"][YES], 67)

    def test_bad_verdict(self):
        with self.assertRaises(ValueError):
            feedback_accounting(["Harmful"])

    def test_empty(self):
        with self.assertRaises(ValueError):
            feedback_accounting([])


class TestPerBandMeans(unittest.TestCase):
    def test_grouping(self):
        records = [
            {"command_count": 10, "iogt": 0.7},
            {"command_count": 20, "iogt": 0.9},
            {"command_count": 50, "iogt": 0.6},
            {"command_count": 100, "iogt": 0.5},
        ]
        means = per_band_means(records, "iogt")
        self.assertAlmostEqual(means["Low"], 0.8)
        self.assertAlmostEqual(means["Medium"], 0.6)
        self.assertAlmostEqual(means["High"], 0.5)

    def test_missing_band_omitted(self):
        means = per_band_means([{"command_count": 5, "v": 1.0}], "v")
        self.assertEqual(set(means), {"Low"})


if __name__ == "__main__":
    unittest.main()
