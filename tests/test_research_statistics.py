import unittest

from research.statistics import compare_samples, effect_magnitude


class ResearchStatisticsTests(unittest.TestCase):
    def test_effect_and_interval(self):
        report = compare_samples([5, 6, 7, 8], [1, 2, 3, 4])
        self.assertEqual(report.difference, 4)
        self.assertGreater(report.cohen_d, 0)
        self.assertLess(report.ci95_low, report.difference)
        self.assertGreater(report.ci95_high, report.difference)
        self.assertEqual(effect_magnitude(report.cohen_d), "large")

    def test_requires_replicates(self):
        with self.assertRaises(ValueError):
            compare_samples([1], [2, 3])
