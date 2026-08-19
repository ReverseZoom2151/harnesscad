"""Hand-computed checks for the ReCAD IR / CAD-RL Executability module.

Every expected number below is worked out by hand from the stated definition
(``IR = 100 * #did-not-build / #outputs``), not read back off the code.
"""

from __future__ import annotations

import unittest

from harnesscad.eval.bench.harness import build_invalidity as bi


#: 8 outputs, 3 of which did not build -> IR is exactly 37.5%.
POPULATION = (
    {"id": "a", "built": True},
    {"id": "b", "built": False, "error": "SyntaxError"},
    {"id": "c", "built": True},
    {"id": "d", "built": False, "error": "SyntaxError"},
    {"id": "e", "built": True},
    {"id": "f", "built": False, "error": "KernelError"},
    {"id": "g", "built": True},
    {"id": "h", "built": True},
)


class InvalidityRatioTest(unittest.TestCase):
    def test_ir_is_a_percentage_of_the_whole_population(self):
        self.assertEqual(bi.invalidity_ratio_percent(POPULATION), 37.5)

    def test_executability_is_the_exact_complement(self):
        self.assertEqual(bi.executability_percent(POPULATION), 62.5)
        self.assertAlmostEqual(
            bi.invalidity_ratio_percent(POPULATION)
            + bi.executability_percent(POPULATION), 100.0, places=12)

    def test_a_perfect_population_scores_zero_not_one(self):
        self.assertEqual(bi.invalidity_ratio_percent([{"built": True}]), 0.0)
        self.assertEqual(bi.executability_percent([{"built": True}]), 100.0)

    def test_error_key_alone_decides_when_built_is_absent(self):
        self.assertFalse(bi.is_invalid({"error": None}))
        self.assertFalse(bi.is_invalid({"error": ""}))
        self.assertTrue(bi.is_invalid({"error": "NameError"}))

    def test_built_wins_over_error_when_both_are_present(self):
        self.assertTrue(bi.is_invalid({"built": False, "error": None}))

    def test_a_record_with_neither_signal_is_refused_not_assumed_valid(self):
        with self.assertRaises(ValueError):
            bi.is_invalid({"id": "silent"})

    def test_an_empty_population_is_undefined_not_perfect(self):
        with self.assertRaises(ValueError):
            bi.invalidity_ratio_percent([])


class BuildReportTest(unittest.TestCase):
    def test_report_carries_both_polarities_and_both_scales(self):
        report = bi.build_report(POPULATION)
        self.assertEqual(report.n_outputs, 8)
        self.assertEqual(report.n_invalid, 3)
        self.assertEqual(report.ir_percent, 37.5)
        self.assertEqual(report.executability_percent, 62.5)
        self.assertEqual(report.ir_ratio, 0.375)
        self.assertEqual(report.executability_ratio, 0.625)

    def test_failure_census_counts_only_the_invalid_outputs(self):
        self.assertEqual(bi.failure_census(POPULATION),
                         {"KernelError": 1, "SyntaxError": 2})

    def test_unlabelled_failures_are_named_not_dropped(self):
        census = bi.failure_census([{"built": False}, {"built": True}])
        self.assertEqual(census, {"unlabelled": 1})

    def test_as_dict_and_summary_report_the_pair_together(self):
        report = bi.build_report(POPULATION)
        payload = report.as_dict()
        self.assertEqual(payload["ir_percent"], 37.5)
        self.assertEqual(payload["executability_percent"], 62.5)
        self.assertEqual(payload["failures"], {"KernelError": 1, "SyntaxError": 2})
        self.assertIn("IR 37.50%", report.summary())
        self.assertIn("Executability 62.50%", report.summary())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
