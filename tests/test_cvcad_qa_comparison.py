import math
import unittest

from verifiers.cvcad_qa_comparison import (
    DimensionCheck,
    QAReport,
    check_by_percent,
    check_dimension,
    compare_dimensions,
    percentage_error,
    qa_report,
)


class TestPercentageError(unittest.TestCase):
    def test_paper_cube_case(self):
        # 29.9 nominal, 29.8 measured -> 0.3%.
        self.assertAlmostEqual(percentage_error(29.9, 29.8), 0.3344, places=3)

    def test_zero_nominal_raises(self):
        with self.assertRaises(ValueError):
            percentage_error(0.0, 1.0)

    def test_sign(self):
        # measured larger than nominal -> negative signed error.
        self.assertLess(percentage_error(10.0, 12.0), 0.0)


class TestCheckDimension(unittest.TestCase):
    def test_within_tolerance(self):
        c = check_dimension("width", 30.0, 29.8, tolerance=0.5)
        self.assertTrue(c.within_tolerance)
        self.assertAlmostEqual(c.deviation, -0.2)
        self.assertAlmostEqual(c.abs_deviation, 0.2)

    def test_out_of_tolerance(self):
        c = check_dimension("width", 30.0, 31.0, tolerance=0.5)
        self.assertFalse(c.within_tolerance)

    def test_boundary_passes(self):
        c = check_dimension("width", 30.0, 30.5, tolerance=0.5)
        self.assertTrue(c.within_tolerance)

    def test_negative_tolerance_raises(self):
        with self.assertRaises(ValueError):
            check_dimension("x", 1.0, 1.0, tolerance=-0.1)

    def test_check_by_percent(self):
        # 1% band on nominal 30 -> 0.3 mm; measured 29.8 deviates 0.2 -> pass.
        c = check_by_percent("width", 30.0, 29.8, percent_tolerance=1.0)
        self.assertTrue(c.within_tolerance)
        c2 = check_by_percent("width", 30.0, 29.5, percent_tolerance=1.0)
        self.assertFalse(c2.within_tolerance)


class TestQAReport(unittest.TestCase):
    def test_all_pass(self):
        checks = [
            check_dimension("w", 30.0, 29.9, 0.5),
            check_dimension("h", 20.0, 20.1, 0.5),
        ]
        rep = qa_report(checks)
        self.assertTrue(rep.all_pass)
        self.assertEqual(rep.num_pass, 2)
        self.assertEqual(rep.num_fail, 0)
        self.assertAlmostEqual(rep.max_abs_deviation, 0.1)
        self.assertAlmostEqual(rep.mean_abs_error, 0.1)

    def test_with_failure(self):
        checks = [
            check_dimension("w", 30.0, 29.9, 0.5),
            check_dimension("h", 20.0, 25.0, 0.5),
        ]
        rep = qa_report(checks)
        self.assertFalse(rep.all_pass)
        self.assertEqual(rep.num_fail, 1)
        self.assertEqual(len(rep.failures()), 1)
        self.assertEqual(rep.failures()[0].name, "h")
        self.assertAlmostEqual(rep.max_abs_deviation, 5.0)

    def test_rmse(self):
        checks = [
            check_dimension("a", 0.0, 3.0, 10.0),   # dev 3
            check_dimension("b", 0.0, 4.0, 10.0),   # dev 4
        ]
        rep = qa_report(checks)
        # RMSE = sqrt((9+16)/2) = sqrt(12.5).
        self.assertAlmostEqual(rep.rms_error, math.sqrt(12.5))

    def test_empty_report(self):
        rep = qa_report([])
        self.assertTrue(rep.all_pass)
        self.assertEqual(rep.num_checks, 0)
        self.assertEqual(rep.rms_error, 0.0)

    def test_mape(self):
        checks = [
            check_dimension("a", 100.0, 99.0, 5.0),   # 1% error
            check_dimension("b", 100.0, 103.0, 5.0),  # -3% error
        ]
        rep = qa_report(checks)
        self.assertAlmostEqual(rep.mean_abs_percent_error, 2.0)

    def test_compare_dimensions(self):
        nominal = {"w": 30.0, "h": 20.0, "extra": 5.0}
        measured = {"w": 29.9, "h": 20.6, "other": 9.0}
        rep = compare_dimensions(nominal, measured, tolerance=0.5)
        # Only shared keys w, h checked.
        self.assertEqual(rep.num_checks, 2)
        self.assertEqual(rep.num_fail, 1)  # h off by 0.6 > 0.5
        names = sorted(c.name for c in rep.checks)
        self.assertEqual(names, ["h", "w"])


if __name__ == "__main__":
    unittest.main()
