"""Tests for the forgent3d printability verdict and issue taxonomy."""

import unittest

from harnesscad.domain.fabrication import printability_verdict as pv


PROFILE = pv.PrinterProfile()


class FitTest(unittest.TestCase):
    def test_fits_within_volume(self):
        res = pv.check_fit((50, 50, 50), PROFILE)
        self.assertTrue(res.fits)
        self.assertTrue(res.rotated_fits)

    def test_over_axis_flagged(self):
        # z far exceeds 256 - 4 usable.
        res = pv.check_fit((10, 10, 400), PROFILE)
        self.assertFalse(res.fits)
        self.assertGreater(res.over_mm[2], 0)

    def test_rotation_permutation_fits(self):
        # Long thin part: 300 x 10 x 10 does not fit z=300>252 axis-aligned in a
        # cube volume it always does; use asymmetric volume.
        prof = pv.PrinterProfile(build_volume_mm=(300, 100, 50), margin_mm=0.0)
        # 250 tall won't fit z=50, but rotated (250 into 300 axis) fits.
        res = pv.check_fit((10, 10, 250), prof)
        self.assertFalse(res.fits)
        self.assertTrue(res.rotated_fits)


class IssueTest(unittest.TestCase):
    def test_clean_part_print_ok(self):
        m = pv.Measurements(
            size_mm=(50, 50, 50), is_valid_solid=True, is_watertight=True,
            min_wall_mm=3.0, overhang_area_ratio=0.0,
        )
        issues = pv.classify_issues(m, PROFILE)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "PRINT_OK")

    def test_thin_wall_warning_vs_error(self):
        warn = pv.classify_issues(pv.Measurements((50, 50, 50), min_wall_mm=0.6), PROFILE)
        self.assertTrue(any(i.code == "THIN_WALL" and i.severity == "warning" for i in warn))
        err = pv.classify_issues(pv.Measurements((50, 50, 50), min_wall_mm=0.3), PROFILE)
        self.assertTrue(any(i.code == "THIN_WALL" and i.severity == "error" for i in err))

    def test_not_solid_error(self):
        issues = pv.classify_issues(pv.Measurements((50, 50, 50), is_valid_solid=False), PROFILE)
        self.assertTrue(any(i.code == "NOT_SOLID" for i in issues))

    def test_overhang_warning(self):
        issues = pv.classify_issues(pv.Measurements((50, 50, 50), overhang_area_ratio=0.3), PROFILE)
        self.assertTrue(any(i.code == "OVERHANG" for i in issues))

    def test_small_feature_warning(self):
        issues = pv.classify_issues(pv.Measurements((50, 50, 50), short_edges=2, tiny_faces=1), PROFILE)
        self.assertTrue(any(i.code == "SMALL_FEATURE" for i in issues))

    def test_unmeasured_wall_no_issue(self):
        issues = pv.classify_issues(pv.Measurements((50, 50, 50), min_wall_mm=None), PROFILE)
        self.assertFalse(any(i.code == "THIN_WALL" for i in issues))


class ScoreTest(unittest.TestCase):
    def test_clean_is_100_printable(self):
        m = pv.Measurements((50, 50, 50), is_valid_solid=True, is_watertight=True, min_wall_mm=3.0)
        verdict = pv.printability_verdict(m, PROFILE)
        self.assertTrue(verdict["printable"])
        self.assertEqual(verdict["score"], 100)

    def test_error_makes_unprintable(self):
        m = pv.Measurements((50, 50, 50), is_valid_solid=False)
        verdict = pv.printability_verdict(m, PROFILE)
        self.assertFalse(verdict["printable"])
        self.assertLessEqual(verdict["score"], 65)

    def test_score_clamped_nonnegative(self):
        # Several errors: 3 errors * 35 = 105 penalty -> clamp to 0.
        m = pv.Measurements((400, 400, 400), is_valid_solid=False, min_wall_mm=0.1)
        _, score = pv.score_issues(pv.classify_issues(m, PROFILE))
        self.assertGreaterEqual(score, 0)


if __name__ == "__main__":
    unittest.main()
