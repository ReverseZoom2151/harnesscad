"""Tests for the cadsmith solid-usability gate."""

import unittest

from harnesscad.eval.quality.geometry import solid_usability as su


def _good():
    return su.Measurement(volume=24000.0, bbox=(60.0, 40.0, 10.0), n_solids=1, free_edges=0, is_valid=True)


class GoodSolidTest(unittest.TestCase):
    def test_passes(self):
        report = su.assess(_good())
        self.assertTrue(report.ok)
        self.assertEqual(report.findings, [])
        self.assertIsNone(report.failure_message())

    def test_describe(self):
        report = su.assess(_good())
        text = report.describe()
        self.assertIn("volume", text)
        self.assertIn("solids=1", text)


class DefectTest(unittest.TestCase):
    def test_no_solid(self):
        report = su.assess(su.Measurement(volume=0.0, n_solids=0))
        self.assertFalse(report.ok)
        self.assertIn("no_solid", report.codes())

    def test_below_min_volume(self):
        m = su.Measurement(volume=0.2, bbox=(1, 1, 0.001), n_solids=1, free_edges=0)
        report = su.assess(m)
        self.assertFalse(report.ok)
        self.assertIn("below_min_volume", report.codes())

    def test_degenerate_bbox(self):
        m = su.Measurement(volume=5.0, bbox=(10, 10, 0.0), n_solids=1, free_edges=0)
        report = su.assess(m)
        self.assertIn("degenerate_bbox", report.codes())

    def test_units_mistake(self):
        m = su.Measurement(volume=1e9, bbox=(60000.0, 40000.0, 10000.0), n_solids=1, free_edges=0)
        report = su.assess(m)
        self.assertFalse(report.ok)
        self.assertIn("units_mistake", report.codes())

    def test_malformed_brep(self):
        m = su.Measurement(volume=100.0, bbox=(10, 10, 10), n_solids=1, free_edges=0, is_valid=False)
        report = su.assess(m)
        self.assertIn("malformed_brep", report.codes())

    def test_not_watertight(self):
        m = su.Measurement(volume=100.0, bbox=(10, 10, 10), n_solids=1, free_edges=4, is_valid=True)
        report = su.assess(m)
        self.assertFalse(report.ok)
        self.assertIn("not_watertight", report.codes())

    def test_failure_message_feedback(self):
        m = su.Measurement(volume=0.0, n_solids=0)
        report = su.assess(m)
        msg = report.failure_message()
        self.assertIsNotNone(msg)
        self.assertIn("invalid geometry", msg)


class SkipUnmeasuredTest(unittest.TestCase):
    def test_missing_fields_skipped(self):
        # Only n_solids known; validity/watertight not measured -> still passes.
        m = su.Measurement(n_solids=1)
        report = su.assess(m)
        self.assertTrue(report.ok)


if __name__ == "__main__":
    unittest.main()
