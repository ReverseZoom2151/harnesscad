"""Tests for eval.bench.geometry.identity_preservation."""

import unittest

from harnesscad.eval.bench.geometry.identity_preservation import (
    edit_locality,
    edit_report,
    identity_preservation,
)


class PreservationTest(unittest.TestCase):
    def test_perfect_preservation(self):
        # region = {top}; everything else preserved.
        p = identity_preservation(
            before={"top", "side", "bottom"},
            after={"top2", "side", "bottom"},
            intended_region={"top", "top2"},
        )
        self.assertAlmostEqual(p, 1.0)

    def test_collateral_damage(self):
        # "side" removed though not in region -> preservation drops.
        p = identity_preservation(
            before={"top", "side", "bottom"},
            after={"top2", "bottom"},
            intended_region={"top", "top2"},
        )
        self.assertAlmostEqual(p, 0.5)  # of {side, bottom}, only bottom preserved

    def test_modified_counts_as_not_preserved(self):
        p = identity_preservation(
            before={"a", "b"}, after={"a", "b"},
            intended_region={"a"}, modified={"b"},
        )
        self.assertAlmostEqual(p, 0.0)  # b modified, outside region

    def test_no_outside_entities(self):
        p = identity_preservation({"a"}, {"a2"}, intended_region={"a", "a2"})
        self.assertEqual(p, 1.0)


class LocalityTest(unittest.TestCase):
    def test_fully_local(self):
        loc = edit_locality(
            before={"top", "side"},
            after={"top2", "side"},
            intended_region={"top", "top2"},
        )
        self.assertAlmostEqual(loc, 1.0)

    def test_nonlocal_change(self):
        loc = edit_locality(
            before={"top", "side"},
            after={"top", "side2"},
            intended_region={"top"},
        )
        self.assertAlmostEqual(loc, 0.0)  # changed {side, side2}, none in region

    def test_noop_is_local(self):
        self.assertEqual(edit_locality({"a"}, {"a"}, {"a"}), 1.0)


class ReportTest(unittest.TestCase):
    def test_report_fields(self):
        r = edit_report(
            before={"top", "side", "bottom"},
            after={"top2", "side", "bottom"},
            intended_region={"top", "top2"},
        )
        self.assertEqual(r["added"], {"top2"})
        self.assertEqual(r["removed"], {"top"})
        self.assertAlmostEqual(r["identity_score"], 1.0)

    def test_bad_weight(self):
        with self.assertRaises(ValueError):
            edit_report({"a"}, {"a"}, {"a"}, w_preservation=2.0)


if __name__ == "__main__":
    unittest.main()
