"""Tests for geometry.cadgpt_module_series (standard module snapping)."""

import unittest

from geometry.cadgpt_module_series import (
    PREFERRED_MODULES,
    SECONDARY_MODULES,
    ALL_MODULES,
    standard_module,
    nearest_module,
    is_standard_module,
)


class TestSeries(unittest.TestCase):
    def test_all_modules_sorted_and_merged(self):
        self.assertEqual(ALL_MODULES, sorted(ALL_MODULES))
        for m in PREFERRED_MODULES + SECONDARY_MODULES:
            self.assertIn(m, ALL_MODULES)

    def test_is_standard(self):
        self.assertTrue(is_standard_module(2.5))
        self.assertTrue(is_standard_module(1.75))
        self.assertFalse(is_standard_module(1.75, allow_secondary=False))
        self.assertFalse(is_standard_module(2.4))


class TestStandardModule(unittest.TestCase):
    def test_rounds_up_preferred(self):
        # 2.3 -> next preferred >= 2.3 is 2.5
        self.assertEqual(standard_module(2.3), 2.5)

    def test_exact_value_kept(self):
        self.assertEqual(standard_module(2.5), 2.5)

    def test_just_above_bumps_next(self):
        self.assertEqual(standard_module(2.6), 3.0)

    def test_secondary_can_win_when_allowed(self):
        # 2.6 -> preferred gives 3.0, secondary offers 2.75 which is closer & >=
        self.assertEqual(standard_module(2.6, allow_secondary=True), 2.75)

    def test_secondary_ignored_by_default(self):
        self.assertEqual(standard_module(2.6, allow_secondary=False), 3.0)

    def test_small_value(self):
        self.assertEqual(standard_module(0.5), 1.0)

    def test_too_large_raises(self):
        with self.assertRaises(ValueError):
            standard_module(1000.0)

    def test_non_positive_raises(self):
        with self.assertRaises(ValueError):
            standard_module(0.0)


class TestNearestModule(unittest.TestCase):
    def test_nearest_below(self):
        self.assertEqual(nearest_module(2.45), 2.5)

    def test_nearest_can_go_down(self):
        # 2.05 is closest to 2.0
        self.assertEqual(nearest_module(2.05), 2.0)

    def test_tie_rounds_up(self):
        # midpoint between 2.0 and 2.5 is 2.25, which is itself standard(sec).
        # Use a tie against preferred-only series: 2.25 equidistant 2.0/2.5 -> 2.5
        self.assertEqual(nearest_module(2.25, allow_secondary=False), 2.5)

    def test_non_positive_raises(self):
        with self.assertRaises(ValueError):
            nearest_module(-1.0)


if __name__ == "__main__":
    unittest.main()
