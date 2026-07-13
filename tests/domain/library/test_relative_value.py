"""Tests for the relative property-edit resolver."""

import unittest

from harnesscad.domain.library.relative_value import (
    ABSOLUTE,
    DELTA,
    PERCENT,
    SCALE,
    Resolution,
    classify,
    resolve,
)


class ClassifyTest(unittest.TestCase):
    def test_percent(self):
        self.assertEqual(classify("+10%"), PERCENT)
        self.assertEqual(classify("-20%"), PERCENT)

    def test_scale(self):
        self.assertEqual(classify("*1.5"), SCALE)

    def test_delta_positive(self):
        self.assertEqual(classify("+5"), DELTA)

    def test_delta_negative(self):
        self.assertEqual(classify("-3"), DELTA)

    def test_absolute(self):
        self.assertEqual(classify("50"), ABSOLUTE)
        self.assertEqual(classify("12.5"), ABSOLUTE)

    def test_non_numeric(self):
        self.assertIsNone(classify("wide"))
        self.assertIsNone(classify(""))
        self.assertIsNone(classify("+abc%"))
        self.assertIsNone(classify(None))


class ResolvePercentTest(unittest.TestCase):
    def test_grow(self):
        r = resolve(100, "+10%")
        self.assertEqual(r.kind, PERCENT)
        self.assertAlmostEqual(r.resolved, 110.0)
        self.assertAlmostEqual(r.delta, 10.0)
        self.assertTrue(r.changed)

    def test_shrink(self):
        r = resolve(50, "-20%")
        self.assertAlmostEqual(r.resolved, 40.0)


class ResolveScaleTest(unittest.TestCase):
    def test_scale_up(self):
        r = resolve(10, "*1.5")
        self.assertEqual(r.kind, SCALE)
        self.assertAlmostEqual(r.resolved, 15.0)

    def test_scale_down(self):
        r = resolve(10, "*0.5")
        self.assertAlmostEqual(r.resolved, 5.0)


class ResolveDeltaTest(unittest.TestCase):
    def test_add(self):
        r = resolve(10, "+5")
        self.assertEqual(r.kind, DELTA)
        self.assertAlmostEqual(r.resolved, 15.0)

    def test_subtract(self):
        r = resolve(10, "-3")
        self.assertEqual(r.kind, DELTA)
        self.assertAlmostEqual(r.resolved, 7.0)


class ResolveAbsoluteTest(unittest.TestCase):
    def test_set(self):
        r = resolve(10, "50")
        self.assertEqual(r.kind, ABSOLUTE)
        self.assertAlmostEqual(r.resolved, 50.0)
        self.assertAlmostEqual(r.previous, 10.0)

    def test_no_change(self):
        r = resolve(50, "50")
        self.assertFalse(r.changed)
        self.assertEqual(r.delta, 0.0)


class NonNumericTest(unittest.TestCase):
    def test_non_numeric_token(self):
        self.assertIsNone(resolve(10, "wider"))

    def test_non_numeric_current(self):
        self.assertIsNone(resolve("notnum", "+5"))

    def test_returns_resolution_type(self):
        self.assertIsInstance(resolve(1, "+1"), Resolution)


class ClampTest(unittest.TestCase):
    def test_min_clamp(self):
        r = resolve(10, "-50%", minimum=6)
        self.assertEqual(r.resolved, 6.0)

    def test_max_clamp(self):
        r = resolve(10, "*10", maximum=50)
        self.assertEqual(r.resolved, 50.0)

    def test_within_bounds_unclamped(self):
        r = resolve(10, "+5", minimum=0, maximum=100)
        self.assertEqual(r.resolved, 15.0)


if __name__ == "__main__":
    unittest.main()
