"""Tests for drawings.autocad_linetype_dash."""

import unittest

from harnesscad.domain.drawings.linetypes import (
    NAMED_PATTERNS,
    StrokedLine,
    pattern_length,
    point_at_arclen,
    apply_pattern,
    apply_named,
    dashed_length,
)


class TestPatternLength(unittest.TestCase):
    def test_abs_sum(self):
        self.assertAlmostEqual(pattern_length([0.5, -0.25]), 0.75)

    def test_empty(self):
        self.assertEqual(pattern_length([]), 0.0)


class TestPointAtArclen(unittest.TestCase):
    def test_midpoint(self):
        p = point_at_arclen([(0.0, 0.0), (10.0, 0.0)], 5.0)
        self.assertAlmostEqual(p[0], 5.0)
        self.assertAlmostEqual(p[1], 0.0)

    def test_clamped_ends(self):
        line = [(0.0, 0.0), (10.0, 0.0)]
        self.assertEqual(point_at_arclen(line, -1.0), (0.0, 0.0))
        self.assertEqual(point_at_arclen(line, 999.0), (10.0, 0.0))

    def test_across_vertex(self):
        # L-shape: 10 right then 10 up; arclen 15 -> (10,5)
        p = point_at_arclen([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)], 15.0)
        self.assertAlmostEqual(p[0], 10.0)
        self.assertAlmostEqual(p[1], 5.0)


class TestApplyPattern(unittest.TestCase):
    def test_continuous_returns_original(self):
        st = apply_pattern([(0.0, 0.0), (4.0, 0.0)], [])
        self.assertEqual(st.segments, [(0.0, 0.0, 4.0, 0.0)])
        self.assertEqual(st.dots, [])

    def test_dash_gap_tiling(self):
        # pattern dash 1, gap 1, over length 4 -> dashes [0,1] and [2,3]
        st = apply_pattern([(0.0, 0.0), (4.0, 0.0)], [1.0, -1.0])
        self.assertEqual(len(st.segments), 2)
        self.assertAlmostEqual(st.segments[0][0], 0.0)
        self.assertAlmostEqual(st.segments[0][2], 1.0)
        self.assertAlmostEqual(st.segments[1][0], 2.0)
        self.assertAlmostEqual(st.segments[1][2], 3.0)

    def test_inked_length(self):
        st = apply_pattern([(0.0, 0.0), (4.0, 0.0)], [1.0, -1.0])
        self.assertAlmostEqual(dashed_length(st), 2.0)

    def test_dots(self):
        # dot then gap 1 over length 3 -> dots at 0,1,2
        st = apply_pattern([(0.0, 0.0), (3.0, 0.0)], [0.0, -1.0])
        self.assertEqual(len(st.dots), 3)
        self.assertAlmostEqual(st.dots[0][0], 0.0)
        self.assertAlmostEqual(st.dots[1][0], 1.0)

    def test_scale(self):
        st = apply_pattern([(0.0, 0.0), (8.0, 0.0)], [1.0, -1.0], scale=2.0)
        # scaled dash length 2 -> first dash [0,2]
        self.assertAlmostEqual(st.segments[0][2], 2.0)

    def test_short_line_raises(self):
        with self.assertRaises(ValueError):
            apply_pattern([(0.0, 0.0)], [1.0, -1.0])


class TestApplyNamed(unittest.TestCase):
    def test_known(self):
        st = apply_named([(0.0, 0.0), (10.0, 0.0)], "dashed")
        self.assertIsInstance(st, StrokedLine)
        self.assertTrue(len(st.segments) >= 1)

    def test_all_named_run(self):
        line = [(0.0, 0.0), (20.0, 0.0)]
        for name in NAMED_PATTERNS:
            st = apply_named(line, name)
            self.assertIsInstance(st, StrokedLine)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            apply_named([(0.0, 0.0), (1.0, 0.0)], "nope")


if __name__ == "__main__":
    unittest.main()
