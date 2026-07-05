"""Tests for creft_view_consistency."""

import unittest

from drawings.creft_projection import Box, View, project_three_views
from drawings.creft_view_consistency import (
    DimensionConstraint,
    check_dimension_constraints,
    check_view_consistency,
    contour_has_overlap,
    contour_is_connected,
    contour_is_valid,
    implied_dimensions,
    views_match,
)


class ConsistencyTest(unittest.TestCase):
    def test_consistent_projection(self):
        views = project_three_views([Box(0, 0, 0, 4, 6, 8)])
        res = check_view_consistency(views)
        self.assertTrue(res.consistent)
        self.assertEqual(res.mismatches, ())

    def test_inconsistent_width(self):
        # Hand-build views whose front/top X extents disagree.
        views = project_three_views([Box(0, 0, 0, 4, 6, 8)])
        bad_top = View("top", "x", "y", (type(views["top"].rects[0])(0, 0, 9, 6),))
        views["top"] = bad_top
        res = check_view_consistency(views)
        self.assertFalse(res.consistent)
        axes = {m["axis"] for m in res.mismatches}
        self.assertIn("x", axes)

    def test_missing_view(self):
        views = project_three_views([Box(0, 0, 0, 4, 6, 8)])
        del views["side"]
        res = check_view_consistency(views)
        self.assertFalse(res.consistent)

    def test_implied_dimensions(self):
        views = project_three_views([Box(0, 0, 0, 4, 6, 8)])
        dims = implied_dimensions(views)
        self.assertEqual(dims, {"width": 4, "height": 8, "depth": 6})


class ViewsMatchTest(unittest.TestCase):
    def test_same_object(self):
        a = project_three_views([Box(0, 0, 0, 4, 6, 8)])
        b = project_three_views([Box(2, 2, 2, 4, 6, 8)])  # translated, same size
        self.assertTrue(views_match(a, b))

    def test_different_object(self):
        a = project_three_views([Box(0, 0, 0, 4, 6, 8)])
        b = project_three_views([Box(0, 0, 0, 4, 6, 9)])
        self.assertFalse(views_match(a, b))


class ContourTest(unittest.TestCase):
    def test_connected_touching(self):
        # two boxes sharing a face -> connected, no overlap
        boxes = [Box(0, 0, 0, 4, 2, 4), Box(4, 0, 0, 4, 2, 4)]
        top = project_three_views(boxes)["top"]
        self.assertTrue(contour_is_connected(top))
        self.assertFalse(contour_has_overlap(top))
        self.assertTrue(contour_is_valid(top))

    def test_gap_disconnected(self):
        boxes = [Box(0, 0, 0, 2, 2, 4), Box(5, 0, 0, 2, 2, 4)]
        top = project_three_views(boxes)["top"]
        self.assertFalse(contour_is_connected(top))
        self.assertFalse(contour_is_valid(top))

    def test_overlap_detected(self):
        boxes = [Box(0, 0, 0, 4, 4, 4), Box(1, 1, 0, 4, 4, 4)]
        top = project_three_views(boxes)["top"]
        self.assertTrue(contour_has_overlap(top))
        self.assertFalse(contour_is_valid(top))


class DimensionConstraintTest(unittest.TestCase):
    def test_less_than_satisfied(self):
        c = DimensionConstraint("spacing", "<", "cap_beam")
        self.assertTrue(c.evaluate({"spacing": 2.0, "cap_beam": 5.0}))
        self.assertFalse(c.evaluate({"spacing": 6.0, "cap_beam": 5.0}))

    def test_check_returns_violations(self):
        constraints = [
            DimensionConstraint("a", "<", "b"),
            DimensionConstraint("b", "<=", "c"),
        ]
        violated = check_dimension_constraints({"a": 1, "b": 3, "c": 2}, constraints)
        self.assertEqual(len(violated), 1)
        self.assertEqual(violated[0].left, "b")

    def test_missing_param_skipped(self):
        constraints = [DimensionConstraint("a", "<", "missing")]
        self.assertEqual(check_dimension_constraints({"a": 1}, constraints), [])

    def test_all_ops(self):
        p = {"a": 2.0, "b": 2.0}
        self.assertTrue(DimensionConstraint("a", "==", "b").evaluate(p))
        self.assertTrue(DimensionConstraint("a", "<=", "b").evaluate(p))
        self.assertTrue(DimensionConstraint("a", ">=", "b").evaluate(p))
        self.assertFalse(DimensionConstraint("a", ">", "b").evaluate(p))
        with self.assertRaises(ValueError):
            DimensionConstraint("a", "~", "b").evaluate(p)


if __name__ == "__main__":
    unittest.main()
