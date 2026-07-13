import unittest

from harnesscad.domain.editing.mrcad2_edit_degeneracy import (
    canonicalize_design,
    move_point_resolved,
    resolve_curve,
)
from harnesscad.domain.editing.mrcad_schema import Curve, Design, arc, circle, line


class TestResolveCurve(unittest.TestCase):
    def test_valid_line_unchanged(self):
        c = line((0, 0), (1, 1))
        self.assertEqual(resolve_curve(c), c)

    def test_zero_length_line_dropped(self):
        self.assertIsNone(resolve_curve(Curve("line", ((2, 2), (2, 2)))))

    def test_degenerate_circle_dropped(self):
        self.assertIsNone(resolve_curve(Curve("circle", ((3, 3), (3, 3)))))

    def test_arc_start_equals_end_becomes_circle(self):
        a = Curve("arc", ((0, 0), (1, 1), (0, 0)))
        r = resolve_curve(a)
        self.assertIsNotNone(r)
        self.assertEqual(r.kind, "circle")
        self.assertEqual(r.points, ((0.0, 0.0), (1.0, 1.0)))

    def test_arc_start_equals_mid_dropped(self):
        self.assertIsNone(resolve_curve(Curve("arc", ((0, 0), (0, 0), (2, 2)))))

    def test_arc_mid_equals_end_dropped(self):
        self.assertIsNone(resolve_curve(Curve("arc", ((0, 0), (2, 2), (2, 2)))))

    def test_valid_arc_unchanged(self):
        a = arc((1, 0), (0, 1), (-1, 0))
        self.assertEqual(resolve_curve(a), a)


class TestMovePointResolved(unittest.TestCase):
    def test_line_collapses_and_is_dropped(self):
        # Move (0,0) onto (10,0): the line degenerates and disappears.
        d = Design((line((0, 0), (10, 0)),))
        out = move_point_resolved(d, (0, 0), (10, 0))
        self.assertEqual(len(out), 0)

    def test_arc_collapses_to_circle(self):
        # Move the arc's end onto its start -> a circle.
        d = Design((arc((0, 0), (5, 5), (10, 0)),))
        out = move_point_resolved(d, (10, 0), (0, 0))
        self.assertEqual(len(out), 1)
        self.assertEqual(out.curves[0].kind, "circle")

    def test_shared_point_moves_all_curves(self):
        # Two lines share (0,0); moving it relocates both.
        d = Design((line((0, 0), (5, 0)), line((0, 0), (0, 5))))
        out = move_point_resolved(d, (0, 0), (1, 1))
        for c in out.curves:
            self.assertIn((1.0, 1.0), c.points)

    def test_untouched_curve_unchanged(self):
        d = Design((line((0, 0), (5, 0)), line((9, 9), (8, 8))))
        out = move_point_resolved(d, (0, 0), (2, 2))
        self.assertIn(line((9, 9), (8, 8)), out.curves)

    def test_no_matching_point_is_noop(self):
        d = Design((line((0, 0), (5, 0)),))
        out = move_point_resolved(d, (100, 100), (1, 1))
        self.assertEqual(out, d)


class TestCanonicalizeDesign(unittest.TestCase):
    def test_drops_and_collapses(self):
        d = Design(
            (
                line((0, 0), (5, 0)),                     # keep
                Curve("line", ((2, 2), (2, 2))),          # drop
                Curve("arc", ((0, 0), (3, 3), (0, 0))),   # -> circle
            )
        )
        out = canonicalize_design(d)
        kinds = sorted(c.kind for c in out.curves)
        self.assertEqual(kinds, ["circle", "line"])


if __name__ == "__main__":
    unittest.main()
