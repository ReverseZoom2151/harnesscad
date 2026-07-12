"""Tests for reconstruction.gencad2_loop_reorder."""

import math
import unittest

from reconstruction.gencad2_loop_reorder import (
    ARC,
    CIRCLE,
    LINE,
    Curve,
    canonicalize_profile,
    circle,
    curve_bbox,
    curve_direction,
    is_counter_clockwise,
    leftmost_index,
    loop_bbox,
    loop_bbox_size,
    loop_is_closed,
    reorder_loop,
    reorder_profile,
    repair_orientation,
    reverse_curve,
)


def line(a, b):
    return Curve(kind=LINE, start=a, end=b)


def square_ccw(x=0.0, y=0.0, s=1.0):
    p = [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    return [line(p[i], p[(i + 1) % 4]) for i in range(4)]


def square_cw(x=0.0, y=0.0, s=1.0):
    p = [(x, y), (x, y + s), (x + s, y + s), (x + s, y)]
    return [line(p[i], p[(i + 1) % 4]) for i in range(4)]


class TestCurve(unittest.TestCase):
    def test_arc_requires_mid(self):
        with self.assertRaises(ValueError):
            Curve(kind=ARC, start=(0.0, 0.0), end=(1.0, 0.0))

    def test_circle_requires_center(self):
        with self.assertRaises(ValueError):
            Curve(kind=CIRCLE, radius=1.0)

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            Curve(kind="Spline")

    def test_circle_helper_endpoints(self):
        c = circle((0.0, 0.0), 2.0)
        self.assertEqual(c.start, (-2.0, 0.0))
        self.assertEqual(c.end, (2.0, 0.0))

    def test_reverse_line(self):
        r = reverse_curve(line((0.0, 0.0), (1.0, 2.0)))
        self.assertEqual(r.start, (1.0, 2.0))
        self.assertEqual(r.end, (0.0, 0.0))

    def test_reverse_circle_is_noop(self):
        c = circle((1.0, 1.0), 1.0)
        self.assertEqual(reverse_curve(c), c)

    def test_direction(self):
        self.assertEqual(curve_direction(line((0.0, 0.0), (1.0, 2.0))), (1.0, 2.0))
        arc = Curve(kind=ARC, start=(1.0, 0.0), mid=(0.0, 1.0), end=(-1.0, 0.0),
                    center=(0.0, 0.0), radius=1.0)
        self.assertEqual(curve_direction(arc, from_start=True), (-1.0, 1.0))
        self.assertEqual(curve_direction(arc, from_start=False), (-1.0, -1.0))


class TestOrientationRepair(unittest.TestCase):
    def test_second_curve_reversed(self):
        curves = [line((0.0, 0.0), (1.0, 0.0)), line((1.0, 1.0), (1.0, 0.0))]
        fixed = repair_orientation(curves)
        self.assertEqual(fixed[1].start, (1.0, 0.0))
        self.assertEqual(fixed[1].end, (1.0, 1.0))

    def test_first_curve_reversed(self):
        # first curve starts where the second one starts -> flip the first
        curves = [line((1.0, 0.0), (0.0, 0.0)), line((1.0, 0.0), (1.0, 1.0))]
        fixed = repair_orientation(curves)
        self.assertEqual(fixed[0].start, (0.0, 0.0))
        self.assertEqual(fixed[0].end, (1.0, 0.0))

    def test_already_chained_untouched(self):
        curves = square_ccw()
        self.assertEqual(repair_orientation(curves), curves)

    def test_single_curve_untouched(self):
        c = [circle((0.0, 0.0), 1.0)]
        self.assertEqual(repair_orientation(c), c)


class TestReorderLoop(unittest.TestCase):
    def test_leftmost_index(self):
        curves = square_ccw()
        rotated = curves[2:] + curves[:2]
        self.assertEqual(leftmost_index(rotated), 2)

    def test_rotation_to_leftmost_start(self):
        curves = square_ccw()
        rotated = curves[2:] + curves[:2]
        out = reorder_loop(rotated)
        self.assertEqual(out[0].start, (0.0, 0.0))
        self.assertEqual(out[0].end, (1.0, 0.0))

    def test_ccw_loop_is_preserved(self):
        out = reorder_loop(square_ccw())
        self.assertTrue(is_counter_clockwise(out))
        self.assertEqual([c.start for c in out],
                         [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    def test_cw_loop_is_flipped_to_ccw(self):
        out = reorder_loop(square_cw())
        self.assertTrue(is_counter_clockwise(out))
        self.assertEqual([c.start for c in out],
                         [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    def test_reordered_loop_stays_closed(self):
        for src in (square_ccw(), square_cw(), square_cw()[1:] + square_cw()[:1]):
            self.assertTrue(loop_is_closed(reorder_loop(src)))

    def test_ties_broken_by_lower_y(self):
        # two starts share x = 0; the one with the lower y must win
        out = reorder_loop(square_ccw())
        self.assertEqual(out[0].start, (0.0, 0.0))

    def test_circle_loop_untouched(self):
        c = [circle((3.0, 3.0), 1.0)]
        self.assertEqual(reorder_loop(c), c)

    def test_idempotent(self):
        once = reorder_loop(square_cw())
        twice = reorder_loop(once)
        self.assertEqual(once, twice)


class TestBBox(unittest.TestCase):
    def test_loop_bbox(self):
        self.assertEqual(loop_bbox(square_ccw()), (0.0, 0.0, 1.0, 1.0))

    def test_curve_bbox_arc_includes_bulge(self):
        arc = Curve(kind=ARC, start=(1.0, 0.0), mid=(0.0, 1.0), end=(-1.0, 0.0),
                    center=(0.0, 0.0), radius=1.0)
        box = curve_bbox(arc)
        self.assertAlmostEqual(box[3], 1.0, places=6)

    def test_bbox_size_relative_to_start_point(self):
        # square (0,0)..(1,1) starting at (0,0) -> size is 1, not the diagonal
        self.assertAlmostEqual(loop_bbox_size(square_ccw()), 1.0, places=9)

    def test_bbox_size_offset_start(self):
        curves = reorder_loop(square_ccw(x=-2.0, y=-2.0, s=4.0))
        # start point is (-2, -2); farthest corner offset is 4
        self.assertAlmostEqual(loop_bbox_size(curves), 4.0, places=9)

    def test_empty_loop_bbox_raises(self):
        with self.assertRaises(ValueError):
            loop_bbox([])


class TestProfile(unittest.TestCase):
    def test_loops_sorted_by_bbox_min_x_then_y(self):
        right = square_ccw(x=5.0)
        left = square_ccw(x=0.0)
        out = reorder_profile([right, left])
        self.assertEqual(loop_bbox(out[0])[0], 0.0)
        self.assertEqual(loop_bbox(out[1])[0], 5.0)

    def test_tie_on_x_broken_by_y(self):
        top = square_ccw(x=0.0, y=9.0)
        bottom = square_ccw(x=0.0, y=0.0)
        out = reorder_profile([top, bottom])
        self.assertEqual(loop_bbox(out[0])[1], 0.0)

    def test_canonicalize_reorders_and_sorts(self):
        out = canonicalize_profile([square_cw(x=5.0), square_cw(x=0.0)])
        self.assertEqual(loop_bbox(out[0])[0], 0.0)
        for lp in out:
            self.assertTrue(is_counter_clockwise(lp))
            self.assertTrue(loop_is_closed(lp))

    def test_single_loop_profile(self):
        lp = square_ccw()
        self.assertEqual(reorder_profile([lp]), [lp])

    def test_deterministic(self):
        a = canonicalize_profile([square_cw(x=2.0), square_cw()])
        b = canonicalize_profile([square_cw(x=2.0), square_cw()])
        self.assertEqual(a, b)


class TestClosure(unittest.TestCase):
    def test_open_loop_detected(self):
        curves = [line((0.0, 0.0), (1.0, 0.0)), line((1.0, 0.0), (1.0, 1.0))]
        self.assertFalse(loop_is_closed(curves))

    def test_circle_is_closed(self):
        self.assertTrue(loop_is_closed([circle((0.0, 0.0), 1.0)]))

    def test_empty_not_closed(self):
        self.assertFalse(loop_is_closed([]))


if __name__ == "__main__":
    unittest.main()
