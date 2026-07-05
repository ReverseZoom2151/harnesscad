import unittest

from drawings.cad2program_view_lifting import (
    Rect, parse_rect, parse_view, lift_three_views, check_three_view_consistency,
    extrude_profile, lift_matched_components, FRONT, TOP, SIDE,
)
from reconstruction.cad2program_shape_program import Bbox


class ParseTest(unittest.TestCase):
    def test_parse_rect(self):
        r = parse_rect((1, 2, 3, 4))
        self.assertEqual((r.h0, r.v0, r.hw, r.vh), (1, 2, 3, 4))
        self.assertEqual((r.h1, r.v1), (4, 6))

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            parse_rect((0, 0, -1, 2))

    def test_parse_view(self):
        rects = parse_view([(0, 0, 1, 1), (2, 2, 3, 3)])
        self.assertEqual(len(rects), 2)


class LiftThreeViewsTest(unittest.TestCase):
    def test_lift_unit_box(self):
        # A box spanning X:[0,4], Y:[0,2], Z:[0,6].
        front = Rect(0, 0, 4, 6)   # X-Z
        top = Rect(0, 0, 4, 2)     # X-Y
        side = Rect(0, 0, 2, 6)    # Y-Z
        box = lift_three_views(front, top, side)
        self.assertEqual(box, Bbox(2, 1, 3, 4, 2, 6, 0))

    def test_offset_box(self):
        # X:[10,14], Y:[5,7], Z:[1,7]
        front = Rect(10, 1, 4, 6)
        top = Rect(10, 5, 4, 2)
        side = Rect(5, 1, 2, 6)
        box = lift_three_views(front, top, side)
        self.assertEqual(box.position, (12, 6, 4))
        self.assertEqual(box.size, (4, 2, 6))

    def test_inconsistent_x(self):
        front = Rect(0, 0, 4, 6)
        top = Rect(0, 0, 5, 2)   # X extent 5 != 4
        side = Rect(0, 0, 2, 6)
        with self.assertRaises(ValueError):
            lift_three_views(front, top, side)
        self.assertFalse(check_three_view_consistency(front, top, side))

    def test_consistency_true(self):
        front = Rect(0, 0, 4, 6)
        top = Rect(0, 0, 4, 2)
        side = Rect(0, 0, 2, 6)
        self.assertTrue(check_three_view_consistency(front, top, side))


class ExtrudeTest(unittest.TestCase):
    def test_extrude_front_along_y(self):
        prof = Rect(0, 0, 4, 6)   # X-Z profile
        box = extrude_profile(prof, FRONT, depth=2)
        self.assertEqual(box.size, (4, 2, 6))
        self.assertEqual(box.scale_y, 2)

    def test_extrude_top_along_z(self):
        prof = Rect(0, 0, 4, 2)   # X-Y profile
        box = extrude_profile(prof, TOP, depth=6)
        self.assertEqual(box.size, (4, 2, 6))

    def test_extrude_side_along_x(self):
        prof = Rect(0, 0, 2, 6)   # Y-Z profile
        box = extrude_profile(prof, SIDE, depth=4)
        self.assertEqual(box.size, (4, 2, 6))

    def test_bad_view(self):
        with self.assertRaises(ValueError):
            extrude_profile(Rect(0, 0, 1, 1), "back", 1)


class MultiComponentTest(unittest.TestCase):
    def test_two_components(self):
        # Component A: X[0,4] Y[0,2] Z[0,6]; Component B: X[0,4] Y[0,2] Z[6,8]
        front = [Rect(0, 0, 4, 6), Rect(0, 6, 4, 2)]
        top = [Rect(0, 0, 4, 2), Rect(0, 0, 4, 2)]
        side = [Rect(0, 0, 2, 6), Rect(0, 6, 2, 2)]
        boxes = lift_matched_components(front, top, side,
                                        [(0, 0, 0), (1, 1, 1)])
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0].size, (4, 2, 6))
        self.assertEqual(boxes[1], Bbox(2, 1, 7, 4, 2, 2, 0))


if __name__ == "__main__":
    unittest.main()
