import math
import unittest

from harnesscad.domain.reconstruction.fitting.solid_regeneration import (
    Solid,
    box_from_corners,
    close_contour,
    extrude_contour,
    is_closed,
    polygon_area,
    polygon_perimeter,
    regenerate_box,
)


class TestPolygon(unittest.TestCase):
    def test_area_square(self):
        sq = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
        self.assertAlmostEqual(polygon_area(sq), 16.0)

    def test_area_cw_negative(self):
        sq = [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0)]
        self.assertAlmostEqual(polygon_area(sq), -16.0)

    def test_perimeter(self):
        sq = [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)]
        self.assertAlmostEqual(polygon_perimeter(sq), 12.0)

    def test_is_closed(self):
        self.assertTrue(is_closed([(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]))
        self.assertFalse(is_closed([(0.0, 0.0), (1.0, 0.0)]))

    def test_close_contour_drops_duplicate(self):
        pts = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 0.0)]
        closed = close_contour(pts, tol=1e-6)
        self.assertEqual(len(closed), 3)


class TestExtrude(unittest.TestCase):
    def test_box_volume_and_area(self):
        solid = regenerate_box(2.0, 3.0, 4.0)
        self.assertAlmostEqual(solid.volume, 24.0)
        # Surface area of box: 2*(2*3 + 2*4 + 3*4) = 52.
        self.assertAlmostEqual(solid.surface_area, 52.0)
        self.assertEqual(len(solid.vertices), 8)
        # 2 caps + 4 sides.
        self.assertEqual(len(solid.faces), 6)

    def test_bounding_box(self):
        solid = regenerate_box(2.0, 3.0, 4.0)
        lo, hi = solid.bounding_box
        self.assertEqual(lo, (0.0, 0.0, 0.0))
        self.assertEqual(hi, (2.0, 3.0, 4.0))

    def test_extrude_triangle(self):
        tri = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]  # area 6
        solid = extrude_contour(tri, 10.0)
        self.assertAlmostEqual(solid.volume, 60.0)
        self.assertEqual(len(solid.vertices), 6)
        self.assertEqual(len(solid.faces), 5)  # 2 caps + 3 sides

    def test_extrude_l_shape_uniform_cross_section(self):
        # L-shaped cross-section (the paper's "complex uniform cross-section").
        L = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0),
             (2.0, 2.0), (2.0, 4.0), (0.0, 4.0)]
        area = abs(polygon_area(L))
        self.assertAlmostEqual(area, 12.0)
        solid = extrude_contour(L, 5.0)
        self.assertAlmostEqual(solid.volume, 60.0)
        self.assertEqual(len(solid.faces), 2 + 6)

    def test_invalid_depth(self):
        with self.assertRaises(ValueError):
            regenerate_box(1.0, 1.0, 0.0)

    def test_degenerate_contour(self):
        with self.assertRaises(ValueError):
            extrude_contour([(0.0, 0.0), (1.0, 1.0)], 5.0)
        with self.assertRaises(ValueError):
            # collinear -> zero area
            extrude_contour([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], 5.0)

    def test_box_from_corners(self):
        solid = box_from_corners((1.0, 1.0), (6.0, 4.0), 2.0)
        # width 5, height 3, depth 2 -> volume 30.
        self.assertAlmostEqual(solid.volume, 30.0)

    def test_closed_input_no_double_count(self):
        # Explicitly-closed square (last == first) should give same volume.
        sq = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)]
        solid = extrude_contour(sq, 3.0)
        self.assertAlmostEqual(solid.volume, 12.0)
        self.assertEqual(len(solid.vertices), 8)


if __name__ == "__main__":
    unittest.main()
