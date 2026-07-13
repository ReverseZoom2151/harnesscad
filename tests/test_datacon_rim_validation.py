"""Tests for the deterministic rim feasibility validation module."""

import math
import unittest

from harnesscad.eval.verifiers.rim_feasibility import (
    ValidationReport,
    jaccard_similarity,
    point_in_polygon,
    polygon_area,
    polygon_centroid,
    rotate_polygon,
    rotational_symmetry_ok,
    single_outer_contour,
    spoke_position_ok,
    to_polar,
    validate_design,
)


def square(cx, cy, half):
    """Axis-aligned square centered at (cx, cy) with the given half-side."""
    return [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]


class TestPrimitives(unittest.TestCase):

    def test_polygon_area_unit_square(self):
        poly = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.assertAlmostEqual(polygon_area(poly), 1.0, places=5)

    def test_polygon_area_rectangle(self):
        poly = [(0, 0), (2, 0), (2, 3), (0, 3)]
        self.assertAlmostEqual(polygon_area(poly), 6.0, places=5)

    def test_polygon_area_clockwise_same(self):
        cw = [(0, 1), (1, 1), (1, 0), (0, 0)]
        self.assertAlmostEqual(polygon_area(cw), 1.0, places=5)

    def test_centroid_square(self):
        poly = square(5, 5, 1)
        cx, cy = polygon_centroid(poly)
        self.assertAlmostEqual(cx, 5.0, places=5)
        self.assertAlmostEqual(cy, 5.0, places=5)

    def test_to_polar(self):
        r, theta = to_polar(3.0, 4.0)
        self.assertAlmostEqual(r, 5.0, places=5)
        self.assertAlmostEqual(theta, math.atan2(4.0, 3.0), places=5)

    def test_jaccard_identical(self):
        s = square(0, 0, 2)
        self.assertAlmostEqual(jaccard_similarity(s, s), 1.0, places=5)

    def test_jaccard_disjoint(self):
        a = square(0, 0, 1)
        b = square(100, 100, 1)
        self.assertAlmostEqual(jaccard_similarity(a, b), 0.0, places=5)

    def test_jaccard_half_overlap(self):
        a = square(0, 0, 2)
        b = square(2, 0, 2)
        j = jaccard_similarity(a, b)
        self.assertGreater(j, 0.0)
        self.assertLess(j, 1.0)

    def test_point_in_polygon_inside(self):
        poly = square(0, 0, 5)
        self.assertTrue(point_in_polygon(0.0, 0.0, poly))

    def test_point_in_polygon_outside(self):
        poly = square(0, 0, 5)
        self.assertFalse(point_in_polygon(100.0, 100.0, poly))

    def test_rotate_polygon_quarter_turn(self):
        rotated = rotate_polygon([(1.0, 0.0)], math.pi / 2.0)
        rx, ry = rotated[0]
        self.assertAlmostEqual(rx, 0.0, places=5)
        self.assertAlmostEqual(ry, 1.0, places=5)


class TestSpokePosition(unittest.TestCase):

    def setUp(self):
        # Feasible annulus: inner = 50/2 + 5 + 10 = 40; outer = 400/2 - 30 - 4 = 166.
        self.kw = dict(pcd=50.0, bolt_radius=5.0, rim_diameter_D=400.0,
                       well_depth_H=30.0, rim_thickness_tc=4.0)

    def test_point_inside_annulus(self):
        pts = square(100, 0, 2)  # ~100 units from center, within [40, 166]
        self.assertTrue(spoke_position_ok(pts, **self.kw))

    def test_point_too_close_to_center(self):
        pts = square(5, 0, 1)  # ~5 units from center, below inner bound
        self.assertFalse(spoke_position_ok(pts, **self.kw))

    def test_point_beyond_outer_bound(self):
        pts = square(180, 0, 1)  # ~180 units, beyond outer bound 166
        self.assertFalse(spoke_position_ok(pts, **self.kw))


class TestSymmetry(unittest.TestCase):

    def _spoke_at(self, radius, angle, half=3.0):
        cx = radius * math.cos(angle)
        cy = radius * math.sin(angle)
        base = square(0, 0, half)
        return rotate_polygon(
            [(x + cx, y + cy) for (x, y) in base], angle, about=(cx, cy)
        )

    def test_four_fold_symmetry_true(self):
        r = 100.0
        spokes = [self._spoke_at(r, a)
                  for a in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)]
        self.assertTrue(rotational_symmetry_ok(spokes))

    def test_broken_symmetry_removed_spoke(self):
        r = 100.0
        spokes = [self._spoke_at(r, a)
                  for a in (0.0, math.pi / 2, math.pi)]  # 3 spokes at 90deg steps
        self.assertFalse(rotational_symmetry_ok(spokes))

    def test_broken_symmetry_shifted_angle(self):
        r = 100.0
        spokes = [self._spoke_at(r, a)
                  for a in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2 + 0.4)]
        self.assertFalse(rotational_symmetry_ok(spokes))

    def test_single_spoke_naturally_balanced(self):
        self.assertTrue(rotational_symmetry_ok([square(0, 0, 1)]))

    def test_no_spokes_naturally_balanced(self):
        self.assertTrue(rotational_symmetry_ok([]))


class TestSingleOuterContour(unittest.TestCase):

    def test_one_enclosing_contour(self):
        outer = square(0, 0, 50)
        inner_a = square(20, 0, 3)
        inner_b = square(-20, 0, 3)
        self.assertTrue(single_outer_contour([outer, inner_a, inner_b]))

    def test_two_overlapping_big_contours(self):
        a = square(0, 0, 50)
        b = square(40, 0, 50)  # neither bbox encloses the other
        self.assertFalse(single_outer_contour([a, b]))


class TestValidateDesign(unittest.TestCase):

    def _spoke_at(self, radius, angle, half=3.0):
        cx = radius * math.cos(angle)
        cy = radius * math.sin(angle)
        base = square(0, 0, half)
        return rotate_polygon(
            [(x + cx, y + cy) for (x, y) in base], angle, about=(cx, cy)
        )

    def test_happy_path_feasible(self):
        r = 100.0
        spokes = [self._spoke_at(r, a)
                  for a in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)]
        outer = square(0, 0, 195)  # encloses all spokes, within outer bound
        contours = [outer] + spokes
        spec = dict(pcd=50.0, bolt_radius=5.0, rim_diameter_D=400.0,
                    well_depth_H=30.0, rim_thickness_tc=4.0)
        report = validate_design(contours, spokes, spec)
        self.assertIsInstance(report, ValidationReport)
        self.assertTrue(report.single_contour)
        self.assertTrue(report.position_ok)
        self.assertTrue(report.symmetry_ok)
        self.assertTrue(report.feasible)
        self.assertEqual(report.reasons, [])


if __name__ == "__main__":
    unittest.main()
