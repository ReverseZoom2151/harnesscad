"""Tests for programs.paramgeom_handles (parametric primitive handle grids)."""

import unittest
from fractions import Fraction

from harnesscad.domain.programs.expressions.handles import (
    circle_handles,
    cube_handles,
    cylinder_handles,
    handle_role,
    sphere_handles,
    sphere_handles_from_radius,
    square_handles,
)
from harnesscad.domain.programs.expressions.linear_form import LinearForm


class CubeHandlesTest(unittest.TestCase):
    def test_count_is_27(self):
        self.assertEqual(len(cube_handles("sx", "sy", "sz")), 27)

    def test_center_is_zero(self):
        h = cube_handles("sx", "sy", "sz")
        cx, cy, cz = h["center"]
        self.assertTrue(cx.is_zero and cy.is_zero and cz.is_zero)

    def test_bottom_face_center_z(self):
        # middle of the bottom face -> z = -sz/2, x=y=0
        h = cube_handles("sx", "sy", "sz")
        dx, dy, dz = h["xmid_ymid_zmin"]
        self.assertTrue(dx.is_zero)
        self.assertTrue(dy.is_zero)
        self.assertEqual(dz.coefficient("sz"), Fraction(-1, 2))

    def test_corner_offset(self):
        h = cube_handles("sx", "sy", "sz")
        dx, dy, dz = h["xmax_ymax_zmax"]
        self.assertEqual(dx.coefficient("sx"), Fraction(1, 2))
        self.assertEqual(dy.coefficient("sy"), Fraction(1, 2))
        self.assertEqual(dz.coefficient("sz"), Fraction(1, 2))

    def test_numeric_size(self):
        h = cube_handles(10, 10, 10)
        dx, _, _ = h["xmax_ymid_zmid"]
        self.assertEqual(dx.constant, Fraction(5))

    def test_roles(self):
        self.assertEqual(handle_role("center"), "center")
        self.assertEqual(handle_role("xmax_ymid_zmid"), "face")
        self.assertEqual(handle_role("xmax_ymax_zmid"), "edge")
        self.assertEqual(handle_role("xmax_ymax_zmax"), "corner")

    def test_role_counts(self):
        roles = [handle_role(n) for n in cube_handles("a", "b", "c")]
        self.assertEqual(roles.count("center"), 1)
        self.assertEqual(roles.count("face"), 6)
        self.assertEqual(roles.count("edge"), 12)
        self.assertEqual(roles.count("corner"), 8)


class SphereHandlesTest(unittest.TestCase):
    def test_count_is_27(self):
        self.assertEqual(len(sphere_handles("d")), 27)

    def test_boundary_uses_diameter(self):
        h = sphere_handles("d")
        dx, _, _ = h["xmax_ymid_zmid"]
        self.assertEqual(dx.coefficient("d"), Fraction(1, 2))

    def test_from_radius(self):
        h = sphere_handles_from_radius("r")
        dx, _, _ = h["xmax_ymid_zmid"]
        # boundary at +r == +d/2 with d = 2r
        self.assertEqual(dx.coefficient("r"), Fraction(1))


class CylinderHandlesTest(unittest.TestCase):
    def test_count_is_27(self):
        self.assertEqual(len(cylinder_handles("r1", "r2", "h")), 27)

    def test_bottom_radius_uses_r1(self):
        h = cylinder_handles("r1", "r2", "h")
        dx, _, dz = h["xmax_ymid_zmin"]
        self.assertEqual(dx.coefficient("r1"), Fraction(1))  # +d1/2 = +r1
        self.assertEqual(dz.coefficient("h"), Fraction(-1, 2))

    def test_top_radius_uses_r2(self):
        h = cylinder_handles("r1", "r2", "h")
        dx, _, dz = h["xmax_ymid_zmax"]
        self.assertEqual(dx.coefficient("r2"), Fraction(1))
        self.assertEqual(dz.coefficient("h"), Fraction(1, 2))

    def test_mid_plane_mean_radius(self):
        h = cylinder_handles("r1", "r2", "h")
        dx, _, dz = h["xmax_ymid_zmid"]
        # mid diameter = r1 + r2 -> half extent = (r1+r2)/2
        self.assertEqual(dx.coefficient("r1"), Fraction(1, 2))
        self.assertEqual(dx.coefficient("r2"), Fraction(1, 2))
        self.assertTrue(dz.is_zero)


class SquareCircleTest(unittest.TestCase):
    def test_square_count(self):
        self.assertEqual(len(square_handles("sx", "sy")), 9)

    def test_square_center_zero(self):
        cx, cy, cz = square_handles("sx", "sy")["center"]
        self.assertTrue(cx.is_zero and cy.is_zero and cz.is_zero)

    def test_square_corner(self):
        dx, dy, dz = square_handles("sx", "sy")["xmax_ymax"]
        self.assertEqual(dx.coefficient("sx"), Fraction(1, 2))
        self.assertEqual(dy.coefficient("sy"), Fraction(1, 2))
        self.assertTrue(dz.is_zero)

    def test_circle_has_five_points(self):
        self.assertEqual(len(circle_handles("d")), 5)

    def test_circle_extremes(self):
        h = circle_handles("d")
        dx, _, _ = h["xmax"]
        self.assertEqual(dx.coefficient("d"), Fraction(1, 2))
        cx, cy, cz = h["center"]
        self.assertTrue(cx.is_zero and cy.is_zero and cz.is_zero)


if __name__ == "__main__":
    unittest.main()
