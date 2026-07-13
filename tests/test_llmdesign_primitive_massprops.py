"""Tests for quality.llmdesign_primitive_massprops."""

import unittest
from math import pi

from harnesscad.eval.quality.physics.llmdesign_primitive_massprops import (
    Assembly,
    Box,
    Cylinder,
    radially_symmetric_table,
    table_static_stability_zcm,
)


class TestPrimitiveMassProps(unittest.TestCase):
    def test_box_volume_mass_centroid(self):
        b = Box(1.0, 2.0, 3.0, 2.0, 4.0, 5.0)
        self.assertAlmostEqual(b.volume, 40.0)
        self.assertAlmostEqual(b.mass, 40.0)  # density 1.0
        self.assertEqual(b.centroid, (1.0, 2.0, 3.0))

    def test_cylinder_volume_and_centroid_each_axis(self):
        # volume independent of axis
        cz = Cylinder(0.0, 0.0, 0.0, 2.0, 10.0, axis="z")
        self.assertAlmostEqual(cz.volume, pi * 4.0 * 10.0)
        # base convention: centroid advances height/2 along the axis
        self.assertEqual(cz.centroid, (0.0, 0.0, 5.0))

        cx = Cylinder(1.0, 2.0, 3.0, 2.0, 10.0, axis="x")
        self.assertAlmostEqual(cx.volume, pi * 4.0 * 10.0)
        self.assertEqual(cx.centroid, (6.0, 2.0, 3.0))

        cy = Cylinder(1.0, 2.0, 3.0, 2.0, 10.0, axis="y")
        self.assertAlmostEqual(cy.volume, pi * 4.0 * 10.0)
        self.assertEqual(cy.centroid, (1.0, 7.0, 3.0))

    def test_density_scales_mass_not_centroid(self):
        b1 = Box(1.0, 1.0, 1.0, 2.0, 2.0, 2.0, density=1.0)
        b2 = Box(1.0, 1.0, 1.0, 2.0, 2.0, 2.0, density=3.0)
        self.assertAlmostEqual(b2.mass, 3.0 * b1.mass)
        self.assertEqual(b1.centroid, b2.centroid)
        self.assertAlmostEqual(b1.volume, b2.volume)

    def test_two_body_weighted_com_by_hand(self):
        # mass 2 at x=0, mass 6 at x=8  -> x_cm = (2*0 + 6*8)/8 = 6.0
        a = Box(0.0, 0.0, 0.0, 2.0, 1.0, 1.0, density=1.0)  # mass 2
        b = Box(8.0, 0.0, 0.0, 6.0, 1.0, 1.0, density=1.0)  # mass 6
        asm = Assembly([a, b])
        self.assertAlmostEqual(asm.total_mass, 8.0)
        cx, cy, cz = asm.center_of_mass
        self.assertAlmostEqual(cx, 6.0)
        self.assertAlmostEqual(cy, 0.0)
        self.assertAlmostEqual(cz, 0.0)

    def test_table_zcm_closed_form_hand_number(self):
        # R=2, r=0.2, H=1, h=3, rho=1
        R, r, H, h, rho = 2.0, 0.2, 1.0, 3.0, 1.0
        m_top = rho * pi * R * R * H          # 4*pi
        m_legs = 4 * rho * pi * r * r * h      # 4*pi*0.04*3 = 0.48*pi
        expected = (m_top * (h + H / 2) + m_legs * (h / 2)) / (m_top + m_legs)
        # numeric: (4pi*3.5 + 0.48pi*1.5)/(4pi+0.48pi) = (14+0.72)/4.48 = 3.285714...
        self.assertAlmostEqual(expected, 3.2857142857142856)
        self.assertAlmostEqual(table_static_stability_zcm(R, r, H, h, rho), expected)

    def test_table_zcm_matches_assembly_com(self):
        R, r, H, h, rho = 2.0, 0.2, 1.0, 3.0, 1.0
        asm = radially_symmetric_table(R, r, H, h, rho)
        cx, cy, cz = asm.center_of_mass
        self.assertAlmostEqual(cz, table_static_stability_zcm(R, r, H, h, rho))

    def test_table_symmetry_x_y_cm_zero(self):
        asm = radially_symmetric_table(2.0, 0.2, 1.0, 3.0)
        cx, cy, _ = asm.center_of_mass
        self.assertAlmostEqual(cx, 0.0)
        self.assertAlmostEqual(cy, 0.0)

    def test_zcm_monotonic_increasing_in_tabletop_height(self):
        # Increasing tabletop height H raises z_cm -> minimizing H lowers CoM.
        base = table_static_stability_zcm(2.0, 0.2, 1.0, 3.0)
        taller = table_static_stability_zcm(2.0, 0.2, 2.0, 3.0)
        tallest = table_static_stability_zcm(2.0, 0.2, 4.0, 3.0)
        self.assertLess(base, taller)
        self.assertLess(taller, tallest)

    def test_density_cancels_in_zcm(self):
        z1 = table_static_stability_zcm(2.0, 0.2, 1.0, 3.0, rho=1.0)
        z2 = table_static_stability_zcm(2.0, 0.2, 1.0, 3.0, rho=7.5)
        self.assertAlmostEqual(z1, z2)

    def test_aabb_and_footprint(self):
        asm = Assembly([Box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)])
        lo, hi = asm.aabb
        self.assertEqual(lo, (-1.0, -1.0, -1.0))
        self.assertEqual(hi, (1.0, 1.0, 1.0))
        self.assertEqual(asm.footprint_centers, ((0.0, 0.0),))

    def test_empty_assembly_raises(self):
        asm = Assembly([])
        with self.assertRaises(ValueError):
            _ = asm.center_of_mass
        with self.assertRaises(ValueError):
            asm.mass_properties()
        with self.assertRaises(ValueError):
            _ = asm.aabb

    def test_non_positive_dimensions_raise(self):
        with self.assertRaises(ValueError):
            Box(0.0, 0.0, 0.0, -1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            Box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, density=0.0)
        with self.assertRaises(ValueError):
            Cylinder(0.0, 0.0, 0.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            Cylinder(0.0, 0.0, 0.0, 1.0, -1.0)
        with self.assertRaises(ValueError):
            Cylinder(0.0, 0.0, 0.0, 1.0, 1.0, axis="w")
        with self.assertRaises(ValueError):
            table_static_stability_zcm(-1.0, 0.2, 1.0, 3.0)


if __name__ == "__main__":
    unittest.main()
