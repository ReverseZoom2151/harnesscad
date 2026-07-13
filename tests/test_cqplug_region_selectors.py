"""Tests for geometry.cqplug_region_selectors."""

import math
import unittest

from harnesscad.domain.geometry.cqplug_region_selectors import (
    CylinderRegion,
    HollowCylinderRegion,
    HollowInfiniteCylinderRegion,
    HollowSphereRegion,
    InfiniteCylinderRegion,
    RegionError,
    SphereRegion,
    axis_vector,
    orthogonal_vector,
    select,
)


class TestAxisVector(unittest.TestCase):
    def test_named_axes(self):
        self.assertEqual(axis_vector("X"), (1.0, 0.0, 0.0))
        self.assertEqual(axis_vector("-Z"), (0.0, 0.0, -1.0))

    def test_tuple_is_normalised(self):
        v = axis_vector((0.0, 0.0, 5.0))
        self.assertAlmostEqual(v[2], 1.0, places=12)

    def test_unknown_name_raises(self):
        with self.assertRaises(RegionError):
            axis_vector("Q")

    def test_zero_vector_raises(self):
        with self.assertRaises(RegionError):
            axis_vector((0.0, 0.0, 0.0))


class TestOrthogonalVector(unittest.TestCase):
    def test_is_unit_and_perpendicular(self):
        for axis in ["X", "Y", "Z", (1.0, 2.0, 3.0), (-4.0, 0.5, 2.0)]:
            hat = axis_vector(axis)
            o = orthogonal_vector(axis)
            self.assertAlmostEqual(math.sqrt(sum(c * c for c in o)), 1.0, places=12)
            dot = sum(a * b for a, b in zip(hat, o))
            self.assertAlmostEqual(dot, 0.0, places=12)


class TestInfiniteCylinder(unittest.TestCase):
    def setUp(self):
        self.reg = InfiniteCylinderRegion((0.0, 0.0, 0.0), "Z", 2.0)

    def test_on_axis_is_inside_regardless_of_height(self):
        self.assertTrue(self.reg.contains((0.0, 0.0, 1000.0)))
        self.assertTrue(self.reg.contains((0.0, 0.0, -1000.0)))

    def test_radial_boundary_is_strict(self):
        self.assertTrue(self.reg.contains((1.9, 0.0, 0.0)))
        self.assertFalse(self.reg.contains((2.0, 0.0, 0.0)))
        self.assertFalse(self.reg.contains((2.1, 0.0, 0.0)))

    def test_offset_origin_and_x_axis(self):
        reg = InfiniteCylinderRegion((10.0, 0.0, 0.0), "X", 1.0)
        # radial distance measured perpendicular to X
        self.assertTrue(reg.contains((-50.0, 0.5, 0.5)))
        self.assertFalse(reg.contains((-50.0, 0.8, 0.8)))  # rho ~1.13

    def test_bad_radius(self):
        with self.assertRaises(RegionError):
            InfiniteCylinderRegion((0, 0, 0), "Z", 0.0)


class TestHollowInfiniteCylinder(unittest.TestCase):
    def setUp(self):
        self.reg = HollowInfiniteCylinderRegion((0.0, 0.0, 0.0), "Z", 3.0, 1.0)

    def test_annulus(self):
        self.assertFalse(self.reg.contains((0.5, 0.0, 0.0)))   # inside hole
        self.assertTrue(self.reg.contains((2.0, 0.0, 0.0)))    # in wall
        self.assertFalse(self.reg.contains((3.5, 0.0, 0.0)))   # outside

    def test_radii_validation(self):
        with self.assertRaises(RegionError):
            HollowInfiniteCylinderRegion((0, 0, 0), "Z", 1.0, 2.0)


class TestFiniteCylinder(unittest.TestCase):
    def setUp(self):
        self.reg = CylinderRegion((0.0, 0.0, 0.0), "Z", 10.0, 2.0)

    def test_height_bounds_are_strict_and_one_sided(self):
        self.assertTrue(self.reg.contains((0.0, 0.0, 5.0)))
        self.assertFalse(self.reg.contains((0.0, 0.0, 0.0)))    # h == 0
        self.assertFalse(self.reg.contains((0.0, 0.0, 10.0)))   # h == height
        self.assertFalse(self.reg.contains((0.0, 0.0, -1.0)))   # below origin

    def test_radius_and_height_together(self):
        self.assertTrue(self.reg.contains((1.5, 0.0, 5.0)))
        self.assertFalse(self.reg.contains((2.5, 0.0, 5.0)))

    def test_validation(self):
        with self.assertRaises(RegionError):
            CylinderRegion((0, 0, 0), "Z", 0.0, 2.0)


class TestHollowCylinder(unittest.TestCase):
    def setUp(self):
        self.reg = HollowCylinderRegion((0.0, 0.0, 0.0), "Z", 10.0, 3.0, 1.0)

    def test_tube(self):
        self.assertTrue(self.reg.contains((2.0, 0.0, 5.0)))
        self.assertFalse(self.reg.contains((0.5, 0.0, 5.0)))   # inside hole
        self.assertFalse(self.reg.contains((2.0, 0.0, 20.0)))  # above


class TestSphere(unittest.TestCase):
    def setUp(self):
        self.reg = SphereRegion((1.0, 2.0, 3.0), 5.0)

    def test_membership(self):
        self.assertTrue(self.reg.contains((1.0, 2.0, 3.0)))
        self.assertTrue(self.reg.contains((1.0, 2.0, 7.9)))
        self.assertFalse(self.reg.contains((1.0, 2.0, 8.0)))   # exactly r
        self.assertFalse(self.reg.contains((1.0, 2.0, 9.0)))


class TestHollowSphere(unittest.TestCase):
    def setUp(self):
        self.reg = HollowSphereRegion((0.0, 0.0, 0.0), 5.0, 2.0)

    def test_shell(self):
        self.assertFalse(self.reg.contains((1.0, 0.0, 0.0)))
        self.assertTrue(self.reg.contains((3.0, 0.0, 0.0)))
        self.assertFalse(self.reg.contains((6.0, 0.0, 0.0)))


class TestFilterAndSelect(unittest.TestCase):
    def test_filter_preserves_order(self):
        reg = SphereRegion((0.0, 0.0, 0.0), 2.0)
        pts = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        self.assertEqual(reg.filter(pts), [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])

    def test_select_with_center_accessor(self):
        class Shape:
            def __init__(self, c):
                self.c = c

        reg = InfiniteCylinderRegion((0.0, 0.0, 0.0), "Z", 1.0)
        shapes = [Shape((0.0, 0.0, 5.0)), Shape((5.0, 0.0, 0.0))]
        kept = select(reg, shapes, center=lambda s: s.c)
        self.assertEqual([s.c for s in kept], [(0.0, 0.0, 5.0)])

    def test_select_identity_default(self):
        reg = SphereRegion((0.0, 0.0, 0.0), 2.0)
        self.assertEqual(select(reg, [(0.0, 0.0, 0.0), (9.0, 0.0, 0.0)]),
                         [(0.0, 0.0, 0.0)])


if __name__ == "__main__":
    unittest.main()
