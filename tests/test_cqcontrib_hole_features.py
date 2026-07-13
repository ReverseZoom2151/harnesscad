import math
import unittest

from harnesscad.domain.geometry.features.cqcontrib_hole_features import (
    HoleError,
    counterbore_hole,
    countersink_depth,
    countersink_hole,
    hole_breaks_wall,
    profile_points,
    profile_volume,
    simple_hole,
)


class TestSimpleHole(unittest.TestCase):
    def test_volume_is_cylinder(self):
        f = simple_hole(4.0, 10.0)
        self.assertAlmostEqual(f.volume, math.pi * 4.0 * 10.0)
        self.assertAlmostEqual(f.max_radius, 2.0)

    def test_profile(self):
        self.assertEqual(simple_hole(2.0, 5.0).profile(),
                         [(1.0, 0.0), (1.0, -5.0)])

    def test_bad_params(self):
        with self.assertRaises(HoleError):
            simple_hole(0.0, 5.0)
        with self.assertRaises(HoleError):
            simple_hole(2.0, -1.0)


class TestCounterbore(unittest.TestCase):
    def test_volume(self):
        # contrib Parametric_Enclosure: screwpostID=4, bore=8, boreDepth=1, depth=6
        f = counterbore_hole(4.0, 8.0, 1.0, 6.0)
        expect = math.pi * 16.0 * 1.0 + math.pi * 4.0 * 5.0
        self.assertAlmostEqual(f.volume, expect)
        self.assertEqual(f.kind, "cbore")

    def test_profile_steps(self):
        pts = counterbore_hole(4.0, 8.0, 1.0, 6.0).profile()
        self.assertEqual(pts, [(4.0, 0.0), (4.0, -1.0), (2.0, -1.0), (2.0, -6.0)])

    def test_validation(self):
        with self.assertRaises(HoleError):
            counterbore_hole(8.0, 8.0, 1.0, 6.0)
        with self.assertRaises(HoleError):
            counterbore_hole(4.0, 8.0, 6.0, 6.0)
        with self.assertRaises(HoleError):
            counterbore_hole(4.0, 8.0, 0.0, 6.0)


class TestCountersink(unittest.TestCase):
    def test_depth_90_degrees(self):
        # 90 deg included -> depth equals radial difference
        self.assertAlmostEqual(countersink_depth(4.0, 8.0, 90.0), 2.0)

    def test_depth_60_degrees(self):
        d = countersink_depth(2.0, 4.0, 60.0)
        self.assertAlmostEqual(d, 1.0 / math.tan(math.radians(30.0)))

    def test_cone_plus_cylinder_volume(self):
        f = countersink_hole(4.0, 8.0, 90.0, 10.0)
        h = 2.0
        cone = math.pi * h * (16.0 + 8.0 + 4.0) / 3.0
        cyl = math.pi * 4.0 * (10.0 - h)
        self.assertAlmostEqual(f.volume, cone + cyl)

    def test_profile_is_taper(self):
        pts = countersink_hole(4.0, 8.0, 90.0, 10.0).profile()
        self.assertEqual(pts[0], (4.0, 0.0))
        self.assertAlmostEqual(pts[1][0], 2.0)
        self.assertAlmostEqual(pts[1][1], -2.0)

    def test_validation(self):
        with self.assertRaises(HoleError):
            countersink_depth(4.0, 8.0, 0.0)
        with self.assertRaises(HoleError):
            countersink_depth(4.0, 8.0, 180.0)
        with self.assertRaises(HoleError):
            countersink_hole(4.0, 8.0, 90.0, 1.0)  # cone deeper than hole


class TestProfileUtils(unittest.TestCase):
    def test_profile_volume_matches_feature(self):
        f = counterbore_hole(3.0, 6.0, 2.0, 8.0)
        self.assertAlmostEqual(profile_volume(f.sections), f.volume)

    def test_profile_points_monotone_z(self):
        pts = profile_points(((2.0, 2.0, 1.0), (1.0, 1.0, 3.0)))
        zs = [z for _, z in pts]
        self.assertEqual(zs, sorted(zs, reverse=True))

    def test_breaks_wall(self):
        f = simple_hole(3.0, 6.0)
        self.assertTrue(hole_breaks_wall(f, 6.0))
        self.assertFalse(hole_breaks_wall(f, 7.0))
        with self.assertRaises(HoleError):
            hole_breaks_wall(f, 0.0)


if __name__ == "__main__":
    unittest.main()
