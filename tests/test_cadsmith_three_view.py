import math
import unittest

from generation.cadsmith_three_view import (
    ViewSpec, ISOMETRIC, HIGH_ANGLE_REAR, FRONT_PROFILE, THREE_VIEWS,
    view_by_name, all_directions, render_resolution, fit_distance,
    RENDER_WIDTH, RENDER_HEIGHT,
)


class TestViewSpecs(unittest.TestCase):
    def test_three_canonical_views(self):
        self.assertEqual([v.name for v in THREE_VIEWS],
                         ["isometric", "high_angle_rear", "front_profile"])

    def test_angles(self):
        self.assertEqual((ISOMETRIC.elevation_deg, ISOMETRIC.azimuth_deg), (35.0, 45.0))
        self.assertEqual((HIGH_ANGLE_REAR.elevation_deg, HIGH_ANGLE_REAR.azimuth_deg),
                         (65.0, 220.0))
        self.assertEqual((FRONT_PROFILE.elevation_deg, FRONT_PROFILE.azimuth_deg),
                         (10.0, 0.0))

    def test_lookup_by_name(self):
        self.assertIs(view_by_name("isometric"), ISOMETRIC)
        with self.assertRaises(KeyError):
            view_by_name("nope")


class TestDirection(unittest.TestCase):
    def test_unit_length(self):
        for v in THREE_VIEWS:
            d = v.direction()
            self.assertAlmostEqual(math.sqrt(sum(c * c for c in d)), 1.0)

    def test_front_profile_direction(self):
        # elevation 10, azimuth 0 -> mostly +X, small +Z, ~zero Y.
        d = FRONT_PROFILE.direction()
        self.assertAlmostEqual(d[1], 0.0)
        self.assertGreater(d[0], 0.9)
        self.assertGreater(d[2], 0.0)

    def test_isometric_symmetry(self):
        # azimuth 45 -> x == y.
        d = ISOMETRIC.direction()
        self.assertAlmostEqual(d[0], d[1])

    def test_high_angle_rear_points_back(self):
        # azimuth 220 is in the third quadrant -> negative x and y.
        d = HIGH_ANGLE_REAR.direction()
        self.assertLess(d[0], 0.0)
        self.assertLess(d[1], 0.0)
        self.assertGreater(d[2], 0.0)

    def test_all_directions_deterministic(self):
        self.assertEqual(all_directions(), all_directions())


class TestCamera(unittest.TestCase):
    def test_camera_position(self):
        pos = FRONT_PROFILE.camera_position((1.0, 2.0, 3.0), distance=10.0)
        d = FRONT_PROFILE.direction()
        self.assertAlmostEqual(pos[0], 1.0 + 10.0 * d[0])

    def test_bad_distance(self):
        with self.assertRaises(ValueError):
            ISOMETRIC.camera_position(distance=0.0)


class TestRender(unittest.TestCase):
    def test_resolution(self):
        self.assertEqual(render_resolution(), (2400, 800))
        self.assertEqual((RENDER_WIDTH, RENDER_HEIGHT), (2400, 800))

    def test_fit_distance_scales_with_size(self):
        d1 = fit_distance((10.0, 10.0, 10.0))
        d2 = fit_distance((20.0, 20.0, 20.0))
        self.assertAlmostEqual(d2, 2 * d1)

    def test_fit_distance_bad_margin(self):
        with self.assertRaises(ValueError):
            fit_distance((1.0, 1.0, 1.0), margin=0.0)


if __name__ == "__main__":
    unittest.main()
