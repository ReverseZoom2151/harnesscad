"""Tests for FDM overhang detection and orientation search (AgentsCAD)."""

import math
import unittest

from harnesscad.domain.fabrication import overhang as oh


# A normal at 143 deg from +z (points down-and-out) is an actionable overhang;
# arccos(-0.8) ~= 143.1 deg > 90 + 45.
STEEP_UNDER = (0.6, 0.0, -0.8)


class AngleTest(unittest.TestCase):
    def test_up_face_zero_angle(self):
        self.assertAlmostEqual(oh.overhang_angle_deg((0, 0, 1)), 0.0)

    def test_side_face_ninety(self):
        self.assertAlmostEqual(oh.overhang_angle_deg((1, 0, 0)), 90.0)

    def test_down_face_180(self):
        self.assertAlmostEqual(oh.overhang_angle_deg((0, 0, -1)), 180.0)


class IsOverhangTest(unittest.TestCase):
    def test_vertical_wall_not_overhang(self):
        self.assertFalse(oh.is_overhang((1, 0, 0)))

    def test_steep_downward_is_overhang(self):
        self.assertTrue(oh.is_overhang(STEEP_UNDER))

    def test_bed_face_excluded(self):
        self.assertFalse(oh.is_overhang((0, 0, -1)))

    def test_threshold_boundary_not_past_45(self):
        # a face exactly at 135 deg from up (= 45 deg overhang) is NOT past threshold.
        n = (0.0, math.sin(math.radians(45)), -math.cos(math.radians(45)))
        self.assertFalse(oh.is_overhang(n, threshold_deg=45.0))


class OverhangFacesTest(unittest.TestCase):
    def test_flags_and_areas(self):
        faces = [
            {"id": "top", "normal": (0, 0, 1), "area": 10.0},
            {"id": "wall", "normal": (1, 0, 0), "area": 5.0},
            {"id": "under", "normal": STEEP_UNDER, "area": 4.0},
        ]
        flagged = oh.overhang_faces(faces)
        ids = {f.face_id for f in flagged}
        self.assertEqual(ids, {"under"})

    def test_missing_normal_raises(self):
        with self.assertRaises(ValueError):
            oh.overhang_faces([{"id": "x", "area": 1.0}])


class OrientationTest(unittest.TestCase):
    def test_best_orientation_minimizes_overhang(self):
        # A single down-and-out patch: flipping the build direction removes it.
        faces = [{"id": "under", "normal": STEEP_UNDER, "area": 9.0}]
        _direction, area = oh.best_orientation(faces)
        self.assertEqual(area, 0.0)

    def test_deterministic(self):
        faces = [{"id": "a", "normal": (0.2, 0.3, -0.93), "area": 3.0}]
        self.assertEqual(oh.best_orientation(faces), oh.best_orientation(faces))


class StabilityTest(unittest.TestCase):
    def test_radius_of_gyration(self):
        self.assertAlmostEqual(oh.radius_of_gyration(4.0, 16.0), 2.0)

    def test_elongation_index_symmetric(self):
        self.assertAlmostEqual(oh.elongation_index(5.0, 5.0), 1.0)

    def test_elongation_index_lopsided(self):
        self.assertAlmostEqual(oh.elongation_index(9.0, 1.0), 3.0)

    def test_bad_area(self):
        with self.assertRaises(ValueError):
            oh.radius_of_gyration(0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
