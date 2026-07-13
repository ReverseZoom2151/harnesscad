"""Tests for geometry.gaussiancad_camera."""

from __future__ import annotations

import unittest
from math import isclose

from harnesscad.domain.geometry.views import camera as cam


def _det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


class TestRotations(unittest.TestCase):
    def test_rot_z_90(self):
        r = cam.rot_z(90.0)
        # x-axis maps to y-axis
        self.assertAlmostEqual(r[0][0], 0.0, places=9)
        self.assertAlmostEqual(r[1][0], 1.0, places=9)

    def test_each_axis_rotation_is_proper(self):
        for f in (cam.rot_x(37.0), cam.rot_y(-12.0), cam.rot_z(200.0)):
            self.assertAlmostEqual(_det3(f), 1.0, places=9)

    def test_euler_zyx_composition(self):
        r = cam.euler_zyx_to_matrix(10.0, 20.0, 30.0)
        self.assertAlmostEqual(_det3(r), 1.0, places=9)
        # orthonormal
        for i in range(3):
            col = [r[k][i] for k in range(3)]
            self.assertAlmostEqual(sum(c * c for c in col), 1.0, places=9)

    def test_euler_zero_is_identity(self):
        r = cam.euler_zyx_to_matrix(0.0, 0.0, 0.0)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(r[i][j], 1.0 if i == j else 0.0, places=12)


class TestIntrinsic(unittest.TestCase):
    def test_paper_intrinsic(self):
        k = cam.GAUSSIANCAD_INTRINSIC
        self.assertEqual(k[0][0], 2480.0)
        self.assertEqual(k[1][1], 2080.0)
        self.assertEqual(k[0][2], 960.0)
        self.assertEqual(k[1][2], 540.0)
        self.assertEqual(k[2][2], 1.0)

    def test_intrinsic_shape(self):
        k = cam.intrinsic_matrix(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(k[0][1], 0.0)
        self.assertEqual(k[2], (0.0, 0.0, 1.0))


class TestExtrinsic(unittest.TestCase):
    def test_extrinsic_assembly(self):
        r = cam.euler_zyx_to_matrix(0.0, 0.0, 0.0)
        t = cam.extrinsic_matrix(r, (1.0, 2.0, 3.0))
        self.assertEqual(t[0][3], 1.0)
        self.assertEqual(t[1][3], 2.0)
        self.assertEqual(t[2][3], 3.0)
        self.assertEqual(t[3], (0.0, 0.0, 0.0, 1.0))

    def test_bad_translation(self):
        r = cam.euler_zyx_to_matrix(0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            cam.extrinsic_matrix(r, (1.0, 2.0))


class TestThreeView(unittest.TestCase):
    def test_view_names(self):
        self.assertEqual(cam.THREE_VIEW_NAMES, ("front", "left", "bottom"))

    def test_three_view_rotations_proper(self):
        for name in cam.THREE_VIEW_NAMES:
            r = cam.three_view_rotation(name)
            self.assertAlmostEqual(_det3(r), 1.0, places=9)

    def test_three_view_extrinsic_translation(self):
        t = cam.three_view_extrinsic("front")
        self.assertEqual((t[0][3], t[1][3], t[2][3]), (0.0, 0.0, 5.0))
        t = cam.three_view_extrinsic("left")
        self.assertEqual((t[0][3], t[1][3], t[2][3]), (0.0, -5.0, 0.0))

    def test_unknown_view_raises(self):
        with self.assertRaises(ValueError):
            cam.three_view_rotation("rear")


class TestBlenderColmap(unittest.TestCase):
    def test_involution(self):
        r = cam.three_view_rotation("front")
        t = (1.0, 2.0, 3.0)
        r2, t2 = cam.blender_to_colmap(r, t)
        r3, t3 = cam.blender_to_colmap(r2, t2)
        for i in range(3):
            self.assertAlmostEqual(t3[i], t[i], places=12)
            for j in range(3):
                self.assertAlmostEqual(r3[i][j], r[i][j], places=12)

    def test_flips_y_and_z(self):
        r = cam.euler_zyx_to_matrix(0.0, 0.0, 0.0)
        r2, t2 = cam.blender_to_colmap(r, (1.0, 2.0, 3.0))
        self.assertEqual(t2, (1.0, -2.0, -3.0))


class TestProjectPoint(unittest.TestCase):
    def test_principal_axis_maps_to_principal_point(self):
        r = cam.euler_zyx_to_matrix(0.0, 0.0, 0.0)
        # point straight ahead at z=1 -> pixel at principal point (960,540)
        u, v = cam.project_point((0.0, 0.0, 1.0), cam.GAUSSIANCAD_INTRINSIC, r, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(u, 960.0, places=6)
        self.assertAlmostEqual(v, 540.0, places=6)

    def test_offset_moves_pixel(self):
        r = cam.euler_zyx_to_matrix(0.0, 0.0, 0.0)
        u, _ = cam.project_point((1.0, 0.0, 2.0), cam.GAUSSIANCAD_INTRINSIC, r, (0.0, 0.0, 0.0))
        # x=1 at depth 2: u = 960 + 2480*1/2
        self.assertAlmostEqual(u, 960.0 + 1240.0, places=6)

    def test_w_zero_raises(self):
        r = cam.euler_zyx_to_matrix(0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            cam.project_point((1.0, 1.0, 0.0), cam.GAUSSIANCAD_INTRINSIC, r, (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
