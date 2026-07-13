import math
import unittest

from harnesscad.domain.geometry.cq_plane_frame import Plane, PlaneError


def close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


class TestNamedPresets(unittest.TestCase):
    def test_xy(self):
        p = Plane.named("XY")
        self.assertTrue(close(p.xDir, (1, 0, 0)))
        self.assertTrue(close(p.yDir, (0, 1, 0)))
        self.assertTrue(close(p.zDir, (0, 0, 1)))

    def test_yz(self):
        p = Plane.named("YZ")
        self.assertTrue(close(p.xDir, (0, 1, 0)))
        self.assertTrue(close(p.yDir, (0, 0, 1)))
        self.assertTrue(close(p.zDir, (1, 0, 0)))

    def test_zx(self):
        p = Plane.named("ZX")
        self.assertTrue(close(p.xDir, (0, 0, 1)))
        self.assertTrue(close(p.yDir, (1, 0, 0)))
        self.assertTrue(close(p.zDir, (0, 1, 0)))

    def test_xz_ydir(self):
        # table: XZ has yDir +z, zDir -y
        p = Plane.named("XZ")
        self.assertTrue(close(p.yDir, (0, 0, 1)))
        self.assertTrue(close(p.zDir, (0, -1, 0)))

    def test_all_frames_orthonormal(self):
        for name in ("XY", "YZ", "ZX", "XZ", "YX", "ZY",
                     "front", "back", "left", "right", "top", "bottom"):
            p = Plane.named(name)
            self.assertAlmostEqual(sum(a * a for a in p.xDir), 1.0)
            self.assertAlmostEqual(sum(a * a for a in p.zDir), 1.0)
            # right-handed: yDir == zDir x xDir already; check orthogonality
            self.assertAlmostEqual(
                p.xDir[0] * p.zDir[0] + p.xDir[1] * p.zDir[1] + p.xDir[2] * p.zDir[2],
                0.0,
            )

    def test_view_directions(self):
        # front/top/right normals match the reference table
        self.assertTrue(close(Plane.named("front").zDir, (0, 0, 1)))
        self.assertTrue(close(Plane.named("back").zDir, (0, 0, -1)))
        self.assertTrue(close(Plane.named("left").zDir, (-1, 0, 0)))
        self.assertTrue(close(Plane.named("right").zDir, (1, 0, 0)))
        self.assertTrue(close(Plane.named("top").zDir, (0, 1, 0)))
        self.assertTrue(close(Plane.named("bottom").zDir, (0, -1, 0)))

    def test_unknown_name(self):
        with self.assertRaises(PlaneError):
            Plane.named("QQ")


class TestClassmethods(unittest.TestCase):
    def test_classmethod_matches_named(self):
        self.assertEqual(Plane.XY(), Plane.named("XY"))
        self.assertEqual(Plane.top((1, 2, 3)), Plane.named("top", (1, 2, 3)))

    def test_custom_xdir(self):
        p = Plane.XY(xDir=(0, 1, 0))
        self.assertTrue(close(p.xDir, (0, 1, 0)))
        # yDir = zDir x xDir = z x y = -x
        self.assertTrue(close(p.yDir, (-1, 0, 0)))


class TestCoords(unittest.TestCase):
    def test_roundtrip_xy(self):
        p = Plane.XY(origin=(10, 20, 5))
        w = p.toWorldCoords((1, 2))
        self.assertTrue(close(w, (11, 22, 5)))
        back = p.toLocalCoords(w)
        self.assertTrue(close(back, (1, 2, 0)))

    def test_roundtrip_rotated_frame(self):
        p = Plane.YZ(origin=(1, 1, 1))
        local = (3.0, -2.0, 4.0)
        w = p.toWorldCoords(local)
        self.assertTrue(close(p.toLocalCoords(w), local))

    def test_two_tuple_is_z_zero(self):
        p = Plane.named("YZ")
        w = p.toWorldCoords((5, 7))
        # x-local -> +Y, y-local -> +Z
        self.assertTrue(close(w, (0, 5, 7)))

    def test_setorigin2d(self):
        p = Plane.XY()
        p.setOrigin2d(2, 2)
        p.setOrigin2d(2, 2)
        self.assertTrue(close(p.origin, (4, 4, 0)))


class TestRotated(unittest.TestCase):
    def test_rotate_z_keeps_normal(self):
        p = Plane.XY()
        r = p.rotated((0, 0, 90))
        self.assertTrue(close(r.zDir, (0, 0, 1)))
        self.assertTrue(close(r.xDir, (0, 1, 0), tol=1e-9))

    def test_rotate_x_tilts_normal(self):
        p = Plane.XY()
        r = p.rotated((90, 0, 0))
        # z tilts onto -y
        self.assertTrue(close(r.zDir, (0, -1, 0), tol=1e-9))

    def test_origin_unchanged(self):
        p = Plane.XY(origin=(5, 6, 7))
        r = p.rotated((30, 0, 0))
        self.assertTrue(close(r.origin, (5, 6, 7)))


class TestEquality(unittest.TestCase):
    def test_eq_and_ne(self):
        self.assertEqual(Plane.XY(), Plane.XY())
        self.assertNotEqual(Plane.XY(), Plane.YZ())
        self.assertNotEqual(Plane.XY(origin=(0, 0, 1)), Plane.XY())

    def test_arbitrary_normal_autoxdir(self):
        p = Plane(origin=(0, 0, 0), normal=(0, 0, 1))
        self.assertAlmostEqual(
            p.xDir[0] ** 2 + p.xDir[1] ** 2 + p.xDir[2] ** 2, 1.0
        )
        self.assertAlmostEqual(
            p.xDir[0] * p.zDir[0] + p.xDir[1] * p.zDir[1] + p.xDir[2] * p.zDir[2], 0.0
        )

    def test_null_normal(self):
        with self.assertRaises(PlaneError):
            Plane(normal=(0, 0, 0))


if __name__ == "__main__":
    unittest.main()
