"""Tests for port/mate assembly interfaces (ASSEMCAD / ArtiCAD)."""

import math
import unittest

from harnesscad.domain.geometry.assembly import mates as m


class PortTest(unittest.TestCase):
    def test_frame_is_orthonormal(self):
        p = m.Port("a", (1.0, 2.0, 3.0), z_axis=(0.0, 0.0, 2.0), x_axis=(3.0, 0.0, 0.0), kind="flat_face")
        # x and z normalized
        self.assertAlmostEqual(math.hypot(*p.z_axis), 1.0)
        self.assertAlmostEqual(math.hypot(*p.x_axis), 1.0)
        f = m.port_frame(p)
        self.assertEqual((f[3], f[7], f[11]), (1.0, 2.0, 3.0))

    def test_x_orthogonalized_against_z(self):
        # x deliberately not orthogonal to z -> Gram-Schmidt fixes it.
        p = m.Port("a", (0, 0, 0), z_axis=(0, 0, 1), x_axis=(1, 0, 1))
        self.assertAlmostEqual(p.x_axis[2], 0.0, places=9)

    def test_bad_type_rejected(self):
        with self.assertRaises(ValueError):
            m.Port("a", (0, 0, 0), kind="nonsense")


class FrameMathTest(unittest.TestCase):
    def test_invert_round_trip(self):
        p = m.Port("a", (2.0, -1.0, 0.5), z_axis=(0, 1, 0), x_axis=(1, 0, 0))
        f = m.port_frame(p)
        prod = m.compose(f, m.invert_frame(f))
        ident = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
        for a, b in zip(prod, ident):
            self.assertAlmostEqual(a, b, places=9)


class MateTransformTest(unittest.TestCase):
    def test_face_to_face_makes_axes_antiparallel(self):
        base = m.Port("b", (0, 0, 0), z_axis=(0, 0, 1), x_axis=(1, 0, 0), kind="flat_face")
        inc = m.Port("c", (5, 5, 5), z_axis=(0, 0, 1), x_axis=(1, 0, 0), kind="flat_face")
        T = m.mate_transform(base, inc, flip=True)
        # incoming port origin should map onto base origin
        moved_origin = m.transform_point(T, inc.origin)
        for a in moved_origin:
            self.assertAlmostEqual(a, 0.0, places=9)
        # incoming z (0,0,1) as a direction should become (0,0,-1) (anti-parallel)
        z_dir = (
            T[0] * 0 + T[1] * 0 + T[2] * 1,
            T[4] * 0 + T[5] * 0 + T[6] * 1,
            T[8] * 0 + T[9] * 0 + T[10] * 1,
        )
        self.assertAlmostEqual(z_dir[2], -1.0, places=9)

    def test_transform_is_deterministic(self):
        base = m.Port("b", (1, 2, 3), z_axis=(0, 1, 0), kind="bore")
        inc = m.Port("c", (4, 0, 0), z_axis=(1, 0, 0), kind="shaft_seat")
        self.assertEqual(m.mate_transform(base, inc), m.mate_transform(base, inc))


class CompatibilityTest(unittest.TestCase):
    def test_compatible_ports(self):
        a = m.Port("a", (0, 0, 0), kind="bore")
        b = m.Port("b", (0, 0, 0), kind="shaft_seat")
        self.assertTrue(m.ports_compatible("coaxial", a, b))

    def test_incompatible_ports(self):
        a = m.Port("a", (0, 0, 0), kind="gear_teeth")
        b = m.Port("b", (0, 0, 0), kind="thread_male")
        self.assertFalse(m.ports_compatible("gear_mesh", a, b))

    def test_mate_is_valid_and_dangling(self):
        ports = {
            "p1": m.Port("p1", (0, 0, 0), kind="flat_face"),
            "p2": m.Port("p2", (0, 0, 1), kind="flat_face"),
        }
        ok, reason = m.mate_is_valid(m.Mate("face_to_face", "p1", "p2"), ports)
        self.assertTrue(ok)
        self.assertEqual(reason, "")
        ok2, reason2 = m.mate_is_valid(m.Mate("face_to_face", "p1", "missing"), ports)
        self.assertFalse(ok2)
        self.assertIn("unresolved", reason2)

    def test_type_mismatch_rejected(self):
        ports = {
            "p1": m.Port("p1", (0, 0, 0), kind="gear_teeth"),
            "p2": m.Port("p2", (0, 0, 1), kind="flat_face"),
        }
        ok, reason = m.mate_is_valid(m.Mate("gear_mesh", "p1", "p2"), ports)
        self.assertFalse(ok)
        self.assertIn("incompatible", reason)

    def test_contact_mate_set(self):
        self.assertIn("press_fit", m.CONTACT_MATES)
        self.assertNotIn("face_to_face", m.CONTACT_MATES)


if __name__ == "__main__":
    unittest.main()
