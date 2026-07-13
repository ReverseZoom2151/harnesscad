import math
import unittest

from harnesscad.domain.reconstruction.sketch import pointercad_sketchplane as sp


class DirectionMapTest(unittest.TestCase):
    def test_table14_entries(self):
        # spot-check Table 14: X+ -> aux Y+, X- -> aux Z+, Z- -> aux Y+
        self.assertEqual(sp.direction_vectors("X+"), ((1, 0, 0), (0, 1, 0)))
        self.assertEqual(sp.direction_vectors("X-"), ((-1, 0, 0), (0, 0, 1)))
        self.assertEqual(sp.direction_vectors("Z-"), ((0, 0, -1), (0, 1, 0)))

    def test_unknown_symbol(self):
        with self.assertRaises(sp.SketchPlaneError):
            sp.direction_vectors("W+")

    def test_all_six_present(self):
        self.assertEqual(set(sp.DIRECTION_MAP), {"X+", "X-", "Y+", "Y-", "Z+", "Z-"})


class OrientNormalTest(unittest.TestCase):
    def test_normal_flipped_to_match_primary(self):
        # face normal points -Z, dr=Z+ -> W should flip to +Z
        w = sp.orient_normal((0, 0, -1), "Z+")
        self.assertEqual(w, (0.0, 0.0, 1.0))

    def test_normal_kept_when_aligned(self):
        w = sp.orient_normal((0, 0, 2), "Z+")
        self.assertEqual(w, (0.0, 0.0, 1.0))


class BuildFrameTest(unittest.TestCase):
    def test_frame_orthonormal_and_right_handed(self):
        for sym in sp.DIRECTION_MAP:
            primary, _ = sp.direction_vectors(sym)
            frame = sp.build_frame(primary, sym)
            self.assertTrue(sp.is_orthonormal(frame), sym)

    def test_w_aligns_with_primary(self):
        frame = sp.build_frame((0, 0, 1), "Z+")
        self.assertEqual(frame.w, (0.0, 0.0, 1.0))
        # U' is aux X+ projected -> X axis
        self.assertAlmostEqual(frame.u[0], 1.0)

    def test_rotation_90_maps_u_to_v(self):
        base = sp.build_frame((0, 0, 1), "Z+", rotation_deg=0.0)
        rot = sp.build_frame((0, 0, 1), "Z+", rotation_deg=90.0)
        # after +90 CCW, new U should equal old V
        for i in range(3):
            self.assertAlmostEqual(rot.u[i], base.v[i], places=9)

    def test_scale_applied_in_lift(self):
        frame = sp.build_frame((0, 0, 1), "Z+", scale=2.0)
        p = frame.to_world(1.0, 0.0)
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in p)), 2.0)

    def test_to_world_lifts_into_plane(self):
        frame = sp.build_frame((0, 0, 1), "Z+")
        p = frame.to_world(3.0, 4.0)
        self.assertAlmostEqual(p[2], 0.0)  # lies in z=0 plane
        self.assertAlmostEqual(math.sqrt(p[0] ** 2 + p[1] ** 2), 5.0)

    def test_invalid_scale(self):
        with self.assertRaises(sp.SketchPlaneError):
            sp.build_frame((0, 0, 1), "Z+", scale=0.0)


if __name__ == "__main__":
    unittest.main()
